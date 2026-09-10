"""OpenAI Batch API vision pipeline v2 — batch-exclusive, photo-level, prompt-once.

Design (see also worker.py orchestration):
- ALL photo analysis flows through the Batch API; the sync path never analyzes photos.
- Work is packed at the PHOTO level: a property's photos may span several batches.
  Validated per-photo results accumulate in `vision_results`, and the worker ingests a
  property only when EVERY photo it currently needs has exactly one result.
- Token saving: each request carries the shared system prompt ONCE plus a group of up
  to vision_group_max_images images labeled "IMAGE 1..N"; the model must echo each
  image's index in its JSON reply.
- Exact matching (the critical invariant): request-level identity via custom_id plus an
  ordered URL manifest persisted per batch; within a request, the index-echo reply is
  validated by parse_group_output — exactly one entry per index or the WHOLE group is
  rejected and retried at a smaller size (group → 5 → single → Unknown stub after
  _MAX_GROUP_ATTEMPTS), never guessed.
- Queue budget: batches of ≤ vision_batch_max_tokens ESTIMATED INPUT tokens (the org
  queue counts input only) are submitted while the total in-flight estimate fits
  vision_batch_queue_tokens; completions free budget and the next wave refills it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

import asyncpg

from config.settings import settings
from src.img_analyzer.analyzer import build_grouped_system_prompt, parse_group_output
from src.img_analyzer.models import PhotoResult
from src.img_analyzer.raw_db import CATALOG_STATUSES, extract_photo_urls_from_data, primary_photo_urls
from src.llm_client import get_async_client

logger = logging.getLogger(__name__)

# Batch API hard limit is 200MB/file; grouped requests are tiny (~20KB) so this never
# binds in practice, kept as a guard.
MAX_BATCH_BYTES = 120 * 1024 * 1024

# Estimated INPUT tokens per high-detail image (observed ~1.5k; padded). The org batch
# queue counts input tokens only (verified live via probes on 2026-07-03).
IMAGE_TOKENS_EST = 1_600

# Escalation ladder for groups that fail matching validation: attempts→group size.
# After _MAX_GROUP_ATTEMPTS failures a photo gets an Unknown stub result so its
# property can complete (mirrors the old sync path's behavior on a dead photo).
_TIER_GROUP_SIZE = {0: None, 1: 5, 2: 1}  # None → settings.vision_group_max_images
_MAX_GROUP_ATTEMPTS = 3

# Per-request output ceiling: ~500 tokens per image + headroom (billed only if used;
# does NOT count toward the queue).
_COMPLETION_PER_IMAGE = 500
_COMPLETION_HEADROOM = 300

_OPEN_STATUSES = ("submitted", "validating", "in_progress", "finalizing", "cancelling")

_CUSTOM_ID_MAX = 64

# After a submit error or a 'failed' batch, pause submissions so a deterministic
# failure can't loop; pending photos simply wait (batch-exclusive: no sync drain).
FAILURE_COOLDOWN_SECONDS = 600.0
_backoff_until_monotonic = 0.0

# Poll throttle for batch status checks (the worker loops every ~5s).
_last_poll_monotonic = 0.0

# Throttle for the queue-exhausted log (fires every worker cycle while full otherwise).
_last_exhausted_log_monotonic = 0.0

_tables_ready = False


async def _ensure_tables(conn: asyncpg.Connection) -> None:
    """Create/migrate the batch tracker, per-photo result store, and attempts ledger."""
    global _tables_ready
    if _tables_ready:
        return
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS vision_batches (
            batch_id     TEXT PRIMARY KEY,
            status       TEXT NOT NULL DEFAULT 'submitted',
            items        JSONB NOT NULL,
            est_tokens   BIGINT NOT NULL DEFAULT 0,
            submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at TIMESTAMPTZ
        );
    """)
    await conn.execute(
        "ALTER TABLE vision_batches ADD COLUMN IF NOT EXISTS est_tokens BIGINT NOT NULL DEFAULT 0;"
    )
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS vision_results (
            property_id TEXT NOT NULL,
            photo_url   TEXT NOT NULL,
            result      JSONB NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (property_id, photo_url)
        );
    """)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS vision_attempts (
            property_id TEXT NOT NULL,
            photo_url   TEXT NOT NULL,
            attempts    INT NOT NULL DEFAULT 0,
            PRIMARY KEY (property_id, photo_url)
        );
    """)
    _tables_ready = True


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

_cached_prompt_tokens: int | None = None


def _prompt_tokens_est() -> int:
    global _cached_prompt_tokens
    if _cached_prompt_tokens is None:
        _cached_prompt_tokens = len(build_grouped_system_prompt(20)) // 4 + 100
    return _cached_prompt_tokens


def _request_tokens_est(n_images: int) -> int:
    """Estimated ENQUEUED (input) tokens for one grouped request."""
    return _prompt_tokens_est() + n_images * IMAGE_TOKENS_EST


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------

def _grouped_request_line(custom_id: str, urls: list[str]) -> str:
    """One JSONL line: shared prompt ONCE + labeled images IMAGE 1..N of one property."""
    content: list[dict] = []
    for i, u in enumerate(urls, 1):
        content.append({"type": "text", "text": f"IMAGE {i}"})
        content.append({"type": "image_url", "image_url": {"url": u, "detail": "high"}})
    return json.dumps({
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": settings.openai_model,
            "max_completion_tokens": _COMPLETION_HEADROOM + _COMPLETION_PER_IMAGE * len(urls),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": build_grouped_system_prompt(len(urls))},
                {"role": "user", "content": content},
            ],
        },
    })


# ---------------------------------------------------------------------------
# Pending work discovery
# ---------------------------------------------------------------------------

async def _pending_rows(conn: asyncpg.Connection) -> list[asyncpg.Record]:
    """Rows whose vision work the batch pipeline owns ('batch_submitted' = legacy rows
    from the previous design, treated like unprocessed)."""
    return await conn.fetch(
        """
        SELECT id, data, status, updated_at FROM raw_properties
        WHERE status IN ('unprocessed', 'partial_image_only_processed', 'batch_submitted')
          AND data->>'homeStatus' = ANY($2)  -- catalog statuses only (see prune_non_for_sale)
        ORDER BY updated_at ASC
        LIMIT $1
        """,
        settings.vision_batch_max_items,
        list(CATALOG_STATUSES),
    )


async def _needed_urls(conn: asyncpg.Connection, row: asyncpg.Record) -> list[str]:
    """Photo URLs this row still requires vision for, in original photo order.
    Full path: all photos. Partial path: only photos not yet in the primary tables."""
    data = row["data"] if isinstance(row["data"], dict) else json.loads(row["data"])
    urls = extract_photo_urls_from_data(data)
    if row["status"] == "partial_image_only_processed":
        have = await primary_photo_urls(conn, row["id"])
        urls = [u for u in urls if u not in have]
    # dedupe, order-preserving
    seen: set[str] = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


async def _resulted_urls(conn: asyncpg.Connection, property_id: str) -> set[str]:
    rows = await conn.fetch(
        "SELECT photo_url FROM vision_results WHERE property_id = $1", property_id
    )
    return {r["photo_url"] for r in rows}


async def _inflight_urls(conn: asyncpg.Connection) -> set[tuple[str, str]]:
    """(property_id, url) pairs currently inside open batches."""
    rows = await conn.fetch(
        "SELECT items FROM vision_batches WHERE status = ANY($1)", list(_OPEN_STATUSES)
    )
    out: set[tuple[str, str]] = set()
    for r in rows:
        items = r["items"] if isinstance(r["items"], dict) else json.loads(r["items"])
        for req in (items.get("requests") or {}).values():
            for u in req.get("urls", []):
                out.add((req["p"], u))
    return out


# ---------------------------------------------------------------------------
# Submission (photo-level packing, queue-budget refill)
# ---------------------------------------------------------------------------

def _max_images_per_request() -> int:
    """Largest group the caps can carry: a request must fit within BOTH the per-batch
    cap and the whole queue budget (prevents unsubmittable groups that would stall)."""
    cap = min(settings.vision_batch_max_tokens, settings.vision_batch_queue_tokens)
    fit = (cap - _prompt_tokens_est()) // IMAGE_TOKENS_EST
    return max(1, min(settings.vision_group_max_images, fit))


def _chunk_by_attempts(
    urls: list[str], attempts: dict[str, int], max_group: int | None = None
) -> tuple[list[list[str]], list[str]]:
    """Split needed urls into request groups sized by their retry tier (clamped so any
    group fits the caps). Returns (groups, exhausted_urls) — exhausted urls get stubs."""
    if max_group is None:
        max_group = _max_images_per_request()
    tiers: dict[int, list[str]] = {}
    exhausted: list[str] = []
    for u in urls:
        a = attempts.get(u, 0)
        if a >= _MAX_GROUP_ATTEMPTS:
            exhausted.append(u)
        else:
            tiers.setdefault(a, []).append(u)
    groups: list[list[str]] = []
    for a, tier_urls in sorted(tiers.items()):
        size = _TIER_GROUP_SIZE.get(a) or settings.vision_group_max_images
        size = max(1, min(size, max_group))
        for i in range(0, len(tier_urls), size):
            groups.append(tier_urls[i:i + size])
    return groups, exhausted


async def submit_waves(pool: asyncpg.Pool) -> int:
    """Pack pending photos into grouped requests and submit as many batches as fit the
    queue budget (fill-until-full; refills as prior batches complete). Returns the
    number of batches submitted."""
    global _backoff_until_monotonic
    if time.monotonic() < _backoff_until_monotonic:
        return 0

    async with pool.acquire() as conn:
        await _ensure_tables(conn)
        in_flight_tokens = await conn.fetchval(
            "SELECT COALESCE(SUM(est_tokens), 0) FROM vision_batches WHERE status = ANY($1)",
            list(_OPEN_STATUSES),
        ) or 0
        headroom = settings.vision_batch_queue_tokens - in_flight_tokens
        if headroom < _request_tokens_est(1):
            return 0  # queue full — wait for a batch to finish

        rows = await _pending_rows(conn)
        if not rows:
            return 0
        inflight = await _inflight_urls(conn)
        max_group = _max_images_per_request()

        # Build the global chunk list (photo-level; properties may span batches).
        chunks: list[tuple[str, list[str]]] = []  # (property_id, urls)
        for row in rows:
            pid = row["id"]
            needed = await _needed_urls(conn, row)
            if not needed:
                continue  # complete or photo-less → worker applies it
            done = await _resulted_urls(conn, pid)
            todo = [u for u in needed if u not in done and (pid, u) not in inflight]
            if not todo:
                continue
            arows = await conn.fetch(
                "SELECT photo_url, attempts FROM vision_attempts WHERE property_id = $1", pid
            )
            attempts = {r["photo_url"]: r["attempts"] for r in arows}
            groups, exhausted = _chunk_by_attempts(todo, attempts, max_group)
            for u in exhausted:
                # Photo failed matching _MAX_GROUP_ATTEMPTS times even solo → stub it
                # so the property can complete (never blocks forever).
                logger.warning("Batch v2: stubbing Unknown for %s photo %s (retries exhausted)", pid, u[:90])
                await conn.execute(
                    """
                    INSERT INTO vision_results (property_id, photo_url, result)
                    VALUES ($1, $2, $3::jsonb) ON CONFLICT (property_id, photo_url) DO NOTHING
                    """,
                    pid, u, json.dumps({"room_type": "Unknown", "color": None, "features": []}),
                )
            chunks.extend((pid, g) for g in groups)

    if not chunks:
        return 0

    # Pack chunks into batches of ≤ vision_batch_max_tokens, submit while headroom lasts.
    submitted = 0
    batch_lines: list[str] = []
    batch_manifest: dict[str, dict] = {}
    batch_props: set[str] = set()
    batch_tokens = 0
    batch_bytes = 0
    seq = 0  # global across this call — custom_ids stay unique through flushes

    async def _flush() -> bool:
        nonlocal batch_lines, batch_manifest, batch_props, batch_tokens, batch_bytes, headroom, submitted
        if not batch_lines:
            return True
        ok = await _create_batch(pool, batch_lines, batch_manifest, batch_props, batch_tokens)
        if ok:
            submitted += 1
            headroom -= batch_tokens
        batch_lines, batch_manifest, batch_props = [], {}, set()
        batch_tokens = batch_bytes = 0
        return ok

    for pid, group in chunks:
        est = _request_tokens_est(len(group))
        # Hash-based custom_id: the manifest maps cid -> (property, urls), so the id
        # needn't embed the (arbitrary-length) property id — 64-char limit safe always.
        cid = f"{hashlib.sha1(pid.encode()).hexdigest()[:16]}#{seq}"
        line = _grouped_request_line(cid, group)
        # Flush the current batch first if this chunk would overflow it (batch cap,
        # queue budget, or the byte guard) — then pack the chunk into the fresh batch.
        if batch_lines and (
            batch_tokens + est > settings.vision_batch_max_tokens
            or batch_tokens + est > headroom
            or batch_bytes + len(line) + 1 > MAX_BATCH_BYTES
        ):
            if not await _flush():
                return submitted  # submit error → backoff set; stop this cycle
        if est > headroom:
            global _last_exhausted_log_monotonic
            if time.monotonic() - _last_exhausted_log_monotonic > 300:
                _last_exhausted_log_monotonic = time.monotonic()
                logger.info("Batch v2: queue budget exhausted (%dk headroom < %dk chunk) — "
                            "later waves take the rest", headroom // 1000, est // 1000)
            break
        batch_lines.append(line)
        batch_manifest[cid] = {"p": pid, "urls": list(group)}
        batch_props.add(pid)
        batch_tokens += est
        batch_bytes += len(line) + 1
        seq += 1

    await _flush()
    return submitted


async def _create_batch(
    pool: asyncpg.Pool, lines: list[str], manifest: dict, props: set[str], est_tokens: int
) -> bool:
    """Upload one JSONL + create the OpenAI batch + persist the manifest."""
    global _backoff_until_monotonic
    client = get_async_client()
    f = None
    try:
        f = await client.files.create(
            file=("vision_batch.jsonl", "\n".join(lines).encode("utf-8")), purpose="batch"
        )
        batch = await client.batches.create(
            input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h"
        )
    except Exception as e:  # noqa: BLE001
        _backoff_until_monotonic = time.monotonic() + FAILURE_COOLDOWN_SECONDS
        logger.error("Batch v2 submit failed (%s); backing off %.0fs", e, FAILURE_COOLDOWN_SECONDS)
        if f is not None:
            try:
                await client.files.delete(f.id)
            except Exception:  # noqa: BLE001
                pass
        return False

    items = {"requests": manifest, "properties": sorted(props)}
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO vision_batches (batch_id, status, items, est_tokens) "
                "VALUES ($1, $2, $3::jsonb, $4)",
                batch.id, batch.status, json.dumps(items), est_tokens,
            )
    except Exception as e:  # noqa: BLE001 — untracked batch would silently eat the queue
        logger.error("Batch v2 %s: manifest INSERT failed (%s); cancelling batch", batch.id, e)
        try:
            await client.batches.cancel(batch.id)
        except Exception:  # noqa: BLE001 — best effort
            pass
        return False
    n_photos = sum(len(r["urls"]) for r in manifest.values())
    logger.info("Batch v2 %s submitted: %d requests / %d photos / %d properties (~%dk est tokens)",
                batch.id, len(manifest), n_photos, len(props), est_tokens // 1000)
    return True


# ---------------------------------------------------------------------------
# Collection: poll → download → validate matching → store per-photo results
# ---------------------------------------------------------------------------

async def collect_completed(pool: asyncpg.Pool) -> bool:
    """Poll open batches (throttled). For each terminal batch, validate every grouped
    reply and store per-photo results; failed groups get attempts++ (smaller groups next
    wave). Returns True if any batch reached a terminal state."""
    global _last_poll_monotonic, _backoff_until_monotonic

    async with pool.acquire() as conn:
        await _ensure_tables(conn)
        open_rows = await conn.fetch(
            "SELECT batch_id, items FROM vision_batches WHERE status = ANY($1)",
            list(_OPEN_STATUSES),
        )
    if not open_rows:
        return False
    if time.monotonic() - _last_poll_monotonic < settings.vision_batch_poll_seconds:
        return False
    _last_poll_monotonic = time.monotonic()

    client = get_async_client()
    any_terminal = False
    for brow in open_rows:
        batch_id = brow["batch_id"]
        items = brow["items"] if isinstance(brow["items"], dict) else json.loads(brow["items"])
        manifest: dict[str, dict] = items.get("requests") or {}
        try:
            batch = await client.batches.retrieve(batch_id)
        except Exception as e:  # noqa: BLE001
            if getattr(e, "status_code", None) == 404:
                logger.warning("Batch v2 %s: gone at OpenAI — photos will repack", batch_id)
                await _mark_terminal(pool, batch_id, "failed")
                any_terminal = True
            else:
                logger.warning("Batch v2 %s: status check failed: %s", batch_id, e)
            continue

        if batch.status in ("validating", "in_progress", "finalizing", "cancelling"):
            counts = getattr(batch, "request_counts", None)
            logger.info("Batch v2 %s: %s (%s/%s requests done)", batch_id, batch.status,
                        getattr(counts, "completed", "?"), getattr(counts, "total", "?"))
            continue

        if batch.status in ("failed", "cancelled"):
            if batch.status == "failed":
                _backoff_until_monotonic = time.monotonic() + FAILURE_COOLDOWN_SECONDS
            logger.warning("Batch v2 %s: %s — %d requests will repack next wave",
                           batch_id, batch.status, len(manifest))
            await _mark_terminal(pool, batch_id, batch.status)
            any_terminal = True
            continue

        # completed (or expired with partial output)
        output_lines: dict[str, dict] = {}
        error_cids: set[str] = set()
        output_file_id = getattr(batch, "output_file_id", None)
        if output_file_id:
            try:
                content = await client.files.content(output_file_id)
                for line in content.text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        output_lines[rec.get("custom_id") or ""] = rec
                    except (json.JSONDecodeError, AttributeError, TypeError):
                        continue
            except Exception as e:  # noqa: BLE001
                if getattr(e, "status_code", None) == 404:
                    logger.warning("Batch v2 %s: output file gone — photos will repack", batch_id)
                    await _mark_terminal(pool, batch_id, "failed")
                    any_terminal = True
                else:
                    logger.warning("Batch v2 %s: output download failed: %s (will retry)", batch_id, e)
                continue
        # Failed requests (dead image URL, per-request API error) land in the ERROR
        # file, not the output file — those groups must climb the retry ladder, or a
        # single dead photo URL would repack at the same size forever.
        error_file_id = getattr(batch, "error_file_id", None)
        if error_file_id:
            try:
                err_content = await client.files.content(error_file_id)
                for line in err_content.text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        error_cids.add(json.loads(line).get("custom_id") or "")
                    except (json.JSONDecodeError, AttributeError, TypeError):
                        continue
            except Exception as e:  # noqa: BLE001
                logger.warning("Batch v2 %s: error-file download failed: %s (will retry)", batch_id, e)
                continue

        # Atomic claim so two replicas can't both store this batch's results.
        async with pool.acquire() as conn:
            claimed = await conn.fetchval(
                "UPDATE vision_batches SET status = $2, completed_at = NOW() "
                "WHERE batch_id = $1 AND status = ANY($3) RETURNING batch_id",
                batch_id, batch.status, list(_OPEN_STATUSES),
            )
        if claimed is None:
            continue
        any_terminal = True

        stored = failed_groups = missing = 0
        async with pool.acquire() as conn:
            for cid, req in manifest.items():
                pid, urls = req["p"], req["urls"]
                rec = output_lines.get(cid)
                if rec is None:
                    if cid in error_cids or batch.status == "completed":
                        # Request-level failure (dead image URL, API error): the group
                        # must climb the ladder (smaller size → stub), not loop as-is.
                        failed_groups += 1
                        for u in urls:
                            await conn.execute(
                                """
                                INSERT INTO vision_attempts (property_id, photo_url, attempts)
                                VALUES ($1, $2, 1)
                                ON CONFLICT (property_id, photo_url)
                                DO UPDATE SET attempts = vision_attempts.attempts + 1
                                """,
                                pid, u,
                            )
                    else:
                        missing += 1  # infra gap (expired mid-run) — repack at the SAME tier
                    continue
                resp = rec.get("response") or {}
                content_text = ""
                if isinstance(resp, dict) and resp.get("status_code") == 200:
                    body = resp.get("body") or {}
                    choices = body.get("choices") or [{}]
                    try:
                        content_text = (choices[0].get("message") or {}).get("content") or ""
                    except (AttributeError, IndexError, TypeError):
                        content_text = ""
                parsed = parse_group_output(content_text, len(urls)) if content_text else None
                if parsed is None:
                    # Matching NOT proven → reject the whole group, retry smaller.
                    failed_groups += 1
                    for u in urls:
                        await conn.execute(
                            """
                            INSERT INTO vision_attempts (property_id, photo_url, attempts)
                            VALUES ($1, $2, 1)
                            ON CONFLICT (property_id, photo_url)
                            DO UPDATE SET attempts = vision_attempts.attempts + 1
                            """,
                            pid, u,
                        )
                    continue
                for u, res in zip(urls, parsed):
                    await conn.execute(
                        """
                        INSERT INTO vision_results (property_id, photo_url, result)
                        VALUES ($1, $2, $3::jsonb)
                        ON CONFLICT (property_id, photo_url) DO UPDATE SET result = EXCLUDED.result
                        """,
                        pid, u, json.dumps(res),
                    )
                    stored += 1
        logger.info("Batch v2 %s: %s — %d photo results stored, %d groups failed matching, %d missing",
                    batch_id, batch.status, stored, failed_groups, missing)
    return any_terminal


async def _mark_terminal(pool: asyncpg.Pool, batch_id: str, status: str) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE vision_batches SET status = $2, completed_at = NOW() WHERE batch_id = $1",
            batch_id, status,
        )


# ---------------------------------------------------------------------------
# Property completion for the worker
# ---------------------------------------------------------------------------

async def ready_properties(pool: asyncpg.Pool) -> list[dict[str, Any]]:
    """Pending rows whose EVERY needed photo has a validated result (incl. photo-less
    rows). Returns [{item_id, status, data, updated_at, results{url: PhotoResult}}]."""
    out: list[dict[str, Any]] = []
    async with pool.acquire() as conn:
        await _ensure_tables(conn)
        rows = await _pending_rows(conn)
        for row in rows:
            needed = await _needed_urls(conn, row)
            results: dict[str, PhotoResult] = {}
            if needed:
                rrows = await conn.fetch(
                    "SELECT photo_url, result FROM vision_results "
                    "WHERE property_id = $1 AND photo_url = ANY($2)",
                    row["id"], needed,
                )
                found = {}
                for r in rrows:
                    res = r["result"] if isinstance(r["result"], dict) else json.loads(r["result"])
                    found[r["photo_url"]] = PhotoResult(
                        photo_url=r["photo_url"],
                        room_type=res.get("room_type") or "Unknown",
                        color=res.get("color"),
                        features=res.get("features") or [],
                    )
                if len(found) < len(needed):
                    continue  # still accumulating
                results = found
            data = row["data"] if isinstance(row["data"], dict) else json.loads(row["data"])
            out.append({
                "item_id": row["id"],
                "status": row["status"],
                "data": data,
                "updated_at": row["updated_at"],
                "results": results,
            })
    return out


async def clear_property(pool: asyncpg.Pool, property_id: str) -> None:
    """Drop accumulated results/attempts after a property is successfully ingested."""
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM vision_results WHERE property_id = $1", property_id)
        await conn.execute("DELETE FROM vision_attempts WHERE property_id = $1", property_id)


async def gc(pool: asyncpg.Pool) -> None:
    """Remove accumulated results for properties no longer pending (deleted/processed)."""
    async with pool.acquire() as conn:
        await _ensure_tables(conn)
        for table in ("vision_results", "vision_attempts"):
            await conn.execute(f"""
                DELETE FROM {table} t
                WHERE NOT EXISTS (
                    SELECT 1 FROM raw_properties rp
                    WHERE rp.id = t.property_id
                      AND rp.status IN ('unprocessed', 'partial_image_only_processed', 'batch_submitted')
                )
            """)


async def recover_when_disabled(pool: asyncpg.Pool) -> int:
    """Requeue legacy 'batch_submitted' rows when the batch flag is off at startup."""
    async with pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE raw_properties SET status = 'unprocessed', updated_at = NOW() "
            "WHERE status = 'batch_submitted'"
        )
    try:
        n = int(result.rsplit(" ", 1)[-1])
    except ValueError:
        n = 0
    if n:
        logger.warning("Batch vision disabled: requeued %d stranded row(s) to the sync path", n)
    return n
