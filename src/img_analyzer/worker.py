"""Background worker: loops forever draining raw_properties → primary tables in WORKER_BATCH_SIZE-row batches, marking rows 'processed' (errors leave them for retry)."""

from __future__ import annotations

import asyncio
import json
import logging

import asyncpg

from config.settings import settings
from src.data.database import get_pool
from src.data.import_pois import ensure_coverage
from src.img_analyzer import batch_vision
from src.img_analyzer.analyzer import (
    _pick_jpeg_url,
    analyze_photos,
    inject_features,
    verify_unit_pool_photos,
)
from src.img_analyzer.db_ingest import (
    _build_rooms_from_photos,
    _canonical_photo_url,
    _ensure_community_pool_tag,
    _extract_property_fields,
    _extract_schools,
    _get_room_counts,
    _insert_children,
    assign_region_ids,
    refresh_property_room_counts,
    update_property_scalars,
    update_property_with_children,
)
from src.img_analyzer.models import Photo, PhotoResult, PropertyItem
from src.img_analyzer.raw_db import (
    claim_pending_batch,
    extract_photo_urls_from_data,
    mark_processed,
    primary_photo_urls,
    prune_non_for_sale,
)

logger = logging.getLogger(__name__)

WORKER_BATCH_SIZE = 1            # rows processed in parallel per iteration
WORKER_IDLE_SLEEP_SECONDS = 5    # sleep when no work
WORKER_ERROR_SLEEP_SECONDS = 10  # sleep after an errored iteration

# Background POI auto-refresh: after new properties land, import POIs for any
# newly-seen county (debounced — one task at a time; no-op when all covered).
_poi_refresh_task: "asyncio.Task | None" = None


def _trigger_poi_refresh(pool: asyncpg.Pool) -> None:
    global _poi_refresh_task
    if _poi_refresh_task and not _poi_refresh_task.done():
        return

    async def _run() -> None:
        try:
            n = await ensure_coverage(pool)
            if n:
                logger.info("Worker: auto-imported POIs for %d new county region(s)", n)
        except Exception as e:  # noqa: BLE001 — never let POI refresh disrupt ingestion
            logger.warning("Worker: POI auto-refresh failed: %s", e)

    _poi_refresh_task = asyncio.create_task(_run())


async def _process_unprocessed(
    pool: asyncpg.Pool, item_id: str, data: dict, expected_updated_at,
    precomputed: dict[str, PhotoResult] | None = None,
) -> bool:
    """Full sync: Vision API → primary tables (INSERT or full re-INSERT of children); returns False if a concurrent /process write raced us.
    precomputed = {photo_url: PhotoResult} from a Batch API run; photos it doesn't cover
    (batch error lines) are analyzed synchronously here."""
    item = {"Id": item_id, "ZillowPropertyRecord": data}

    p = PropertyItem(**item)  # raises on broken structure; caller logs
    photos = p.ZillowPropertyRecord.originalPhotos

    if photos:
        results = list(precomputed.values()) if precomputed else []
        to_analyze = photos
        if precomputed:
            # Residue can only be jpeg-less photos (analyze_photos drops them without
            # API calls) — filter here so the log doesn't cry wolf about sync analysis.
            to_analyze = [ph for ph in photos
                          if _pick_jpeg_url(ph) and _pick_jpeg_url(ph) not in precomputed]
            if to_analyze:
                logger.info(
                    f"Worker {item_id}: {len(to_analyze)} photo(s) missing from batch output; analyzing synchronously"
                )
        if to_analyze:
            results += await analyze_photos(property_id=p.Id, photos=to_analyze)
        # Unit properties: second-opinion check on non-community Pool photos
        # BEFORE ingestion writes them (see verify_unit_pool_photos).
        await verify_unit_pool_photos(data, results)
        inject_features([item], {p.Id: results})

    async with pool.acquire() as conn:
        # Transaction: child rebuild is DELETE-then-INSERT, so a mid-write crash must not leave children deleted-but-not-reinserted.
        async with conn.transaction():
            existing_id = await conn.fetchval(
                "SELECT id FROM properties WHERE guid = $1", item_id
            )
            if existing_id is None:
                # The scraper re-issues NEW GUIDs for homes it re-scrapes. The
                # zpid is the stable Zillow identity, so a zpid we already hold
                # is the SAME listing: adopt it (move the row to the new GUID and
                # refresh everything) instead of inserting a duplicate — which
                # would violate idx_properties_zpid on every retry, forever.
                zpid = _extract_property_fields(item).get("zpid")
                dup = await conn.fetchrow(
                    "SELECT id, guid FROM properties WHERE zpid = $1", zpid
                ) if zpid is not None else None
                if dup is not None:
                    await conn.execute(
                        "UPDATE properties SET guid = $1 WHERE id = $2", item_id, dup["id"]
                    )
                    # The superseded raw row is dead weight: nothing references
                    # its GUID any more, and keeping it would make the
                    # catalog prune / photo joins see a phantom listing.
                    await conn.execute(
                        "DELETE FROM raw_properties WHERE id = $1 AND id <> $2",
                        dup["guid"], item_id,
                    )
                    logger.info(
                        f"Worker {item_id}: zpid {zpid} already catalogued as {dup['guid']} — "
                        f"adopted under the new GUID and refreshed"
                    )
                    await update_property_with_children(conn, dup["id"], item)
                else:
                    await _insert_primary_full(conn, item)
            else:
                await update_property_with_children(conn, existing_id, item)
        return await mark_processed(conn, item_id, expected_updated_at)


async def _process_image_only(
    pool: asyncpg.Pool, item_id: str, data: dict, expected_updated_at,
) -> bool:
    """Metadata-only sync: scalars + Bedroom/Bathroom + schools, rooms untouched; uses update_property_scalars to avoid the count-zeroing bug, falling back to full path if primary row missing."""
    async with pool.acquire() as conn:
        existing_id = await conn.fetchval(
            "SELECT id FROM properties WHERE guid = $1", item_id
        )
    if existing_id is None:
        if settings.vision_use_batch:
            # Batch-exclusive: a missing primary row means full vision is needed —
            # hand the row to the batch pipeline instead of full-price sync analysis.
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE raw_properties SET status = 'unprocessed', updated_at = NOW() "
                    "WHERE id = $1 AND status = 'image_only_processed'",
                    item_id,
                )
            return False
        return await _process_unprocessed(pool, item_id, data, expected_updated_at)

    item = {"Id": item_id, "ZillowPropertyRecord": data}
    async with pool.acquire() as conn:
        async with conn.transaction():
            await update_property_scalars(conn, existing_id, item)
            # Path-independent recompute of Kitchen/Living/Dining/Garage counts.
            await refresh_property_room_counts(conn, existing_id, item)
            await conn.execute(
                "DELETE FROM property_schools WHERE property_id = $1", existing_id
            )
            for s in _extract_schools(item):
                await conn.execute(
                    """
                    INSERT INTO property_schools (
                        property_id, school_name, rating, grades, distance_miles, link
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    existing_id, s["name"], s["rating"], s["grades"],
                    s["distance"], s["link"],
                )
        return await mark_processed(conn, item_id, expected_updated_at)


async def _process_partial(
    pool: asyncpg.Pool,
    item_id: str,
    data: dict,
    expected_updated_at,
    precomputed: dict[str, PhotoResult] | None = None,
) -> bool:
    """Partial image sync: recompute the photo diff against live primary state, re-analyze only added photos, drop removed ones, refresh metadata + schools; idempotent.
    precomputed = {photo_url: PhotoResult} from the Batch API pipeline; added photos it
    doesn't cover are analyzed synchronously here (shouldn't happen in batch mode —
    the worker only calls this once every added photo has a result)."""
    item = {"Id": item_id, "ZillowPropertyRecord": data}

    # Step 1: existing_id + recompute diff against live primary state.
    async with pool.acquire() as conn:
        existing_id = await conn.fetchval(
            "SELECT id FROM properties WHERE guid = $1", item_id
        )
        if existing_id is None:
            primary_urls: set[str] = set()
        else:
            primary_urls = await primary_photo_urls(conn, item_id)

    if existing_id is None:
        # Defensive: partial implies primary exists; if not, full-process.
        return await _process_unprocessed(pool, item_id, data, expected_updated_at,
                                          precomputed=precomputed)

    desired_urls: set[str] = set(extract_photo_urls_from_data(data))
    actual_added = desired_urls - primary_urls
    actual_removed = sorted(primary_urls - desired_urls)

    # Step 2: locate Photo dicts for actual_added (dedupe by URL).
    photos_to_analyze: list[dict] = []
    seen_urls: set[str] = set()
    for photo_dict in (data.get("originalPhotos") or []):
        url = _canonical_photo_url(photo_dict)
        if url and url in actual_added and url not in seen_urls:
            seen_urls.add(url)
            photos_to_analyze.append(photo_dict)

    # Step 3: vision results for the freshly-needed photos (batch-accumulated first).
    photo_results = []
    if photos_to_analyze:
        to_analyze = photos_to_analyze
        if precomputed:
            covered = [ph for ph in photos_to_analyze
                       if (_canonical_photo_url(ph) or "") in precomputed]
            photo_results = [precomputed[_canonical_photo_url(ph)] for ph in covered]
            to_analyze = [ph for ph in photos_to_analyze
                          if (_canonical_photo_url(ph) or "") not in precomputed]
            photos_to_analyze = covered + to_analyze  # keep zip alignment below
        if to_analyze:
            photo_models = [Photo(**ph) for ph in to_analyze]
            photo_results = photo_results + await analyze_photos(
                property_id=item_id, photos=photo_models, concurrency=5
            )
    await verify_unit_pool_photos(data, photo_results)

    if not photos_to_analyze and not actual_removed:
        # Empty diff (earlier attempt already applied it): metadata-only refresh.
        # update_property_scalars for the same count-zeroing reason as image_only.
        logger.info(
            f"Worker {item_id}: partial diff empty after recompute (idempotent skip)"
        )
        async with pool.acquire() as conn:
            async with conn.transaction():
                await update_property_scalars(conn, existing_id, item)
                await refresh_property_room_counts(conn, existing_id, item)
                await conn.execute(
                    "DELETE FROM property_schools WHERE property_id = $1", existing_id
                )
                for s in _extract_schools(item):
                    await conn.execute(
                        """
                        INSERT INTO property_schools (
                            property_id, school_name, rating, grades, distance_miles, link
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        existing_id, s["name"], s["rating"], s["grades"],
                        s["distance"], s["link"],
                    )
            return await mark_processed(conn, item_id, expected_updated_at)

    async with pool.acquire() as conn:
        async with conn.transaction():
            # 1. Drop room_instances for removed photos.
            if actual_removed:
                await conn.execute(
                    """
                    DELETE FROM room_instances
                    WHERE property_id = $1 AND photo_url = ANY($2)
                    """,
                    existing_id,
                    actual_removed,
                )

            # 2. Insert room_instances for added photos, grouped by room_type (Vision-failed photos get a stub Unknown instance to claim their URL).
            new_rows: dict[str, list[dict]] = {}
            for photo, result in zip(photos_to_analyze, photo_results):
                url = _canonical_photo_url(photo)
                if not url:
                    continue  # no JPEG URL — untrackable
                if result.room_type and result.room_type != "Unknown" and result.features:
                    new_rows.setdefault(result.room_type, []).append({
                        "photo_url": url,
                        # Same community-pool backstop as the full-ingest path —
                        # partial photo updates must not bypass it.
                        "features": _ensure_community_pool_tag(
                            result.room_type, list(result.features)
                        ),
                        "color": result.color,
                    })
                else:
                    new_rows.setdefault("Unknown", []).append({
                        "photo_url": url,
                        "features": [],
                        "color": None,
                    })

            for room_type, instances in new_rows.items():
                # Find or create the rooms row for (property, room_type).
                room_id = await conn.fetchval(
                    "SELECT id FROM rooms WHERE property_id = $1 AND room_type = $2",
                    existing_id, room_type,
                )
                if room_id is None:
                    room_id = await conn.fetchval(
                        """
                        INSERT INTO rooms (property_id, room_type, count)
                        VALUES ($1, $2, 0) RETURNING id
                        """,
                        existing_id, room_type,
                    )

                # Append after the highest existing instance_index.
                max_idx = await conn.fetchval(
                    """
                    SELECT COALESCE(MAX(instance_index), -1)
                    FROM room_instances
                    WHERE property_id = $1 AND room_type = $2
                    """,
                    existing_id, room_type,
                )
                for offset, inst in enumerate(instances):
                    features_text = ", ".join(inst["features"])
                    await conn.execute(
                        """
                        INSERT INTO room_instances (
                            room_id, property_id, room_type, instance_index,
                            features, features_text, color, photo_url
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """,
                        room_id, existing_id, room_type,
                        max_idx + 1 + offset,
                        inst["features"], features_text,
                        inst["color"], inst["photo_url"],
                    )

            # 3. Resync rooms.count from room_instances; drop now-empty rooms.
            await conn.execute(
                """
                UPDATE rooms
                SET count = (
                    SELECT COUNT(*) FROM room_instances ri
                    WHERE ri.property_id = rooms.property_id
                      AND ri.room_type = rooms.room_type
                )
                WHERE property_id = $1
                """,
                existing_id,
            )
            await conn.execute(
                "DELETE FROM rooms WHERE property_id = $1 AND count = 0",
                existing_id,
            )

            # 4. Refresh scalars + Bed/Bath and recompute Kitchen/Living/Dining/Garage (path-independent, same algorithm as full-process).
            await update_property_scalars(conn, existing_id, item)
            await refresh_property_room_counts(conn, existing_id, item)

            # 5. Refresh schools.
            await conn.execute(
                "DELETE FROM property_schools WHERE property_id = $1", existing_id
            )
            for s in _extract_schools(item):
                await conn.execute(
                    """
                    INSERT INTO property_schools (
                        property_id, school_name, rating, grades, distance_miles, link
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    existing_id, s["name"], s["rating"], s["grades"],
                    s["distance"], s["link"],
                )

        # 6. Optimistic-concurrency mark_processed.
        return await mark_processed(conn, item_id, expected_updated_at)


async def _insert_primary_full(conn, item: dict) -> None:
    """INSERT a new property + all child rows; used for new arrivals and the rare missing-primary-row case."""
    record = item.get("ZillowPropertyRecord", {}) or {}
    fields = _extract_property_fields(item)
    rooms_from_photos = _build_rooms_from_photos(record.get("originalPhotos", []) or [])
    room_counts = _get_room_counts(
        record, {rt: len(insts) for rt, insts in rooms_from_photos.items()}
    )

    prop_id = await conn.fetchval("""
        INSERT INTO properties (
            guid, name, street, district, city, state, postal_code, country,
            geom, area_sqft, price_usd,
            bedroom_count, bathroom_count, kitchen_count,
            living_room_count, dining_room_count, garage_count,
            home_type, rent_estimate, year_built,
            lot_size_sqft, stories,
            has_pool, has_waterfront, description, financing,
            county, locality, neighborhood, zpid
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8,
            ST_MakePoint($9, $10)::geography,
            $11, $12, $13, $14, $15, $16, $17, $18,
            $19, $20, $21, $22, $23, $24, $25, $26, $27,
            $28, $29, $30, $31
        ) RETURNING id
    """,
        fields["guid"], fields["name"], fields["street"], fields["district"],
        fields["city"], fields["state"], fields["postal_code"], fields["country"],
        fields["longitude"], fields["latitude"],
        fields["area_sqft"], fields["price_usd"],
        room_counts.get("Bedroom", 0), room_counts.get("Bathroom", 0),
        room_counts.get("Kitchen", 0), room_counts.get("Living Room", 0),
        room_counts.get("Dining Room", 0), room_counts.get("Garage", 0),
        fields["home_type"], fields["rent_estimate"], fields["year_built"],
        fields["lot_size_sqft"], fields["stories"],
        fields["has_pool"], fields["has_waterfront"], fields["description"],
        fields["financing"],
        fields["county"], fields["locality"], fields["neighborhood"],
        fields["zpid"],
    )
    await _insert_children(conn, prop_id, rooms_from_photos, _extract_schools(item))
    await assign_region_ids(conn, prop_id)


MAX_INGEST_ATTEMPTS = 5


async def _record_failure(pool: asyncpg.Pool, item_id: str, status: str) -> None:
    """A row that raised: push it to the back of the queue (claim is oldest-first)
    and count the attempt. After MAX_INGEST_ATTEMPTS it is parked as 'failed' so a
    poison row can never loop forever — the 2026-08 incident was 80 duplicate
    rows retried every cycle for three days, logging 18 GB and filling the disk.
    A re-upload via /process resets the row to 'unprocessed' (attempts reset too)."""
    try:
        async with pool.acquire() as conn:
            attempts = await conn.fetchval(
                """
                UPDATE raw_properties
                SET updated_at = NOW(), ingest_attempts = ingest_attempts + 1
                WHERE id = $1 AND status = $2
                RETURNING ingest_attempts
                """,
                item_id, status,
            )
            if attempts is not None and attempts >= MAX_INGEST_ATTEMPTS:
                await conn.execute(
                    "UPDATE raw_properties SET status = 'failed', updated_at = NOW() "
                    "WHERE id = $1 AND status = $2",
                    item_id, status,
                )
                logger.error(
                    f"Worker: {item_id} failed {attempts} times — parked as 'failed' "
                    f"(re-upload via /process to retry)"
                )
    except Exception:  # noqa: BLE001 — best-effort bookkeeping
        pass


async def _process_one_row(pool: asyncpg.Pool, row: asyncpg.Record) -> bool:
    """Dispatch a single raw row to the right processor; returns False if raced or if processing raised (row stays pending)."""
    item_id = row["id"]
    raw_data = row["data"]
    if not isinstance(raw_data, dict):
        raw_data = json.loads(raw_data)
    status = row["status"]
    expected_updated_at = row["updated_at"]
    try:
        if status == "unprocessed":
            marked = await _process_unprocessed(pool, item_id, raw_data, expected_updated_at)
        elif status == "image_only_processed":
            marked = await _process_image_only(pool, item_id, raw_data, expected_updated_at)
        elif status == "partial_image_only_processed":
            # No stored diff; _process_partial recomputes against primary.
            marked = await _process_partial(
                pool, item_id, raw_data, expected_updated_at
            )
        else:
            return False
        if marked:
            logger.info(f"Worker processed {item_id} (was {status})")
        else:
            logger.info(
                f"Worker re-queue {item_id}: concurrent /process write since claim"
            )
        return marked
    except Exception as e:
        logger.exception(f"Worker failed for {item_id} (status={status}): {e}")
        await _record_failure(pool, item_id, status)
        return False


async def _signal_features_changed(pool: asyncpg.Pool) -> None:
    """Post-ingest bookkeeping: NOTIFY the web tier to rebuild the feature registry and kick the POI auto-refresh."""
    try:
        async with pool.acquire() as conn:
            await conn.execute("NOTIFY feature_change")
    except Exception as e:
        logger.warning(f"Worker: NOTIFY feature_change failed: {e}")
    _trigger_poi_refresh(pool)


async def _batch_step(pool: asyncpg.Pool) -> int:
    """One cycle of the batch-exclusive vision pipeline:
    1) collect finished batches (validate matching, store per-photo results),
    2) ingest every property whose needed photos are all resulted,
    3) submit new grouped batches while the queue budget has headroom.
    Returns an activity count (terminal batches + ingested properties + submissions)."""
    activity = 0
    try:
        if await batch_vision.collect_completed(pool):
            activity += 1
    except Exception as e:
        logger.warning(f"Worker: batch collect failed: {e}")

    try:
        ready = await batch_vision.ready_properties(pool)
    except Exception as e:
        logger.warning(f"Worker: batch readiness check failed: {e}")
        ready = []

    ingested = 0
    for entry in ready:
        item_id = entry["item_id"]
        try:
            if entry["status"] == "partial_image_only_processed":
                ok = await _process_partial(pool, item_id, entry["data"],
                                            entry["updated_at"], precomputed=entry["results"])
            else:  # unprocessed (+ legacy batch_submitted)
                ok = await _process_unprocessed(pool, item_id, entry["data"],
                                                entry["updated_at"], precomputed=entry["results"])
            if ok:
                ingested += 1
                logger.info(f"Worker processed {item_id} (batch vision v2)")
                await batch_vision.clear_property(pool, item_id)
            # ok=False → a /process write raced us; row re-evaluated next cycle with
            # fresh data (URL-keyed results stay valid for unchanged photos).
        except Exception as e:
            logger.exception(f"Worker: ingesting {item_id} from batch results failed: {e}")
            await _record_failure(pool, item_id, entry["status"])
    if ingested:
        await _signal_features_changed(pool)
        try:
            await batch_vision.gc(pool)
        except Exception as e:
            logger.warning(f"Worker: batch GC failed: {e}")
    activity += ingested

    try:
        activity += await batch_vision.submit_waves(pool)
    except Exception as e:
        logger.warning(f"Worker: batch submit failed: {e}")
    return activity


async def _worker_iteration(pool: asyncpg.Pool) -> tuple[int, int]:
    """Process one iteration; returns (claimed, succeeded), claimed=0 meaning no work.
    Batch mode is BATCH-EXCLUSIVE: all photo analysis flows through _batch_step, and the
    sync claim below only handles metadata-only rows (image_only_processed)."""
    # Catalog rule (FOR_SALE/PENDING): drop listings that sold or otherwise left the
    # catalog since ingest and park their pending rows BEFORE any vision work below.
    try:
        async with pool.acquire() as conn:
            await prune_non_for_sale(conn)
    except Exception as e:  # noqa: BLE001 — never let the prune stall ingestion
        logger.warning(f"Worker: catalog prune failed: {e}")

    batch_activity = 0
    if settings.vision_use_batch:
        batch_activity = await _batch_step(pool)

    async with pool.acquire() as conn:
        async with conn.transaction():
            rows = await claim_pending_batch(
                conn, limit=WORKER_BATCH_SIZE,
                metadata_only=settings.vision_use_batch,
            )
    if not rows:
        return (batch_activity, batch_activity) if batch_activity else (0, 0)

    results = await asyncio.gather(
        *[_process_one_row(pool, row) for row in rows],
        return_exceptions=True,
    )
    succeeded = sum(1 for ok in results if ok is True)

    features_may_have_changed = any(
        ok is True
        and row["status"] in ("unprocessed", "partial_image_only_processed")
        for ok, row in zip(results, rows)
    )
    if features_may_have_changed:
        # Worker has no in-memory registry; signal the web tier to rebuild via Postgres NOTIFY on the feature_change channel.
        await _signal_features_changed(pool)

    return len(rows), succeeded


async def run_worker_forever() -> None:
    """Top-level worker loop. Started by worker_main in the worker process."""
    logger.info("Background worker starting")
    recovery_done = False
    while True:
        try:
            pool = await get_pool()
            if not recovery_done:
                # Flag off but rows left 'batch_submitted' by a previous run → requeue
                # once, or nothing would ever claim them.
                if not settings.vision_use_batch:
                    await batch_vision.recover_when_disabled(pool)
                recovery_done = True
            claimed, succeeded = await _worker_iteration(pool)
            if claimed == 0:
                await asyncio.sleep(WORKER_IDLE_SLEEP_SECONDS)
            elif succeeded == 0:
                # Claimed rows but none made progress (all failed or raced); back off instead of hot-looping.
                await asyncio.sleep(WORKER_ERROR_SLEEP_SECONDS)
        except asyncio.CancelledError:
            logger.info("Background worker received cancellation; exiting")
            raise
        except Exception as e:
            logger.exception(f"Background worker iteration crashed: {e}")
            await asyncio.sleep(WORKER_ERROR_SLEEP_SECONDS)
