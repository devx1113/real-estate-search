"""Raw_properties staging for /process; worker syncs to primary tables via diff (data vs room_instances.photo_url), idempotent under concurrent writes."""

from __future__ import annotations

import json
import logging
from datetime import datetime

import asyncpg

logger = logging.getLogger(__name__)


def extract_photo_urls_from_data(data: dict) -> list[str]:
    """Highest-res JPEG URL per photo; same extractor as room_instances.photo_url so diffs round-trip."""
    urls: list[str] = []
    for photo in (data.get("originalPhotos") or []):
        jpegs = ((photo.get("mixedSources") or {}).get("jpeg") or [])
        if not jpegs:
            continue
        best = max(jpegs, key=lambda j: j.get("width", 0))
        url = best.get("url", "")
        if url:
            urls.append(url)
    return urls


async def primary_photo_urls(conn, property_guid: str) -> set[str]:
    """Photo URLs already in primary (room_instances.photo_url); empty if absent/legacy NULL (→ full re-analyze)."""
    rows = await conn.fetch(
        """
        SELECT DISTINCT ri.photo_url
        FROM room_instances ri
        JOIN properties p ON p.id = ri.property_id
        WHERE p.guid = $1 AND ri.photo_url IS NOT NULL
        """,
        property_guid,
    )
    return {r["photo_url"] for r in rows}


async def upsert_raw_property(
    conn,
    item_id: str,
    new_data: dict,
) -> tuple[str, dict | None, bool]:
    """Insert/update a raw_properties row; returns (final_status, diff_info_or_None, existed_before)."""
    existing = await conn.fetchrow(
        "SELECT status FROM raw_properties WHERE id = $1", item_id
    )
    new_urls = extract_photo_urls_from_data(new_data)
    new_data_json = json.dumps(new_data)
    new_url_set = set(new_urls)

    if existing is None:
        # ON CONFLICT: two concurrent POSTs with the same NEW id both see "no row";
        # the loser of the insert race must not error out (payload would be dropped).
        result = await conn.execute(
            """
            INSERT INTO raw_properties (id, data, status)
            VALUES ($1, $2::jsonb, 'unprocessed')
            ON CONFLICT (id) DO NOTHING
            """,
            item_id,
            new_data_json,
        )
        if result.endswith(" 1"):
            return "unprocessed", None, False
        # Lost the race — a row now exists; fall through to the update path below.
        existing = await conn.fetchrow(
            "SELECT status FROM raw_properties WHERE id = $1", item_id
        )

    primary_urls = await primary_photo_urls(conn, item_id)

    added = sorted(new_url_set - primary_urls)
    removed = sorted(primary_urls - new_url_set)
    unchanged = new_url_set & primary_urls

    if not primary_urls:
        new_status = "unprocessed"
        diff_info: dict | None = None
    elif not added and not removed:
        new_status = "image_only_processed"
        diff_info = None
    elif not unchanged:
        new_status = "unprocessed"
        diff_info = None
    else:
        new_status = "partial_image_only_processed"
        diff_info = {"added": added, "removed": removed}

    # Never silently drop queued vision work.
    if existing["status"] == "unprocessed":
        new_status = "unprocessed"
        diff_info = None

    await conn.execute(
        """
        UPDATE raw_properties
        SET data = $2::jsonb,
            status = $3,
            updated_at = NOW()
        WHERE id = $1
        """,
        item_id,
        new_data_json,
        new_status,
    )
    return new_status, diff_info, True


async def claim_pending_batch(
    conn,
    limit: int = 5,
    metadata_only: bool = False,
) -> list[asyncpg.Record]:
    """Claim up to `limit` pending rows via FOR UPDATE SKIP LOCKED; returns updated_at
    for optimistic-concurrency guarding. metadata_only=True (batch-exclusive mode):
    claim ONLY rows needing no vision (image_only_processed) — everything with photo
    work belongs to the Batch API pipeline and must never drain at full price."""
    statuses = ["image_only_processed"]
    if not metadata_only:
        statuses += ["unprocessed", "partial_image_only_processed"]
    rows = await conn.fetch(
        """
        SELECT id, data, status, updated_at
        FROM raw_properties
        WHERE status = ANY($2)
          AND data->>'homeStatus' = ANY($3)  -- catalog statuses only (see prune_non_for_sale)
        ORDER BY updated_at ASC
        LIMIT $1
        FOR UPDATE SKIP LOCKED
        """,
        limit,
        statuses,
        list(CATALOG_STATUSES),
    )
    return rows


async def mark_processed(conn, item_id: str, expected_updated_at: datetime) -> bool:
    """Set status='processed' only if updated_at unchanged since claim; returns False if a concurrent write moved it."""
    result = await conn.execute(
        """
        UPDATE raw_properties
        SET status = 'processed', updated_at = NOW()
        WHERE id = $1 AND updated_at = $2
        """,
        item_id,
        expected_updated_at,
    )
    try:
        n = int(result.rsplit(" ", 1)[-1])
    except ValueError:
        n = 0
    return n > 0


async def get_status_counts(conn) -> dict[str, int]:
    rows = await conn.fetch("SELECT status, COUNT(*) AS n FROM raw_properties GROUP BY status")
    return {r["status"]: r["n"] for r in rows}


# Client requirement (2026-09-10/11, MLS feed): the catalog carries listings that
# are for sale — StandardStatus Active / Active Under Contract / Pending on
# PropertyType A/B/C (homeStatus FOR_SALE or PENDING) — and, since 2026-09-11,
# rentals: PropertyType E (commercial lease) and F (residential lease), homeStatus
# FOR_RENT. Anything else (SOLD, OTHER, commercial SALE — and any status added
# upstream later) must not be searchable. Enforced as a keep-only rule so a new
# upstream status can never leak in. Search separates sale from rent (listing_mode).
CATALOG_STATUSES = ("FOR_SALE", "PENDING", "FOR_RENT")
SKIPPED_NOT_FOR_SALE = "skipped_not_for_sale"

_PENDING_STATUSES = ("unprocessed", "partial_image_only_processed", "image_only_processed")


async def prune_non_for_sale(conn) -> tuple[int, int]:
    """Enforce the catalog-status rule (FOR_SALE/PENDING); returns (properties_deleted, rows_skipped).

    Three parts, all idempotent and cheap when there is nothing to do:
      1. DELETE properties whose OWN raw record has left the catalog statuses — a
         listing re-sent under the same GUID after it sold / was withdrawn (children cascade).
      2. DELETE properties superseded by a NEWER raw record with the same zpid that is
         off-catalog. The scraper re-issues GUIDs on re-scrape, so a status change
         usually arrives under a new GUID: it gets parked by part 3 and the old
         catalog row would otherwise stay searchable forever (1550 Mars St, 2026-08).
         "Newer" is decided by upload time, so an OLD pending record can never delete
         a listing that has since been re-listed and adopted under a new GUID.
      3. Park still-pending raw rows of off-catalog listings in a terminal status so
         the worker never claims them: no vision spend on homes we would delete anyway.
    A listing that returns to a catalog status is re-uploaded as 'unprocessed' by /process and
    flows through normally, so this never permanently blacklists a property.
    """
    deleted = await conn.fetchval(
        """
        WITH gone AS (
            DELETE FROM properties p
            USING raw_properties r
            WHERE r.id = p.guid
              AND coalesce(r.data->>'homeStatus', '') <> ALL($1::text[])
            RETURNING p.id
        )
        SELECT count(*) FROM gone
        """,
        list(CATALOG_STATUSES),
    )
    superseded = await conn.fetchval(
        """
        WITH gone AS (
            DELETE FROM properties p
            USING raw_properties own, raw_properties newer
            WHERE own.id = p.guid
              AND p.zpid IS NOT NULL
              AND newer.id <> own.id
              AND newer.data->>'zpid' = p.zpid::text
              AND newer.created_at > own.created_at
              AND coalesce(newer.data->>'homeStatus', '') <> ALL($1::text[])
            RETURNING p.id
        )
        SELECT count(*) FROM gone
        """,
        list(CATALOG_STATUSES),
    )
    deleted = (deleted or 0) + (superseded or 0)
    skipped = await conn.fetchval(
        """
        WITH parked AS (
            UPDATE raw_properties
            SET status = $1, updated_at = NOW()
            WHERE status = ANY($2)
              AND coalesce(data->>'homeStatus', '') <> ALL($3::text[])
            RETURNING id
        )
        SELECT count(*) FROM parked
        """,
        SKIPPED_NOT_FOR_SALE,
        list(_PENDING_STATUSES),
        list(CATALOG_STATUSES),
    )
    if deleted or skipped:
        logger.info(
            "Catalog prune: %d off-catalog property(ies) removed, %d pending row(s) skipped",
            deleted, skipped,
        )
    return deleted or 0, skipped or 0
