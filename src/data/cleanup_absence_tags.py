"""Remove absence tags ("ceiling fan not visible", "shutters absent", "ceiling fan
light omitted") written by the vision model before ingest started dropping them
(src.img_analyzer.db_ingest.drop_absence_tags). Such a tag says what a photo does
NOT show; it is noise for every search and made "without fan" exclude homes that
have no fan.

Rewrites room_instances.features / features_text, deletes the tags from
feature_embeddings, strips them from feature_phrase_cache and NOTIFYs the app to
rebuild its registry. Idempotent. Every changed row is backed up as JSON under
LOG_DIR first (restore = write the saved features back by room_instances id).

  docker exec realestatesearch-app-1 python -m src.data.cleanup_absence_tags --dry-run
  docker exec realestatesearch-app-1 python -m src.data.cleanup_absence_tags
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings
from src.data.database import close_pool, get_pool
from src.img_analyzer.db_ingest import drop_absence_tags

logger = logging.getLogger(__name__)


async def cleanup(conn, dry_run: bool) -> dict:
    tags = [r["f"] for r in await conn.fetch("SELECT DISTINCT unnest(features) AS f FROM room_instances")]
    bad = sorted(set(tags) - set(drop_absence_tags(tags)))
    rows = await conn.fetch(
        "SELECT id, property_id, room_type, photo_url, features FROM room_instances WHERE features && $1::text[]",
        bad,
    ) if bad else []
    per_tag = Counter(t for r in rows for t in r["features"] if t in set(bad))
    homes = len({r["property_id"] for r in rows})
    emptied = sum(1 for r in rows if not drop_absence_tags(list(r["features"])))
    summary = {"absence_tags": len(bad), "rows": len(rows), "homes": homes, "rows_left_without_tags": emptied,
               "top": per_tag.most_common(12), "dry_run": dry_run}
    if dry_run or not rows:
        return summary

    backup = Path(settings.log_dir) / f"absence_tags_backup_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text(json.dumps(
        {"removed_tags": bad,
         "rows": [{"id": r["id"], "property_id": r["property_id"], "room_type": r["room_type"],
                   "photo_url": r["photo_url"], "features": list(r["features"])} for r in rows]},
        ensure_ascii=False))
    summary["backup"] = str(backup)

    async with conn.transaction():
        await conn.executemany(
            "UPDATE room_instances SET features = $1::text[], features_text = $2 WHERE id = $3",
            [(kept, ", ".join(kept), r["id"]) for r in rows for kept in [drop_absence_tags(list(r["features"]))]],
        )
        emb = await conn.execute("DELETE FROM feature_embeddings WHERE feature = ANY($1::text[])", bad)
        cache = await conn.execute(
            "UPDATE feature_phrase_cache SET alternatives = ARRAY(SELECT a FROM unnest(alternatives) a WHERE a <> ALL($1::text[])) "
            "WHERE alternatives && $1::text[]", bad)
    await conn.execute("NOTIFY feature_change")
    summary["embeddings_deleted"] = emb
    summary["phrase_cache_rows_updated"] = cache
    return summary


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dry_run = "--dry-run" in sys.argv
    pool = await get_pool()
    async with pool.acquire() as conn:
        summary = await cleanup(conn, dry_run)
    await close_pool()
    logger.info("Absence-tag cleanup %s: %s", "DRY RUN" if dry_run else "APPLIED", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
