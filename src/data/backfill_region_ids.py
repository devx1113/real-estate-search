"""Re-assign region identity for every property with the location-first rules of
src.img_analyzer.db_ingest.assign_region_ids (county / city / zipcode /
neighborhood from the map pin and the regions polygons, with the pin-trust,
coastline and overlap safety rules), including city_region_ids,
location_trusted and the derived neighborhood name.

Re-runnable and idempotent. Run after changing the rules or the regions table:

  docker exec realestatesearch-app-1 python -m src.data.backfill_region_ids
"""

from __future__ import annotations

import asyncio
import logging
import time

from src.data.database import close_pool, get_pool
from src.img_analyzer.db_ingest import assign_region_ids, ensure_property_columns

logger = logging.getLogger(__name__)


async def backfill(conn) -> dict[str, int]:
    """Recompute region columns for all properties; returns summary counts."""
    await ensure_property_columns(conn)
    ids = [r["id"] for r in await conn.fetch("SELECT id FROM properties ORDER BY id")]
    started = time.monotonic()
    for n, pid in enumerate(ids, 1):
        await assign_region_ids(conn, pid)
        if n % 500 == 0:
            logger.info("region backfill: %d/%d (%.0fs)", n, len(ids), time.monotonic() - started)
    row = await conn.fetchrow(
        """
        SELECT count(*) AS total,
               count(city_region_id) AS city,
               count(*) FILTER (WHERE cardinality(city_region_ids) > 1) AS multi_city,
               count(county_region_id) AS county,
               count(zipcode_region_id) AS zipcode,
               count(neighborhood_region_id) AS neighborhood,
               count(*) FILTER (WHERE location_trusted IS FALSE) AS untrusted_pins
        FROM properties
        """
    )
    return dict(row)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pool = await get_pool()
    async with pool.acquire() as conn:
        counts = await backfill(conn)
    logger.info(f"Region backfill complete: {counts}")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
