"""Resolve the place a query searched INSIDE to a row of the regions table.

Geo-location contract: /search returns regionId + canonical regionName + the
boundary ONLY when that region's polygon actually filtered the results
(resolve_search_region succeeded and the region has geometry). Every fallback —
no polygon data, subdivision/neighborhood searches, ambiguous names — returns
regionId null, with the parsed place name as a bare label. regionId is a promise
that the pins are that region's geometry; never a guess.
"""

import logging

import asyncpg

from config.settings import settings
from src.data.us_states import abbrev_state, expand_state
from src.models.search import LocationCriterion

logger = logging.getLogger(__name__)

# regiontype -> the properties column holding that level's assigned region id
# (see src/data/backfill_region_ids.py for how assignment works).
REGION_ID_COLUMNS: dict[str, str] = {
    "0": "city_region_id",
    "3": "county_region_id",
    "2": "zipcode_region_id",
    "1": "neighborhood_region_id",
}

# regions.regiontype values a given LocationCriterion field can map to, tried in
# order. city/locality fall through '0' -> '1' because the parser deliberately
# files ANY doubtful sub-city place under city (its prompt: "when in doubt, put
# the place here") — so "Silver Lake" arrives as city even when the real region
# row is a neighborhood. Mirrors the hard filter, which matches city across all
# place columns for the same reason.
_FIELD_REGION_TYPES: dict[str, tuple[str, ...]] = {
    "neighborhood": ("1",),
    "district": ("1",),
    # '3' at the end: users say bare "brevard" for the county, which the parser
    # files under city; the only type-0 'Brevard' is a town in NC, so without the
    # county fall-through the obvious local meaning would never resolve.
    "locality": ("0", "1", "3"),
    "city": ("0", "1", "3"),
    "county": ("3",),
    "state": ("4",),
}


# Polygon membership is static between ingests, but ST_Covers over the whole
# catalog costs ~0.5s for a county-sized boundary — and it was recomputed on EVERY
# request. Cache the matched property ids per region (TTL: newly ingested homes
# appear in polygon searches within 5 minutes; same trade as the parse cache).
_REGION_IDS_CACHE: dict[int, tuple[float, list[int]]] = {}
_REGION_IDS_TTL_SEC = 300
_REGION_IDS_MAX = 64


def region_membership_sql(region_type: str, placeholder: str) -> str | None:
    """SQL condition: the property belongs to the region `placeholder` of this
    type; None when the type has no column. Cities use city_region_ids — a home
    inside overlapping city polygons (Viera over Rockledge / Melbourne) belongs to
    each. Rows not yet re-assigned (city_region_ids NULL) fall back to the primary."""
    col = REGION_ID_COLUMNS.get(region_type)
    if col is None:
        return None
    if region_type == "0":
        return (f"(city_region_ids @> ARRAY[{placeholder}::bigint] "
                f"OR (city_region_ids IS NULL AND city_region_id = {placeholder}))")
    return f"{col} = {placeholder}"


async def _region_id_count(pool: asyncpg.Pool, region_type: str, region_id: int) -> int:
    """How many properties belong to region_id at this level; 0 when the type
    has no column. Indexed count — cheap enough to skip caching."""
    cond = region_membership_sql(region_type, "$1")
    if cond is None:
        return 0
    async with pool.acquire() as conn:
        return await conn.fetchval(f"SELECT count(*) FROM properties WHERE {cond}", region_id)


async def _candidate_population(
    pool: asyncpg.Pool, region_type: str, candidate
) -> int:
    """Tie-break weight for same-named regions: how many catalog homes the
    candidate actually holds — by stored region ids in ID mode, by polygon
    membership otherwise (polygon-less candidates weigh 0 there)."""
    if settings.search_use_region_ids:
        return await _region_id_count(pool, region_type, candidate["regionid"])
    if not candidate["has_geom"]:
        return 0
    return len(await region_property_ids(pool, candidate["regionid"]))


async def region_property_ids(pool: asyncpg.Pool, region_id: int) -> list[int]:
    """All property ids inside the region's polygon (cached)."""
    import time
    hit = _REGION_IDS_CACHE.get(region_id)
    if hit and time.monotonic() - hit[0] < _REGION_IDS_TTL_SEC:
        return hit[1]
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT p.id FROM properties p, regions r "
            "WHERE r.regionid = $1 AND ST_Covers(r.geom, p.geom)",
            region_id,
        )
    ids = [r["id"] for r in rows]
    if len(_REGION_IDS_CACHE) >= _REGION_IDS_MAX:
        _REGION_IDS_CACHE.pop(min(_REGION_IDS_CACHE, key=lambda k: _REGION_IDS_CACHE[k][0]))
    _REGION_IDS_CACHE[region_id] = (time.monotonic(), ids)
    return ids


def search_area_target(criteria) -> tuple[str, str, LocationCriterion] | None:
    """(field, value, criterion) for the place the user is searching INSIDE, or None
    when the query names no such place OR is RELATIONAL — "near X", "between X and Y"
    — because those cover a ring/corridor, not X's own polygon. Most-specific
    location field wins."""
    for c in criteria:
        if isinstance(c, LocationCriterion):
            for field in ("neighborhood", "locality", "district", "city", "county"):
                value = getattr(c, field)
                if value and str(value).strip():
                    return field, str(value).strip(), c
            if c.state and c.state.strip():
                return "state", c.state.strip(), c
    return None


def extract_search_area(criteria) -> str | None:
    """The PARSED area name — the display label when no region polygon filtered the
    search (the canonical name then comes from resolve_search_region instead)."""
    target = search_area_target(criteria)
    if target is None:
        return None
    field, value, _ = target
    # State expanded to its full name — "Florida" geocodes where "FL" is ambiguous.
    return expand_state(value) if field == "state" else value


async def resolve_search_region(pool: asyncpg.Pool, criteria) -> dict | None:
    """PRE-FILTER region resolution for geo-location search: map the searched place
    name to one regions row BEFORE any properties are matched, so its polygon can
    replace name matching entirely. Returns
    {"region_id", "region_name", "has_geom"} or None.

    No matched results exist yet, so candidates narrow by the query's own
    state/parent-city context, then by which candidate's POLYGON actually contains
    catalog properties ("Rockledge" with no state: the FL polygon holds 300+ homes,
    the GA one zero). Ambiguous, unknown, or polygon-less -> None, and the caller
    falls back to name matching with regionId null. Never raises."""
    try:
        target = search_area_target(criteria)
        if target is None:
            return None
        field, value, crit = target
        state_code = abbrev_state(crit.state).upper() if crit.state and crit.state.strip() else None
        parent_city = crit.city if field in ("neighborhood", "district", "locality") else None
        # Parser variance: the LLM sometimes duplicates the place into BOTH city and
        # neighborhood ("viera west" -> city="Viera West", neighborhood="Viera West").
        # That's not a real parent constraint — requiring a city of the same name
        # would wrongly kill resolution (Viera West's parent city is Melbourne).
        if parent_city and parent_city.strip().lower() == value.strip().lower():
            parent_city = None

        # A region the NAME resolved to but that cannot filter (no property
        # carries its id): returned as display-only so /search can still show
        # regionId + boundary while the results come from name matching (Viera).
        display_fallback: dict | None = None
        async with pool.acquire() as conn:
            for region_type in _FIELD_REGION_TYPES[field]:
                names = [value]
                if field == "state":
                    names = [expand_state(value)]
                elif region_type == "3" and not value.lower().endswith("county"):
                    names.append(f"{value} County")  # bare "Brevard" -> the county row

                candidates = await conn.fetch(
                    """
                    SELECT regionid, regionname, statecode,
                           (geom IS NOT NULL) AS has_geom
                    FROM regions
                    WHERE regiontype = $1
                      AND lower(regionname) = ANY($2)
                      AND ($3::text IS NULL OR statecode = $3)
                      AND ($4::text IS NULL OR lower(city) = lower($4))
                    """,
                    region_type,
                    [n.lower() for n in names],
                    state_code,
                    parent_city if region_type == "1" else None,
                )
                if not candidates:
                    continue
                if len(candidates) == 1:
                    chosen = candidates[0]
                else:
                    # Tie-break by population: which same-named candidate actually
                    # holds catalog homes (stored ids in ID mode, polygon otherwise).
                    populated = []
                    for c in candidates:
                        n = await _candidate_population(pool, region_type, c)
                        if n:
                            populated.append((n, c))
                    if len(populated) > 1:
                        return None  # genuinely ambiguous -> no geo mode
                    if not populated:
                        continue  # nothing usable at this level, try the next type
                    chosen = populated[0][1]
                if settings.search_use_region_ids:
                    # ID mode: membership was precomputed at ingest — filter by the
                    # stored column. Works with or without a polygon (Viera East).
                    # NO region-polygon fallback here: region polygons are drawing
                    # data only (the map's hand-drawn polygon is the sole
                    # polygon-based filter); a region no property carries falls
                    # through to the next type, then to name matching.
                    n = await _region_id_count(pool, region_type, chosen["regionid"])
                    if n:
                        return {
                            "region_id": chosen["regionid"],
                            "region_name": f"{chosen['regionname']}, {chosen['statecode']}",
                            "has_geom": chosen["has_geom"],
                            "region_type": region_type,
                            "id_mode": True,
                        }
                    if display_fallback is None and region_type != "4":
                        display_fallback = {
                            "region_id": chosen["regionid"],
                            "region_name": f"{chosen['regionname']}, {chosen['statecode']}",
                            "has_geom": chosen["has_geom"],
                            "region_type": region_type,
                            "id_mode": False,
                            "display_only": True,
                        }
                    continue
                if not chosen["has_geom"]:
                    # A polygon-less match can't power geo filtering; try the next
                    # type first (bare "brevard" hits the polygon-less NC town at
                    # type '0' but Brevard County's polygon at '3') before falling
                    # back to name matching.
                    continue
                return {
                    "region_id": chosen["regionid"],
                    "region_name": f"{chosen['regionname']}, {chosen['statecode']}",
                    "has_geom": True,
                    "region_type": region_type,
                    "id_mode": False,
                }
        return display_fallback
    except Exception:
        logger.exception("Pre-filter region resolution failed")
        return None


async def keep_majority_location(
    pool: asyncpg.Pool, criteria, property_ids: list[int]
) -> list[int]:
    """When a place NAME was searched WITHOUT a state, same-named places in different
    states can all match ("Rockledge" hits FL and GA listings) — one map, two places.
    Trim the results to the single state holding the most matches so the response
    describes one location; ties break deterministically (highest count, then
    statecode A→Z) so identical requests return identical results.

    Untouched cases: no place searched (feature-only queries legitimately span the
    catalog), a state was given (the hard filter already pinned it), or the results
    already sit in one state. Never raises — trimming is enrichment, so failures
    return the ids unchanged."""
    target = search_area_target(criteria)
    if target is None or not property_ids:
        return property_ids
    _, _, crit = target
    if crit.state and crit.state.strip():
        return property_ids

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, state FROM properties WHERE id = ANY($1)", property_ids
            )
        by_state: dict[str, list[int]] = {}
        for r in rows:
            by_state.setdefault(abbrev_state(r["state"]).upper(), []).append(r["id"])
        if len(by_state) <= 1:
            return property_ids
        winner = min(by_state, key=lambda s: (-len(by_state[s]), s))
        logger.info(
            f"Majority-location trim: keeping {winner} "
            f"({len(by_state[winner])}/{len(property_ids)} results; "
            f"dropped states: {sorted(s for s in by_state if s != winner)})"
        )
        return by_state[winner]
    except Exception:
        logger.exception("Majority-location trim failed — keeping all results")
        return property_ids


