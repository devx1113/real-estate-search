"""Deterministic filter engine: PostgreSQL indexed queries for room counts, price, area, location."""

import logging
import re

import asyncpg

from src.data.us_states import country_variants, state_variants
from src.search.region_resolver import REGION_ID_COLUMNS, region_property_ids, search_area_target
from src.models.search import (
    AreaCriterion,
    Criterion,
    LocationCriterion,
    PriceCriterion,
    PropertyCriterion,
    RoomCountCriterion,
)

logger = logging.getLogger(__name__)

def _word_match_pattern(value: str) -> str:
    """Case-insensitive WHOLE-WORD regex for a place name: 'viera' must match
    'Bridgewater at Viera' but NOT 'Riviera' (plain %contains% matched both —
    'viera' is a substring of 'Riviera'). Regex metacharacters in user input are
    escaped; whitespace runs are normalized; \\m/\\M (PostgreSQL word boundaries)
    are added only next to word characters so values like 'St.' still match."""
    cleaned = value.strip()
    escaped = re.sub(r"([^A-Za-z0-9\s])", r"\\\1", cleaned)
    escaped = re.sub(r"\s+", r"\\s+", escaped)
    if re.match(r"[A-Za-z0-9]", cleaned):
        escaped = r"\m" + escaped
    if re.search(r"[A-Za-z0-9]$", cleaned):
        escaped = escaped + r"\M"
    return escaped


async def apply_hard_filters(
    pool: asyncpg.Pool,
    criteria: list[Criterion],
    bounds: dict | None = None,
    filters: dict | None = None,
    area_region_id: int | None = None,
    area_region_type: str | None = None,
    drawn_polygon: list[tuple[float, float]] | None = None,
    homes_only: bool = True,
) -> list[int]:
    """Return property IDs passing ALL criteria; bounds is an optional bbox, filters
    are per-field overrides that suppress matching LLM sub-conditions.

    homes_only: when no home type is requested, exclude land (home_type LOT) so a
    generic "homes in X" search is not padded with lots. Callers pass False when
    the query is an exact street address — an address names one listing, lot or not.

    area_region_id: geo-location mode — the searched place resolved to this regions
    row, so membership replaces the target criterion's PLACE-NAME condition (its
    other fields — state, street... — still apply). With area_region_type set
    (region-ID search), membership is the stored <level>_region_id column; without
    it, legacy polygon membership via ST_Covers. Name matching remains for every
    other criterion and as the fallback when no region resolves."""
    hard_criteria = [
        c for c in criteria
        if isinstance(c, (RoomCountCriterion, PriceCriterion, AreaCriterion,
                          LocationCriterion, PropertyCriterion))
    ]

    # Which criterion+field the polygon replaces (the same target the region was
    # resolved from) — computed once so exactly one name condition is suppressed.
    polygon_target: tuple[int, str] | None = None
    if area_region_id is not None:
        t = search_area_target(criteria)
        if t is not None:
            field, _, crit = t
            polygon_target = (id(crit), field)

    conditions = []
    params = []
    param_idx = 1

    if polygon_target is not None:
        if area_region_type is not None:
            # Region-ID mode: membership precomputed at ingest — one indexed
            # integer equality, no geometry and no cache.
            col = REGION_ID_COLUMNS[area_region_type]
            conditions.append(f"{col} = ${param_idx}")
            params.append(area_region_id)
        else:
            # Polygon mode: membership via the cached precomputed id set —
            # ST_Covers over the catalog cost ~0.5s per request for county-sized
            # polygons (see region_property_ids).
            conditions.append(f"id = ANY(${param_idx}::int[])")
            params.append(await region_property_ids(pool, area_region_id))
        param_idx += 1

    if drawn_polygon:
        # Hand-drawn map polygon — the only polygon-based filtering in the system
        # (named regions filter by region id). Ring is closed automatically;
        # ST_MakeValid absorbs self-intersections from freehand drawing.
        ring = list(drawn_polygon)
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        wkt = "POLYGON((" + ", ".join(f"{lng} {lat}" for lat, lng in ring) + "))"
        conditions.append(
            f"ST_Covers(ST_MakeValid(ST_SetSRID(ST_GeomFromText(${param_idx}), 4326))::geography, geom)"
        )
        params.append(wkt)
        param_idx += 1

    if bounds:
        try:
            south = float(bounds["south"])
            north = float(bounds["north"])
            west = float(bounds["west"])
            east = float(bounds["east"])
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Invalid bounds ignored: {e}")
        else:
            conditions.append(
                f"ST_Covers(ST_MakeEnvelope(${param_idx}, ${param_idx + 1}, "
                f"${param_idx + 2}, ${param_idx + 3}, 4326)::geography, geom)"
            )
            params.extend([west, south, east, north])
            param_idx += 4

    # Apply explicit filters first; track covered fields so LLM criteria skip them.
    covered: set[str] = set()
    if filters:
        if filters.get("price_min") is not None:
            conditions.append(f"price_usd >= ${param_idx}")
            params.append(filters["price_min"])
            param_idx += 1
            covered.add("price_min")
        if filters.get("price_max") is not None:
            conditions.append(f"price_usd <= ${param_idx}")
            params.append(filters["price_max"])
            param_idx += 1
            covered.add("price_max")
        if filters.get("beds_min") is not None:
            conditions.append(f"bedroom_count >= ${param_idx}")
            params.append(filters["beds_min"])
            param_idx += 1
            covered.add("beds_min")
        if filters.get("baths_min") is not None:
            conditions.append(f"bathroom_count >= ${param_idx}")
            params.append(filters["baths_min"])
            param_idx += 1
            covered.add("baths_min")
        if filters.get("sqft_min") is not None:
            conditions.append(f"area_sqft >= ${param_idx}")
            params.append(filters["sqft_min"])
            param_idx += 1
            covered.add("sqft_min")
        if filters.get("sqft_max") is not None:
            conditions.append(f"area_sqft <= ${param_idx}")
            params.append(filters["sqft_max"])
            param_idx += 1
            covered.add("sqft_max")
        if filters.get("year_from") is not None:
            conditions.append(f"year_built >= ${param_idx}")
            params.append(filters["year_from"])
            param_idx += 1
            covered.add("year_from")
        if filters.get("year_to") is not None:
            conditions.append(f"year_built <= ${param_idx}")
            params.append(filters["year_to"])
            param_idx += 1
            covered.add("year_to")
        if filters.get("property_types"):
            conditions.append(f"UPPER(home_type) = ANY(${param_idx}::text[])")
            params.append(filters["property_types"])
            param_idx += 1
            covered.add("property_types")
        if filters.get("financing"):
            conditions.append(f"financing && ${param_idx}::text[]")
            params.append(filters["financing"])
            param_idx += 1
            covered.add("financing")

    # LLM-extracted criteria; skip sub-conditions already covered by filters.
    for criterion in hard_criteria:
        if isinstance(criterion, RoomCountCriterion):
            col = _room_type_to_column(criterion.room_type)
            # NB: not named `bounds` — that would shadow the bbox parameter and make
            # the summary log below claim bounds=yes on every room-count query.
            count_bounds = ((criterion.exact_count, "="),
                            (criterion.min_count, ">="),
                            (criterion.max_count, "<="))
            if col is None and any(b is not None for b, _ in count_bounds):
                # Room type without a denormalized column (Pool, Exterior, Office…):
                # count via the rooms table instead of silently dropping the criterion
                # (which made "2 pools" match the whole catalog).
                count_sub = (
                    f"COALESCE((SELECT r.count FROM rooms r "
                    f"WHERE r.property_id = properties.id "
                    f"AND lower(r.room_type) = lower(${param_idx})), 0)"
                )
                params.append(criterion.room_type)
                param_idx += 1
                for bound, op in count_bounds:
                    if bound is not None:
                        conditions.append(f"{count_sub} {op} ${param_idx}")
                        params.append(bound)
                        param_idx += 1
            if col:
                room_min_field = {
                    "bedroom": "beds_min",
                    "bathroom": "baths_min",
                }.get(criterion.room_type.lower())
                # A UI room filter REPLACES the parsed criterion entirely — exact
                # count included ("4 bedrooms" typed + filter 3 must search >=3,
                # not ==4 AND >=3).
                overridden = room_min_field in covered
                if criterion.exact_count is not None and not overridden:
                    conditions.append(f"{col} = ${param_idx}")
                    params.append(criterion.exact_count)
                    param_idx += 1
                if criterion.min_count is not None and not overridden:
                    conditions.append(f"{col} >= ${param_idx}")
                    params.append(criterion.min_count)
                    param_idx += 1
                if criterion.max_count is not None and not overridden:
                    conditions.append(f"{col} <= ${param_idx}")
                    params.append(criterion.max_count)
                    param_idx += 1

        elif isinstance(criterion, PriceCriterion):
            if criterion.min_price is not None and "price_min" not in covered:
                conditions.append(f"price_usd >= ${param_idx}")
                params.append(criterion.min_price)
                param_idx += 1
            if criterion.max_price is not None and "price_max" not in covered:
                conditions.append(f"price_usd <= ${param_idx}")
                params.append(criterion.max_price)
                param_idx += 1

        elif isinstance(criterion, AreaCriterion):
            if criterion.min_sqft is not None and "sqft_min" not in covered:
                conditions.append(f"area_sqft >= ${param_idx}")
                params.append(criterion.min_sqft)
                param_idx += 1
            if criterion.max_sqft is not None and "sqft_max" not in covered:
                conditions.append(f"area_sqft <= ${param_idx}")
                params.append(criterion.max_sqft)
                param_idx += 1

        elif isinstance(criterion, LocationCriterion):
            # Place-name matching semantics (STRICT municipal):
            #  - city/locality columns: whole-FIELD equality — 'melbourne' must not
            #    return Melbourne Beach / West Melbourne / Melbourne Village homes;
            #    those are different municipalities whose names merely embed the word.
            #  - neighborhood/district/county columns: whole-WORD match — 'viera'
            #    should find neighborhood 'Viera East' and district 'Bridgewater at
            #    Viera' (but not 'Riviera Isles': word, not substring).
            # State/country stay exact via variants.
            # In geo-location mode the polygon condition replaces the TARGET
            # field's name condition (and only that one).
            def _by_polygon(field_name: str) -> bool:
                return polygon_target == (id(criterion), field_name)

            if criterion.city and not _by_polygon("city"):
                conditions.append(
                    f"(lower(trim(city)) = lower(${param_idx}) "
                    f"OR lower(trim(coalesce(locality, ''))) = lower(${param_idx}) "
                    f"OR neighborhood ~* ${param_idx + 1} OR district ~* ${param_idx + 1} "
                    f"OR county ~* ${param_idx + 1})"
                )
                params.append(criterion.city.strip())
                params.append(_word_match_pattern(criterion.city))
                param_idx += 2
            if criterion.locality and not _by_polygon("locality"):
                conditions.append(
                    f"(lower(trim(coalesce(locality, ''))) = lower(${param_idx}) "
                    f"OR lower(trim(city)) = lower(${param_idx}))"
                )
                params.append(criterion.locality.strip())
                param_idx += 1
            if criterion.neighborhood and not _by_polygon("neighborhood"):
                conditions.append(f"neighborhood ~* ${param_idx}")
                params.append(_word_match_pattern(criterion.neighborhood))
                param_idx += 1
            if criterion.county and not _by_polygon("county"):
                conditions.append(f"county ~* ${param_idx}")
                params.append(_word_match_pattern(criterion.county))
                param_idx += 1
            if criterion.street:
                conditions.append(f"street ~* ${param_idx}")
                params.append(_word_match_pattern(criterion.street))
                param_idx += 1
            if criterion.district and not _by_polygon("district"):
                conditions.append(f"district ~* ${param_idx}")
                params.append(_word_match_pattern(criterion.district))
                param_idx += 1
            # State/country are matched against ALL accepted forms: users type
            # "Florida" while records store "FL" (an exact compare returned 0 results).
            # An empty variant list would make `= ANY(ARRAY[])` always FALSE and zero
            # the whole query, so a blank value adds no condition at all.
            if criterion.state and (sv := state_variants(criterion.state)):
                conditions.append(f"LOWER(state) = ANY(${param_idx}::text[])")
                params.append(sv)
                param_idx += 1
            if criterion.country and (cv := country_variants(criterion.country)):
                conditions.append(f"LOWER(country) = ANY(${param_idx}::text[])")
                params.append(cv)
                param_idx += 1

        elif isinstance(criterion, PropertyCriterion):
            if criterion.home_type and "property_types" not in covered:
                conditions.append(f"UPPER(home_type) = UPPER(${param_idx})")
                params.append(criterion.home_type)
                param_idx += 1
            if criterion.min_rent is not None:
                conditions.append(f"rent_estimate >= ${param_idx}")
                params.append(criterion.min_rent)
                param_idx += 1
            if criterion.max_rent is not None:
                conditions.append(f"rent_estimate <= ${param_idx}")
                params.append(criterion.max_rent)
                param_idx += 1
            if criterion.min_year_built is not None and "year_from" not in covered:
                conditions.append(f"year_built >= ${param_idx}")
                params.append(criterion.min_year_built)
                param_idx += 1
            if criterion.max_year_built is not None and "year_to" not in covered:
                conditions.append(f"year_built <= ${param_idx}")
                params.append(criterion.max_year_built)
                param_idx += 1
            if criterion.min_lot_sqft is not None:
                conditions.append(f"lot_size_sqft >= ${param_idx}")
                params.append(criterion.min_lot_sqft)
                param_idx += 1
            if criterion.max_lot_sqft is not None:
                conditions.append(f"lot_size_sqft <= ${param_idx}")
                params.append(criterion.max_lot_sqft)
                param_idx += 1
            if criterion.min_stories is not None:
                conditions.append(f"stories >= ${param_idx}")
                params.append(criterion.min_stories)
                param_idx += 1
            if criterion.max_stories is not None:
                conditions.append(f"stories <= ${param_idx}")
                params.append(criterion.max_stories)
                param_idx += 1
            # has_pool / has_waterfront deliberately skipped: handled as features so positive + negative sum to the total.

    # Homes by default: land (home_type LOT) only appears when a type filter or the
    # parsed query asked for a type — "homes in palm bay" must not fill up with lots.
    type_requested = bool((filters or {}).get("property_types")) or any(
        isinstance(c, PropertyCriterion) and c.home_type for c in hard_criteria
    )
    if homes_only and not type_requested:
        conditions.append("(home_type IS NULL OR UPPER(home_type) <> 'LOT')")

    where_clause = " AND ".join(conditions) if conditions else "TRUE"
    query = f"SELECT id FROM properties WHERE {where_clause}"

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    property_ids = [row["id"] for row in rows]
    logger.info(
        f"Hard filter: {len(property_ids)} properties match "
        f"({len(conditions)} conditions, bounds={'yes' if bounds else 'no'}, "
        f"filters={'yes' if filters else 'no'})"
    )
    return property_ids


def _room_type_to_column(room_type: str) -> str | None:
    mapping = {
        "bedroom": "bedroom_count",
        "bathroom": "bathroom_count",
        "kitchen": "kitchen_count",
        "living room": "living_room_count",
        "dining room": "dining_room_count",
        "garage": "garage_count",
    }
    return mapping.get(room_type.lower())


async def drop_district_name_outliers(
    pool: asyncpg.Pool, criteria: list[Criterion], property_ids: list[int]
) -> list[int]:
    """Drop properties whose ONLY match for a searched place name is the district
    (subdivision plat) column AND whose own city/locality differs from every strong
    match's. A plat name proves NAMING, not location — 'MELBOURNE HEIGHTS SEC C'
    sits in Malabar, and Palm Bay's plats are named 'Port Malabar Unit NN' — while
    city/locality/neighborhood/county are location facts.

    The rule: when strong matches exist, a district-only match survives only if its
    own city or locality appears among the strong matches' cities/localities
    ('Bridgewater at Viera' homes share mailing city Melbourne with the direct Viera
    matches; 'Port Malabar' homes are Palm Bay's, not Malabar's). With NO strong
    matches at all, district hits stand alone unchecked — that is a pure subdivision
    search ('homes in Bridgewater') and the only evidence there is. Never raises;
    failures keep all ids."""
    place_names = [
        c.city.strip() for c in criteria
        if isinstance(c, LocationCriterion) and c.city and c.city.strip()
    ]
    if not place_names or not property_ids:
        return property_ids

    try:
        dropped: set[int] = set()
        async with pool.acquire() as conn:
            for name in place_names:
                # Mirrors apply_hard_filters' strict semantics: city/locality by
                # whole-field equality, neighborhood/county by whole-word match.
                rows = await conn.fetch(
                    """
                    WITH m AS (
                        SELECT id,
                               lower(trim(city)) AS own_city,
                               lower(trim(coalesce(locality, ''))) AS own_locality,
                               lower(trim(city)) = lower($2) AS city_eq,
                               lower(trim(coalesce(locality, ''))) = lower($2) AS locality_eq,
                               (lower(trim(city)) = lower($2)
                                OR lower(trim(coalesce(locality, ''))) = lower($2)
                                OR coalesce(neighborhood, '') ~* $3
                                OR coalesce(county, '') ~* $3) AS strong,
                               coalesce(district, '') ~* $3 AS via_district
                        FROM properties
                        WHERE id = ANY($1)
                    ),
                    -- Anchor places = mailing cities of city-equality matches; only
                    -- when the name is not a mailing city at all (Viera) do
                    -- locality-equality matches anchor instead. Never both: homes
                    -- whose Photon locality fuzzily names the searched city would
                    -- donate THEIR cities and re-admit sibling-town plat homes.
                    allowed AS (
                        SELECT DISTINCT own_city AS place FROM m WHERE city_eq
                        UNION ALL
                        SELECT DISTINCT own_city FROM m
                        WHERE locality_eq AND NOT EXISTS (SELECT 1 FROM m WHERE city_eq)
                    )
                    SELECT m.id
                    FROM m
                    WHERE m.via_district AND NOT m.strong
                      AND EXISTS (SELECT 1 FROM allowed)
                      AND m.own_city NOT IN (SELECT place FROM allowed)
                      AND (m.own_locality = ''
                           OR m.own_locality NOT IN (SELECT place FROM allowed))
                    """,
                    property_ids,
                    name,
                    _word_match_pattern(name),
                )
                if rows:
                    dropped.update(r["id"] for r in rows)
                    logger.info(
                        f"District-name outliers for '{name}': dropped {len(rows)} "
                        f"properties whose city/locality matches no strong result"
                    )
        if not dropped:
            return property_ids
        return [pid for pid in property_ids if pid not in dropped]
    except Exception:
        logger.exception("District outlier trim failed — keeping all results")
        return property_ids
