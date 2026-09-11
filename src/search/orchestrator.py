"""Search orchestrator: parse query -> hard filters -> proximity -> color -> feature matching."""

import asyncio
import json
import logging

import asyncpg

from config.settings import settings
from src.data.feature_registry import registry
from src.data.geolocate import locate_by_point
from src.data.us_states import state_variants
from src.models.search import (
    AreaCriterion,
    ParsedQuery,
    AreaRelationCriterion,
    ColorRoomCriterion,
    FeatureCriterion,
    LocationCriterion,
    PriceCriterion,
    PropertyCriterion,
    ProximityCriterion,
    RoomCountCriterion,
)
from src.search.address_lookup import known_cities, match_address, parse_address
from src.search.feature_resolver import resolve_feature_phrases
from src.search.filter_engine import apply_hard_filters, drop_district_name_outliers
from src.search.geo_search import apply_area_relation_filters, apply_proximity_filters
from src.search.query_parser import parse_query
from src.search.region_resolver import (
    extract_search_area,
    keep_majority_location,
    resolve_search_region,
)

logger = logging.getLogger(__name__)


async def _match_feature_set(
    conn,
    property_ids: list[int],
    feature: str,
    alternatives: list[str],
    room_context: str | None,
) -> set[int]:
    """Match feature + alternatives against room_instances via GIN-indexed `features &&`."""
    terms = alternatives if alternatives else [feature]
    if not terms:
        return set()

    if room_context:
        rows = await conn.fetch("""
            SELECT DISTINCT property_id FROM room_instances
            WHERE property_id = ANY($1)
              AND room_type = $2
              AND features && $3::text[]
        """, property_ids, room_context, terms)
    else:
        rows = await conn.fetch("""
            SELECT DISTINCT property_id FROM room_instances
            WHERE property_id = ANY($1)
              AND features && $2::text[]
        """, property_ids, terms)
    return {row["property_id"] for row in rows}


async def _match_features(
    pool: asyncpg.Pool,
    property_ids: list[int],
    feature_criteria: list[FeatureCriterion],
    feature_alternatives: dict[str, list[str]] | None = None,
) -> list[int]:
    """Filter IDs by feature criteria: match each in parallel, then intersect (positive) / subtract (negated)."""
    if not property_ids or not feature_criteria:
        return property_ids

    feature_alternatives = feature_alternatives or {}
    initial_ids = property_ids

    async def _run(fc: FeatureCriterion) -> set[int]:
        alts = feature_alternatives.get(fc.feature, [fc.feature])
        async with pool.acquire() as conn:
            return await _match_feature_set(
                conn, initial_ids, fc.feature, alts, fc.room_context
            )

    matches = await asyncio.gather(*[_run(fc) for fc in feature_criteria])

    result_ids = set(property_ids)
    for fc, matched in zip(feature_criteria, matches):
        alts = feature_alternatives.get(fc.feature, [fc.feature])
        if fc.negated:
            logger.info(
                f"NEGATED '{fc.feature}' (alts={len(alts)}) excluded {len(matched)} properties"
            )
            result_ids = result_ids - matched
        else:
            logger.info(
                f"POSITIVE '{fc.feature}' (alts={len(alts)}) matched {len(matched)} properties"
            )
            result_ids = result_ids & matched

    return list(result_ids)


def _build_alternatives(
    feature_criteria: list[FeatureCriterion],
    reconstructed_queries: list[str],  # kept for API compat, unused
) -> dict[str, list[str]]:
    """Map each feature -> alternatives from the registry (deterministic; reconstructed_queries ignored)."""
    if not feature_criteria:
        return {}

    return {
        fc.feature: registry.get_feature_alternatives(fc.feature)
        for fc in feature_criteria
    }


_POOL_PHRASES = ("pool", "swimming pool")

# Below this many results, hard feature requirements are relaxed one at a time
# (weakest evidence first) into ranking signals — a conversational query must
# never come back empty because one phrase had no photo evidence.
RELAX_FLOOR = 5


def _extract_pool_filters(
    feature_criteria: list[FeatureCriterion],
) -> tuple[bool | None, bool | None, bool, list[FeatureCriterion]]:
    """Extract pool intent (want_pool, want_covered, no_uncovered) for structured-column filtering, leaving other criteria in `remaining`."""
    want_pool: bool | None = None
    want_covered: bool | None = None
    no_uncovered = False
    remaining: list[FeatureCriterion] = []
    for fc in feature_criteria:
        nf = fc.feature.strip().lower()
        if nf in _POOL_PHRASES:                    # "pool" / "swimming pool"
            want_pool = not fc.negated
        elif nf == "covered pool":
            # Served by has_covered_pool alone (itself pool evidence); no existence gate.
            want_covered = not fc.negated          # positive → True, "without" → False
        elif nf == "uncovered pool":
            if fc.negated:
                # "no uncovered pool" = no-pool OR covered — an OR, so it can't be
                # expressed by AND-ing the two flags (that wrongly required a pool).
                no_uncovered = True
            else:
                want_covered = False
                want_pool = True                   # uncovered pool still requires a pool
        else:
            remaining.append(fc)
    return want_pool, want_covered, no_uncovered, remaining


# "Beachside" is WHERE a home is, not what a photo shows — the barrier-island
# towns between the Indian River and the Atlantic. Matched by city region id
# (with the city text as a fallback for rows whose region assignment differs).
_BEACHSIDE_CITIES = (
    "cocoa beach", "cape canaveral", "satellite beach", "indian harbour beach",
    "indialantic", "melbourne beach",
)
_BEACHSIDE_PHRASES = ("beachside", "beach side", "oceanside", "ocean side",
                      "barrier island", "on the beach", "beach area")


def _extract_beachside(
    feature_criteria: list[FeatureCriterion],
) -> tuple[bool | None, list[FeatureCriterion]]:
    """Pull the beach-area intent out of the feature list: (want_beachside, remaining)."""
    want: bool | None = None
    remaining: list[FeatureCriterion] = []
    for fc in feature_criteria:
        nf = fc.feature.strip().lower()
        if nf in _BEACHSIDE_PHRASES or nf.replace("-", " ") in _BEACHSIDE_PHRASES:
            want = not fc.negated
        else:
            remaining.append(fc)
    return want, remaining


async def _filter_by_beachside(
    pool: asyncpg.Pool, property_ids: list[int], want: bool
) -> list[int]:
    if not property_ids:
        return property_ids
    neg = "" if want else "NOT "
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id FROM properties p
            WHERE id = ANY($1) AND {neg}(
                p.city_region_id IN (
                    SELECT regionid FROM regions
                    WHERE lower(regionname) = ANY($2::text[]) AND statecode = 'FL'
                )
                OR lower(p.city) = ANY($2::text[])
            )
            """,
            property_ids, list(_BEACHSIDE_CITIES),
        )
    return [r["id"] for r in rows]


async def _filter_by_pool_evidence(
    pool: asyncpg.Pool, property_ids: list[int], want: bool
) -> list[int]:
    """Keep (want=True) / drop (want=False) IDs with a pool by IMAGE evidence
    ONLY: a private (non-community) Pool photo, or the image-derived covered
    flag. Listing metadata (resoFacts.hasPrivatePool) is NOT trusted — agents
    routinely set it for the community's pool; what the camera saw decides."""
    if not property_ids:
        return property_ids
    neg = "" if want else "NOT "
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id FROM properties p
            WHERE id = ANY($1) AND {neg}(
                p.has_covered_pool OR EXISTS (
                    SELECT 1 FROM room_instances ri
                    WHERE ri.property_id = p.id AND ri.room_type = 'Pool'
                      AND NOT EXISTS (SELECT 1 FROM unnest(ri.features) cf WHERE cf ILIKE '%community%pool%')
                )
            )
            """,
            property_ids,
        )
    return [r["id"] for r in rows]


async def _filter_by_has_covered_pool(
    pool: asyncpg.Pool, property_ids: list[int], want: bool
) -> list[int]:
    """Keep property_ids whose properties.has_covered_pool == want."""
    if not property_ids:
        return property_ids
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id FROM properties WHERE id = ANY($1) AND has_covered_pool = $2",
            property_ids, want,
        )
    return [r["id"] for r in rows]


async def _filter_by_no_uncovered_pool(
    pool: asyncpg.Pool, property_ids: list[int]
) -> list[int]:
    """'No uncovered pool': keep properties with NO pool at all OR a covered one.
    Community-tagged Pool photos are not the property's pool (see
    _filter_by_pool_evidence)."""
    if not property_ids:
        return property_ids
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id FROM properties p
            WHERE id = ANY($1) AND (
                p.has_covered_pool
                OR NOT EXISTS (
                    SELECT 1 FROM room_instances ri
                    WHERE ri.property_id = p.id AND ri.room_type = 'Pool'
                      AND NOT EXISTS (SELECT 1 FROM unnest(ri.features) cf WHERE cf ILIKE '%community%pool%')
                )
            )
            """,
            property_ids,
        )
    return [r["id"] for r in rows]


# Nearest palette colors to fall back to when a requested color is never assigned
# to a room type (e.g. "gold" kitchens don't exist; warm ones read as yellow/beige/brown).
_COLOR_FALLBACKS = {
    "gold":   ["yellow", "beige"],
    "yellow": ["gold", "beige", "orange"],
    "orange": ["red", "brown", "yellow"],
    "beige":  ["brown", "white", "gray"],
    "brown":  ["beige", "gray"],
    "white":  ["beige", "gray"],
    "gray":   ["white", "beige", "black"],
    "black":  ["gray"],
    "blue":   ["gray", "green"],
    "green":  ["gray", "blue"],
    "red":    ["orange", "brown", "pink"],
    "pink":   ["red", "purple", "white"],
    "purple": ["blue", "pink"],
}


async def _color_room_matched(
    conn, ids: list[int], crit: ColorRoomCriterion
) -> tuple[set[int], list[str]]:
    """Property ids whose DOMINANT `crit.room_type` color (most-frequent non-Unknown color across that room's photos; RANK ties keep all co-top colors) is `crit.color` — a single mislabeled photo can't qualify a property. For a POSITIVE search, widens to nearest palette colors when no property is dominantly that color. NEGATED stays literal: "no white kitchen" must exclude white kitchens only, so if no kitchen is white it excludes nothing (widening would wrongly drop beige/gray kitchens). Returns (matched ids, colors used)."""
    colors = [crit.color]
    if not crit.negated:
        # Scope the "is this color dominant anywhere?" probe to the SAME candidate set as
        # the match below — otherwise an unrelated property elsewhere in the DB being
        # dominantly this color would suppress the fallback for the actual candidates.
        dom_exists = await conn.fetchval("""
            SELECT EXISTS(
                SELECT 1 FROM (
                    SELECT color, RANK() OVER (PARTITION BY property_id ORDER BY count(*) DESC) AS rnk
                    FROM room_instances
                    WHERE property_id = ANY($1) AND room_type = $2
                      AND color IS NOT NULL AND color <> 'Unknown'
                    GROUP BY property_id, color
                ) t WHERE rnk = 1 AND color = $3
            )
        """, ids, crit.room_type, crit.color)
        if not dom_exists and _COLOR_FALLBACKS.get(crit.color):
            colors = _COLOR_FALLBACKS[crit.color]
            logger.info(f"COLOR ROOM fallback: {crit.color} {crit.room_type} not dominant -> {colors}")
    rows = await conn.fetch("""
        SELECT property_id FROM (
            SELECT property_id, color,
                   RANK() OVER (PARTITION BY property_id ORDER BY count(*) DESC) AS rnk
            FROM room_instances
            WHERE property_id = ANY($1) AND room_type = $2
              AND color IS NOT NULL AND color <> 'Unknown'
            GROUP BY property_id, color
        ) t WHERE rnk = 1 AND color = ANY($3)
    """, ids, crit.room_type, colors)
    return {row["property_id"] for row in rows}, colors


async def _match_color_rooms(
    pool: asyncpg.Pool,
    property_ids: list[int],
    color_room_criteria: list[ColorRoomCriterion],
) -> list[int]:
    """Filter IDs by room color: intersect (positive) / subtract (negated), with nearest-color fallback for a palette color absent from that room type."""
    if not property_ids or not color_room_criteria:
        return property_ids
    result_ids = set(property_ids)
    async with pool.acquire() as conn:
        for crit in color_room_criteria:
            matched, _ = await _color_room_matched(conn, list(result_ids), crit)
            result_ids = result_ids - matched if crit.negated else result_ids & matched
    return list(result_ids)


async def _load_pins(pool: asyncpg.Pool, property_ids: list[int]) -> list[dict]:
    """Lightweight map pins for ALL matched properties (the card list is paginated,
    the map is not): id + coords + price only. 0,0-placeholder coords -> null."""
    if not property_ids:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT guid, ST_Y(geom::geometry) AS lat, ST_X(geom::geometry) AS lon, price_usd
            FROM properties WHERE id = ANY($1) ORDER BY id
            """,
            property_ids,
        )
    return [
        {
            "propertyId": r["guid"],
            "latitude": r["lat"] if not (r["lat"] == 0 and r["lon"] == 0) else None,
            "longitude": r["lon"] if not (r["lat"] == 0 and r["lon"] == 0) else None,
            "price": r["price_usd"],
        }
        for r in rows
    ]


# Fixed presentation order for photo groups — every property's strip reads the
# same way (client feedback: predictable placement; "Other" always last).
# "Main" is prepended separately. Legacy room types from older vision prompts
# (Laundry Room, Utility Room, ...) still live in the DB — they take the
# unlisted rank, after the named groups but strictly before Other.
_GROUP_ORDER = {
    rt: i
    for i, rt in enumerate([
        "Exterior", "Pool", "Kitchen", "Living Room", "Dining Room",
        "Bedroom", "Bathroom", "Garage", "Community Pool",
    ])
}
_GROUP_UNLISTED_RANK = len(_GROUP_ORDER)
_GROUP_ORDER["Other"] = _GROUP_UNLISTED_RANK + 1


def _photo_groups(
    photos_json: str | None,
    room_type_by_url: dict[str, str],
    full_size: bool = False,
    water_urls: set[str] | None = None,
) -> list[dict]:
    """originalPhotos JSON -> [{"roomType": "Bathroom", "urls": [...]}, ...], grouped
    by the vision-derived room type of each image (looked up by the photo's canonical
    highest-width url — the id room_instances stores). roomType is the native name
    ("Living Room", "Pool"); unclassified images group under "Other". Groups appear
    in the fixed _GROUP_ORDER (Exterior first, Other last). A synthetic "Main" group
    leads the list — the FIRST image of every group in that same order, so
    Main[0] == Exterior[0]: the card's cover photo is the house front (client
    requirement); listings with no Exterior photo fall back to the listing's own
    lead photo as cover. Within the Pool group, photos in `water_urls` (canonical
    urls whose tags show actual pool water) sort ahead of equipment-only shots.
    urls are the SMALLEST-width jpegs (search results are cards/thumbnails;
    full-size urls live on the property detail endpoint)."""
    if not photos_json:
        return []
    try:
        photos = json.loads(photos_json)
    except (json.JSONDecodeError, TypeError):
        return []
    # rt -> [(serve_url, canonical_url)] — canonical kept alongside because the
    # water-first sort keys on it while the served url may be the thumbnail.
    groups: dict[str, list[tuple[str, str]]] = {}
    first_serve_url: str | None = None
    for photo in photos or []:
        if not isinstance(photo, dict):
            continue
        jpegs = ((photo.get("mixedSources") or {}).get("jpeg") or [])
        candidates = [s for s in jpegs if isinstance(s, dict) and s.get("url")]
        if not candidates:
            continue
        smallest = min(candidates, key=lambda s: s.get("width") or 10**9)
        canonical = max(candidates, key=lambda s: s.get("width") or 0)["url"]
        room_type = room_type_by_url.get(canonical)
        if not room_type or room_type in ("Unknown", "Main"):
            # "Unknown" = vision couldn't classify. "Main" is reserved for the
            # synthetic highlight group below (impossible with the current vision
            # prompt, but cheap insurance against a duplicate group).
            room_type = "Other"
        # full_size (detail/lightbox endpoint) serves the canonical highest-width
        # url; search results serve the smallest (cards/thumbnails).
        serve_url = canonical if full_size else smallest["url"]
        if first_serve_url is None:
            first_serve_url = serve_url
        groups.setdefault(room_type, []).append((serve_url, canonical))
    if not groups:
        return []
    if water_urls and "Pool" in groups:
        # Stable sort: water shots lead, equipment/utility shots trail, listing
        # order preserved within each half.
        groups["Pool"].sort(key=lambda p: p[1] not in water_urls)
    ordered = sorted(
        groups.items(), key=lambda kv: _GROUP_ORDER.get(kv[0], _GROUP_UNLISTED_RANK)
    )
    main_urls = [photos_[0][0] for _, photos_ in ordered]
    if "Exterior" not in groups and first_serve_url is not None:
        # No exterior photo at all (rare) — cover falls back to the listing's
        # lead photo instead of whichever group happens to sort first.
        if first_serve_url in main_urls:
            main_urls.remove(first_serve_url)
        main_urls.insert(0, first_serve_url)
    main = {"roomType": "Main", "urls": main_urls}
    return [main] + [
        {"roomType": rt, "urls": [u for u, _ in photos_]} for rt, photos_ in ordered
    ]


def _blank_to_none(value: str | None) -> str | None:
    return value.strip() if value and value.strip() else None


async def _load_results(pool: asyncpg.Pool, property_ids: list[int]) -> list[dict]:
    """Load the matched properties for /search as briefproperty dicts, in one SELECT
    (properties + the raw feed record for fields never promoted to columns:
    currency, homeStatus, daysOnZillow, photos), ordered by id so the result list is
    stable across identical requests. Coordinates come back as null for the rare
    'undisclosed address' listings whose feed record carries no latitude/longitude
    (stored as 0,0 because the column is NOT NULL); returning 0,0 would drop a map
    pin in the Atlantic."""
    if not property_ids:
        return []
    async with pool.acquire() as conn:
        # Room type per photo (for photo captions), keyed by the canonical url the
        # ingest stores. One query for the whole page.
        room_rows = await conn.fetch(
            """
            SELECT property_id, photo_url,
                   CASE WHEN room_type = 'Pool' AND EXISTS (SELECT 1 FROM unnest(features) cf WHERE cf ILIKE '%community%pool%')
                        THEN 'Community Pool' ELSE room_type END AS room_type,
                   (room_type = 'Pool'
                    AND features && ARRAY['pool','covered pool','uncovered pool']) AS pool_water
            FROM room_instances
            WHERE property_id = ANY($1) AND photo_url IS NOT NULL
            """,
            property_ids,
        )
        rows = await conn.fetch(
            """
            SELECT p.id AS internal_id,
                   p.guid,
                   ST_Y(p.geom::geometry) AS lat,
                   ST_X(p.geom::geometry) AS lon,
                   p.price_usd, p.bedroom_count, p.bathroom_count, p.area_sqft,
                   p.street, p.city, p.state, p.postal_code,
                   p.neighborhood, p.locality, p.district,
                   p.home_type, p.year_built, p.county, p.lot_size_sqft,
                   r.data->>'currency'     AS currency,
                   r.data->>'homeStatus'   AS home_status,
                   r.data->>'daysOnZillow' AS days_on_market,
                   r.data->>'originalPhotos' AS photos_json
            FROM properties p
            LEFT JOIN raw_properties r ON r.id = p.guid
            WHERE p.id = ANY($1)
            ORDER BY p.id
            """,
            property_ids,
        )
    room_types: dict[int, dict[str, str]] = {}
    water_urls: dict[int, set[str]] = {}
    for rr in room_rows:
        room_types.setdefault(rr["property_id"], {})[rr["photo_url"]] = rr["room_type"]
        if rr["pool_water"]:
            water_urls.setdefault(rr["property_id"], set()).add(rr["photo_url"])

    results = []
    for r in rows:
        lat, lon = r["lat"], r["lon"]
        placeable = not (lat == 0 and lon == 0)
        try:
            days = int(r["days_on_market"]) if r["days_on_market"] else None
        except ValueError:
            days = None
        results.append({
            "_internal_id": r["internal_id"],
            "propertyId": r["guid"],
            "price": r["price_usd"],
            "currency": r["currency"] or "USD",
            "latitude": lat if placeable else None,
            "longitude": lon if placeable else None,
            "bedrooms": r["bedroom_count"],
            "bathrooms": r["bathroom_count"],
            # 0 is the ingest's "unknown" sentinel (land lots, undisclosed) — null it
            # rather than showing "0 sqft", matching lotAreaValue's treatment.
            "livingArea": r["area_sqft"] or None,
            "address": {
                "streetAddress": _blank_to_none(r["street"]),
                "city": _blank_to_none(r["city"]),
                "state": _blank_to_none(r["state"]),
                "zipcode": _blank_to_none(r["postal_code"]),
                "neighborhood": _blank_to_none(r["neighborhood"]),
                # community = the OSM place (locality): the level Zillow's mailing
                # address can't express (Viera, Suntree...).
                "community": _blank_to_none(r["locality"]),
                "subdivision": _blank_to_none(r["district"]),
            },
            "homeStatus": r["home_status"],
            "homeType": r["home_type"],
            "propertyphotos": _photo_groups(
                r["photos_json"], room_types.get(r["internal_id"], {}),
                water_urls=water_urls.get(r["internal_id"]),
            ),
            # Frozen at scrape time — does NOT tick daily after ingest.
            "daysOnmarket": days,
            "yearBuilt": r["year_built"],
            "county": _blank_to_none(r["county"]),
            "lotAreaValue": float(r["lot_size_sqft"]) if r["lot_size_sqft"] else None,
        })
    return results


def _criterion_labels(criterion) -> list[str]:
    """Human-readable labels for each SQL condition a criterion produces."""
    labels: list[str] = []
    if isinstance(criterion, RoomCountCriterion):
        if criterion.exact_count is not None:
            labels.append(f"{criterion.room_type}=={criterion.exact_count}")
        if criterion.min_count is not None:
            labels.append(f"{criterion.room_type}>={criterion.min_count}")
        if criterion.max_count is not None:
            labels.append(f"{criterion.room_type}<={criterion.max_count}")
    elif isinstance(criterion, PriceCriterion):
        if criterion.min_price is not None:
            labels.append(f"price>={criterion.min_price}")
        if criterion.max_price is not None:
            labels.append(f"price<={criterion.max_price}")
    elif isinstance(criterion, AreaCriterion):
        if criterion.min_sqft is not None:
            labels.append(f"area>={criterion.min_sqft}")
        if criterion.max_sqft is not None:
            labels.append(f"area<={criterion.max_sqft}")
    elif isinstance(criterion, LocationCriterion):
        for attr in ("city", "state", "country", "district",
                     "county", "neighborhood", "locality", "street"):
            val = getattr(criterion, attr)
            if val:
                labels.append(f"{attr}={val}")
    elif isinstance(criterion, PropertyCriterion):
        for attr, op in [
            ("home_type", "="), ("min_rent", ">="), ("max_rent", "<="),
            ("min_year_built", ">="), ("max_year_built", "<="),
            ("min_lot_sqft", ">="), ("max_lot_sqft", "<="),
            ("min_stories", ">="), ("max_stories", "<="),
        ]:
            val = getattr(criterion, attr)
            if val is not None:
                labels.append(f"{attr}{op}{val}")
    return labels


_FILTER_STEP_LABELS = [
    ("price_min", "price>={v}"),
    ("price_max", "price<={v}"),
    ("beds_min", "bedrooms>={v}"),
    ("baths_min", "bathrooms>={v}"),
    ("sqft_min", "area>={v}"),
    ("sqft_max", "area<={v}"),
    ("year_from", "year>={v}"),
    ("year_to", "year<={v}"),
    ("property_types", "home_type∈{v}"),
    ("financing", "financing∋{v}"),
]


async def _collect_hard_filter_steps(
    pool: asyncpg.Pool,
    criteria: list,
    bounds: dict | None,
    filters: dict | None = None,
    area_region_id: int | None = None,
    area_region_type: str | None = None,
    drawn_polygon: list[tuple[float, float]] | None = None,
) -> list[dict]:
    """Debug-only: apply bounds/filters/criteria one at a time, recording count per
    step. area_region_id (+area_region_type in region-ID mode) mirrors the real
    pipeline's geo-location mode so the location step's count matches what the
    search actually did (region ids / polygon, not name)."""
    steps: list[dict] = []

    async with pool.acquire() as conn:
        total = await conn.fetchval("SELECT count(*) FROM properties")
    steps.append({"step": "total_properties", "count": int(total)})

    applied: list = []
    partial_filters: dict = {}
    prev = int(total)

    if drawn_polygon:
        count = len(await apply_hard_filters(pool, applied, drawn_polygon=drawn_polygon))
        steps.append({
            "step": f"drawn polygon ({len(drawn_polygon)} points)",
            "count": count,
            "dropped": prev - count,
        })
        prev = count

    if bounds:
        count = len(await apply_hard_filters(pool, applied, bounds=bounds,
                                             drawn_polygon=drawn_polygon))
        steps.append({
            "step": "bounds",
            "count": count,
            "dropped": prev - count,
        })
        prev = count

    if filters:
        for key, label_tpl in _FILTER_STEP_LABELS:
            value = filters.get(key)
            if value is None or value == [] or value == "":
                continue
            partial_filters[key] = value
            count = len(await apply_hard_filters(
                pool, applied, bounds=bounds, filters=partial_filters,
                drawn_polygon=drawn_polygon,
            ))
            steps.append({
                "step": f"filter: {label_tpl.format(v=value)}",
                "count": count,
                "dropped": prev - count,
            })
            prev = count

    hard_types = (RoomCountCriterion, PriceCriterion, AreaCriterion,
                  LocationCriterion, PropertyCriterion)
    for c in criteria:
        if not isinstance(c, hard_types):
            continue
        applied.append(c)
        labels = _criterion_labels(c) or [c.type.value if hasattr(c, "type") else type(c).__name__]
        count = len(await apply_hard_filters(
            pool, applied, bounds=bounds, filters=filters,
            area_region_id=area_region_id, area_region_type=area_region_type,
            drawn_polygon=drawn_polygon,
        ))
        steps.append({
            "step": ", ".join(labels),
            "count": count,
            "dropped": prev - count,
        })
        prev = count

    return steps


async def search(
    query: str,
    pool: asyncpg.Pool,
    bounds: dict | None = None,
    filters: dict | None = None,
    debug: bool = False,
    page: int = 1,
    page_size: int = 50,
    client_location: dict | None = None,
    ignore_region: bool = False,
    drawn_polygon: list[tuple[float, float]] | None = None,
) -> dict:
    """Run the full search pipeline; bounds restricts to a bbox, filters override LLM
    hard-filter sub-fields, debug adds a per-step count breakdown. The heavy
    briefproperty payload is paginated (page/page_size); `pins` always covers ALL
    matches so the map never loses markers. client_location ({lat, lng} from the
    browser geolocation API) feeds the detected-location default for queries that
    name no place. ignore_region (the frontend's "remove boundaries" button) drops
    the region part of the query entirely: place criteria are stripped after
    parsing and location detection is suppressed, so the search is scoped only by
    the map bounds (when sent) — the response then carries no region at all.
    drawn_polygon ([(lat, lng), ...] from the map's draw tool) restricts results
    to homes inside that ring — the only polygon-based filtering; like bounds it
    suppresses detected-location injection."""
    # Phase 0: Exact address. "1845 Hidden Lake Dr[, Rockledge, FL 32955]" scopes
    # the search to that house (all its units when none is given); the rest of the
    # text is parsed as usual and applied STRICTLY (no relaxation). Deterministic —
    # a bare street / place name is NOT an address and takes the normal path.
    address = parse_address(query, await known_cities(pool))
    address_ids: list[int] | None = None
    if address is not None:
        address_ids = await match_address(pool, address)
        logger.info(
            "Phase 0: exact address '%s' -> %d home(s); remainder '%s'",
            address.matched_text.strip(), len(address_ids), address.remainder,
        )
        query = address.remainder
        # A typed address is the strongest location intent there is: it overrides
        # the map viewport and any drawn polygon (the frontend sends `bounds` on
        # every search — a home outside the current view must still be found).
        if bounds is not None or drawn_polygon is not None:
            logger.info("Phase 0: address query — ignoring map bounds / drawn polygon")
            bounds = None
            drawn_polygon = None

    # Phase 1: Parse. A blank query skips the LLM entirely — the structured
    # request fields (filters/bounds/drawnPolygon) carry the search instead
    # (e.g. the filter panel after "remove boundaries" emptied the bar).
    if not query.strip():
        logger.info("Phase 1: blank query — skipping parse, structured fields only")
        parsed_query = ParsedQuery(
            original_query=query, criteria=[],
            understood_intent="blank query: structured filters only",
        )
    else:
        logger.info(f"Phase 1: Parsing query: '{query}'")
        parsed_query = await parse_query(query)
        logger.info(f"Parsed {len(parsed_query.criteria)} criteria: {parsed_query.understood_intent}")

    # Phase 1.2: "Remove boundaries" — strip the region part BEFORE any location
    # machinery runs. With no LocationCriterion left, region resolution, region
    # filtering, place name-matching, and majority-state trimming all no-op
    # downstream by construction (works identically under polygon or region-id
    # search), and the guard below keeps detected-location injection off too.
    stripped_locations: list[str] = []
    if ignore_region:
        before = len(parsed_query.criteria)
        kept = []
        for c in parsed_query.criteria:
            if isinstance(c, LocationCriterion):
                # Collect the place strings so /search can also remove them from
                # updated_query — the search bar must not keep showing a place
                # the search no longer uses.
                for field in ("neighborhood", "locality", "district", "city",
                              "county", "state", "street"):
                    v = getattr(c, field, None)
                    if v and str(v).strip():
                        stripped_locations.append(str(v).strip())
                        if field == "state":
                            # The parser normalizes states ("florida" -> "FL"), so
                            # the typed text may spell it either way — strip both.
                            stripped_locations.extend(state_variants(str(v)))
            else:
                kept.append(c)
        parsed_query.criteria = kept
        if before != len(parsed_query.criteria):
            logger.info(
                f"Phase 1.2: ignore_region — dropped "
                f"{before - len(parsed_query.criteria)} location criterion(s)"
            )

    # Phase 1.4: Detected-location default. A query that names NO place searches the
    # user's own area: browser coordinates -> containing region polygon; otherwise
    # (no permission, no coordinates, or a point outside coverage) the configured
    # default area (Brevard County). Skipped when the user named a place — including
    # relational "near X"/"between X and Y" (AreaRelationCriterion), where injecting
    # the user's city would wrongly intersect the corridor — or sent map bounds
    # (the map IS the location), or the parse produced nothing at all (gibberish
    # must stay a 422, not become a Brevard search).
    location_detected = False
    has_location = any(
        isinstance(c, (LocationCriterion, AreaRelationCriterion))
        for c in parsed_query.criteria
    )
    if (not ignore_region and not has_location and address is None
            and parsed_query.criteria and bounds is None
            and drawn_polygon is None):
        detected = None
        if client_location is not None:
            detected = await locate_by_point(
                pool, client_location.get("lat"), client_location.get("lng")
            )
        if detected is None:
            detected = {
                "county": settings.default_search_county,
                "state": settings.default_search_state,
            }
        parsed_query.criteria.append(LocationCriterion(**detected))
        location_detected = True
        logger.info(f"Phase 1.4: No place in query — scoping to detected area {detected}")

    filter_steps: list[dict] = []

    # Phase 1.6 (overlapped): feature-phrase resolution depends only on the parsed
    # criteria, and its cold path (embedding + LLM #2 filter, ~5s for novel phrases)
    # dominated feature queries — start it NOW so it runs concurrently with region
    # resolution and all SQL filtering below; Phase 4 awaits the result.
    feature_criteria_all = [
        c for c in parsed_query.criteria if isinstance(c, FeatureCriterion)
    ]
    want_pool, want_covered, no_uncovered, feature_criteria = _extract_pool_filters(
        feature_criteria_all
    )
    want_beachside, feature_criteria = _extract_beachside(feature_criteria)
    feature_task: asyncio.Task | None = None
    if feature_criteria and settings.search_use_embedding_retrieval:
        feature_task = asyncio.create_task(
            resolve_feature_phrases(pool, [fc.feature for fc in feature_criteria])
        )
        # If the pipeline raises before Phase 4 awaits this task, mark its eventual
        # exception as retrieved so asyncio doesn't log an orphaned-task warning.
        feature_task.add_done_callback(
            lambda t: t.exception() if not t.cancelled() else None
        )

    # Phase 1.5: Geo-location resolution — if the searched place maps to a regions
    # row WITH a polygon, membership is decided geometrically (ST_Covers) and all
    # name-based place matching/trimming for that place is skipped.
    search_region = await resolve_search_region(pool, parsed_query.criteria)
    id_mode = bool(search_region and search_region.get("id_mode"))
    display_only = bool(search_region and search_region.get("display_only"))
    area_region_id = (
        search_region["region_id"]
        if search_region and not display_only
        and (id_mode or search_region["has_geom"]) else None
    )
    area_region_type = (
        search_region["region_type"] if search_region and id_mode else None
    )
    if search_region:
        mode = ("region-id filter" if id_mode
                else "polygon filter" if area_region_id
                else "display-only region — name matching" if display_only
                else "no polygon — name matching")
        logger.info(
            f"Phase 1.5: Region '{search_region['region_name']}' "
            f"(id {search_region['region_id']}, {mode})"
        )

    # Phase 2: Hard filters (incl. bounds + filters)
    logger.info(
        f"Phase 2: Hard filters (PostgreSQL)"
        f"{' + map bounds' if bounds else ''}"
        f"{' + filters' if filters else ''}"
    )
    if debug:
        filter_steps.extend(
            await _collect_hard_filter_steps(
                pool, parsed_query.criteria, bounds, filters,
                area_region_id=area_region_id, area_region_type=area_region_type,
                drawn_polygon=drawn_polygon,
            )
        )
    property_ids = await apply_hard_filters(
        pool, parsed_query.criteria, bounds=bounds, filters=filters,
        area_region_id=area_region_id, area_region_type=area_region_type,
        drawn_polygon=drawn_polygon,
    )
    if debug and area_region_id:
        filter_steps.append({
            "step": f"{'region-id' if id_mode else 'polygon'}: region {search_region['region_name']}",
            "count": len(property_ids),
            "dropped": 0,
        })
    if address_ids is not None:
        before_addr = len(property_ids)
        keep = set(address_ids)
        property_ids = [pid for pid in property_ids if pid in keep]
        if debug:
            filter_steps.append({
                "step": f"exact address: {address.matched_text.strip()}",
                "count": len(property_ids),
                "dropped": before_addr - len(property_ids),
            })
    if area_region_id is None:
        # Name-matching mode only: subdivision plat names embed nearby-city names
        # ("MELBOURNE HEIGHTS" sits in Malabar), so drop district-only matches whose
        # own city contradicts the strong matches. Pointless (and wrong) under a
        # polygon: geometry already decided membership.
        before_district_trim = len(property_ids)
        property_ids = await drop_district_name_outliers(pool, parsed_query.criteria, property_ids)
        if debug and len(property_ids) != before_district_trim:
            filter_steps.append({
                "step": "district_name_outliers",
                "count": len(property_ids),
                "dropped": before_district_trim - len(property_ids),
            })
    after_hard_filter_count = len(property_ids)

    # Phase 3: Proximity (one at a time so each step is recorded for debug)
    logger.info("Phase 3: Proximity filters (PostGIS)")
    proximity_criteria = [c for c in parsed_query.criteria if isinstance(c, ProximityCriterion)]
    if debug and not proximity_criteria:
        filter_steps.append({
            "step": "proximity_skipped",
            "count": after_hard_filter_count,
            "dropped": 0,
        })
    for pc in proximity_criteria:
        before = len(property_ids)
        property_ids = await apply_proximity_filters(pool, property_ids, [pc])
        if debug:
            filter_steps.append({
                "step": f"proximity: {pc.landmark_name} (<= {pc.max_distance_miles}mi)",
                "count": len(property_ids),
                "dropped": before - len(property_ids),
            })
    after_proximity_count = len(property_ids)

    # Phase 3.6: Area relations — "near A" / "between A and B" (PostGIS centroids)
    area_rel_criteria = [c for c in parsed_query.criteria if isinstance(c, AreaRelationCriterion)]
    for rc in area_rel_criteria:
        before = len(property_ids)
        property_ids = await apply_area_relation_filters(pool, property_ids, [rc])
        if debug:
            if rc.relation == "between":
                label = f"between {rc.place_a} and {rc.place_b}"
            elif rc.relation == "neighbors":
                label = f"neighbors of {rc.place_a}"
            else:
                label = f"near {rc.place_a}"
            filter_steps.append({
                "step": f"area_relation: {label}",
                "count": len(property_ids),
                "dropped": before - len(property_ids),
            })

    # Phase 3.5: Color-room matching
    color_room_criteria = [
        c for c in parsed_query.criteria if isinstance(c, ColorRoomCriterion)
    ]
    if color_room_criteria and property_ids:
        logger.info("Phase 3.5: Color-room matching (PostgreSQL)")
        if debug:
            current = set(property_ids)
            async with pool.acquire() as conn:
                for crit in color_room_criteria:
                    before = len(current)
                    matched, colors = await _color_room_matched(conn, list(current), crit)
                    if crit.negated:
                        current = current - matched
                        op = "NOT"
                    else:
                        current = current & matched
                        op = "HAS"
                    shown = crit.color if colors == [crit.color] else f"{crit.color}->{'/'.join(colors)}"
                    filter_steps.append({
                        "step": f"color_room: {op} color={shown} in {crit.room_type}",
                        "count": len(current),
                        "dropped": before - len(current),
                    })
            property_ids = list(current)
        else:
            property_ids = await _match_color_rooms(pool, property_ids, color_room_criteria)
    after_color_room_count = len(property_ids)

    # Phase 3.7: Pool — answered by IMAGE-derived evidence (private 'Pool' room photos, has_covered_pool), not fuzzy 'pool' tags. Listing metadata is not trusted.
    if no_uncovered and property_ids:
        before = len(property_ids)
        property_ids = await _filter_by_no_uncovered_pool(pool, property_ids)
        logger.info("Phase 3.7: no_uncovered_pool -> %d", len(property_ids))
        if debug:
            filter_steps.append({
                "step": "no_uncovered_pool (no pool OR covered)",
                "count": len(property_ids),
                "dropped": before - len(property_ids),
            })
    if want_pool is not None and property_ids:
        before = len(property_ids)
        property_ids = await _filter_by_pool_evidence(pool, property_ids, want_pool)
        logger.info("Phase 3.7: pool_evidence = %s -> %d", want_pool, len(property_ids))
        if debug:
            filter_steps.append({
                "step": f"pool image-evidence = {want_pool}",
                "count": len(property_ids),
                "dropped": before - len(property_ids),
            })
    if want_covered is not None and property_ids:
        before = len(property_ids)
        property_ids = await _filter_by_has_covered_pool(pool, property_ids, want_covered)
        logger.info("Phase 3.7: has_covered_pool = %s -> %d", want_covered, len(property_ids))
        if debug:
            filter_steps.append({
                "step": f"has_covered_pool = {want_covered}",
                "count": len(property_ids),
                "dropped": before - len(property_ids),
            })

    if want_beachside is not None and property_ids:
        before = len(property_ids)
        property_ids = await _filter_by_beachside(pool, property_ids, want_beachside)
        logger.info("Phase 3.8: beachside = %s -> %d", want_beachside, len(property_ids))
        if debug:
            filter_steps.append({
                "step": f"beachside towns = {want_beachside}",
                "count": len(property_ids),
                "dropped": before - len(property_ids),
            })

    relaxed: list[str] = []
    soft_criteria: list[str] = []
    soft_hits: dict[int, list[str]] = {}
    alternatives: dict[str, list[str]] = {}
    if feature_task is not None and not (feature_criteria and property_ids):
        # Filters emptied the pool (or only pool-phrases remained) — the overlapped
        # resolution is no longer needed; don't wait up to seconds for its answer.
        feature_task.cancel()
    if feature_criteria and property_ids:
        logger.info("Phase 4: Feature matching (PostgreSQL)")
        if feature_task is not None:
            # Resolution ran CONCURRENTLY with region resolution + SQL filtering
            # (started right after parse) — by now it's finished or nearly so.
            alternatives = await feature_task
            # Union each phrase's embedding list with deterministic word-subset alternatives for completeness/consistency.
            for fc in feature_criteria:
                ws = registry.get_feature_alternatives(fc.feature)
                if ws:
                    merged = set(alternatives.get(fc.feature, [])) | set(ws)
                    alternatives[fc.feature] = sorted(merged)
        else:
            # Legacy: deterministic word-subset alternatives from the registry.
            alternatives = _build_alternatives(
                feature_criteria, parsed_query.reconstructed_queries
            )
        if alternatives:
            logger.info(f"Feature alternatives: {alternatives}")
        # Sort positives-first, base-before-modifier for logical debug narration (set ops commute, so result is identical).
        feature_criteria = sorted(
            feature_criteria,
            key=lambda fc: (fc.negated, len(fc.feature.split())),
        )
        # Natural-language queries mix REQUIREMENTS with WISHES. Hard features
        # (and every negation) filter; soft ones only rank. A hard feature that
        # would starve the page below RELAX_FLOOR is relaxed — weakest evidence
        # first — into a ranking signal, and reported in `relaxed` so the UI can
        # say "showing homes without: ...". Structured criteria (place, price,
        # beds, pool) are never relaxed: those were explicit.
        # A negated SOFT phrase ("doesn't need to be big") is the absence of a
        # wish, not a ban — it must never exclude homes. Drop it outright.
        dropped = [fc for fc in feature_criteria if fc.negated and fc.strength == "soft"]
        if dropped:
            logger.info("Phase 4: ignoring negated soft wish(es): %s", [fc.feature for fc in dropped])
        feature_criteria = [fc for fc in feature_criteria if fc not in dropped]
        hard = [fc for fc in feature_criteria if fc.negated or fc.strength != "soft"]
        soft = [fc for fc in feature_criteria if not fc.negated and fc.strength == "soft"]
        pool_ids = list(property_ids)
        async with pool.acquire() as conn:
            matched: dict[str, set[int]] = {}
            for fc in feature_criteria:
                alts = alternatives.get(fc.feature, [fc.feature])
                matched[fc.feature] = await _match_feature_set(
                    conn, pool_ids, fc.feature, alts, fc.room_context
                )

        def _apply(hards: list[FeatureCriterion]) -> set[int]:
            ids = set(pool_ids)
            for fc in hards:
                ids = ids - matched[fc.feature] if fc.negated else ids & matched[fc.feature]
            return ids

        current_ids = _apply(hard)
        if debug:
            for fc in hard:
                alts = alternatives.get(fc.feature, [fc.feature])
                room_ctx = f" in {fc.room_context}" if fc.room_context else ""
                filter_steps.append({
                    "step": f"feature: {'NOT' if fc.negated else 'HAS'} '{fc.feature}'{room_ctx} (alts={len(alts)})",
                    "count": len(matched[fc.feature]) if not fc.negated else len(pool_ids) - len(matched[fc.feature]),
                    "dropped": 0,
                })
        while len(current_ids) < RELAX_FLOOR and address is None:
            relaxable = [fc for fc in hard if not fc.negated]
            # Nothing to gain: no positive requirement left, or the search area
            # itself holds fewer homes than the floor (relaxing can't add any).
            if not relaxable or current_ids == set(pool_ids):
                break
            weakest = min(relaxable, key=lambda fc: (len(matched[fc.feature]), fc.feature))
            hard.remove(weakest)
            soft.append(weakest)
            relaxed.append(weakest.feature)
            current_ids = _apply(hard)
            logger.info("Phase 4: relaxed '%s' (matched %d) -> %d results",
                        weakest.feature, len(matched[weakest.feature]), len(current_ids))
            if debug:
                filter_steps.append({
                    "step": f"relaxed '{weakest.feature}' to ranking-only",
                    "count": len(current_ids),
                    "dropped": 0,
                })
        property_ids = list(current_ids)
        if debug:
            filter_steps.append({
                "step": "hard features combined",
                "count": len(property_ids),
                "dropped": len(pool_ids) - len(property_ids),
            })
        soft_criteria = [fc.feature for fc in soft]
        for fc in soft:
            for pid in matched[fc.feature] & current_ids:
                soft_hits.setdefault(pid, []).append(fc.feature)
    after_feature_count = len(property_ids)

    # Replace LLM reconstructed_queries with the alternatives actually used (honest debug view)
    parsed_query.reconstructed_queries = sorted(
        {alt for alts in alternatives.values() for alt in alts}
    )

    # A place name searched without a state can match same-named places in several
    # states at once ("Rockledge" -> FL + GA pins on one map). Keep only the state
    # with the most matches; no-ops for feature-only or state-qualified queries.
    if area_region_id is None:
        # Name-matching mode only — under a polygon the results are one place by
        # construction and majority trimming has nothing legitimate to remove.
        before_majority = len(property_ids)
        property_ids = await keep_majority_location(pool, parsed_query.criteria, property_ids)
        if debug and len(property_ids) != before_majority:
            filter_steps.append({
                "step": "majority_location",
                "count": len(property_ids),
                "dropped": before_majority - len(property_ids),
            })

    # Pagination: sort ids once (set operations above lose order) so pages are
    # deterministic and disjoint, slice BEFORE the heavy raw-JSON/photos join —
    # page 2+ skips that cost entirely. Pins stay un-paginated for the map.
    # Ranking: homes matching more soft wishes first, then id (deterministic pages).
    property_ids = sorted(property_ids, key=lambda pid: (-len(soft_hits.get(pid, [])), pid))
    total_count = len(property_ids)
    page_ids = property_ids[(page - 1) * page_size : page * page_size]

    pins, results = await asyncio.gather(
        _load_pins(pool, property_ids), _load_results(pool, page_ids)
    )
    # _load_results orders by id; restore rank order and attach the wishes each
    # card satisfies so the UI can badge them ("great kitchen ✓").
    by_id = {r["_internal_id"]: r for r in results}
    results = [by_id[pid] for pid in page_ids if pid in by_id]
    for r in results:
        r["matchedSoft"] = soft_hits.get(r.pop("_internal_id"), [])

    stats = {
        "after_hard_filters": after_hard_filter_count,
        "after_proximity_filters": after_proximity_count,
        "after_color_room_match": after_color_room_count,
        "after_feature_match": after_feature_count,
        "after_majority_location": total_count,
        "final_results": total_count,
    }

    logger.info(f"Pipeline complete: {stats} (page {page}, {len(results)} items)")

    # Region info is returned when the region filtered the results (id/polygon
    # mode) OR when the searched name resolved to a known region row that merely
    # can't filter (display-only: no property carries its id — Viera). In every
    # case the polygon is a drawing aid, not a membership guarantee; only
    # unresolvable places (subdivisions, unknown names) return region_id None
    # with the parsed place name as a bare label.
    region = search_region if (area_region_id is not None
                               or (search_region and search_region.get("display_only"))) else None

    # The polygon the results were filtered by, as rings of {Lat, Lng} points (the
    # frontend's native boundary format), so the client can draw EXACTLY the
    # boundary that decided membership — pins can never disagree.
    polygons = None
    if region is not None:
        async with pool.acquire() as conn:
            gj = await conn.fetchval(
                "SELECT ST_AsGeoJSON(geom) FROM regions WHERE regionid = $1",
                region["region_id"],
            )
        if gj:
            multi = json.loads(gj)["coordinates"]  # [polygon][ring][[lng, lat], ...]
            polygons = [
                [{"Lat": lat, "Lng": lng} for lng, lat in ring]
                for poly in multi
                for ring in poly
            ]

    return {
        "results": results,
        "pins": pins,
        "total_count": total_count,
        "parsed_query": parsed_query,
        "stats": stats,
        "filter_steps": filter_steps if debug else None,
        "region_id": region["region_id"] if region else None,
        "region_name": region["region_name"] if region
                       else extract_search_area(parsed_query.criteria),
        "polygons": polygons,
        "location_detected": location_detected,
        "stripped_locations": stripped_locations,
        "relaxed": relaxed,
        "soft_criteria": soft_criteria,
        "address_mode": address is not None,
        # Complete address AND exactly one home survived every criterion.
        "exact_address": address is not None and total_count == 1,
    }
