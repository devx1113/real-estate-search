"""Database ingestion: convert processed Zillow data into the PostgreSQL schema."""

import logging
import math
from collections import defaultdict

import asyncpg

logger = logging.getLogger(__name__)

INT4_MAX = 2_147_483_647          # Postgres INTEGER range (clamp to avoid overflow either way)
INT4_MIN = -2_147_483_648
_NULLISH = {"", "n/a", "na", "none", "null", "-"}


def _to_int(value, default: int = 0, clamp_max: int | None = None) -> int:
    """Tolerant int coercion: handles None / 'N/A' / comma-formatted / float-strings
    ('2.5' bath -> 2). Bad input (incl. 'inf'/'nan'/overflow) -> default. Always clamped
    into INT4 range so asyncpg can never overflow an INTEGER column."""
    try:
        s = str(value).strip().replace(",", "")
        n = default if (value is None or s.lower() in _NULLISH) else int(float(s))
    except (ValueError, TypeError, OverflowError):  # int(float('inf')) raises OverflowError
        n = default
    if clamp_max is not None:
        n = min(n, clamp_max)
    return max(INT4_MIN, min(n, INT4_MAX))


def _to_int_or_none(value, clamp_max: int | None = None):
    """Like _to_int but returns None (not a default) for missing/invalid — for nullable columns."""
    if value is None:
        return None
    try:
        s = str(value).strip().replace(",", "")
        if s.lower() in _NULLISH:
            return None
        n = int(float(s))
    except (ValueError, TypeError, OverflowError):
        return None
    if clamp_max is not None:
        n = min(n, clamp_max)
    return max(INT4_MIN, min(n, INT4_MAX))


def _to_float(value, default: float = 0.0) -> float:
    """Tolerant float coercion: None / 'N/A' / comma-formatted / non-finite (inf/nan) -> default."""
    try:
        s = str(value).strip().replace(",", "")
        if value is None or s.lower() in _NULLISH:
            return default
        n = float(s)
    except (ValueError, TypeError, OverflowError):
        return default
    return n if math.isfinite(n) else default  # reject inf/nan so geom/aggregates stay valid


# resoFacts.rooms roomType → our room types
RESO_ROOM_MAP = {
    "MasterBedroom": "Bedroom",
    "Bedroom": "Bedroom",
    "MasterBathroom": "Bathroom",
    "Bathroom": "Bathroom",
    "Kitchen": "Kitchen",
    "DiningRoom": "Dining Room",
    "LivingRoom": "Living Room",
    "FamilyRoom": "Living Room",
    "Garage": "Garage",
}

# room types → DB columns
ROOM_COUNT_COLUMNS = {
    "Bedroom": "bedroom_count",
    "Bathroom": "bathroom_count",
    "Kitchen": "kitchen_count",
    "Living Room": "living_room_count",
    "Dining Room": "dining_room_count",
    "Garage": "garage_count",
}


def _normalize_listing_terms(raw: str | None) -> list[str]:
    """'Cash,Conventional,VA Loan' -> ['cash','conventional','va_loan']."""
    if not raw or not isinstance(raw, str):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for term in raw.split(","):
        norm = "_".join(term.strip().lower().split())
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _canonical_photo_url(photo: dict) -> str | None:
    """Highest-res JPEG URL for a photo; canonical id for diffing and room_instances.photo_url."""
    jpegs = ((photo.get("mixedSources") or {}).get("jpeg") or [])
    if not jpegs:
        return None
    best = max(jpegs, key=lambda j: j.get("width", 0))
    return best.get("url") or None


# High-precision amenity signals: when the vision model describes a Pool photo
# with these but forgets the "community pool" tag (prompt rule 8), the ingest
# adds it deterministically — a clubhouse pool must never count as the home's
# private pool. Deliberately conservative: signals that also appear on genuine
# private-pool photos (cabana, tennis court, "…community" context tags) are NOT
# here; the prompt handles those, this guard only enforces the unambiguous ones.
_COMMUNITY_POOL_SIGNALS = (
    "clubhouse", "fitness center", "amenity", "onsite", "lap lanes",
)


def _ensure_community_pool_tag(room_type: str, features: list) -> list:
    """Append 'community pool' to a Pool photo whose own tags carry unambiguous
    amenity signals — trust-but-verify backstop for prompt rule 8."""
    if room_type != "Pool" or not features:
        return features
    lowered = [str(f).lower() for f in features]
    if any("community pool" in f for f in lowered):
        return features
    if any(sig in f for f in lowered for sig in _COMMUNITY_POOL_SIGNALS):
        return list(features) + ["community pool"]
    return features


def _build_rooms_from_photos(photos: list[dict]) -> dict[str, list[dict]]:
    """Group features by RoomType → {room_type: [{"features","color","photo_url"}]}; unusable results become empty "Unknown" stubs to claim their URL."""
    rooms: dict[str, list[dict]] = defaultdict(list)
    for photo in photos:
        room_type = photo.get("RoomType", "Unknown")
        features = _ensure_community_pool_tag(room_type, photo.get("Features", []))
        color = photo.get("Color")
        if isinstance(color, str):
            color = color.strip().lower() or None
            if color in {"unknown", "n/a", "none", "null"}:
                color = None
        photo_url = _canonical_photo_url(photo)

        if room_type and room_type != "Unknown" and features:
            rooms[room_type].append({
                "features": features,
                "color": color,
                "photo_url": photo_url,
            })
        elif photo_url:
            # Unusable Vision result: claim URL as stub to avoid infinite re-analyze.
            rooms["Unknown"].append({
                "features": [],
                "color": None,
                "photo_url": photo_url,
            })
    return dict(rooms)


def _get_room_counts(record: dict, room_counts_by_type: dict[str, int]) -> dict[str, int]:
    """Compute the 6 denormalized room-count columns; Bedroom/Bathroom from Zillow, others from resoFacts.rooms > room_counts_by_type > hasGarage capacity."""
    counts: dict[str, int] = {
        "Bedroom": _to_int(record.get("bedrooms")),
        "Bathroom": _to_int(record.get("bathrooms")),
    }

    reso_facts = record.get("resoFacts", {}) or {}
    reso_rooms = reso_facts.get("rooms", []) or []
    reso_counts: dict[str, int] = defaultdict(int)
    for r in reso_rooms:
        raw_type = r.get("roomType", "")
        mapped = RESO_ROOM_MAP.get(raw_type)
        if mapped:
            reso_counts[mapped] += 1

    for room_type in ["Kitchen", "Dining Room", "Living Room", "Garage"]:
        counts[room_type] = reso_counts.get(room_type, 0)

    # Fallback to provided source when resoFacts had no count.
    for room_type, n in room_counts_by_type.items():
        if room_type in ROOM_COUNT_COLUMNS and counts.get(room_type, 0) == 0:
            counts[room_type] = n

    # Garage fallback from hasGarage flag when no other count exists.
    if counts.get("Garage", 0) == 0:
        if reso_facts.get("hasGarage") or reso_facts.get("hasAttachedGarage"):
            capacity = _to_int(reso_facts.get("garageParkingCapacity"), 1) or 1
            counts["Garage"] = capacity

    return counts


async def query_room_instance_counts(conn, property_id: int) -> dict[str, int]:
    """{room_type: count} for a property's room_instances, excluding 'Unknown' stubs."""
    rows = await conn.fetch(
        """
        SELECT room_type, COUNT(*) AS n FROM room_instances
        WHERE property_id = $1 AND room_type != 'Unknown'
        GROUP BY room_type
        """,
        property_id,
    )
    return {r["room_type"]: r["n"] for r in rows}


async def refresh_property_room_counts(conn, existing_id: int, item: dict) -> None:
    """Recompute Kitchen/Living/Dining/Garage from current room_instances; call after update_property_scalars."""
    record = item.get("ZillowPropertyRecord", {}) or {}
    current_counts = await query_room_instance_counts(conn, existing_id)
    counts = _get_room_counts(record, current_counts)
    await conn.execute(
        """
        UPDATE properties SET
            kitchen_count = $2,
            living_room_count = $3,
            dining_room_count = $4,
            garage_count = $5,
            updated_at = NOW()
        WHERE id = $1
        """,
        existing_id,
        counts.get("Kitchen", 0),
        counts.get("Living Room", 0),
        counts.get("Dining Room", 0),
        counts.get("Garage", 0),
    )


# Helpers for single-property create/update (POST/PUT /properties)

def _extract_neighborhood(record: dict) -> str | None:
    """Zillow neighborhood name from neighborhoodSearchUrl, resolved against nearbyNeighborhoods or de-slugged from the URL path; None when not in a named neighborhood."""
    u = record.get("neighborhoodSearchUrl") or {}
    path = u.get("path") if isinstance(u, dict) else None
    if not path:
        return None
    for nn in (record.get("nearbyNeighborhoods") or []):
        if isinstance(nn, dict) and (nn.get("regionUrl") or {}).get("path") == path and nn.get("name"):
            return nn["name"]
    # fallback: "/viera-east-melbourne-fl/" -> "Viera East"
    slug = path.strip("/")
    slug = slug[:-3] if slug.endswith("-fl") else slug.split("-fl/")[0]
    parts = slug.split("-")
    cities = {"melbourne", "titusville", "rockledge", "mims"}
    while parts and parts[-1].lower() in cities:
        parts.pop()
    return " ".join(p.capitalize() for p in parts) or None


def _extract_property_fields(item: dict) -> dict:
    """Extract DB-relevant scalar fields from a Zillow item, keyed like properties columns."""
    record = item.get("ZillowPropertyRecord", {}) or {}
    address = record.get("address") if isinstance(record.get("address"), dict) else {}
    reso_facts = record.get("resoFacts") if isinstance(record.get("resoFacts"), dict) else {}

    lot_units = record.get("lotAreaUnits", "")
    if lot_units == "Acres":
        lot_size = int(_to_float(record.get("lotAreaValue")) * 43560)
    else:
        lot_size = _to_int(record.get("lotSize"))
    lot_size = min(lot_size, INT4_MAX)   # avoid INTEGER overflow on huge acreage

    return {
        "guid": item.get("Id", ""),
        # Zillow's stable property id — the only cross-batch identity the feed has.
        # Present in only some exporter versions; None when absent.
        "zpid": _to_int_or_none(record.get("zpid")),
        "name": address.get("streetAddress", "Unknown Property"),
        "street": address.get("streetAddress", ""),
        "district": address.get("subdivision", ""),
        "city": address.get("city", ""),
        "state": address.get("state", ""),
        "postal_code": address.get("zipcode", ""),
        "country": "US",
        "county": str(record.get("county") or "").strip() or None,
        # Locality was only ever populated from Photon reverse geocoding, which is
        # removed — the Zillow feed has no readable locality source.
        "locality": None,
        "neighborhood": _extract_neighborhood(record),
        "latitude": _to_float(record.get("latitude")),
        "longitude": _to_float(record.get("longitude")),
        "area_sqft": _to_int(record.get("livingArea"), clamp_max=INT4_MAX),
        "price_usd": _to_int(record.get("price"), clamp_max=INT4_MAX),
        "home_type": record.get("homeType"),
        "rent_estimate": _to_int_or_none(record.get("rentZestimate"), clamp_max=INT4_MAX),
        "year_built": _to_int_or_none(record.get("yearBuilt")),
        "lot_size_sqft": lot_size,
        "stories": _to_int_or_none(reso_facts.get("stories")),
        "has_pool": bool(reso_facts.get("hasPrivatePool")),
        "has_waterfront": bool(reso_facts.get("hasWaterfrontView")),
        "description": record.get("description"),
        "financing": _normalize_listing_terms(reso_facts.get("listingTerms")),
    }


def _extract_schools(item: dict) -> list[dict]:
    """Normalized schools list from incoming item."""
    record = item.get("ZillowPropertyRecord", {}) or {}
    out: list[dict] = []
    for s in record.get("schools", []) or []:
        out.append({
            "name": s.get("name", ""),
            "rating": _to_int_or_none(s.get("rating")),  # "8"/8.5 → 8; junk → NULL
            "grades": s.get("grades", ""),
            "distance": _to_float(s.get("distance")),
            "link": s.get("link", ""),
        })
    return out


# Feature tags meaning the pool water itself is roofed/screened/caged; used to derive properties.has_covered_pool.
_COVERED_POOL_TAGS = [
    "covered pool", "screened pool", "screened-in pool", "screen-enclosed pool",
    "screen enclosed pool", "enclosed pool", "caged pool", "pool cage",
    "covered pool cage", "pool cage enclosure", "pool enclosure",
    "screened pool cage", "screened pool enclosure", "screened pool enclosures",
]


async def _apply_pool_metadata_guard(conn, prop_id: int) -> None:
    """Reject a WRONG metadata pool claim: listing agents in amenity communities
    sometimes set resoFacts.hasPrivatePool for the COMMUNITY's pool. Demote
    has_pool when (a) the description attributes the pool to the community and
    nothing shows a private pool, or (b) the LLM description sweep flagged the
    listing (pool_override). Runs after every metadata refresh, so a re-ingest
    of the same wrong data cannot resurrect the false pool. Call BEFORE
    _refresh_has_covered_pool — the covered flag corroborates against has_pool."""
    await conn.execute(
        """
        UPDATE properties p SET has_pool = false
        WHERE p.id = $1 AND p.has_pool AND (
            p.pool_override
            OR (
                -- No private-pool photo anywhere in the listing, AND either the
                -- description attributes the pool to the community, OR the
                -- property is a UNIT in a complex (condo, or a unit-numbered
                -- townhouse/multifamily) — where "private pool" metadata means
                -- the building's shared pool.
                NOT EXISTS (
                    SELECT 1 FROM room_instances ri
                    WHERE ri.property_id = p.id AND ri.room_type = 'Pool'
                      AND NOT EXISTS (SELECT 1 FROM unnest(ri.features) cf
                                      WHERE cf ILIKE '%community%pool%')
                )
                AND COALESCE((SELECT r.data->>'description' FROM raw_properties r
                              WHERE r.id = p.guid), '') NOT ILIKE '%private pool%'
                AND (
                    COALESCE((SELECT r.data->>'description' FROM raw_properties r
                              WHERE r.id = p.guid), '') ILIKE '%community pool%'
                    OR p.home_type = 'CONDO'
                    OR (p.home_type IN ('TOWNHOUSE', 'MULTI_FAMILY')
                        AND p.street ~* '\y(apt|unit)\y|#')
                )
            )
        )
        """,
        prop_id,
    )


async def _refresh_has_covered_pool(conn, prop_id: int) -> None:
    """Recompute properties.has_covered_pool from current room instances: TRUE iff
    a covered-pool tag is present AND the property has independent pool evidence
    (listing metadata has_pool, or a Pool-classified photo of its own); idempotent.

    The corroboration guard exists because vision tags leak from NEIGHBORING
    homes: aerial/drone shots and over-the-fence backyard photos get tagged
    "screened pool enclosure" for a pool that isn't the subject property's. A
    lone enclosure tag with no other pool signal anywhere in the listing is that
    failure signature, not a pool."""
    await conn.execute(
        """
        UPDATE properties p SET has_covered_pool = (
            EXISTS (
                SELECT 1 FROM room_instances ri
                WHERE ri.property_id = p.id AND ri.features && $2::text[]
                  AND NOT EXISTS (SELECT 1 FROM unnest(ri.features) cf WHERE cf ILIKE '%community%pool%')
            )
            AND (
                p.has_pool
                OR EXISTS (
                    SELECT 1 FROM room_instances ri
                    WHERE ri.property_id = p.id AND ri.room_type = 'Pool'
                      AND NOT EXISTS (SELECT 1 FROM unnest(ri.features) cf WHERE cf ILIKE '%community%pool%')
                )
            )
        )
        WHERE p.id = $1
        """,
        prop_id, _COVERED_POOL_TAGS,
    )


async def ensure_property_columns(conn) -> None:
    """Self-migrating additions to `properties` (the schema file only runs on a
    fresh database). home_status mirrors the raw record's homeStatus so search can
    split sale (FOR_SALE/PENDING) from rent (FOR_RENT) with a plain indexed column.
    Idempotent; the backfill touches only rows still NULL. The app and the worker
    both run this at startup, so it is serialized with a SESSION advisory lock —
    not one transaction: ALTER TABLE takes an exclusive lock on properties even when
    the column exists, and each statement must commit (and release it) at once
    instead of blocking live searches for the whole migration."""
    await conn.execute("SELECT pg_advisory_lock(724113)")
    try:
        await _ensure_property_columns_locked(conn)
    finally:
        await conn.execute("SELECT pg_advisory_unlock(724113)")


async def _ensure_property_columns_locked(conn) -> None:
    await conn.execute("ALTER TABLE properties ADD COLUMN IF NOT EXISTS home_status TEXT")
    # Incremental catalog prune walks raw rows by write time.
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_raw_properties_updated_at ON raw_properties(updated_at)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_properties_home_status ON properties(home_status)"
    )
    # Location-first regions (2026-09-14): every city polygon covering the pin, and
    # whether the pin is trustworthy. NULL until assign_region_ids / the backfill
    # runs — search falls back to city_region_id for such rows.
    await conn.execute("ALTER TABLE properties ADD COLUMN IF NOT EXISTS city_region_ids BIGINT[]")
    await conn.execute("ALTER TABLE properties ADD COLUMN IF NOT EXISTS location_trusted BOOLEAN")
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_properties_city_region_ids ON properties USING GIN (city_region_ids)"
    )
    # Open houses (2026-09-16): one row per event, searchable by local date/time.
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS property_open_houses (
            id          BIGSERIAL PRIMARY KEY,
            property_id INTEGER NOT NULL REFERENCES properties(id) ON DELETE CASCADE,
            starts_at   TIMESTAMPTZ NOT NULL,
            ends_at     TIMESTAMPTZ NOT NULL,
            host        TEXT,
            livestream  BOOLEAN NOT NULL DEFAULT FALSE
        )""")
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_open_houses_property ON property_open_houses(property_id)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_open_houses_ends ON property_open_houses(ends_at)"
    )
    n = await conn.fetchval(
        f"""
        WITH filled AS (
            INSERT INTO property_open_houses (property_id, starts_at, ends_at, host, livestream)
            {_OPEN_HOUSE_ROWS_SQL}
              AND NOT EXISTS (SELECT 1 FROM property_open_houses x WHERE x.property_id = p.id)
            RETURNING 1
        )
        SELECT count(*) FROM filled
        """
    )
    if n:
        logger.info("property_open_houses backfilled with %d event(s)", n)
    n = await conn.fetchval(
        """
        WITH filled AS (
            UPDATE properties p SET home_status = r.data->>'homeStatus'
            FROM raw_properties r
            WHERE r.id = p.guid AND p.home_status IS NULL
            RETURNING p.id
        )
        SELECT count(*) FROM filled
        """
    )
    if n:
        logger.info("properties.home_status backfilled for %d row(s)", n)


# Events of the stored raw record, as property_open_houses rows (the adapter writes
# ISO timestamps; the pattern guard keeps a malformed value from failing the ingest).
_OPEN_HOUSE_ROWS_SQL = r"""
    SELECT p.id, (e->>'start')::timestamptz, (e->>'end')::timestamptz,
           nullif(trim(e->>'host'), ''),
           coalesce(CASE WHEN jsonb_typeof(e->'livestream') = 'boolean' THEN (e->>'livestream')::boolean END, false)
    FROM properties p
    JOIN raw_properties r ON r.id = p.guid
    CROSS JOIN LATERAL jsonb_array_elements(
        CASE WHEN jsonb_typeof(r.data->'openHouses') = 'array' THEN r.data->'openHouses' ELSE '[]'::jsonb END
    ) e
    WHERE jsonb_typeof(e) = 'object'
      AND (e->>'start') ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}'
      AND (e->>'end')   ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}'
"""


async def refresh_open_houses(conn, prop_id: int) -> None:
    """Replace the property's open-house rows with the events of its raw record."""
    await conn.execute("DELETE FROM property_open_houses WHERE property_id = $1", prop_id)
    await conn.execute(
        "INSERT INTO property_open_houses (property_id, starts_at, ends_at, host, livestream) "
        + _OPEN_HOUSE_ROWS_SQL + " AND p.id = $1",
        prop_id,
    )


async def refresh_home_status(conn, prop_id: int) -> None:
    """Copy the raw record's homeStatus onto the property row (call after any
    write that may change it; idempotent)."""
    await conn.execute(
        """
        UPDATE properties p SET home_status = r.data->>'homeStatus'
        FROM raw_properties r
        WHERE r.id = p.guid AND p.id = $1
        """,
        prop_id,
    )


# Location-first region assignment (2026-09-14). The MLS "City" is the USPS
# mailing city, which spills far past legal city limits (USPS "Melbourne" covers
# Viera, Suntree and West Melbourne), so filing by name put homes outside the
# boundary the map draws. The pin now decides, with these safety rules:
#   - a pin is TRUSTED only when present (not 0,0) and within PIN_TRUST_TOLERANCE_M
#     of the listing's own county (any county of its state when the county is
#     unknown) — MLS pins are sometimes hundreds of km off (a Melbourne house in
#     Pensacola), while real border parcels sit a few km across a county line.
#     Untrusted pins fall back to the address names and are hidden on the map
#     (location_trusted = false). The county name matches in the stated state
#     first, then any state (agents mistype the state: "Cape Canaveral, NC");
#     the mailing city is looked up in the resolved county's state;
#   - county: covering polygon, else the nearest county within
#     COUNTY_SNAP_TOLERANCE_M — the coarse coastline cuts off beachfront homes by
#     up to ~1 km — else the county name;
#   - city: EVERY covering city polygon goes into city_region_ids, so a home inside
#     overlapping areas (Viera over Rockledge / Melbourne) counts for both
#     searches; city_region_id keeps one primary (the mailing city when it is
#     among them). A pin covered by no city polygon keeps its mailing city only
#     within CITY_EDGE_TOLERANCE_M of that city's boundary, else no city;
#   - zipcode / neighborhood: covering polygon (ZIP falls back to the postal text).
PIN_TRUST_TOLERANCE_M = 5000.0
COUNTY_SNAP_TOLERANCE_M = 2000.0
CITY_EDGE_TOLERANCE_M = 250.0


async def assign_region_ids(conn, prop_id: int) -> None:
    """Assign county / city / zipcode / neighborhood region ids for ONE property
    from its map location (rules above), plus city_region_ids, location_trusted
    and the derived neighborhood name. Idempotent; call after any write that may
    move the point or change the raw record. Bulk re-run:
    python -m src.data.backfill_region_ids"""
    await conn.execute(
        """
        WITH base AS (
            SELECT p.id, p.geom, p.postal_code, r.data AS raw, p.neighborhood AS cur_nbhd,
                   (p.geom IS NULL OR (ST_X(p.geom::geometry) = 0 AND ST_Y(p.geom::geometry) = 0)) AS nocoord,
                   upper(trim(coalesce(r.data->'address'->>'state', p.state, ''))) AS st,
                   lower(trim(coalesce(r.data->'address'->>'city', p.city, ''))) AS mcity,
                   lower(trim(coalesce(r.data->>'county', p.county, ''))) AS mcounty
            FROM properties p LEFT JOIN raw_properties r ON r.id = p.guid
            WHERE p.id = $1
        ),
        county_named AS (
            SELECT b.*,
                (SELECT g.regionid FROM regions g
                  WHERE g.regiontype = '3' AND b.mcounty <> '' AND lower(g.regionname) = b.mcounty
                  ORDER BY coalesce(g.statecode = b.st, false) DESC, g.regionid LIMIT 1) AS county_by_name
            FROM base b
        ),
        named AS (
            SELECT c.*, s.eff_state,
                (SELECT g.regionid FROM regions g
                  WHERE g.regiontype = '0' AND c.mcity <> '' AND lower(g.regionname) = c.mcity
                    AND (s.eff_state IS NULL OR g.statecode = s.eff_state)
                  ORDER BY g.regionid LIMIT 1) AS city_by_name
            FROM county_named c,
                 LATERAL (SELECT coalesce((SELECT g.statecode FROM regions g WHERE g.regionid = c.county_by_name),
                                          nullif(c.st, '')) AS eff_state) s
        ),
        trusted AS (
            SELECT n.*,
                (NOT n.nocoord AND CASE
                    WHEN EXISTS (SELECT 1 FROM regions g WHERE g.regionid = n.county_by_name AND g.geom IS NOT NULL)
                    THEN EXISTS (SELECT 1 FROM regions g WHERE g.regionid = n.county_by_name
                                   AND ST_DWithin(g.geom, n.geom, $2))
                    ELSE EXISTS (SELECT 1 FROM regions g WHERE g.regiontype = '3' AND g.geom IS NOT NULL
                                   AND (n.eff_state IS NULL OR g.statecode = n.eff_state)
                                   AND ST_DWithin(g.geom, n.geom, $2))
                END) AS ok
            FROM named n
        ),
        geo AS (
            SELECT t.*,
                CASE WHEN t.ok THEN coalesce(
                    (SELECT g.regionid FROM regions g WHERE g.regiontype = '3' AND g.geom IS NOT NULL
                       AND ST_Covers(g.geom, t.geom) ORDER BY ST_Area(g.geom), g.regionid LIMIT 1),
                    (SELECT g.regionid FROM regions g WHERE g.regiontype = '3' AND g.geom IS NOT NULL
                       AND ST_DWithin(g.geom, t.geom, $4) ORDER BY ST_Distance(g.geom, t.geom), g.regionid LIMIT 1)
                ) END AS county_geo,
                CASE WHEN t.ok THEN ARRAY(
                    SELECT g.regionid FROM regions g WHERE g.regiontype = '0' AND g.geom IS NOT NULL
                      AND ST_Covers(g.geom, t.geom) ORDER BY ST_Area(g.geom), g.regionid
                ) ELSE ARRAY[]::bigint[] END AS cities_geo,
                EXISTS (SELECT 1 FROM regions g WHERE g.regionid = t.city_by_name AND g.geom IS NOT NULL) AS mailing_has_geom,
                (t.ok AND EXISTS (SELECT 1 FROM regions g WHERE g.regionid = t.city_by_name AND g.geom IS NOT NULL
                                    AND ST_DWithin(g.geom, t.geom, $3))) AS mailing_near,
                CASE WHEN t.ok THEN (SELECT g.regionid FROM regions g WHERE g.regiontype = '2' AND g.geom IS NOT NULL
                       AND ST_Covers(g.geom, t.geom) ORDER BY ST_Area(g.geom), g.regionid LIMIT 1) END AS zip_geo,
                CASE WHEN t.ok THEN (SELECT g.regionid FROM regions g WHERE g.regiontype = '1' AND g.geom IS NOT NULL
                       AND ST_Covers(g.geom, t.geom) ORDER BY ST_Area(g.geom), g.regionid LIMIT 1) END AS nbhd_geo,
                (SELECT g.regionid FROM regions g WHERE g.regiontype = '2' AND g.regionname = t.postal_code
                  ORDER BY g.regionid LIMIT 1) AS zip_by_text,
                CASE WHEN (t.raw->>'cityId') ~ '^[0-9]+$' THEN (t.raw->>'cityId')::bigint END AS raw_city,
                CASE WHEN (t.raw->>'countyId') ~ '^[0-9]+$' THEN (t.raw->>'countyId')::bigint END AS raw_county,
                CASE WHEN (t.raw->>'zipcodeId') ~ '^[0-9]+$' THEN (t.raw->>'zipcodeId')::bigint END AS raw_zip,
                CASE WHEN (t.raw->>'neighborhoodId') ~ '^[0-9]+$' THEN (t.raw->>'neighborhoodId')::bigint END AS raw_nbhd,
                (t.raw ? 'neighborhoodSearchUrl' AND jsonb_typeof(t.raw->'neighborhoodSearchUrl') = 'object') AS feed_has_nbhd
            FROM trusted t
        ),
        ids AS (
            SELECT g.*,
                CASE WHEN g.ok THEN
                    g.cities_geo
                    || CASE WHEN cardinality(g.cities_geo) = 0 AND g.mailing_near
                            THEN ARRAY[g.city_by_name] ELSE ARRAY[]::bigint[] END
                    -- a mailing city without any polygon cannot contradict the pin
                    || CASE WHEN g.city_by_name IS NOT NULL AND NOT g.mailing_has_geom
                             AND NOT (g.city_by_name = ANY(g.cities_geo))
                            THEN ARRAY[g.city_by_name] ELSE ARRAY[]::bigint[] END
                ELSE
                    CASE WHEN coalesce(g.raw_city, g.city_by_name) IS NOT NULL
                         THEN ARRAY[coalesce(g.raw_city, g.city_by_name)] ELSE ARRAY[]::bigint[] END
                END AS city_ids
            FROM geo g
        )
        UPDATE properties p SET
            location_trusted = i.ok,
            city_region_ids = i.city_ids,
            city_region_id = CASE WHEN i.city_by_name = ANY(i.city_ids) THEN i.city_by_name
                                  ELSE i.city_ids[1] END,
            county_region_id = CASE WHEN i.ok THEN coalesce(i.county_geo, i.county_by_name)
                                    ELSE coalesce(i.raw_county, i.county_by_name) END,
            zipcode_region_id = CASE WHEN i.ok THEN coalesce(i.zip_geo, i.zip_by_text)
                                     ELSE coalesce(i.raw_zip, i.zip_by_text) END,
            neighborhood_region_id = CASE WHEN i.ok THEN i.nbhd_geo ELSE i.raw_nbhd END,
            -- MLS records carry no neighborhood name: it follows the assigned region.
            -- A feed-provided name (legacy Zillow records) is never overwritten.
            neighborhood = CASE WHEN i.feed_has_nbhd AND coalesce(i.cur_nbhd, '') <> '' THEN i.cur_nbhd
                                ELSE (SELECT g.regionname FROM regions g
                                       WHERE g.regionid = CASE WHEN i.ok THEN i.nbhd_geo ELSE i.raw_nbhd END) END
        FROM ids i
        WHERE p.id = i.id
        """,
        prop_id, PIN_TRUST_TOLERANCE_M, CITY_EDGE_TOLERANCE_M, COUNTY_SNAP_TOLERANCE_M,
    )


async def _insert_children(conn, prop_id: int, rooms_from_photos: dict, schools: list[dict]) -> None:
    """Insert rooms, room_instances, and property_schools for a property."""
    for room_type, instances in rooms_from_photos.items():
        room_id = await conn.fetchval("""
            INSERT INTO rooms (property_id, room_type, count)
            VALUES ($1, $2, $3) RETURNING id
        """, prop_id, room_type, len(instances))
        for idx, inst in enumerate(instances):
            features = inst["features"]
            color = inst.get("color")
            photo_url = inst.get("photo_url")
            features_text = ", ".join(features)
            await conn.execute("""
                INSERT INTO room_instances (
                    room_id, property_id, room_type,
                    instance_index, features, features_text, color, photo_url
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, room_id, prop_id, room_type, idx, features, features_text, color, photo_url)
    for s in schools:
        await conn.execute("""
            INSERT INTO property_schools (
                property_id, school_name, rating, grades, distance_miles, link
            ) VALUES ($1, $2, $3, $4, $5, $6)
        """, prop_id, s["name"], s["rating"], s["grades"], s["distance"], s["link"])
    # Derive has_covered_pool from the room_instances just written.
    await _apply_pool_metadata_guard(conn, prop_id)
    await _refresh_has_covered_pool(conn, prop_id)


async def update_property_scalars(
    conn,
    existing_id: int,
    item: dict,
) -> None:
    """Update non-photo-derived columns: scalars plus Bedroom/Bathroom counts from Zillow.

    Bedroom/Bathroom use _get_room_counts' image fallback: a feed value of 0
    (common for listings with sparse metadata) must not zero out a count the
    photos already established — every metadata refresh used to do exactly
    that, silently re-breaking bedroom filters for those listings."""
    record = item.get("ZillowPropertyRecord", {}) or {}
    fields = _extract_property_fields(item)
    room_counts = _get_room_counts(
        record, await query_room_instance_counts(conn, existing_id)
    )
    await conn.execute("""
        UPDATE properties SET
            name=$2, street=$3, district=$4, city=$5, state=$6,
            postal_code=$7, country=$8,
            geom=ST_MakePoint($9, $10)::geography,
            area_sqft=$11, price_usd=$12,
            bedroom_count=$13, bathroom_count=$14,
            home_type=$15, rent_estimate=$16, year_built=$17,
            lot_size_sqft=$18, stories=$19,
            has_pool=$20, has_waterfront=$21, description=$22, financing=$23,
            county=$24, locality=$25, neighborhood=$26,
            zpid=COALESCE($27, zpid),
            updated_at=NOW()
        WHERE id = $1
    """,
        existing_id,
        fields["name"], fields["street"], fields["district"], fields["city"],
        fields["state"], fields["postal_code"], fields["country"],
        fields["longitude"], fields["latitude"],
        fields["area_sqft"], fields["price_usd"],
        room_counts.get("Bedroom", 0),
        room_counts.get("Bathroom", 0),
        fields["home_type"], fields["rent_estimate"], fields["year_built"],
        fields["lot_size_sqft"], fields["stories"],
        fields["has_pool"], fields["has_waterfront"], fields["description"],
        fields["financing"],
        fields["county"], fields["locality"], fields["neighborhood"],
        fields["zpid"],
    )
    # Re-derive has_covered_pool (room features may have changed).
    await _apply_pool_metadata_guard(conn, existing_id)
    await _refresh_has_covered_pool(conn, existing_id)
    # Re-assign region ids (coordinates or the raw record may have moved).
    await assign_region_ids(conn, existing_id)
    await refresh_home_status(conn, existing_id)
    await refresh_open_houses(conn, existing_id)


async def update_property_metadata(
    conn,
    existing_id: int,
    item: dict,
) -> None:
    """Update only the properties row (scalars + room counts); no child rows."""
    record = item.get("ZillowPropertyRecord", {}) or {}
    fields = _extract_property_fields(item)
    rooms_from_photos = _build_rooms_from_photos(record.get("originalPhotos", []) or [])
    room_counts = _get_room_counts(
        record, {rt: len(insts) for rt, insts in rooms_from_photos.items()}
    )
    await conn.execute("""
        UPDATE properties SET
            name=$2, street=$3, district=$4, city=$5, state=$6,
            postal_code=$7, country=$8,
            geom=ST_MakePoint($9, $10)::geography,
            area_sqft=$11, price_usd=$12,
            bedroom_count=$13, bathroom_count=$14, kitchen_count=$15,
            living_room_count=$16, dining_room_count=$17, garage_count=$18,
            home_type=$19, rent_estimate=$20, year_built=$21,
            lot_size_sqft=$22, stories=$23,
            has_pool=$24, has_waterfront=$25, description=$26, financing=$27,
            county=$28, locality=$29, neighborhood=$30,
            zpid=COALESCE($31, zpid),
            updated_at=NOW()
        WHERE id = $1
    """,
        existing_id,
        fields["name"], fields["street"], fields["district"], fields["city"],
        fields["state"], fields["postal_code"], fields["country"],
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
    # Re-derive has_covered_pool (room features may have changed).
    await _apply_pool_metadata_guard(conn, existing_id)
    await _refresh_has_covered_pool(conn, existing_id)
    # Re-assign region ids (coordinates or the raw record may have moved).
    await assign_region_ids(conn, existing_id)
    await refresh_home_status(conn, existing_id)
    await refresh_open_houses(conn, existing_id)


async def update_property_with_children(
    conn,
    existing_id: int,
    item: dict,
) -> None:
    """Full re-process: update properties row and replace all child rows. Used when photos/schools changed."""
    await conn.execute("DELETE FROM property_schools WHERE property_id = $1", existing_id)
    await conn.execute("DELETE FROM room_instances WHERE property_id = $1", existing_id)
    await conn.execute("DELETE FROM rooms WHERE property_id = $1", existing_id)
    await update_property_metadata(conn, existing_id, item)

    record = item.get("ZillowPropertyRecord", {}) or {}
    rooms_from_photos = _build_rooms_from_photos(record.get("originalPhotos", []) or [])
    await _insert_children(conn, existing_id, rooms_from_photos, _extract_schools(item))

