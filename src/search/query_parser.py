"""Query parser: the OpenAI model maps NL queries to structured criteria using the DB's known features/room types."""

import json
import re
import logging

from config.settings import settings
from src.data.feature_registry import registry
from src.llm_client import get_query_client
from src.models.search import (
    AreaCriterion,
    AreaRelationCriterion,
    ColorRoomCriterion,
    FeatureCriterion,
    LocationCriterion,
    ParsedQuery,
    PriceCriterion,
    PropertyCriterion,
    ProximityCriterion,
    RoomCountCriterion,
)

logger = logging.getLogger(__name__)


class QueryParseError(Exception):
    """Raised when the query could not be parsed at all (e.g. LLM outage) — distinct
    from a successful parse that yields zero criteria. Callers should surface an error
    rather than running an unfiltered (whole-catalog) search."""


SYSTEM_PROMPT_TEMPLATE = """\
You are a real estate search query parser. Your job is to extract structured \
search criteria from natural language queries about real estate properties.

You have access to the room types that exist in our database.

KNOWN ROOM TYPES:
{room_types}
{known_features_block}
Extract ALL criteria from the user's query and return them as JSON.

Available criterion types:

1. room_count — The user wants a specific number of rooms.
   Fields: room_type (string), exact_count (int|null), min_count (int|null), max_count (int|null)
   Use ONLY the room types listed above.
   A BARE count with no qualifier means AT LEAST that many — "3 bedrooms" ->
   min_count=3 (NOT exact_count). Use exact_count ONLY for explicit exactness
   ("exactly 2 bedrooms", "only 2 bedrooms"). "at least/or more" -> min_count;
   "at most/up to/no more than" -> max_count.

2. feature — The user wants a specific feature included or excluded.
   Fields: feature (string), room_context (string|null), negated (bool), strength ("hard"|"soft")
{feature_naming_rule}
   STRENGTH — people write in natural language, mixing REQUIREMENTS with WISHES:
     "hard" = a concrete thing a photo can verify: an object, material, amenity,
       room, layout or condition ("screened patio", "water views", "granite
       countertops", "primary suite", "pool", "fireplace", "updated kitchen",
       "recently renovated", "new roof", "fenced yard", "two-car garage").
     "soft" = a subjective opinion, degree or feeling that no photo can prove:
       quality words ("great", "nice", "nicer", "beautiful", "lovely", "amazing",
       "charming", "cozy", "spacious", "bright", "open-feeling"), comparatives
       ("bigger", "better"), and impressions ("feels private", "looks luxurious").
       Strip impression verbs ("feels", "looks", "seems") and classify what is
       left: "backyard that feels private" -> feature="private backyard", soft;
       "kitchen that looks recently renovated" -> "recently renovated kitchen", hard.
     When unsure, prefer "soft" — a soft wish still ranks matching homes first,
     while a wrong "hard" requirement can wipe out every result.
   CONCESSIONS — a thing the user is WILLING TO GIVE UP is NOT a requirement and
   gets NO criterion at all (this overrides "never drop a feature"): "I'd trade a
   smaller yard for a nicer interior" -> ONLY feature="nicer interior" (soft);
   "don't mind an older kitchen", "okay with a small lot", "yard doesn't matter",
   "willing to give up a garage", "doesn't need to be big", "doesn't have to have
   a pool", "not necessarily new" -> nothing for that thing (NOT a negated feature:
   negated=true is reserved for an explicit ban like "no pool" / "without carpet").
   BEACH AREA — "beachside", "oceanside", "on the beach side", "barrier island"
   describe WHERE the home is: emit feature="beachside" (hard); the search maps it
   to the beach-side towns. "ocean view"/"water views" stay ordinary features.
   room_context ties the feature to a specific room type. Set it ONLY when \
the user EXPLICITLY names a room type as a CONTAINER for the feature \
(pattern: "<ROOM TYPE> with <FEATURE>" or "<FEATURE> in the <ROOM TYPE>").
   DO NOT set room_context just because a feature shares its name with a room type \
(e.g. "pool", "kitchen", "garage" used as a FEATURE). These refer to the feature itself, \
not a containing room.
   Correct usage:
     "kitchen with granite countertops"   → feature=granite countertops, room_context="Kitchen"
     "bedrooms with accent walls"         → feature=accent walls, room_context="Bedroom"
     "hardwood floors in the living room" → feature=hardwood floors, room_context="Living Room"
   Incorrect usage (DO NOT DO THIS):
     "with pool"                → room_context=null   (NOT "Pool")
     "with covered pool"        → room_context=null   (NOT "Pool")
     "with uncovered pool"      → room_context=null   (NOT "Pool")
     "with garage"              → room_context=null   (NOT "Garage")
     "3 bedrooms with a fireplace" → room_context=null (bedroom count is separate; fireplace is not IN the bedroom here)
   negated=true means the property must NOT have this feature.

   NEGATION SEMANTICS — READ CAREFULLY:

   There are TWO KINDS of negation. You MUST distinguish them:

   (A) ATOMIC NEGATION — the user doesn't want the thing at all.
       Phrasing cues: "no X", "without X", "not X", "excluding X" \
when X stands alone with no modifier.
       Emit ONE negated criterion for X only.
       Examples:
         "no pool"           → [negated(pool)]
         "without garage"    → [negated(garage)]
         "not waterfront"    → [negated(waterfront)]

   (B) MODIFIER NEGATION — the user DOES want the thing, but NOT with a specific attribute.
       Phrasing cues: a compound noun where a negative prefix/word modifies \
an attribute of a thing the user wants (e.g., "uncovered pool", "unfenced yard", \
"open-air pool", "gas-free fireplace", "pool without cover", "fireplace without gas").
       Emit TWO criteria:
         1. POSITIVE criterion for the BASE thing (it must exist)
         2. NEGATED criterion for the unwanted ATTRIBUTE
       Examples:
         "with uncovered pool"       → [positive(pool), negated(covered pool)]
         "pool without screen"       → [positive(pool), negated(screened pool)]
         "open-air pool"             → [positive(pool), negated(covered pool)]
         "fireplace without gas"     → [positive(fireplace), negated(gas fireplace)]
         "unfenced backyard"         → [positive(backyard), negated(fenced backyard)]
         "non-granite countertops"   → [positive(countertops), negated(granite countertops)]

   HOW TO DECIDE (A) vs (B):
     - If the negated word stands alone with no described object → (A) atomic
     - If the negation attaches to a MODIFIER of a thing the user wants → (B) modifier
     - Rule of thumb: if removing the negation still leaves something the user wants \
(e.g., "uncovered pool" → still wants a pool), it's MODIFIER NEGATION.
     - If removing the negation leaves nothing wanted (e.g., "no pool" → wants no pool), \
it's ATOMIC NEGATION.

   NEVER expand negations to related/similar features the user did not mention. \
For example, "without granite" must NOT also negate "quartz" — only the exact thing \
the user said.

3. price — Price range constraint.
   Fields: min_price (int|null), max_price (int|null)
   A BARE price with no qualifier means THAT PRICE AND BELOW — set max_price
   only, NEVER min_price ("homes 400k" / "$300,000 homes" / "300k houses" ->
   max_price only). Set BOTH fields ONLY for an explicit range ("300k-500k",
   "between 300k and 500k"). Explicit qualifiers keep their meaning: "under/
   below/up to X" -> max_price; "over/above/at least X" -> min_price. NEVER
   set min_price = max_price.

4. area — Square footage constraint.
   Fields: min_sqft (int|null), max_sqft (int|null)
   A BARE size with no qualifier means AT LEAST that size — "2000 sqft homes" ->
   min_sqft=2000 only, NEVER max_sqft. Set BOTH only for an explicit range
   ("1500-2500 sqft"). "under/below/up to X sqft" -> max_sqft; "over/above/
   at least X sqft" -> min_sqft. NEVER set min_sqft = max_sqft.

5. location — A place the property is IN (administrative or named area), matched
   by name (fuzzy). NOT a "near <landmark>" distance (that's proximity #6).
   Fields: city, state, country, district, county, neighborhood, locality, street.
   - city: a city/town OR any community/area name a person searches by
     (e.g. "Melbourne", "Rockledge", "Viera", "Suntree", "Viera East"). When in
     doubt, put the place here — it is matched broadly across place columns.
   - county: a county name, e.g. "Brevard County".
   - street: a street/road name, e.g. "Murrell Road", "Ganton Court".
   - neighborhood: set ONLY to a real named sub-area (e.g. "Viera East",
     "Viera West", "June Park"). NEVER the generic word "neighborhood/
     neighbourhood", and NEVER the same value as city.
   - locality / district: leave null unless the user clearly names that exact
     level; otherwise just use city.
   STRIP FILLER WORDS — extract ONLY the proper place name, never descriptive
   words like "neighbourhood", "neighborhood", "area", "community", "town",
   "of", "the". So "neighbourhood of Viera", "the neighborhoods of Viera",
   "Viera neighborhood", "Viera area", "town of Viera" ALL mean the place
   "Viera" → city="Viera" (leave neighborhood/locality/district null).
   (Note: "near Viera" / "neighborhoods NEAR Viera" is NOT this — that is the
   spatial area_relation #9, not location.)
   All string|null. Use "<city> in <STATE>" only when the user names a state.
{known_regions_block}

6. proximity — Distance to a named landmark OR a common PLACE TYPE (point of interest).
   Fields: landmark_name (string), max_distance_miles (float; if the user gives no \
distance, default to 3).
   USE proximity for:
     - a SPECIFIC named place: "Oak Park Elementary School", "Central Park".
     - schools: "near good schools" → landmark_name="good schools", max_distance_miles=5.
     - a POI CATEGORY: grocery store / supermarket, church (place of worship), \
gas station, pharmacy / drugstore, restaurant, hospital, bank, park, gym.
       Examples:
         "within 2 miles of a grocery store" → landmark_name="grocery store", max_distance_miles=2
         "near a church"                     → landmark_name="church", max_distance_miles=3
         "close to a gas station"            → landmark_name="gas station", max_distance_miles=3
         "homes by a pharmacy"               → landmark_name="pharmacy", max_distance_miles=3
   Do NOT use proximity for vague geographic features like "beach", "lake", \
"waterfront", "downtown" — use the "feature" type for those ("near beach" → feature="beach").

7. property — Property attribute constraints.
   Fields: home_type (string|null), min_rent (int|null), max_rent (int|null), \
min_year_built (int|null), max_year_built (int|null), \
min_lot_sqft (int|null), max_lot_sqft (int|null), \
min_stories (int|null), max_stories (int|null)
   VALID home_type values: SINGLE_FAMILY, CONDO, TOWNHOUSE, MANUFACTURED, MULTI_FAMILY, LOT
   Only set home_type when the user's term CLEARLY maps to one of these values.
   If ambiguous (e.g. "apartment", "home", "house", "property"), do NOT set home_type.
   Mapping:
     "condo" → CONDO
     "townhouse" → TOWNHOUSE
     "single family" / "single-family home" / "family home" / "family house" / "family residence" / "starter home" → SINGLE_FAMILY
     "manufactured home" / "mobile home" → MANUFACTURED
     "duplex" / "multi family" / "multi-family" / "two-family" → MULTI_FAMILY
     "land" / "lot" / "lots" / "vacant land" / "vacant lot" / "acreage" / "parcel" / "building lot" → LOT
   Results include land lots by default; set home_type=LOT only when the user asks for land/lots specifically.
   IMPORTANT: When the user describes a HOME TYPE phrase (e.g. "family home", "starter home", \
"single-family residence"), emit it ONLY as a `property` criterion with `home_type` set. \
Do NOT also emit a `feature` criterion for the same phrase. \
"family home" is a home type, NOT a feature like "family-friendly community".
   "under $2k/mo" or "rent under 2000" → max_rent=2000
   "built after 2000" → min_year_built=2000
   A BARE year built with no qualifier means THAT YEAR OR NEWER — "built in
   2005" / "2005 build" → min_year_built=2005 only, NEVER max_year_built.
   "built before 1990" → max_year_built=1990; "built 1990-2010" → both.
   NEVER set min_year_built = max_year_built.
   "single story" → max_stories=1
   IMPORTANT: There is NO has_pool or has_waterfront field on the `property` criterion.
   "pool", "swimming pool", "waterfront", "on the water" are FEATURES. \
Use the `feature` criterion type for them — never emit them as property attributes.

8. color_room — The user wants a room with a specific dominant COLOR.
   Fields: color (string), room_type (string), negated (bool)
   Use this WHENEVER the user describes a room by its color, e.g. "white kitchen",
   "blue bathroom", "gray bedroom". Color is a room ATTRIBUTE, not a feature.
   VALID color values (exactly one, normalize synonyms):
     white, black, gray, brown, beige, blue, green, red, yellow, purple, pink, orange, gold
   Color synonym normalization:
     ivory/cream/eggshell/off-white → "white"
     navy/teal/turquoise/light blue/dark blue → "blue"
     tan/khaki/taupe/sand → "beige"
     charcoal/silver/light gray/dark gray → "gray"
     wood/walnut/oak/mahogany/wood-tone → "brown"
     sage/olive/forest/light green/dark green → "green"
     burgundy/maroon/coral/light red/dark red → "red"
     mustard/light yellow/pale yellow → "yellow"
     lavender/violet/light purple/dark purple → "purple"
     salmon/rose/light pink → "pink"
   VALID room_type values: Kitchen, Bedroom, Bathroom, Living Room, Dining Room, Exterior, Pool, Garage
   Match pattern: "<color> <room_type>" or "<room_type> in <color>" or "<color>-toned <room_type>".

   CRITICAL DISTINCTION — room color vs object color:
     - A color directly modifying a ROOM TYPE (no object) → color_room.
         "white kitchen" → the KITCHEN is white → color_room(color="white", room_type="Kitchen")
     - A color modifying a SPECIFIC OBJECT (cabinet, countertop, tile, vanity, wall, floor, etc.)
       → feature criterion, with the color KEPT in the feature string and normalized to the 13 palette.
         "blue cabinet"  → feature(feature="blue cabinets", room_context=<room if known>)
         "white countertop" → feature(feature="white countertops", room_context=<room if known>)
     - The color words in BOTH cases must be normalized to the 13 palette (teal→blue, cream→white, etc.).

   Examples:
     "white kitchen"           → color_room(color="white", room_type="Kitchen")
     "blue bathroom"           → color_room(color="blue", room_type="Bathroom")
     "light purple kitchen"    → color_room(color="purple", room_type="Kitchen") (light purple → purple)
     "taupe bathroom"          → color_room(color="beige", room_type="Bathroom") (taupe → beige)
     "navy blue living room"   → color_room(color="blue", room_type="Living Room")
     "no white kitchen"        → color_room(color="white", room_type="Kitchen", negated=true)
     "modern white kitchen"    → color_room(color="white", room_type="Kitchen")
                                  PLUS feature(feature="modern", room_context="Kitchen")
     "white kitchen with blue cabinet"
                               → color_room(color="white", room_type="Kitchen")
                                  PLUS feature(feature="blue cabinets", room_context="Kitchen")
     "white kitchen with cabinet" (object has no color)
                               → color_room(color="white", room_type="Kitchen")
                                  PLUS feature(feature="cabinets", room_context="Kitchen")
     "gray bathroom with white vanity"
                               → color_room(color="gray", room_type="Bathroom")
                                  PLUS feature(feature="white vanity", room_context="Bathroom")
     "kitchen with teal cabinets" (no room color stated)
                               → feature(feature="blue cabinets", room_context="Kitchen") (teal → blue)
   IMPORTANT:
     - color_room is ONLY for a color describing the ROOM ITSELF (no object).
     - A color describing an OBJECT stays in a `feature` (color normalized to the 13 palette).
     - When the user gives BOTH a room color and a colored object (e.g. "white kitchen with blue cabinet"),
       emit a color_room for the room AND a feature for the object.

9. area_relation — A SPATIAL relation to one or two AREAS (city/neighbourhood/
   town). Use this ONLY for nearness/neighbours/between, NOT a plain "in <place>"
   (location #5) and NOT a specific named landmark/school (proximity #6).
   Fields: relation ("near"|"neighbors"|"between"), place_a (string),
           place_b (string|null), radius_miles (float|null, optional).
   - relation="near"  — "near <A>" / "close to <A>" / "around <A>" / "by <A>".
     Means around A and INCLUDES A itself.
   - relation="neighbors" — "neighbours of <A>" / "neighbors of <A>" /
     "<A>'s neighbours" / "neighbouring <A>" / "adjacent to <A>" / "next to <A>" /
     "bordering <A>" / "areas around <A>". Means the areas that NEIGHBOUR A,
     EXCLUDING A itself.
   - relation="between" — "between <A> and <B>" / "in between <A> and <B>".
     Set place_b="<B>".
   Examples:
     "homes near Viera East"            → area_relation(relation="near", place_a="Viera East")
     "homes in neighbours of Viera East"→ area_relation(relation="neighbors", place_a="Viera East")
     "homes adjacent to Rockledge"      → area_relation(relation="neighbors", place_a="Rockledge")
     "3 bed between Rockledge and Viera"→ room_count(Bedroom,3) PLUS
                                          area_relation(relation="between", place_a="Rockledge", place_b="Viera")
   A plain "homes in Viera" is location #5, NOT area_relation.

Return JSON with this exact structure:
{{
  "criteria": [ ... list of criterion objects, each with a "type" field ... ],
  "reconstructed_queries": [],
  "understood_intent": "Brief summary of what you understood the user is looking for"
}}

IMPORTANT: `reconstructed_queries` is DEPRECATED. Always return an empty array `[]` \
for this field. Do NOT expand synonyms or related features — the server computes \
feature alternatives deterministically from the database. Returning anything other \
than `[]` is wasted output that will be discarded.

Important rules:
- If the user says "two bedrooms", that means min_count=2 for Bedroom (bare
  count = at least; exact_count ONLY for "exactly"/"only two bedrooms").
- If the user says "at least 3 bathrooms", that means min_count=3 for Bathroom.
{feature_value_rule}
- Only extract criteria that are explicitly stated or clearly implied.
- Do NOT invent criteria the user did not mention.
- room_context rules (see FEATURE section above for the full rule):
  Set room_context ONLY when the user names a room type as a CONTAINER via the pattern \
"<ROOM TYPE> with <FEATURE>" or "<FEATURE> in the <ROOM TYPE>".
  NEVER set room_context because a feature name coincides with a room type. \
For "pool", "garage", "kitchen", "bedroom" used as features, room_context=null.
- Every feature gets a strength: "hard" for concrete/checkable things, "soft" for
  opinions, degrees and feelings (see STRENGTH). Concession clauses produce no criterion.
- When the user says "without", "no", "exclude", or "not", set negated=true.
  "2 bedrooms without stone tile" → feature="stone tile", negated=true
  "no pool" → the closest known feature, negated=true
"""


# --- Mode-specific prompt fragments -----------------------------------------

# LEGACY mode: the LLM maps user words to canonical DB feature names using the full feature list embedded in the prompt.
_LEGACY_FEATURE_NAMING_RULE = """\
   CRITICAL: Map the user's words to the closest matching known feature name. \
But if the user's word is a GENERIC term that could match MANY features \
(e.g., "cabinet", "tile", "wood", "pool"), keep the generic term as-is \
so it matches broadly. Only map to a specific feature when the user is clearly specific.
   Examples of mapping:
     "wood flooring" → "hardwood floors" (specific → specific)
     "hearth" → "fireplace" (specific → specific)
     "marble counters" → "marble countertops" (specific → specific)
     "cabinet" → "cabinet" (generic → keep generic, matches "white cabinets", "shaker cabinets", etc.)
     "tile" → "tile" (generic → keep generic, matches "tile flooring", "tile backsplash", etc.)
     "pool" → "pool" (generic → keep generic)
   If the user mentions a feature that has NO close match in the known features, \
still include it using the user's original wording. NEVER drop a feature from the query."""

_LEGACY_FEATURE_VALUE_RULE = """\
- CRITICAL: Feature values MUST be from the KNOWN FEATURES list above. \
Map synonyms, abbreviations, and alternate phrasings to the exact known feature name."""

# EMBEDDING mode: no feature list in the prompt; the LLM outputs the user's own wording, mapped downstream by embedding retrieval + a second LLM call.
_EMBEDDING_FEATURE_NAMING_RULE = """\
   Use the user's OWN wording for the feature value — do NOT try to map it to a \
canonical database name. A separate downstream step handles that mapping. \
Keep the phrase concise and search-like (e.g. "swimming pool", "hardwood floors", \
"granite countertops", "covered pool"). Preserve attribute words the user gave \
(colors, materials, styles) since they carry intent. NEVER drop a feature the user mentioned."""

_EMBEDDING_FEATURE_VALUE_RULE = """\
- Feature values are the user's own concise wording (NOT mapped to any canonical \
list). A downstream embedding step resolves them to real database features."""


# Known catalog regions injected into the parse prompt: canonical place names by
# level, so the LLM normalizes spelling ("veira" -> "Viera") and files each place
# in the RIGHT field (Viera West -> neighborhood, not city). Cached — region
# assignments only change on ingest.
_REGIONS_BLOCK_CACHE: dict = {}
_REGIONS_BLOCK_TTL_SEC = 600


async def _known_regions_block() -> str:
    import time
    hit = _REGIONS_BLOCK_CACHE.get("block")
    if hit and time.monotonic() - hit[0] < _REGIONS_BLOCK_TTL_SEC:
        return hit[1]
    try:
        from src.data.database import get_pool
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT g.regiontype, g.regionname FROM regions g
                WHERE g.regionid IN (
                    SELECT city_region_id FROM properties WHERE city_region_id IS NOT NULL
                    UNION SELECT county_region_id FROM properties WHERE county_region_id IS NOT NULL
                    UNION SELECT neighborhood_region_id FROM properties WHERE neighborhood_region_id IS NOT NULL
                ) AND g.regiontype IN ('0', '1', '3')
                ORDER BY g.regiontype, g.regionname
            """)
        by_type: dict[str, list[str]] = {"0": [], "1": [], "3": []}
        for r in rows:
            by_type[r["regiontype"]].append(r["regionname"])
        block = (
            "   KNOWN CATALOG REGIONS — the catalog covers exactly these places. When\n"
            "   the query names one of them (any casing, spacing, or misspelling), use\n"
            "   the EXACT canonical name from this list and put it in the field of its\n"
            "   level. For a NEIGHBORHOOD from the list, set neighborhood to it and\n"
            "   leave city null. Places not listed are still extracted as usual.\n"
            f"   Cities: {', '.join(by_type['0'])}\n"
            f"   Neighborhoods: {', '.join(by_type['1'])}\n"
            f"   Counties: {', '.join(by_type['3'])}\n"
        )
    except Exception as e:  # noqa: BLE001 — parsing must work without the block
        logger.warning(f"Known-regions block unavailable: {e}")
        block = ""
    if len(_REGIONS_BLOCK_CACHE) < 4:
        _REGIONS_BLOCK_CACHE["block"] = (time.monotonic(), block)
    return block


# The vision taxonomy. The registry also carries legacy types from older prompts
# ("Interior": 5 photos, "Closet", "Hallway", ...) — offering those to the parser
# let "nicer interior" get scoped to a room type that effectively doesn't exist,
# zeroing the search. Only these may appear in the prompt or in room_context.
CANONICAL_ROOM_TYPES = (
    "Kitchen", "Bedroom", "Bathroom", "Living Room", "Dining Room",
    "Exterior", "Pool", "Garage",
)
_ROOM_TYPE_BY_LOWER = {rt.lower(): rt for rt in CANONICAL_ROOM_TYPES}


def canonical_room_context(value) -> str | None:
    """Normalize a parsed room_context to the vision taxonomy; anything else -> None
    (the feature then matches in any room instead of in no room)."""
    if not value or not str(value).strip():
        return None
    return _ROOM_TYPE_BY_LOWER.get(str(value).strip().lower())


def _build_system_prompt(use_embedding_retrieval: bool, known_regions_block: str = "") -> str:
    room_types = [rt for rt in registry.get_room_types_list() if rt in CANONICAL_ROOM_TYPES] \
        or list(CANONICAL_ROOM_TYPES)
    if use_embedding_retrieval:
        return SYSTEM_PROMPT_TEMPLATE.format(
            room_types=", ".join(room_types),
            known_features_block="",
            known_regions_block=known_regions_block,
            feature_naming_rule=_EMBEDDING_FEATURE_NAMING_RULE,
            feature_value_rule=_EMBEDDING_FEATURE_VALUE_RULE,
        )
    # Legacy: inline the full DB feature list.
    features = registry.get_features_list()
    known_features_block = (
        "\nKNOWN FEATURES (these are the ONLY valid feature names):\n"
        + ", ".join(features)
        + "\n"
    )
    return SYSTEM_PROMPT_TEMPLATE.format(
        room_types=", ".join(room_types),
        known_features_block=known_features_block,
        known_regions_block=known_regions_block,
        feature_naming_rule=_LEGACY_FEATURE_NAMING_RULE,
        feature_value_rule=_LEGACY_FEATURE_VALUE_RULE,
    )


async def _call_llm(client, system_prompt: str, query: str) -> str | None:
    """Call the query LLM (OpenAI) and return raw JSON text or None (response_format forces strict JSON; fence-stripping is a fallback)."""
    response = await client.chat.completions.create(
        model=settings.openai_model_for_query,
        max_completion_tokens=16384,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )
    raw_text = (response.choices[0].message.content or "").strip()
    return raw_text or None


def _safe_miles(value, default: float | None = 3.0):
    """Coerce a distance/radius to a positive float; missing/non-numeric/zero/negative -> default."""
    try:
        if value is None:
            return default
        n = float(value)
    except (ValueError, TypeError):
        return default
    return n if n > 0 else default


# Parse cache: page navigation re-runs the whole pipeline, and the LLM parse is its
# only slow/expensive stage — cache it so page 2+ (and repeats of popular queries)
# parse once. Keyed by normalized query text; bounded (LRU-ish) so memory stays flat
# under many distinct users; TTL keeps prompt/model changes from serving stale
# criteria forever. Values are deep-copied OUT because the orchestrator mutates
# ParsedQuery (reconstructed_queries, injected criteria).
_PARSE_CACHE: dict[str, tuple[float, ParsedQuery]] = {}
_PARSE_CACHE_TTL_SEC = 300
_PARSE_CACHE_MAX = 500


async def parse_query(query: str, max_retries: int = 2) -> ParsedQuery:
    import time
    key = query.strip().lower()
    hit = _PARSE_CACHE.get(key)
    if hit and time.monotonic() - hit[0] < _PARSE_CACHE_TTL_SEC:
        logger.info("Parse cache hit — skipping LLM call")
        return hit[1].model_copy(deep=True)

    result = await _parse_query_uncached(query, max_retries)
    if len(_PARSE_CACHE) >= _PARSE_CACHE_MAX:
        # Evict the oldest entry; O(n) over a small bounded dict is fine.
        _PARSE_CACHE.pop(min(_PARSE_CACHE, key=lambda k: _PARSE_CACHE[k][0]))
    _PARSE_CACHE[key] = (time.monotonic(), result)
    return result.model_copy(deep=True)


async def _parse_query_uncached(query: str, max_retries: int = 2) -> ParsedQuery:
    client = get_query_client()
    system_prompt = _build_system_prompt(
        settings.search_use_embedding_retrieval, await _known_regions_block()
    )

    raw_text = None
    parsed = None

    for attempt in range(max_retries):
        try:
            raw_text = await _call_llm(client, system_prompt, query)
            if not raw_text:
                logger.warning(f"LLM returned empty response (attempt {attempt + 1}/{max_retries})")
                continue

            logger.debug(f"Raw LLM response: {raw_text[:500]}")

            # Strip markdown code fences
            clean = raw_text
            if clean.startswith("```"):
                lines = clean.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                clean = "\n".join(lines).strip()

            parsed = json.loads(clean)
            break
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(
                f"Failed to parse LLM response (attempt {attempt + 1}/{max_retries}): {e}\n"
                f"Raw text: {raw_text[:500] if raw_text else 'None'}"
            )
            continue
        except Exception as e:
            logger.error(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
            continue

    if not parsed:
        logger.error(f"All {max_retries} attempts failed for query: '{query}'")
        # Hard parse failure (e.g. LLM outage). Raise so /search returns an error
        # instead of an empty-criteria query that would match the ENTIRE catalog.
        raise QueryParseError(f"Could not parse query after {max_retries} attempts")

    if not isinstance(parsed, dict):
        logger.error(f"LLM returned non-object JSON ({type(parsed).__name__}) for query: '{query}'")
        raise QueryParseError("LLM returned a non-object response")

    # Build each criterion in its own try/except so ONE malformed entry (a non-numeric
    # distance, a bad enum, a Pydantic ValidationError) is skipped — not the whole query.
    raw_criteria = parsed.get("criteria")
    if not isinstance(raw_criteria, list):  # e.g. LLM returned {"criteria": 42}
        logger.warning(f"LLM 'criteria' is {type(raw_criteria).__name__}, not a list; treating as empty")
        raw_criteria = []
    criteria = []
    for c in raw_criteria:
        try:
            criterion_type = c["type"]
            if criterion_type == "room_count":
                exact, cmin = c.get("exact_count"), c.get("min_count")
                # Product rule: a bare count means AT LEAST that many. The prompt
                # says so too, but enforce it deterministically — keep exact only
                # when the user actually said so ("exactly"/"only 2 bedrooms").
                if exact is not None and cmin is None and not re.search(
                        r"\b(?:exact|exactly|only|just)\b", query, re.IGNORECASE):
                    cmin, exact = exact, None
                criteria.append(RoomCountCriterion(
                    room_type=c["room_type"],
                    exact_count=exact,
                    min_count=cmin,
                    max_count=c.get("max_count"),
                ))
            elif criterion_type == "feature":
                strength = str(c.get("strength") or "hard").strip().lower()
                criteria.append(FeatureCriterion(
                    feature=c["feature"],
                    room_context=canonical_room_context(c.get("room_context")),
                    negated=c.get("negated", False),
                    strength=strength if strength in ("hard", "soft") else "hard",
                ))
            elif criterion_type == "price":
                pmin, pmax = c.get("min_price"), c.get("max_price")
                # Product rule: a specific price means THAT PRICE AND BELOW.
                # The prompt says so too, but enforce it deterministically —
                # an exact-price parse (min == max) matched almost nothing
                # ("homes 400k" returned 3 results instead of ~900).
                if pmin is not None and pmin == pmax:
                    pmin = None
                criteria.append(PriceCriterion(
                    min_price=pmin,
                    max_price=pmax,
                ))
            elif criterion_type == "area":
                smin, smax = c.get("min_sqft"), c.get("max_sqft")
                # Product rule: a bare size means AT LEAST that size (enforced
                # deterministically like the bare-price guard, but keeping MIN).
                if smin is not None and smin == smax:
                    smax = None
                criteria.append(AreaCriterion(
                    min_sqft=smin,
                    max_sqft=smax,
                ))
            elif criterion_type == "location":
                criteria.append(LocationCriterion(
                    city=c.get("city"),
                    state=c.get("state"),
                    country=c.get("country"),
                    district=c.get("district"),
                    county=c.get("county"),
                    neighborhood=c.get("neighborhood"),
                    locality=c.get("locality"),
                    street=c.get("street"),
                ))
            elif criterion_type == "proximity":
                criteria.append(ProximityCriterion(
                    landmark_name=c["landmark_name"],
                    max_distance_miles=_safe_miles(c.get("max_distance_miles")),
                ))
            elif criterion_type == "area_relation":
                place_a = c.get("place_a")
                if place_a:
                    criteria.append(AreaRelationCriterion(
                        relation=str(c.get("relation") or "near").strip().lower(),
                        place_a=place_a,
                        place_b=c.get("place_b"),
                        radius_miles=_safe_miles(c.get("radius_miles"), default=None),
                    ))
            elif criterion_type == "property":
                ymin, ymax = c.get("min_year_built"), c.get("max_year_built")
                # Product rule: a bare year built means THAT YEAR OR NEWER.
                if ymin is not None and ymin == ymax:
                    ymax = None
                criteria.append(PropertyCriterion(
                    home_type=c.get("home_type"),
                    min_rent=c.get("min_rent"),
                    max_rent=c.get("max_rent"),
                    min_year_built=ymin,
                    max_year_built=ymax,
                    min_lot_sqft=c.get("min_lot_sqft"),
                    max_lot_sqft=c.get("max_lot_sqft"),
                    min_stories=c.get("min_stories"),
                    max_stories=c.get("max_stories"),
                ))
            elif criterion_type == "color_room":
                criteria.append(ColorRoomCriterion(
                    color=str(c["color"]).strip().lower(),
                    room_type=str(c["room_type"]).strip(),
                    negated=c.get("negated", False),
                ))
            else:
                logger.warning(f"Unknown criterion type: {criterion_type}")
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Skipping malformed criterion {c!r}: {e}")
            continue

    return ParsedQuery(
        original_query=query,
        criteria=criteria,
        # NULL-safe: {"understood_intent": null} would fail ParsedQuery validation
        # OUTSIDE the retry loop → opaque 500 instead of a handled parse result.
        reconstructed_queries=(
            rq if isinstance(rq := parsed.get("reconstructed_queries"), list) else []
        ),
        understood_intent=str(parsed.get("understood_intent") or ""),
    )
