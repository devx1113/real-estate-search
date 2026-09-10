"""Real Estate AI Search FastAPI app — multi-phase pipeline (LLM parse → hard filters → proximity → feature matching) returning only properties matching ALL criteria."""

import asyncio
import gzip
import json
import logging
import re
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, field_validator
from starlette.responses import Response

from config.settings import settings
from src.data.database import close_pool, get_pool
from src.data.feature_registry import registry
from src.img_analyzer.router import router as img_analyzer_router
from src.img_analyzer.test_router import router as vision_test_router
from src.models.search import CriterionType, ParsedQuery
from src.search.orchestrator import _load_results as load_brief_properties, search
from src.search.photo_search import detailed_photos
from src.search.query_parser import QueryParseError

logging.basicConfig(
    level=settings.log_level.upper(),  # tolerate LOG_LEVEL=info in .env
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await get_pool()
    logger.info("Database pool initialized")

    # Fail loudly+early on a missing key rather than a cryptic 401 on the first LLM call.
    if not settings.openai_api_key:
        logger.error(
            "OPENAI_API_KEY is empty — vision, /search query parsing, and "
            "embeddings will all fail. Set it in .env."
        )

    logger.info("Building feature registry from database...")
    await registry.build_from_db(pool)
    logger.info(
        f"Registry ready: {len(registry.features)} features, "
        f"{len(registry.room_types)} room types"
    )

    # Incrementally embed features not yet in feature_embeddings; guarded so an embedding outage can't block startup.
    if settings.search_use_embedding_retrieval:
        try:
            from src.search.feature_index import sync_feature_embeddings
            n = await sync_feature_embeddings(pool)
            logger.info(f"Feature-embedding sync at startup: {n} newly embedded")
        except Exception as e:
            logger.error(f"Feature-embedding sync at startup failed: {e}")

    # LISTEN for 'feature_change' to keep the registry fresh; supervisor heartbeats + reconnects so a Postgres blip can't drop NOTIFYs.
    supervisor = _ListenerSupervisor(pool)
    supervisor_task = asyncio.create_task(supervisor.run())
    logger.info("LISTEN feature_change supervisor started")

    yield

    await supervisor.stop()
    supervisor_task.cancel()
    try:
        await supervisor_task
    except asyncio.CancelledError:
        pass
    logger.info("Feature-change listener stopped")

    await close_pool()
    logger.info("Database pool closed")


async def _refresh_registry_and_embeddings() -> None:
    """Rebuild the registry and incrementally embed new features; shared by NOTIFY handling and listener reconnects."""
    pool = await get_pool()
    await registry.build_from_db(pool)
    logger.info(
        f"Registry refreshed: {len(registry.features)} features, "
        f"{len(registry.room_types)} room types"
    )
    # New room features may have appeared → embed them and drop the resolver cache so they surface immediately.
    if settings.search_use_embedding_retrieval:
        from src.search.feature_index import sync_feature_embeddings
        from src.search.feature_resolver import clear_cache, clear_persisted_cache
        n = await sync_feature_embeddings(pool)
        if n:
            # New tags exist: curated phrase lists may be missing them — drop the
            # caches so each phrase re-curates once. Unchanged vocabulary keeps
            # the persisted (deterministic) lists.
            clear_cache()
            await clear_persisted_cache(pool)
        logger.info(f"Embeddings synced: {n} newly embedded; resolver caches {'cleared' if n else 'kept'}")


async def _on_feature_change(_connection, _pid, channel, _payload):
    logger.info(f"Received NOTIFY {channel} → rebuilding feature registry")
    try:
        await _refresh_registry_and_embeddings()
    except Exception as e:
        logger.error(f"Registry/embedding refresh after notify failed: {e}")


class _ListenerSupervisor:
    """Dedicated LISTEN connection that reconnects on failure (probes every HEARTBEAT_SEC with SELECT 1, re-LISTENs on error)."""

    HEARTBEAT_SEC = 30
    RECONNECT_BACKOFF_SEC = 2

    def __init__(self, pool):
        self._pool = pool
        self._stopping = False
        self._connected_before = False

    async def stop(self) -> None:
        self._stopping = True

    async def run(self) -> None:
        while not self._stopping:
            conn = None
            try:
                conn = await self._pool.acquire()
                await conn.add_listener("feature_change", _on_feature_change)
                logger.info("LISTENing for feature_change notifications")
                if self._connected_before:
                    # NOTIFYs fired while we were disconnected are gone — compensate
                    # with one refresh so the registry can't stay stale until the
                    # next ingest happens to fire again.
                    try:
                        await _refresh_registry_and_embeddings()
                    except Exception as e:
                        logger.error(f"Post-reconnect refresh failed: {e}")
                self._connected_before = True
                while not self._stopping:
                    await asyncio.sleep(self.HEARTBEAT_SEC)
                    await conn.fetchval("SELECT 1")  # raises if session is gone
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    f"Listener connection lost ({type(e).__name__}: {e}); "
                    f"reconnecting in {self.RECONNECT_BACKOFF_SEC}s"
                )
            finally:
                if conn is not None:
                    try:
                        await conn.remove_listener("feature_change", _on_feature_change)
                    except Exception:
                        pass
                    try:
                        # No `discard=` kwarg exists on asyncpg Pool.release — passing it
                        # raised TypeError and LEAKED the connection (shutdown hang).
                        await self._pool.release(conn)
                    except Exception:
                        pass
            if not self._stopping:
                await asyncio.sleep(self.RECONNECT_BACKOFF_SEC)


app = FastAPI(
    title="Real Estate AI Search",
    description="Accurate AI-powered real estate search — returns only properties that match ALL criteria.",
    version="0.3.0",
    lifespan=lifespan,
)

# Compress large JSON responses (photo-URL lists shrink ~5x; browsers negotiate via Accept-Encoding).
app.add_middleware(GZipMiddleware, minimum_size=8192)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # Must be False with wildcard origin (CORS spec forbids the combo); API is token/cookie-free.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(img_analyzer_router)
app.include_router(vision_test_router)


LOG_DIR = Path(settings.log_dir)
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
except OSError as e:
    # Don't abort startup if the log dir isn't writable; request logging just no-ops (see _append_log_entry's guard).
    logger.warning(f"Could not create log dir {LOG_DIR}: {e}; request logging disabled")
MAX_BODY_LOG_BYTES = 200_000
_log_write_lock = asyncio.Lock()


def _normalize_bounds(body: dict) -> dict:
    """Coerce bounds values to floats for logging."""
    bounds = body.get("bounds")
    if isinstance(bounds, dict):
        for key in ("north", "south", "east", "west"):
            if key in bounds:
                try:
                    bounds[key] = float(bounds[key])
                except (ValueError, TypeError):
                    pass
    return body


def _decode_body(raw: bytes, content_type: str) -> dict | str | None:
    if not raw:
        return None
    if len(raw) > MAX_BODY_LOG_BYTES:
        return {"_truncated": True, "size_bytes": len(raw)}
    if "application/json" in content_type:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                parsed = _normalize_bounds(parsed)
            return parsed
        except json.JSONDecodeError:
            pass
    if "multipart/form-data" in content_type:
        return {"_binary_upload": True, "size_bytes": len(raw)}
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return {"_binary": True, "size_bytes": len(raw)}


def _weekly_log_path() -> Path:
    year, week, _ = datetime.now(timezone.utc).isocalendar()
    return LOG_DIR / f"{year}-W{week:02d}.jsonl"


async def _append_log_entry(entry: dict) -> None:
    if not LOG_DIR.is_dir():
        return  # log dir unavailable (see startup warning) — skip silently
    # Append one JSON object per line — O(1) per request, no whole-file rewrite.
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    async with _log_write_lock:
        with _weekly_log_path().open("a", encoding="utf-8") as f:
            f.write(line)


@app.middleware("http")
async def log_request_response(request: Request, call_next):
    if request.url.path != "/search":
        return await call_next(request)

    req_body = await request.body()

    async def receive():
        return {"type": "http.request", "body": req_body, "more_body": False}

    request._receive = receive

    response = await call_next(request)

    resp_chunks: list[bytes] = []
    async for chunk in response.body_iterator:
        resp_chunks.append(chunk)
    resp_body = b"".join(resp_chunks)

    new_response = Response(
        content=resp_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )

    try:
        # GZipMiddleware sits inside this middleware, so large responses arrive here
        # compressed — decompress for the log only (the client gets the original bytes).
        log_body = resp_body
        if new_response.headers.get("content-encoding") == "gzip":
            log_body = gzip.decompress(resp_body)
        entry = {
            "req": _decode_body(req_body, request.headers.get("content-type", "")),
            "res": _decode_body(log_body, new_response.headers.get("content-type", "")),
        }
        await _append_log_entry(entry)
    except Exception as e:
        logger.error(f"Failed to write request log: {e}")

    return new_response


class LatLng(BaseModel):
    Lat: float
    Lng: float


class Bounds(BaseModel):
    north: float
    south: float
    east: float
    west: float


ALLOWED_HOME_TYPES = {"SINGLE_FAMILY", "CONDO", "TOWNHOUSE", "MANUFACTURED", "MULTI_FAMILY", "LOT"}


class Filters(BaseModel):
    price_min: int | None = None
    price_max: int | None = None
    beds_min: int | None = None
    baths_min: int | None = None
    property_types: list[str] | None = None
    financing: list[str] | None = None
    sqft_min: int | None = None
    sqft_max: int | None = None
    year_from: int | None = None
    year_to: int | None = None

    @field_validator("property_types")
    @classmethod
    def _norm_property_types(cls, v):
        if v is None:
            return None
        normalized: list[str] = []
        seen: set[str] = set()
        for item in v:
            up = str(item).strip().upper()
            if up not in ALLOWED_HOME_TYPES:
                raise ValueError(
                    f"Invalid property_type '{item}'. "
                    f"Allowed: {sorted(ALLOWED_HOME_TYPES)}"
                )
            if up not in seen:
                seen.add(up)
                normalized.append(up)
        return normalized

    @field_validator("financing")
    @classmethod
    def _norm_financing(cls, v):
        if v is None:
            return None
        out: list[str] = []
        seen: set[str] = set()
        for item in v:
            norm = "_".join(str(item).strip().lower().split())
            if norm and norm not in seen:
                seen.add(norm)
                out.append(norm)
        return out


class ClientLocation(BaseModel):
    """User's own position from the browser geolocation API (sent only when the
    user granted the permission prompt). Scopes location-less queries to their city."""
    lat: float
    lng: float


class SearchRequest(BaseModel):
    query: str
    bounds: Bounds | None = None
    filters: Filters | None = None
    clientLocation: ClientLocation | None = None
    # "Remove boundaries" button: True = ignore the region part of the query
    # entirely (no region filtering, no location detection) and scope only by
    # `bounds` when sent — the map replaces the region. Response then has
    # regionId/regionName/polygons null and detectedLocation false.
    ignore_region: bool = False
    # Hand-drawn map polygon (the "draw" tool): results are restricted to homes
    # INSIDE this ring, same {Lat, Lng} format the response's `polygons` uses.
    # This is the ONLY polygon-based filtering in the system — named regions
    # filter by region id. Combines with the query/filters like `bounds` does,
    # and like bounds it suppresses detected-location injection.
    drawnPolygon: list[LatLng] | None = None
    # Pagination of briefproperties (pins are never paginated). Defaults keep
    # existing callers working: page 1 of 50.
    page: int = 1
    pageSize: int = 50
    debug: bool = False

    @field_validator("page")
    @classmethod
    def _page_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("page must be >= 1")
        return v

    @field_validator("pageSize")
    @classmethod
    def _page_size_range(cls, v: int) -> int:
        if not 1 <= v <= 200:
            raise ValueError("pageSize must be between 1 and 200")
        return v

    @field_validator("drawnPolygon")
    @classmethod
    def _polygon_ring(cls, v: list[LatLng] | None) -> list[LatLng] | None:
        if v is None:
            return v
        if len(v) < 3:
            raise ValueError("drawnPolygon needs at least 3 points")
        if len(v) > 1000:
            raise ValueError("drawnPolygon supports at most 1000 points")
        for p in v:
            if not (-90 <= p.Lat <= 90 and -180 <= p.Lng <= 180):
                raise ValueError("drawnPolygon contains an out-of-range coordinate")
        return v

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, v: str) -> str:
        # Blank is allowed: the structured fields (filters/bounds/drawnPolygon) can
        # carry the whole search — e.g. after "remove boundaries" empties the bar.
        # A blank query with NO other criteria still 422s downstream, so an
        # unfiltered search can't dump the catalog by accident.
        return (v or "").strip()


class DebugInfo(BaseModel):
    parsed_query: ParsedQuery
    stats: dict
    filter_steps: list[dict]
    bounds_applied: bool


class PhotoGroup(BaseModel):
    """A property's images of one room type, e.g. all its Bathroom photos."""
    roomType: str                      # native name: "Bathroom", "Living Room", "Pool";
    #                                    "Main" = highlight strip (first of each type,
    #                                    Exterior leading — Main[0] is the cover photo);
    #                                    "Other" for images the vision pass couldn't classify
    urls: list[str] = []               # /search: smallest-res; /search/photos: full-res


class PhotoSearchRequest(BaseModel):
    propertyIds: list[str]  # property GUIDs

    @field_validator("propertyIds")
    @classmethod
    def _ids_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("propertyIds must contain at least one GUID")
        if len(v) > 500:
            raise ValueError("propertyIds is limited to 500 GUIDs per request")
        return v


class PropertyDetailedPhotos(BaseModel):
    id: str                                    # property GUID
    propertyphotos: list[PhotoGroup] = []      # same structure as /search, full-res urls


class PhotoSearchResponse(BaseModel):
    propertydetailedphotos: list[PropertyDetailedPhotos]


@app.post("/search/photos", response_model=PhotoSearchResponse)
async def search_photos_endpoint(request: PhotoSearchRequest):
    """ALL photos of the given properties, grouped by room type (incl. the "Main"
    highlight group), at FULL resolution — the detail/lightbox companion to
    /search's thumbnail propertyphotos. Unknown guids are omitted; input order is
    preserved. No LLM involved — instant and free."""
    try:
        pool = await get_pool()
        results = await detailed_photos(pool, request.propertyIds)
        return PhotoSearchResponse(propertydetailedphotos=results)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Photo detail failed for {len(request.propertyIds)} ids: {e}")
        raise HTTPException(status_code=500, detail="Photo search failed due to an internal error.")


class PropertyAddress(BaseModel):
    streetAddress: str | None = None
    city: str | None = None            # USPS mailing city
    state: str | None = None
    zipcode: str | None = None
    neighborhood: str | None = None    # Zillow neighborhood page (e.g. "Viera East")
    community: str | None = None       # OSM place/locality (e.g. "Viera") — the level
    #                                    the mailing address can't express
    subdivision: str | None = None     # plat name (e.g. "Bridgewater at Viera")


class BriefProperty(BaseModel):
    """One matched property: pin data + the card fields the result list renders."""
    propertyId: str
    price: float | None = None
    currency: str | None = None
    # null when the listing has no coordinates (undisclosed address) — never a fake 0,0.
    latitude: float | None = None
    longitude: float | None = None
    bedrooms: int | None = None
    bathrooms: int | None = None
    livingArea: int | None = None
    address: PropertyAddress
    homeStatus: str | None = None
    homeType: str | None = None
    # Images grouped by room type: [{roomType: "Bathroom", urls: [...]}, ...].
    propertyphotos: list[PhotoGroup] = []
    # Frozen at scrape time (Zillow's daysOnZillow) — does NOT tick daily after ingest.
    daysOnmarket: int | None = None
    yearBuilt: int | None = None
    county: str | None = None
    lotAreaValue: float | None = None  # square feet (acreage converted at ingest)
    # The query's soft wishes ("great kitchen", "feels private") this home
    # satisfies — results are ranked by how many; lets the UI badge them.
    matchedSoft: list[str] = []


class MapPin(BaseModel):
    """Every matched property's map marker — never paginated, so the map stays
    complete while the heavy briefproperties list pages."""
    propertyId: str
    latitude: float | None = None
    longitude: float | None = None
    price: float | None = None


class SearchFilters(BaseModel):
    """The structured filters the parser extracted from the natural-language query,
    echoed back so the frontend can sync its filter panel with what the search
    actually applied. Unspecified = 0 / [] (client-agreed shape — not null).
    NOTE: an exact count ("2 bedrooms") is reported in beds_min even though the
    search matches it exactly; financing is never parsed today, so always []."""
    price_min: int = 0
    price_max: int = 0
    beds_min: int = 0
    baths_min: int = 0
    property_types: list[str] = []
    financing: list[str] = []
    sqft_min: int = 0
    sqft_max: int = 0
    year_from: int = 0
    year_to: int = 0


def _extract_filters(parsed) -> SearchFilters:
    """ParsedQuery.criteria -> SearchFilters. First criterion of a kind wins;
    location/features/colors are deliberately excluded (covered by regionId and
    updated_query)."""
    f = SearchFilters()
    for c in parsed.criteria:
        t = getattr(c, "type", None)
        if t == CriterionType.PRICE:
            f.price_min = f.price_min or c.min_price or 0
            f.price_max = f.price_max or c.max_price or 0
        elif t == CriterionType.AREA:
            f.sqft_min = f.sqft_min or c.min_sqft or 0
            f.sqft_max = f.sqft_max or c.max_sqft or 0
        elif t == CriterionType.ROOM_COUNT:
            count = c.exact_count or c.min_count or 0
            if c.room_type == "Bedroom":
                f.beds_min = f.beds_min or count
            elif c.room_type == "Bathroom":
                f.baths_min = f.baths_min or count
        elif t == CriterionType.PROPERTY:
            if c.home_type and c.home_type not in f.property_types:
                f.property_types.append(c.home_type)
            f.year_from = f.year_from or c.min_year_built or 0
            f.year_to = f.year_to or c.max_year_built or 0
    return f


class SearchResponse(BaseModel):
    # VERBATIM echo of what the user typed — never modified, safe for the frontend
    # to hold/resend. Filter descriptions live in updated_query instead: merging
    # them into query caused a feedback loop (frontend resent the merged string,
    # clauses accumulated and even polluted LLM parsing).
    query: str
    # The EFFECTIVE search as one readable line: the typed query plus the applied
    # UI filters ("in brevard, 3+ bedrooms"). Display-only — never send it back.
    updated_query: str
    # The current page of full result cards (see page/pageSize/totalCount).
    briefproperties: list[BriefProperty]
    # ALL matches, lightweight — for the map.
    pins: list[MapPin]
    totalCount: int
    page: int
    pageSize: int
    totalPages: int
    # Place the user searched INSIDE (map label); None when the query names no place
    # or is relational ("near X" / "between X and Y"). When regionId is set this is
    # the catalog's canonical "<RegionName>, <StateCode>"; otherwise just the place
    # name as parsed from the query.
    regionName: str | None = None
    # Set ONLY when this region's polygon geometrically filtered the results —
    # regionId is a promise that the pins are that region's geometry. Null on every
    # name-matching fallback (no polygon data, subdivision/neighborhood searches,
    # ambiguous names), even if the place is known.
    regionId: int | None = None
    # The boundary that FILTERED these results, as rings of {Lat, Lng} points.
    # Present exactly when regionId is: draw it and every pin is inside by
    # construction.
    polygons: list[list[LatLng]] | None = None
    # True when the query named no place and the search area came from the user's
    # location (browser coordinates, IP, or the Brevard default) — lets the UI say
    # "showing homes near you".
    detectedLocation: bool = False
    # Structured filters parsed from the query (see SearchFilters) — lets the
    # frontend's filter panel mirror the natural-language query.
    filters: SearchFilters
    # Hard feature requirements the search had to LOOSEN (turned into ranking
    # signals) because they left fewer than a handful of homes — show as
    # "showing homes without: ...". Empty when everything was honored.
    relaxed: list[str] = []
    # The query's subjective wishes (never filters, only ranking) — see
    # BriefProperty.matchedSoft for which ones each result satisfies.
    softCriteria: list[str] = []
    # True only when the query was a COMPLETE street address (house number +
    # street) and exactly one property matched it and every other criterion:
    # the UI opens that home directly. False for every other response — an
    # incomplete address, several units, no match, or a normal search.
    exactAddress: bool = False
    debug: DebugInfo | None = None


def _strip_locations(query: str, places: list[str]) -> str:
    """Remove parsed place names (and their 'in/at/near' connector) from the query
    text — used by ignore_region so the search bar stops showing places the
    search no longer uses. Longest place first so "Viera East" wins over "Viera";
    then whitespace/comma cleanup. May legitimately return "" (place-only query
    -> empty search bar)."""
    import difflib
    out = query
    for place in sorted({p for p in places if p}, key=len, reverse=True):
        before = out
        esc = re.escape(place).replace(r"\ ", r"\s+")
        out = re.sub(rf"\b(?:in|at|near|of)\s+{esc}\b\s*,?", " ", out, flags=re.IGNORECASE)
        out = re.sub(rf"\b{esc}\b\s*,?", " ", out, flags=re.IGNORECASE)
        if out == before:
            # The parser normalizes misspellings ("veira" -> "Viera"), so the
            # canonical value may not appear verbatim in the typed text. Fuzzy
            # fallback: remove the connector phrase whose words nearly match.
            n_words = max(len(place.split()), 1)
            for m in re.finditer(r"\b(?:in|at|near|of)\s+([\w'][\w'\s]*)",
                                 before, flags=re.IGNORECASE):
                cand = " ".join(m.group(1).split()[:n_words])
                if difflib.SequenceMatcher(
                        None, cand.lower(), place.lower()).ratio() >= 0.75:
                    cand_esc = re.escape(cand).replace(r"\ ", r"\s+")
                    out = re.sub(rf"\b(?:in|at|near|of)\s+{cand_esc}\b\s*,?",
                                 " ", before, count=1, flags=re.IGNORECASE)
                    break
            else:
                # Place-only query typed with a misspelling ("veira" alone).
                if difflib.SequenceMatcher(
                        None, before.strip().lower(), place.lower()).ratio() >= 0.75:
                    out = ""
    out = re.sub(r"\s+,", ",", out)          # "homes , pool" -> "homes, pool"
    out = re.sub(r",\s*,", ",", out)         # collapse doubled commas
    out = re.sub(r"\s{2,}", " ", out).strip(" ,")
    # Dangling conjunction after removing a place: "homes or" -> "homes"
    out = re.sub(r"(?:^|\s)(?:or|and)$", "", out, flags=re.IGNORECASE).strip(" ,")
    return out


_MERGE_QUALIFIER = (r"(?:priced?\s+)?(?:above|over|under|below|at\s+least|at\s+most|"
                    r"up\s+to|more\s+than|less\s+than|min(?:imum)?|max(?:imum)?|"
                    r"from|starting\s+at|between)")


def _swap_phrase(text: str, expr: str, desc: str) -> tuple[str, bool]:
    """Replace the FIRST match of `expr` in `text` with `desc` and DELETE any
    further matches (plus a dangling and/or connector) — so the merged line
    mentions the constraint exactly once. Placeholder pass because `desc`
    itself usually matches `expr` ("min price $500,000", "built after 1990")
    and a direct residual-deletion would eat our own replacement."""
    placeholder = "\x00SWAP\x00"
    new, n = re.subn(expr, placeholder, text, count=1, flags=re.IGNORECASE)
    if not n:
        return text, False
    new = re.sub(rf"(?:\s+(?:and|or))?\s*(?:{expr})", " ", new, flags=re.IGNORECASE)
    new = new.replace(placeholder, desc)
    return re.sub(r"\s{2,}", " ", new).strip(), True


def _merge_query_filters(query: str, f: dict) -> str:
    """The EFFECTIVE search as one readable line. Filters that CONFLICT with a
    number in the typed text REPLACE it in place ("4 bedrooms in rockledge" +
    filter 3 -> "3+ bedrooms in rockledge" — matching the pipeline, where a UI
    room filter overrides the parsed criterion); everything else is appended as
    clauses. Deterministic; no LLM."""
    out = query
    consumed: set[str] = set()
    if f.get("beds_min") is not None:
        new, n = re.subn(
            r"\b\d+\s*\+?\s*(?:bed(?:room)?s?|br)\b",
            f"{f['beds_min']}+ bedrooms", out, flags=re.IGNORECASE,
        )
        if n:
            out = new
            consumed.add("beds_min")
    if f.get("baths_min") is not None:
        new, n = re.subn(
            r"\b\d+(?:\.\d+)?\s*\+?\s*(?:bath(?:room)?s?|ba)\b",
            f"{f['baths_min']}+ bathrooms", out, flags=re.IGNORECASE,
        )
        if n:
            out = new
            consumed.add("baths_min")
    if f.get("price_min") is not None or f.get("price_max") is not None:
        # Filter has priority over the query's price (the handler only passes a
        # price here when it DIFFERS from what the query says): the winning value
        # replaces the typed price phrase so the line shows the price ONCE. An
        # amount only counts as a price with a $ prefix or k/m suffix — and not
        # when an area word follows — so "above 2000 sqft", "5k sqft" and
        # "built after 2000" are never swallowed.
        amount = r"(?:\$\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:k|m)\b)?|\d[\d,]*(?:\.\d+)?\s*(?:k|m)\b)"
        price_expr = (rf"(?:{_MERGE_QUALIFIER}\s+)?{amount}(?:\s*\+)?"
                      rf"(?:\s*(?:-|–|to|and)\s*{amount})?"
                      rf"(?!\s*(?:k\b|m\b|sq|sf|square|ft\b|feet|acre))")
        desc = _describe_filters(
            {k: f[k] for k in ("price_min", "price_max") if f.get(k) is not None}
        )
        out, swapped = _swap_phrase(out, price_expr, desc)
        if swapped:
            consumed.update(("price_min", "price_max"))
    if f.get("sqft_min") is not None or f.get("sqft_max") is not None:
        # Same single-mention rule for square footage; the sqft word anchors the
        # match, so plain numbers/prices/years are never touched.
        sq_amount = r"\d[\d,]*(?:\.\d+)?\s*k?"
        sq_word = r"(?:sq\.?\s*\.?\s*ft\.?|sqft|sf|square\s+f(?:ee|oo)t)"
        sqft_expr = (rf"(?:{_MERGE_QUALIFIER}\s+)?{sq_amount}"
                     rf"(?:\s*(?:-|–|to|and)\s*{sq_amount})?\s*{sq_word}\b")
        desc = _describe_filters(
            {k: f[k] for k in ("sqft_min", "sqft_max") if f.get(k) is not None}
        )
        out, swapped = _swap_phrase(out, sqft_expr, desc)
        if swapped:
            consumed.update(("sqft_min", "sqft_max"))
    if f.get("year_from") is not None or f.get("year_to") is not None:
        # Single mention for year built; anchored on built/constructed wording so
        # bare years ("2000 sqft", prices) are never touched.
        yr = r"(?:19|20)\d{2}"
        year_expr = (rf"(?:built|constructed|build)\s*(?:in|after|before|since|"
                     rf"from|between)?\s*{yr}(?:\s*(?:-|–|to|and)\s*{yr})?"
                     rf"(?:\s*(?:or|and)\s+(?:newer|later|older|earlier))?")
        desc = _describe_filters(
            {k: f[k] for k in ("year_from", "year_to") if f.get(k) is not None}
        )
        out, swapped = _swap_phrase(out, year_expr, desc)
        if swapped:
            consumed.update(("year_from", "year_to"))
    clauses = _describe_filters({k: v for k, v in f.items() if k not in consumed})
    if not clauses:
        return out
    # Empty base (ignore_region stripped a place-only query): clauses stand alone.
    return f"{out}, {clauses}" if out.strip() else clauses


def _describe_filters(f: dict) -> str:
    """Human-readable clauses for the applied UI filters, appended to the response's
    query field so it describes the EFFECTIVE search ("homes in viera, max price
    $500,000, 3+ bedrooms"). Deterministic string building — no LLM involved."""
    parts: list[str] = []
    pmin, pmax = f.get("price_min"), f.get("price_max")
    if pmin is not None and pmax is not None:
        parts.append(f"price ${pmin:,}-${pmax:,}")
    elif pmin is not None:
        parts.append(f"min price ${pmin:,}")
    elif pmax is not None:
        parts.append(f"max price ${pmax:,}")
    if f.get("beds_min") is not None:
        parts.append(f"{f['beds_min']}+ bedrooms")
    if f.get("baths_min") is not None:
        parts.append(f"{f['baths_min']}+ bathrooms")
    smin, smax = f.get("sqft_min"), f.get("sqft_max")
    if smin is not None and smax is not None:
        parts.append(f"{smin:,}-{smax:,} sqft")
    elif smin is not None:
        parts.append(f"min {smin:,} sqft")
    elif smax is not None:
        parts.append(f"max {smax:,} sqft")
    yfrom, yto = f.get("year_from"), f.get("year_to")
    if yfrom is not None and yto is not None:
        parts.append(f"built {yfrom}-{yto}")
    elif yfrom is not None:
        parts.append(f"built after {yfrom}")
    elif yto is not None:
        parts.append(f"built before {yto}")
    if f.get("property_types"):
        pretty = [t.replace("_", " ").title() for t in f["property_types"]]
        parts.append("type: " + ", ".join(pretty))
    if f.get("financing"):
        parts.append("financing: " + ", ".join(f["financing"]))
    return ", ".join(parts)


@app.post("/search", response_model=SearchResponse)
async def search_properties(request: SearchRequest):
    try:
        pool = await get_pool()
        bounds_dict = request.bounds.model_dump() if request.bounds else None
        filters_dict = (
            request.filters.model_dump(exclude_none=True) if request.filters else None
        )
        if filters_dict is not None:
            # 0 / [] mean "not specified" — the response's `filters` object uses
            # those as its unset markers (client contract), so a frontend that
            # echoes it back must not turn every search into "max price $0".
            # No real constraint in this shape is expressed by 0 or [].
            filters_dict = {k: v for k, v in filters_dict.items() if v}
        if not filters_dict:
            filters_dict = None  # nothing specified = no filters

        result = await search(
            request.query,
            pool,
            bounds=bounds_dict,
            filters=filters_dict,
            debug=request.debug,
            page=request.page,
            page_size=request.pageSize,
            client_location=(
                request.clientLocation.model_dump() if request.clientLocation else None
            ),
            ignore_region=request.ignore_region,
            drawn_polygon=(
                [(p.Lat, p.Lng) for p in request.drawnPolygon]
                if request.drawnPolygon else None
            ),
        )

        # A query that parses to ZERO criteria (gibberish) with no bounds/filters would
        # otherwise "match" the entire catalog — reject it instead of dumping the DB.
        # Exceptions: ignore_region — a location-only query legitimately strips down to
        # zero criteria there ("remove boundaries" on "homes in rockledge" = browse
        # everything, scoped by the map when the frontend sends bounds) — and a
        # drawn polygon, which IS the search area.
        if (not request.ignore_region
                and request.drawnPolygon is None
                and not result.get("address_mode")
                and not result["parsed_query"].criteria
                and bounds_dict is None and filters_dict is None):
            raise HTTPException(
                status_code=422,
                detail="Could not extract any search criteria from the query; please rephrase.",
            )

        results = result["results"]

        debug_info = None
        if request.debug:
            debug_info = DebugInfo(
                parsed_query=result["parsed_query"],
                stats=result["stats"],
                filter_steps=result["filter_steps"] or [],
                bounds_applied=bounds_dict is not None,
            )

        total = result["total_count"]
        # updated_query = the EFFECTIVE search; query stays the verbatim user input.
        # A detected search area (browser location or the Brevard County default)
        # is appended as "in <area>"; conflicting numbers in the typed text are
        # replaced by filter values (mirroring the pipeline's override behavior).
        base_query = request.query
        if result.get("location_detected") and result.get("region_name"):
            base_query = f"{request.query} in {result['region_name']}".strip()
        if request.ignore_region and result.get("stripped_locations"):
            # "Remove boundaries": the search ignored these places, so the search
            # bar must not keep showing them. A place-only query strips to an
            # EMPTY bar (product decision).
            base_query = _strip_locations(base_query, result["stripped_locations"])
        # Client rule (price, sqft, beds, baths, year built): if the filter
        # RESTATES what the query already expresses ("above 500k" == price_min
        # 500000), the line keeps the user's own phrase and nothing is appended —
        # one mention, verbatim. If they DIFFER, the filter wins and its value
        # replaces the typed phrase (single mention). Display only — result
        # filtering is unchanged either way.
        display_filters = dict(filters_dict) if filters_dict else None
        if display_filters:
            parsed_f = {
                k: (v or None)
                for k, v in _extract_filters(result["parsed_query"]).model_dump().items()
            }
            for pair in (("price_min", "price_max"),
                         ("sqft_min", "sqft_max"),
                         ("year_from", "year_to")):
                sent = [display_filters.get(k) for k in pair]
                if all(v is None for v in sent):
                    continue
                if all(v is None or v == parsed_f.get(k) for k, v in zip(pair, sent)):
                    for k in pair:
                        display_filters.pop(k, None)
                else:
                    # Effective pair (filter priority over parsed) so a partial
                    # override of a typed range still renders coherently.
                    for k in pair:
                        if display_filters.get(k) is None and parsed_f.get(k) is not None:
                            display_filters[k] = parsed_f[k]
            for k in ("beds_min", "baths_min"):
                if display_filters.get(k) is not None and display_filters[k] == parsed_f.get(k):
                    display_filters.pop(k, None)
        updated_query = (
            _merge_query_filters(base_query, display_filters)
            if display_filters else base_query
        )

        return SearchResponse(
            query=request.query,
            updated_query=updated_query,
            briefproperties=results,
            pins=result["pins"],
            totalCount=total,
            page=request.page,
            pageSize=request.pageSize,
            totalPages=(total + request.pageSize - 1) // request.pageSize,
            regionName=result.get("region_name"),
            regionId=result.get("region_id"),
            polygons=result.get("polygons"),
            detectedLocation=result.get("location_detected", False),
            filters=_extract_filters(result["parsed_query"]),
            relaxed=result.get("relaxed", []),
            softCriteria=result.get("soft_criteria", []),
            exactAddress=result.get("exact_address", False),
            debug=debug_info,
        )
    except HTTPException:
        raise  # deliberate 4xx (e.g. zero-criteria 422) — don't convert to 500
    except QueryParseError as e:
        # Couldn't parse at all (e.g. LLM outage). Signal it instead of silently
        # returning every property as a "match".
        logger.warning(f"Query parse failed for '{request.query}': {e}")
        raise HTTPException(status_code=503, detail="Could not interpret the query right now. Please try again.")
    except Exception as e:
        logger.exception(f"Search failed for query '{request.query}': {e}")
        raise HTTPException(status_code=500, detail="Search failed due to an internal error.")


@app.get("/health")
async def health():
    pool = await get_pool()
    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT count(*) FROM properties")
    return {"status": "ok", "properties_in_db": count}


class DiskUsage(BaseModel):
    path: str          # filesystem sampled — the host-mounted log dir, so the
    #                    numbers are the VPS disk, not the container's overlay
    total_gb: float
    used_gb: float
    free_gb: float
    used_pct: float


class DiskStatus(BaseModel):
    """Disk health of the host the API runs on — added after the 2026-08 outage
    where runaway logs filled the disk and took Postgres down. Poll this from a
    monitor: 'warning' below WARN_FREE_PCT free, 'critical' below CRIT_FREE_PCT."""
    status: str                    # ok | warning | critical
    disk: DiskUsage
    database_size_gb: float        # the one consumer that grows with the catalog
    thresholds: dict[str, int]
    checked_at: str


DISK_WARN_FREE_PCT = 20
DISK_CRIT_FREE_PCT = 10


@app.get("/health/disk", response_model=DiskStatus)
async def health_disk():
    """Host disk usage + database size. Cheap (one statvfs + one catalog query)."""
    # LOG_DIR is a bind mount from the host (./log:/app/log), so statvfs on it
    # reports the host's real disk; fall back to / (overlay, same numbers on a
    # default single-disk Docker host) if the mount is missing.
    path = str(LOG_DIR) if LOG_DIR.exists() else "/"
    usage = shutil.disk_usage(path)
    gb = 1024 ** 3
    free_pct = usage.free / usage.total * 100 if usage.total else 0.0
    pool = await get_pool()
    async with pool.acquire() as conn:
        db_bytes = await conn.fetchval("SELECT pg_database_size(current_database())")
    status = ("critical" if free_pct < DISK_CRIT_FREE_PCT
              else "warning" if free_pct < DISK_WARN_FREE_PCT else "ok")
    return DiskStatus(
        status=status,
        disk=DiskUsage(
            path=path,
            total_gb=round(usage.total / gb, 2),
            used_gb=round(usage.used / gb, 2),
            free_gb=round(usage.free / gb, 2),
            used_pct=round(100 - free_pct, 1),
        ),
        database_size_gb=round((db_bytes or 0) / gb, 2),
        thresholds={"warning_free_pct": DISK_WARN_FREE_PCT, "critical_free_pct": DISK_CRIT_FREE_PCT},
        checked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


@app.get("/properties/{propertyId}", response_model=BriefProperty)
async def get_brief_property(propertyId: str):
    """The search-result card for ONE listing, by GUID (the `propertyId` search
    returns — the same id the frontend database uses). Identical shape to a
    /search `briefproperties` entry and built by the same code path, so the two
    never drift. 404 when the id is not in the catalog (unknown, parked as
    not-for-sale, or removed). GUID matching is case-insensitive."""
    pool = await get_pool()
    guid = propertyId.strip()
    async with pool.acquire() as conn:
        # Exact match first (uses the unique guid index); case-insensitive fallback
        # for callers that upper-case GUIDs.
        internal_id = await conn.fetchval("SELECT id FROM properties WHERE guid = $1", guid)
        if internal_id is None:
            internal_id = await conn.fetchval(
                "SELECT id FROM properties WHERE lower(guid) = lower($1)", guid
            )
    if internal_id is None:
        raise HTTPException(status_code=404, detail=f"Property '{propertyId}' not found")
    results = await load_brief_properties(pool, [internal_id])
    if not results:
        raise HTTPException(status_code=404, detail=f"Property '{propertyId}' not found")
    brief = results[0]
    brief.pop("_internal_id", None)
    brief.setdefault("matchedSoft", [])
    return brief


@app.get("/property/{guid}")
async def get_property(guid: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        prop_row = await conn.fetchrow("""
            SELECT
                id, guid, name, street, district, city, state, postal_code, country,
                ST_Y(geom::geometry) AS latitude,
                ST_X(geom::geometry) AS longitude,
                area_sqft, price_usd,
                bedroom_count, bathroom_count, kitchen_count,
                living_room_count, dining_room_count, garage_count,
                home_type, rent_estimate, year_built, lot_size_sqft, stories,
                description, created_at, updated_at
            FROM properties WHERE guid = $1
        """, guid)
        if not prop_row:
            raise HTTPException(status_code=404, detail=f"Property '{guid}' not found")

        room_rows = await conn.fetch("""
            SELECT room_type, instance_index, features
            FROM room_instances
            WHERE property_id = $1
              AND room_type != 'Unknown'    -- hide internal claim stubs
            ORDER BY room_type, instance_index
        """, prop_row["id"])

        school_rows = await conn.fetch("""
            SELECT school_name, rating, grades, distance_miles, link
            FROM property_schools
            WHERE property_id = $1
            ORDER BY distance_miles
        """, prop_row["id"])

    rooms_map: dict[str, list[list[str]]] = {}
    all_features: set[str] = set()
    for r in room_rows:
        rooms_map.setdefault(r["room_type"], []).append(list(r["features"]))
        all_features.update(r["features"])

    rooms = [
        {
            "room_type": rt,
            "instance_count": len(instances),
            "instances": [{"features": feats} for feats in instances],
        }
        for rt, instances in rooms_map.items()
    ]

    schools = [
        {
            "name": s["school_name"],
            "rating": s["rating"],
            "grades": s["grades"],
            "distance_miles": float(s["distance_miles"]),
            "link": s["link"],
        }
        for s in school_rows
    ]

    return {
        "property_id": guid,
        "name": prop_row["name"],
        "address": {
            "street": prop_row["street"],
            "district": prop_row["district"],
            "city": prop_row["city"],
            "state": prop_row["state"],
            "postal_code": prop_row["postal_code"],
            "country": prop_row["country"],
        },
        "location": {
            "latitude": prop_row["latitude"],
            "longitude": prop_row["longitude"],
        },
        "area_sqft": prop_row["area_sqft"],
        "price_usd": prop_row["price_usd"],
        "room_counts": {
            "bedroom": prop_row["bedroom_count"],
            "bathroom": prop_row["bathroom_count"],
            "kitchen": prop_row["kitchen_count"],
            "living_room": prop_row["living_room_count"],
            "dining_room": prop_row["dining_room_count"],
            "garage": prop_row["garage_count"],
        },
        "attributes": {
            "home_type": prop_row["home_type"],
            "rent_estimate": prop_row["rent_estimate"],
            "year_built": prop_row["year_built"],
            "lot_size_sqft": prop_row["lot_size_sqft"],
            "stories": prop_row["stories"],
        },
        "description": prop_row["description"],
        "schools": schools,
        "total_features": len(all_features),
        "all_features": sorted(all_features),
        "rooms": rooms,
        "created_at": prop_row["created_at"].isoformat() if prop_row["created_at"] else None,
        "updated_at": prop_row["updated_at"].isoformat() if prop_row["updated_at"] else None,
    }
