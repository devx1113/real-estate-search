-- ============================================================
-- Real Estate Search — PostgreSQL + PostGIS Schema
-- ============================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;   -- pgvector: feature-embedding similarity search

-- ============================================================
-- TABLES
-- ============================================================

-- Main property table (one row per property)
CREATE TABLE properties (
    id              SERIAL PRIMARY KEY,
    guid            TEXT NOT NULL UNIQUE, -- original UUID from source data
    -- Zillow's stable property id (cross-batch identity). NULL when the exporter
    -- version omitted it; unique where present so a batch replay cannot re-insert
    -- the same listing under a fresh guid.
    zpid            BIGINT,

    -- Address
    name            TEXT NOT NULL,
    street          TEXT,
    district        TEXT,                -- Zillow subdivision/plat name (e.g. "Viera Tract D-4 Phase 3")
    city            TEXT NOT NULL,       -- Zillow mailing city (e.g. "Rockledge")
    state           TEXT NOT NULL,
    postal_code     TEXT,
    country         TEXT NOT NULL,
    county          TEXT,                -- e.g. "Brevard County" (Zillow `county`)
    locality        TEXT,                -- OSM/Photon place name (e.g. "Viera") — catches communities Zillow files under a mailing city
    neighborhood    TEXT,                -- Zillow neighborhood (e.g. "Viera East"/"Viera West")

    -- Location (PostGIS geography point for spatial queries)
    geom            GEOGRAPHY(Point, 4326) NOT NULL,

    -- Numeric fields
    area_sqft       INTEGER NOT NULL,
    price_usd       INTEGER NOT NULL,

    -- Denormalized room counts (for fast filtering without JOINs)
    bedroom_count   INTEGER NOT NULL DEFAULT 0,
    bathroom_count  INTEGER NOT NULL DEFAULT 0,
    kitchen_count   INTEGER NOT NULL DEFAULT 0,
    living_room_count INTEGER NOT NULL DEFAULT 0,
    dining_room_count INTEGER NOT NULL DEFAULT 0,
    garage_count    INTEGER NOT NULL DEFAULT 0,

    -- Property attributes (from Zillow structured data)
    home_type       TEXT,                -- SINGLE_FAMILY, CONDO, TOWNHOUSE, MANUFACTURED, MULTI_FAMILY
    rent_estimate   INTEGER,             -- monthly rent (rentZestimate)
    year_built      INTEGER,
    lot_size_sqft   INTEGER DEFAULT 0,
    stories         INTEGER,
    has_pool        BOOLEAN NOT NULL DEFAULT FALSE,
    -- TRUE = the metadata pool claim (resoFacts.hasPrivatePool) was rejected —
    -- by the community-pool description sweep or manual review. The ingest
    -- guard re-applies the rejection after every metadata refresh, so wrong
    -- listing-agent data cannot resurrect a false pool.
    pool_override   BOOLEAN NOT NULL DEFAULT FALSE,
    -- TRUE = the property has a COVERED pool (water roofed/screened/caged),
    -- derived from images by the ingest pipeline. "Uncovered pool" is not stored
    -- — it is derived at search time as (has a pool) AND NOT has_covered_pool.
    has_covered_pool BOOLEAN NOT NULL DEFAULT FALSE,
    has_waterfront  BOOLEAN NOT NULL DEFAULT FALSE,
    description     TEXT,                -- property description for text search fallback
    financing       TEXT[] NOT NULL DEFAULT '{}',  -- normalized from Zillow listingTerms

    -- Region identity (regions.regionid), assigned at ingest: Zillow's raw region
    -- IDs (cityId/countyId/zipcodeId/parentRegion — the feed is the source of
    -- truth), else polygon containment (smallest-wins), else postal_code text
    -- match (ZIP level only). NULL = unassigned; search falls back to name
    -- matching. Backfill: python -m src.data.backfill_region_ids
    city_region_id         BIGINT,      -- regiontype '0'
    county_region_id       BIGINT,      -- regiontype '3'
    zipcode_region_id      BIGINT,      -- regiontype '2'
    neighborhood_region_id BIGINT,      -- regiontype '1'

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Rooms table (one row per room type per property)
CREATE TABLE rooms (
    id              SERIAL PRIMARY KEY,
    property_id     INTEGER NOT NULL REFERENCES properties(id) ON DELETE CASCADE,
    room_type       TEXT NOT NULL,
    count           INTEGER NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Room instances table (one row per physical room)
CREATE TABLE room_instances (
    id              SERIAL PRIMARY KEY,
    room_id         INTEGER NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    property_id     INTEGER NOT NULL REFERENCES properties(id) ON DELETE CASCADE,
    room_type       TEXT NOT NULL,
    instance_index  INTEGER NOT NULL,
    features        TEXT[] NOT NULL,
    features_text   TEXT NOT NULL,
    color           TEXT,                -- single dominant color from 13-color palette, NULL if not extracted yet
    photo_url       TEXT,                -- canonical (highest-res JPEG) URL of the photo this room was derived from
                                          -- NULL for rows created before partial-update support; treated as "unknown origin"
                                          -- so any new POST /process for that property triggers a full re-analyze.
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- One instance per (property, room_type, index). The worker computes
    -- instance_index carefully; this makes a double-insert a hard error
    -- rather than silent duplicate rooms. (Fresh-DB constraint — dedupe any
    -- legacy duplicates before applying to an existing database.)
    CONSTRAINT uq_room_instance UNIQUE (property_id, room_type, instance_index)
);

-- Raw ingest staging table — receives every incoming property record via
-- POST /process. A background worker continuously picks up rows whose status
-- is not 'processed' and syncs them into the primary tables above.
--
-- The "what changed" diff for the partial-image path is NOT stored here.
-- It's derivable any time from (data.originalPhotos URLs) ⊖ (room_instances.photo_url
-- for this property), so the worker recomputes it at processing time — that
-- makes the worker idempotent under racy /process traffic.
CREATE TABLE raw_properties (
    id           TEXT PRIMARY KEY,
    data         JSONB NOT NULL,
    status       TEXT NOT NULL CHECK (status IN (
                     'unprocessed',
                     'image_only_processed',
                     'partial_image_only_processed',
                     'processed',
                     'batch_submitted',
                     -- Terminal: listing is outside the catalog statuses (FOR_SALE/PENDING), so it is never analyzed or
                     -- ingested (raw_db.prune_non_for_sale). A re-upload via /process
                     -- resets it to 'unprocessed', so this is not a permanent blacklist.
                     'skipped_not_for_sale',
                     -- Terminal: raised MAX_INGEST_ATTEMPTS times (worker._record_failure).
                     -- A re-upload via /process resets it to 'unprocessed'.
                     'failed'
                 )),
    -- Consecutive ingest failures; parked as 'failed' at MAX_INGEST_ATTEMPTS so a
    -- poison row cannot be retried forever (2026-08 disk-full incident).
    ingest_attempts INTEGER NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_raw_properties_status ON raw_properties(status);
-- The catalog carries FOR_SALE and PENDING listings; the prune/claim paths filter on this expression.
CREATE INDEX idx_raw_properties_home_status ON raw_properties ((data->>'homeStatus'));

-- Curated feature alternatives per user phrase (feature_resolver). Created lazily
-- by the app; persisted so a phrase resolves identically across processes and
-- restarts (the LLM curation step is non-deterministic). Cleared whenever new
-- tags are embedded.
CREATE TABLE IF NOT EXISTS feature_phrase_cache (
    phrase       TEXT PRIMARY KEY,
    alternatives TEXT[] NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- OpenAI Batch API vision runs (bulk photo analysis): one row per submitted batch;
-- items = {raw_property_id: {updated_at, urls: [...]}} maps custom_ids back to photos.
CREATE TABLE vision_batches (
    batch_id     TEXT PRIMARY KEY,
    status       TEXT NOT NULL DEFAULT 'submitted',
    items        JSONB NOT NULL,
    est_tokens   BIGINT NOT NULL DEFAULT 0,  -- queue-budget accounting across open batches
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Per-photo vision results accumulated across batches (a property may span several
-- batches); the worker ingests a property once every needed photo has a result.
CREATE TABLE vision_results (
    property_id TEXT NOT NULL,
    photo_url   TEXT NOT NULL,
    result      JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (property_id, photo_url)
);

-- Matching-validation retry ledger: groups that fail index-echo validation retry at
-- smaller sizes (20 -> 5 -> 1); after 3 failures a photo gets an Unknown stub.
CREATE TABLE vision_attempts (
    property_id TEXT NOT NULL,
    photo_url   TEXT NOT NULL,
    attempts    INT NOT NULL DEFAULT 0,
    PRIMARY KEY (property_id, photo_url)
);


-- Nearby schools per property (from Zillow data)
CREATE TABLE property_schools (
    id              SERIAL PRIMARY KEY,
    property_id     INTEGER NOT NULL REFERENCES properties(id) ON DELETE CASCADE,
    school_name     TEXT NOT NULL,
    rating          INTEGER,
    grades          TEXT,
    distance_miles  NUMERIC(5,2) NOT NULL,
    link            TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Feature embeddings — one row per distinct feature string seen in
-- room_instances.features. Populated by the embedding-sync pipeline at app
-- startup and on NOTIFY feature_change. Used by /search Phase 4: the user's
-- raw feature phrase is embedded and cosine-searched here to retrieve the
-- top-K semantically-nearest DB features as candidates (then a second LLM
-- call filters them for precision).
CREATE TABLE feature_embeddings (
    feature      TEXT PRIMARY KEY,
    embedding    vector(1536) NOT NULL,   -- text-embedding-3-small dimension
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
-- HNSW index for fast approximate cosine-similarity search.
CREATE INDEX idx_feature_embeddings_hnsw
    ON feature_embeddings USING hnsw (embedding vector_cosine_ops);

-- ============================================================
-- INDEXES
-- ============================================================

-- Property filters (Phase 2: Hard Filters)
-- Partial unique: zpid is the cross-batch identity, but most legacy records lack it.
CREATE UNIQUE INDEX idx_properties_zpid ON properties(zpid) WHERE zpid IS NOT NULL;

-- Region-ID search: one partial index per level (NULL = unassigned, never queried).
CREATE INDEX idx_properties_city_region ON properties(city_region_id) WHERE city_region_id IS NOT NULL;
CREATE INDEX idx_properties_county_region ON properties(county_region_id) WHERE county_region_id IS NOT NULL;
CREATE INDEX idx_properties_zipcode_region ON properties(zipcode_region_id) WHERE zipcode_region_id IS NOT NULL;
CREATE INDEX idx_properties_neighborhood_region ON properties(neighborhood_region_id) WHERE neighborhood_region_id IS NOT NULL;
CREATE INDEX idx_properties_bedroom_count ON properties(bedroom_count);
CREATE INDEX idx_properties_bathroom_count ON properties(bathroom_count);
CREATE INDEX idx_properties_price ON properties(price_usd);
CREATE INDEX idx_properties_area ON properties(area_sqft);
CREATE INDEX idx_properties_city ON properties(city);
CREATE INDEX idx_properties_state ON properties(state);
CREATE INDEX idx_properties_country ON properties(country);
CREATE INDEX idx_properties_district ON properties(district);

-- Property attribute filters
CREATE INDEX idx_properties_home_type ON properties(home_type);
CREATE INDEX idx_properties_rent ON properties(rent_estimate);
CREATE INDEX idx_properties_year_built ON properties(year_built);
CREATE INDEX idx_properties_lot_size ON properties(lot_size_sqft);
CREATE INDEX idx_properties_has_pool ON properties(has_pool) WHERE has_pool = TRUE;
CREATE INDEX idx_properties_has_covered_pool ON properties(has_covered_pool) WHERE has_covered_pool;
CREATE INDEX idx_properties_has_waterfront ON properties(has_waterfront) WHERE has_waterfront = TRUE;
CREATE INDEX idx_properties_description ON properties USING GIN(description gin_trgm_ops);
-- Trigram indexes for fuzzy (ILIKE '%x%') location search.
CREATE INDEX idx_properties_county_trgm       ON properties USING GIN(county gin_trgm_ops);
CREATE INDEX idx_properties_locality_trgm     ON properties USING GIN(locality gin_trgm_ops);
CREATE INDEX idx_properties_neighborhood_trgm ON properties USING GIN(neighborhood gin_trgm_ops);
CREATE INDEX idx_properties_street_trgm        ON properties USING GIN(street gin_trgm_ops);
CREATE INDEX idx_properties_city_trgm          ON properties USING GIN(city gin_trgm_ops);
CREATE INDEX idx_properties_district_trgm      ON properties USING GIN(district gin_trgm_ops);
CREATE INDEX idx_properties_financing_gin ON properties USING GIN(financing);

-- Spatial index (Phase 3: Proximity)
CREATE INDEX idx_properties_geom ON properties USING GIST(geom);

-- Room instance lookups (Phase 4: Feature Matching)
CREATE INDEX idx_room_instances_property_id ON room_instances(property_id);
CREATE INDEX idx_room_instances_room_type ON room_instances(room_type);
CREATE INDEX idx_room_instances_prop_room ON room_instances(property_id, room_type);
CREATE INDEX idx_room_instances_features_gin ON room_instances USING GIN(features);
CREATE INDEX idx_room_instances_features_text_trgm ON room_instances USING GIN(features_text gin_trgm_ops);
CREATE INDEX idx_room_instances_color ON room_instances(color);
CREATE INDEX idx_room_instances_room_color ON room_instances(room_type, color);

-- School lookups
CREATE INDEX idx_property_schools_property_id ON property_schools(property_id);
CREATE INDEX idx_property_schools_name ON property_schools USING GIN(school_name gin_trgm_ops);
CREATE INDEX idx_property_schools_distance ON property_schools(distance_miles);

-- Points of interest (grocery, church, gas station, etc.) imported from
-- OpenStreetMap per county; proximity uses ST_DWithin on geom (see import_pois.py).
-- poi_coverage tracks which counties are imported so new counties auto-import once.
CREATE TABLE IF NOT EXISTS pois (
    id SERIAL PRIMARY KEY,
    county TEXT,
    category TEXT NOT NULL,
    name TEXT,
    geom GEOGRAPHY(Point, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pois_geom ON pois USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_pois_category ON pois(category);
CREATE INDEX IF NOT EXISTS idx_pois_county ON pois(county);

CREATE TABLE IF NOT EXISTS poi_coverage (
    county TEXT PRIMARY KEY,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Region catalog (city/neighborhood/ZIP/county/state), loaded from
-- src/data/region/regions_data.sql via `python -m src.data.import_regions`.
-- /search resolves the extracted area name to a row here so the client gets a
-- regionid that is unique even when the name isn't ("Downtown" exists 137 times).
-- Column names match the INSERT statements in the data file (lowercased).
CREATE TABLE IF NOT EXISTS regions (
    id          UUID PRIMARY KEY,     -- source GUID
    regionid    INTEGER NOT NULL UNIQUE,  -- the ID clients consume
    regiontype  TEXT NOT NULL,        -- '0' city, '1' neighborhood, '2' ZIP, '3' county, '4' state
    regionname  TEXT NOT NULL,        -- place name (for type '2': the ZIP code itself)
    statecode   TEXT NOT NULL,        -- 2-letter state
    city        TEXT NOT NULL,        -- parent city for types '1'/'2'; = statecode for '0'/'3'; '' for '4'
    -- Region boundary for geo-location search: when present, "homes in <name>"
    -- filters by ST_Covers(geom, property.geom) instead of place-name matching.
    -- Loaded from src/data/region/script.sql (RegionBoundaries JSON) by the import.
    geom        GEOGRAPHY(MultiPolygon, 4326)
);
CREATE INDEX IF NOT EXISTS idx_regions_type_name ON regions (regiontype, lower(regionname));
CREATE INDEX IF NOT EXISTS idx_regions_statecode ON regions (statecode);
CREATE INDEX IF NOT EXISTS idx_regions_geom ON regions USING GIST(geom);
