from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI (api.openai.com) — single key for vision, geocoding, query parsing, feature filtering, and embeddings.
    openai_api_key: str = ""
    openai_model: str = "gpt-5.1"                # vision + geocoding
    openai_model_for_query: str = "gpt-5.1"      # /search query parsing + feature filter
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dim: int = 1536
    # Secondary model for the /test/compare-vision A/B endpoint (same key).
    openai_test_model: str = ""

    # /search feature retrieval: True → embedding top-K + GPT filter; False → legacy dump-all-features; flag enables instant rollback.
    search_use_embedding_retrieval: bool = True
    search_embedding_top_k: int = 200

    # OpenAI Batch API for photo analysis (50% cost, async ≤24h turnaround).
    # True = BATCH-EXCLUSIVE: every photo is analyzed via batches; the sync path only
    # handles metadata-only updates. False = classic sync vision everywhere.
    vision_use_batch: bool = False
    vision_batch_max_items: int = 150    # max pending properties scanned per cycle
    vision_batch_poll_seconds: int = 60  # min seconds between batch status polls
    # TOTAL estimated INPUT-token budget across ALL in-flight batches — size this to
    # your org's Batch Queue Limit (platform.openai.com → Settings → Limits) with
    # ~10% headroom. Plan minimum: 90k. When the plan grows, bump the env var — the
    # uploader automatically keeps the queue full with as many batches as fit.
    vision_batch_queue_tokens: int = 90_000
    # Size of ONE batch (estimated input tokens). At the plan minimum this fills
    # ~the whole queue (serial waves); with a bigger queue, several run concurrently
    # and a new one uploads whenever one completes.
    vision_batch_max_tokens: int = 80_000
    # Max images grouped into ONE request — the shared prompt is sent once per group
    # (the token saving). Groups failing match-validation retry at 5 images, then 1.
    vision_group_max_images: int = 20

    # Photo size served in search-result cards (briefproperties / GET /properties):
    # the SMALLEST rendition at least this wide (MLS ladder 300/640/800/1024/1280/
    # 1600/2048/original; Zillow 384..1536). 640 = the frontend's card size
    # (Mostafa, 2026-09-11); the detail endpoint (/search/photos) always serves
    # the full-size original.
    search_card_photo_min_width: int = 640

    # Region-ID search: True -> the searched place filters by the properties'
    # stored *_region_id columns (assigned at ingest; backfill:
    # python -m src.data.backfill_region_ids). False -> legacy polygon
    # membership (ST_Covers per query). Flag enables instant rollback, and
    # guards deployments whose DB lacks the region-id columns.
    search_use_region_ids: bool = False

    # Search area for location-less queries: browser coordinates -> containing
    # region polygon; otherwise this default.
    default_search_county: str = "Brevard County"
    default_search_state: str = "FL"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "admin"
    postgres_password: str = "admin123"
    postgres_db: str = "real_estate"

    log_level: str = "INFO"
    log_dir: str = "/app/log"  # request/response log location; override for local dev

    # env_ignore_empty: a blank line in .env (e.g. `VISION_USE_BATCH=` copied from
    # .env.example) means "use the default" instead of crashing bool/int parsing.
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "env_ignore_empty": True}


settings = Settings()
