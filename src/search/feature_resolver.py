"""Feature resolver: turn a raw user phrase into canonical DB feature strings via two-stage retrieve-then-rerank (pgvector top-K recall, then a bundled LLM precision filter)."""

from __future__ import annotations

import asyncio
import json
import logging
import re

from config.settings import settings
from src.llm_client import embed_texts, get_query_client
from src.search.feature_index import retrieve_candidates

logger = logging.getLogger(__name__)

# Per-process cache: lowercased phrase -> curated feature list (cleared on embedding change via clear_cache()).
_cache: dict[str, list[str]] = {}
_CACHE_MAX = 5000  # bound memory; phrases are short, this is plenty
_table_ready = False

# The curation step (LLM #2) is non-deterministic: the same phrase curated in two
# processes yields different alternative lists, so the same query returned 18
# homes on one day and 53 on another. Persisting the FIRST curation makes a
# phrase resolve identically forever (and instantly — no embed + LLM round trip),
# which is what the front-page sample queries need. Cleared when new tags are
# embedded (clear_cache), like the in-memory cache.
_ENSURE_TABLE = """
CREATE TABLE IF NOT EXISTS feature_phrase_cache (
    phrase       TEXT PRIMARY KEY,
    alternatives TEXT[] NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


async def _load_persisted(pool, keys: list[str]) -> dict[str, list[str]]:
    global _table_ready
    try:
        async with pool.acquire() as conn:
            if not _table_ready:
                await conn.execute(_ENSURE_TABLE)
                _table_ready = True
            rows = await conn.fetch(
                "SELECT phrase, alternatives FROM feature_phrase_cache WHERE phrase = ANY($1::text[])", keys
            )
        return {r["phrase"]: list(r["alternatives"]) for r in rows}
    except Exception as e:  # noqa: BLE001 — cache is an optimization, never a failure
        logger.warning("feature_phrase_cache read failed: %s", e)
        return {}


async def _persist(pool, entries: dict[str, list[str]]) -> None:
    if not entries:
        return
    try:
        async with pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO feature_phrase_cache (phrase, alternatives) VALUES ($1, $2) "
                "ON CONFLICT (phrase) DO NOTHING",
                list(entries.items()),
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("feature_phrase_cache write failed: %s", e)


def clear_cache() -> None:
    """Drop the in-memory resolver cache; call when feature_embeddings is re-synced so new features appear in retrievals."""
    _cache.clear()


async def clear_persisted_cache(pool) -> None:
    """Drop the persisted curations too — only when NEW tags were embedded, so a
    phrase can pick them up on its next (single) re-curation."""
    global _table_ready
    try:
        async with pool.acquire() as conn:
            await conn.execute(_ENSURE_TABLE)
            _table_ready = True
            await conn.execute("DELETE FROM feature_phrase_cache")
    except Exception as e:  # noqa: BLE001
        logger.warning("feature_phrase_cache clear failed: %s", e)


_RESOLVER_SYSTEM_PROMPT = """\
You are a real estate feature matcher. You are given several USER PHRASES, each \
with a list of CANDIDATE feature names retrieved from a property database.

For each phrase, select EVERY candidate that a listing could be tagged with that \
would satisfy a user searching for that phrase. Be INCLUSIVE: keep all genuine \
matches, including more-specific variants. Only drop candidates that are clearly \
a different thing, a wrong attribute, or merely topically-related noise. Missing \
a real match (under-matching) is worse than keeping a close one.

Decision rules (apply in order):
1. SUPERSET MATCH — if a candidate contains all the meaningful words of the \
phrase (e.g. phrase "covered pool" vs candidate "covered pool deck", "covered \
pool cage", "covered pool area"), KEEP it. These are specific kinds of the thing.
2. SYNONYM / SUBTYPE — keep candidates that are the same thing or a more specific \
kind, even with different words. e.g. phrase "covered pool" → also keep \
"enclosed pool", "screen-enclosed pool", "screened pool", "pool cage", \
"screened pool enclosure", "lanai pool" (a lanai is a covered structure). \
phrase "swimming pool" → keep "pool", "in-ground pool", "saltwater pool".
3. RESPECT ATTRIBUTES — a covered pool is a pool that is sheltered/roofed/ \
screened/enclosed. A "pool cover" or "pool tarp" is a separate accessory, NOT a \
covered pool — DROP those. DROP plain "pool" / "open-air pool" (no cover).
4. RESPECT COLORS / MATERIALS — phrase "yellow cabinets" → keep "yellow \
cabinets", "yellow kitchen cabinets", "mustard cabinets"; DROP "white cabinets", \
"shaker cabinets" (wrong color).
5. DROP TOPICAL NOISE — phrase "swimming pool" → DROP "pool table", "pool bath", \
"poolside shed".

Use ONLY strings that appear VERBATIM in that phrase's candidate list. Do not \
invent or alter strings. If genuinely none of the candidates fit, return an \
empty list for that phrase.

Return STRICT JSON, an object mapping each phrase to its filtered array:
{"phrase one": ["feat", ...], "phrase two": [...]}
No commentary, no markdown."""


async def _retrieve_for_phrases(pool, phrases: list[str], top_k: int) -> dict[str, list[str]]:
    """Embed each phrase (one batched call) and pgvector-retrieve top_k candidates, returning {phrase: [candidate features]}."""
    vectors = await embed_texts(phrases)
    out: dict[str, list[str]] = {}

    async def _one(phrase: str, vec: list[float]):
        async with pool.acquire() as conn:
            out[phrase] = await retrieve_candidates(conn, vec, top_k)

    await asyncio.gather(*[_one(p, v) for p, v in zip(phrases, vectors)])
    return out


async def _llm_filter(
    candidates_by_phrase: dict[str, list[str]],
) -> tuple[dict[str, list[str]], set[str]]:
    """Bundled LLM #2 call: filter each phrase's candidates to the relevant subset,
    falling back to raw candidates on any failure. Returns (filtered, unreliable) —
    `unreliable` phrases used the raw-candidate fallback and must NOT be cached, or a
    single transient LLM hiccup would poison the cache until the next ingest NOTIFY."""
    # Only send phrases that actually retrieved candidates.
    payload = {p: c for p, c in candidates_by_phrase.items() if c}
    if not payload:
        return {p: [] for p in candidates_by_phrase}, set()

    client = get_query_client()
    user_msg = json.dumps(payload, ensure_ascii=False)
    try:
        resp = await client.chat.completions.create(
            model=settings.openai_model_for_query,
            # Generous ceiling: the output echoes a filtered subset of up to
            # (num_phrases * top_k) feature strings. Billing is per token actually
            # generated, so a high ceiling costs nothing on typical small outputs.
            max_completion_tokens=16384,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _RESOLVER_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        filtered = json.loads(text)
    except Exception as e:
        logger.warning("LLM #2 feature filter failed (%s); using raw candidates", e)
        return dict(candidates_by_phrase), set(candidates_by_phrase)

    # Sanitize: keep only strings genuinely in each phrase's candidate list (guards against invented/altered strings).
    result: dict[str, list[str]] = {}
    unreliable: set[str] = set()
    for phrase, cands in candidates_by_phrase.items():
        cand_set = set(cands)
        picked = filtered.get(phrase)
        if isinstance(picked, list):
            # Respect the model's filtering, even an empty result (orchestrator falls back to exact-match on the raw phrase).
            result[phrase] = [f for f in picked if f in cand_set]
        else:
            # Phrase missing from response → treat as filter failure and fall back to raw candidates (recall).
            result[phrase] = cands
            unreliable.add(phrase)
    return result, unreliable


async def resolve_feature_phrases(
    pool,
    phrases: list[str],
    top_k: int | None = None,
) -> dict[str, list[str]]:
    """Map each distinct user feature phrase to its curated DB-feature list {phrase: [canonical strings]}; cached per phrase, cleared on embedding re-sync."""
    if not phrases:
        return {}
    k = top_k or settings.search_embedding_top_k

    # De-dupe by lowercased phrase, honor cache.
    distinct: dict[str, str] = {}        # lowercased -> original phrase
    for p in phrases:
        key = p.strip().lower()
        if key and key not in distinct:
            distinct[key] = p

    to_resolve = [orig for key, orig in distinct.items() if key not in _cache]

    # Persisted curations first: identical answers across processes and restarts.
    if to_resolve:
        persisted = await _load_persisted(pool, [p.strip().lower() for p in to_resolve])
        for key, alts in persisted.items():
            _cache[key] = alts
        to_resolve = [p for p in to_resolve if p.strip().lower() not in persisted]

    # Fallback results are served for THIS request but never cached (see _llm_filter).
    local: dict[str, list[str]] = {}
    if to_resolve:
        retrieved = await _retrieve_for_phrases(pool, to_resolve, k)
        filtered, unreliable = await _llm_filter(retrieved)
        if len(_cache) > _CACHE_MAX:   # simple bound — drop all and refill
            _cache.clear()
        fresh: dict[str, list[str]] = {}
        for phrase in to_resolve:
            val = filtered.get(phrase, [])
            local[phrase.strip().lower()] = val
            if phrase not in unreliable:
                _cache[phrase.strip().lower()] = val
                fresh[phrase.strip().lower()] = val
        await _persist(pool, fresh)

    # Assemble result keyed by the ORIGINAL phrase strings the caller passed.
    return {
        p: _drop_opposite_polarity(
            p, _cache.get(p.strip().lower(), local.get(p.strip().lower(), []))
        )
        for p in phrases
    }


# ---------------------------------------------------------------------------
# Tags that mention the feature without asserting it. Embedding similarity (and
# the word-subset merge in the orchestrator) return them for the phrase:
#   opposite polarity  "needs kitchen renovation" for "renovated kitchen"
#   provision only     "ceiling fan prep", "washer hookup" — wired for it, not there
#   absence            "no pool", "carpet-free bedroom", "screen-ready porch"
# For a positive search they are false matches; for a negated one ("without fan")
# each wrongly excludes a home that has no fan (2026-09-17: 14 homes on
# production). The guard runs at resolve time and again after the orchestrator's
# merge, so it needs no cache invalidation. Pure absence tags ("ceiling fan not
# visible") are additionally dropped at ingest — db_ingest.drop_absence_tags.
_POLARITY_RE = re.compile(
    r"\b(?:needs?|potential|project|unfinished|outdated|dated|damaged|to be|not|"
    r"absent|omitted|missing|unseen|room for|space for|ready for|wired for|plumbed for)\b"
)
# Trailing marker: the tag describes the PROVISION for X, not X itself.
_PROVISION_RE = re.compile(
    r"\s(?:prep|preparation|pre-?wired?|rough-in|roughed-in|hook-?ups?|ready|stub-?outs?)$"
)
_PROVISION_WORD_RE = re.compile(r"\b(?:prep|preparation|pre-?wired?|rough-in|hook-?ups?|ready|stub-?outs?)\b")
# "no X" / "without X" / "X-free" / "X-ready Y": X is absent. Dropped only when X
# overlaps the phrase, so "no rear neighbors" stays a match for "privacy",
# "panel-ready refrigerator" stays a refrigerator and "step-free shower" a shower.
# The missing thing ends at a location qualifier: "no structures on lot" negates
# "structures", not the lot; "no rear neighbors across lake" negates the neighbors.
_ABSENT_HEAD_RE = re.compile(
    r"\b(?:no|without|lacks|lacking|free of)\s+(.+?)"
    r"(?:\s+(?:on|in|at|across|behind|near|from|to|by|over|under|along|beside|next to)\b.*)?$"
)
_ABSENT_SUFFIX_RE = re.compile(r"([a-z']+)-(?:free|ready)\b")
_ABSENT_MARK_RE = re.compile(r"\b(?:no|without|lacks|lacking|free of)\b|-(?:free|ready)\b")
# "-free" compounds that praise rather than negate.
_FREE_IS_POSITIVE = {"maintenance", "hassle", "worry", "care", "stress", "barrier", "step"}
_STOP = {"a", "an", "the", "of", "with", "and", "or", "in", "on", "to", "for", "at"}


def _stems(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z']+", text.lower()):
        if w in _STOP or len(w) < 2:
            continue
        w = w.rstrip("s")
        for suf in ("ing", "ed"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[: -len(suf)]
                break
        out.add(w)
    return out


def _overlap(a: set[str], b: set[str]) -> bool:
    return any(x == y or (min(len(x), len(y)) >= 4 and (x.startswith(y) or y.startswith(x)))
               for x in a for y in b)


def drop_non_asserting(phrase: str, alts: list[str]) -> list[str]:
    """Remove alternatives that do not assert the phrase's feature (opposite
    polarity, provision-only, absence). Each rule is skipped when the user's own
    phrase carries that wording: "needs renovation", "washer hookup" and
    "no rear neighbors" keep their tags."""
    pl = phrase.strip().lower()
    check_polarity = not _POLARITY_RE.search(pl)
    check_provision = not _PROVISION_WORD_RE.search(pl)
    check_absence = not _ABSENT_MARK_RE.search(pl)
    pstems = _stems(pl)
    out: list[str] = []
    for a in alts:
        al = a.lower()
        if check_polarity and _POLARITY_RE.search(al):
            continue
        if check_provision and _PROVISION_RE.search(al) and not al.startswith(("move-in", "move in", "turnkey")):
            continue
        if check_absence:
            m = _ABSENT_HEAD_RE.search(al)
            if m and _overlap(_stems(m.group(1)), pstems):
                continue
            if any(x not in _FREE_IS_POSITIVE and _overlap(_stems(x), pstems)
                   for x in _ABSENT_SUFFIX_RE.findall(al)):
                continue
        out.append(a)
    return out


# Backwards-compatible name.
_drop_opposite_polarity = drop_non_asserting
