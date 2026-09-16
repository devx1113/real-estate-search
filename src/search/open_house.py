"""Open houses: card label + search conditions (frontend requests, 2026-09-14/16).

Storage: property_open_houses — one row per event (UTC start/end, host, livestream),
kept in sync with the raw record by db_ingest.refresh_open_houses.

Card label (computed per request, never stored) in the listing's local US time —
Eastern, except Florida's western panhandle counties on Central time:

  happening now      -> "Open House until 3:00 PM"
  later today        -> "Open House Today, 12:00 PM – 3:00 PM"
  a later day        -> "Open: Sat, 12:00 PM – 3:00 PM (09/20)"
  nothing upcoming   -> None

Only the FIRST event that has not ended is described; ended events are ignored.

Search: OpenHouseCriterion (parsed from "open houses this weekend", "open house
Saturday morning"...) or the request filter `open_house` becomes an EXISTS
condition over the listing's not-yet-ended events; dates, times of day and
weekdays are compared in the listing's local time.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

LISTING_TZ = ZoneInfo("America/New_York")
_CENTRAL_TZ = ZoneInfo("America/Chicago")
# Florida counties entirely on Central time (Gulf County is split; its populated
# south, Port St. Joe, is Eastern).
_CENTRAL_FL_COUNTIES = {
    "escambia", "santa rosa", "okaloosa", "walton", "holmes",
    "washington", "bay", "jackson", "calhoun",
}
DAY_ISO = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7}


def listing_timezone(county: str | None) -> ZoneInfo:
    """Local time zone of a Florida listing from its county name."""
    name = (county or "").strip().lower().removesuffix(" county").strip()
    return _CENTRAL_TZ if name in _CENTRAL_FL_COUNTIES else LISTING_TZ


# ------------------------------------------------------------------ card label
def _parse(ts, tz: ZoneInfo) -> datetime | None:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=tz)
    if not ts or not isinstance(ts, str):
        return None
    try:
        dt = datetime.fromisoformat(ts.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=tz)


def _clock(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0")  # "12:00 PM", "3:00 PM"


def _day(dt: datetime) -> str:
    return f"{dt:%a}, {dt:%b} {dt.day}"  # "Sat, Sep 20"


def open_house_label(events, now: datetime | None = None, tz: ZoneInfo = LISTING_TZ) -> str | None:
    """Display text for the listing's next open house in time zone `tz`, or None.
    events: [{"start": ISO str | datetime, "end": ...}, ...]."""
    now = now or datetime.now(timezone.utc)
    upcoming: list[tuple[datetime, datetime]] = []
    for e in events or []:
        if not isinstance(e, dict):
            continue
        start, end = _parse(e.get("start"), tz), _parse(e.get("end"), tz)
        if start is None or end is None or end <= start or end <= now:
            continue
        upcoming.append((start, end))
    if not upcoming:
        return None
    start, end = min(upcoming)
    ls, le, ln = (d.astimezone(tz) for d in (start, end, now))
    if start <= now:
        if le.date() != ln.date():
            return f"Open House until {_day(le)}, {_clock(le)}"
        return f"Open House until {_clock(le)}"
    if ls.date() == ln.date():
        return f"Open House Today, {_clock(ls)} – {_clock(le)}"
    return f"Open: {ls:%a}, {_clock(ls)} – {_clock(le)} ({ls:%m/%d})"


# ------------------------------------------------------------------ value normalizers
def normalize_date(value) -> str | None:
    """'2026-09-20' (also accepts a datetime/date) -> 'YYYY-MM-DD'; anything else -> None."""
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    try:
        return datetime.strptime(str(value).strip()[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def normalize_time(value) -> str | None:
    """'17:00' / '5:00 PM' / '9' -> 'HH:MM' (24h); anything else -> None."""
    if value is None:
        return None
    s = str(value).strip().upper().replace(".", "")
    for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I %p", "%I:%M%p", "%I%p", "%H"):
        try:
            return datetime.strptime(s, fmt).strftime("%H:%M")
        except ValueError:
            continue
    return None


def normalize_days(values) -> list[str]:
    out: list[str] = []
    for v in values or []:
        code = str(v).strip().lower()[:3]
        if code in DAY_ISO and code not in out:
            out.append(code)
    return out


# ------------------------------------------------------------------ parser context
def local_date_context(now: datetime | None = None) -> str:
    """Today's local date and pre-resolved relative ranges for the query parser, so
    the LLM never does calendar arithmetic ("this weekend" on a Sunday...)."""
    n = (now or datetime.now(timezone.utc)).astimezone(LISTING_TZ)
    today = n.date()
    wd = today.weekday()  # Mon=0 .. Sun=6
    if wd == 6:
        wk_from, wk_to = today, today
    elif wd == 5:
        wk_from, wk_to = today, today + timedelta(days=1)
    else:
        wk_from = today + timedelta(days=5 - wd)
        wk_to = wk_from + timedelta(days=1)
    next_sat = (wk_from if wd not in (5, 6) else today + timedelta(days=(5 - wd) % 7 or 7)) + timedelta(days=7 if wd not in (5, 6) else 0)
    next_week_mon = today + timedelta(days=7 - wd)
    days = ", ".join(
        f"{(today + timedelta(days=i)):%A}={(today + timedelta(days=i)):%Y-%m-%d}" for i in range(7)
    )
    return (
        f"CURRENT LOCAL DATE/TIME (US Eastern): {n:%A} {n:%Y-%m-%d} {n:%H:%M}.\n"
        f"Resolved dates for open_house criteria: today={today}, tomorrow={today + timedelta(days=1)}, "
        f"this weekend={wk_from}..{wk_to}, next weekend={next_sat}..{next_sat + timedelta(days=1)}, "
        f"this week={today}..{today + timedelta(days=6 - wd)}, "
        f"next week={next_week_mon}..{next_week_mon + timedelta(days=6)}, "
        f"this month={today}..{(today.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)}.\n"
        f"The next 7 days (a bare weekday name means the first of these): {days}."
    )


# ------------------------------------------------------------------ search SQL
def _tz_sql(county_col: str) -> str:
    names = ", ".join(f"'{c}'" for c in sorted(_CENTRAL_FL_COUNTIES))
    return (
        rf"(CASE WHEN lower(trim(regexp_replace(coalesce({county_col}, ''), '\s*county\s*$', '', 'i'))) "
        rf"IN ({names}) THEN 'America/Chicago' ELSE 'America/New_York' END)"
    )


def open_house_where(crit, param_idx: int, id_col: str = "properties.id",
                     county_col: str = "properties.county") -> tuple[str, list, int]:
    """WHERE clause (over alias `oh` = property_open_houses) selecting the listing's
    not-yet-ended events that satisfy `crit` (an OpenHouseCriterion). Returns
    (sql, params, next_param_idx)."""
    tz = _tz_sql(county_col)
    conds = [f"oh.property_id = {id_col}", "oh.ends_at > now()"]
    params: list = []

    def add(expr: str, value) -> None:
        nonlocal param_idx
        conds.append(expr.replace("{p}", f"${param_idx}"))
        params.append(value)
        param_idx += 1

    if getattr(crit, "happening_now", False):
        conds.append("oh.starts_at <= now()")
    # asyncpg binds typed parameters: dates/times must be date/time objects.
    d_from, d_to = normalize_date(crit.date_from), normalize_date(crit.date_to)
    if d_from:
        add(f"(oh.starts_at AT TIME ZONE {tz})::date >= {{p}}::date", date.fromisoformat(d_from))
    if d_to:
        add(f"(oh.starts_at AT TIME ZONE {tz})::date <= {{p}}::date", date.fromisoformat(d_to))
    t_from, t_to = normalize_time(crit.time_from), normalize_time(crit.time_to)
    if t_from:  # still running after this local time of day
        add(f"(oh.ends_at AT TIME ZONE {tz})::time > {{p}}::time", time.fromisoformat(t_from))
    if t_to:    # starts before this local time of day
        add(f"(oh.starts_at AT TIME ZONE {tz})::time < {{p}}::time", time.fromisoformat(t_to))
    days = normalize_days(crit.days_of_week)
    if days:
        add(f"extract(isodow FROM (oh.starts_at AT TIME ZONE {tz}))::int = ANY({{p}}::int[])",
            [DAY_ISO[d] for d in days])
    if crit.livestream is not None:
        add("oh.livestream = {p}", bool(crit.livestream))
    words = re.findall(r"[\w'-]+", crit.host or "")
    if words:  # every typed word appears in the host name ("Ashley Langford" ~ "Ashley Aysegul Langford")
        add("oh.host ILIKE ALL({p}::text[])", [f"%{w}%" for w in words])
    return " AND ".join(conds), params, param_idx


def open_house_exists_sql(crit, param_idx: int) -> tuple[str, list, int]:
    """EXISTS condition for apply_hard_filters' `SELECT id FROM properties WHERE ...`."""
    where, params, param_idx = open_house_where(crit, param_idx)
    return f"EXISTS (SELECT 1 FROM property_open_houses oh WHERE {where})", params, param_idx
