"""Open-house label for brief properties (frontend request, 2026-09-14).

The ingest stores each listing's open houses as UTC intervals under the raw
record's `openHouses` key ([{"start": ISO, "end": ISO}, ...]). The label is
computed per request — never stored — so "now", "today" and expiry are always
right. Clock and calendar day use the listing's local US time: Eastern, except
Florida's western panhandle counties, which are on Central time (DST handled by
zoneinfo):

  happening now      -> "Open House until 3:00 PM"
  later today        -> "Open House Today, 12:00 PM – 3:00 PM"
  a later day        -> "Open House: Sat, Sep 20, 12:00 PM – 3:00 PM"
  nothing upcoming   -> None

Only the FIRST event that has not ended is described; ended events are ignored.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

LISTING_TZ = ZoneInfo("America/New_York")
_CENTRAL_TZ = ZoneInfo("America/Chicago")
# Florida counties entirely on Central time (Gulf County is split; its populated
# south, Port St. Joe, is Eastern).
_CENTRAL_FL_COUNTIES = {
    "escambia", "santa rosa", "okaloosa", "walton", "holmes",
    "washington", "bay", "jackson", "calhoun",
}


def listing_timezone(county: str | None) -> ZoneInfo:
    """Local time zone of a Florida listing from its county name."""
    name = (county or "").strip().lower().removesuffix(" county").strip()
    return _CENTRAL_TZ if name in _CENTRAL_FL_COUNTIES else LISTING_TZ


def _parse(ts, tz: ZoneInfo) -> datetime | None:
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
    """Display text for the listing's next open house in time zone `tz`, or None."""
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
    return f"Open House: {_day(ls)}, {_clock(ls)} – {_clock(le)}"
