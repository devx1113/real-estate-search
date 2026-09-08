"""MLS (RESO / Spark API) -> internal record adapter.

The pipeline consumes one record shape (the historical Zillow layout: address
block, originalPhotos.mixedSources.jpeg, homeStatus, resoFacts...). The MLS feed
carries the same information under RESO names (StandardStatus, Photos[].Uri*,
ListPrice...). This module detects an MLS record at /process time and rewrites it
into the internal shape, so EVERYTHING downstream — vision, region assignment,
FOR_SALE pruning, dedup adoption, every search type, response photo groups —
runs unchanged.

Mapping decisions (documented in the 2026-09 field audit):
- identity: the wrapper's id (the exporter-generated GUID shared with the
  frontend database) -> raw-row id, so search responses (propertyId = guid)
  match the frontend DB; a missing/zero-GUID wrapper id falls back to
  SparkId/ListingKey. ListingId -> the zpid slot, so re-uploads of the same
  listing are adopted as updates instead of duplicating.
- status: StandardStatus -> homeStatus (Active=FOR_SALE; everything else maps to
  a non-FOR_SALE value and is pruned/parked by the existing catalog rule).
  Listing CLASS (StandardFieldsJson.PropertyClass) then narrows Active further:
  rentals -> FOR_RENT, land/commercial -> OTHER, so only residential homes for
  sale ever enter the catalog (a $1,800 rental or an empty lot is not a home).
- address: number, direction, name, suffix, direction, "APT unit" — each part
  taken from the flat record, falling back to StandardFieldsJson when the
  exporter nulls it (directions, condo UnitNumber). Duplicated suffix/direction
  words and unit markers ("#", "Unit") are normalized so the stored form is
  what the exact-address matcher expects.
- photos: Photos[] sorted by DisplayOrder -> originalPhotos with a width ladder;
  the highest-width URL is the canonical id room_instances keys on, so it must
  be stable across re-uploads (plain URL passthrough, no rewriting).
- schools: MLS carries NAMES only. We deliberately emit an empty `schools` list
  (no property_schools rows with fake distances) and stash the names under
  `mlsSchools` for the later ratings backfill — school queries return empty,
  never wrong, until that lands.
- the original MLS payload is preserved under `mls` (minus the bulky media
  arrays, which live on in transformed form) so nothing is lost for future use.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# StandardFieldsJson.PropertyClass (Spark) -> listing class. Only residential
# FOR-SALE listings belong in the catalog: rentals carry a MONTHLY price and
# land/commercial have no rooms — all three would surface as absurd "homes".
_CLASS_BY_LABEL = {
    "residential": "residential",
    "rental": "rental",
    "residential lease": "rental",
    "land": "land",
    "commercial": "commercial",
    "commercial lease": "commercial",
    "commercial sale": "commercial",
    "business opportunity": "commercial",
}
# Spark PropertyType letter codes seen in the feed, fallback when the
# StandardFieldsJson labels are absent.
_CLASS_BY_CODE = {"A": "residential", "B": "residential", "C": "land",
                  "E": "commercial", "F": "rental"}

_STATUS_MAP = {
    "active": "FOR_SALE",
    "active under contract": "PENDING",
    "pending": "PENDING",
    "closed": "SOLD",
    "withdrawn": "OTHER",
    "expired": "OTHER",
    "canceled": "OTHER",
    "cancelled": "OTHER",
    "coming soon": "OTHER",
    "hold": "OTHER",
}

# PropertySubType (RESO) -> internal home_type. Substring match, first hit wins.
_TYPE_MAP = [
    ("single family", "SINGLE_FAMILY"),
    ("condo", "CONDO"),
    ("townhouse", "TOWNHOUSE"),
    ("townhome", "TOWNHOUSE"),
    ("manufactured", "MANUFACTURED"),
    ("mobile", "MANUFACTURED"),
    ("duplex", "MULTI_FAMILY"),
    ("triplex", "MULTI_FAMILY"),
    ("quadruplex", "MULTI_FAMILY"),
    ("multi family", "MULTI_FAMILY"),
    ("multi-family", "MULTI_FAMILY"),
    ("apartment", "CONDO"),
    ("villa", "TOWNHOUSE"),
]

# (uri key, width for the jpeg ladder). UriLarge is the original upload — widest.
_PHOTO_URIS = [
    ("Uri300", 300),
    ("Uri800", 800),
    ("Uri1024", 1024),
    ("Uri1600", 1600),
    ("Uri2048", 2048),
    ("UriLarge", 2560),
]

# Media arrays are transformed (photos) or irrelevant; keeping the raw copies too
# would double every record's footprint for no reader.
_MLS_KEEP_SKIP = {"Photos", "FloorPlans", "Documents", "Videos", "VirtualTours",
                  "OpenHouses", "DomainEvents"}


def is_mls_record(data: dict) -> bool:
    """An MLS/RESO record carries ListingKey/SparkId + StandardStatus; the
    internal shape never does."""
    return (
        isinstance(data, dict)
        and ("ListingKey" in data or "SparkId" in data)
        and "StandardStatus" in data
        and "homeStatus" not in data
    )


def _listing_class(data: dict, sf: dict) -> str | None:
    for label in (sf.get("PropertyClass"), sf.get("PropertyTypeLabel")):
        cls = _CLASS_BY_LABEL.get(str(label or "").strip().lower())
        if cls:
            return cls
    return _CLASS_BY_CODE.get(str(data.get("PropertyType") or "").strip().upper())


def _home_type(sub_type: str | None, type_label: str | None) -> str | None:
    s = (sub_type or "").strip().lower()
    for needle, mapped in _TYPE_MAP:
        if needle in s:
            return mapped
    return None


def _street(d: dict, sf: dict | None = None) -> str:
    # The exporter nulls some flat address parts (StreetDirPrefix/Suffix, and
    # UnitNumber on condos) that the embedded StandardFieldsJson still has.
    # Without them "1675 S Fiske Blvd" loses its S and the units of one
    # building collapse to a single street address — exact-address search
    # then returns 2 matches instead of 1. Flat value wins when present.
    emb = sf or {}

    def g(key: str) -> str:
        v = d.get(key)
        if v is None or not str(v).strip():
            v = emb.get(key)
        if isinstance(v, float) and v.is_integer():
            v = int(v)  # a JSON 4875.0 house number is "4875"
        return " ".join(str(v).split()) if v is not None else ""

    name, suffix = g("StreetName"), g("StreetSuffix")
    # Feed quirk: StreetName sometimes already ends with the suffix word
    # ("Long Iron Drive" + "Drive", or "San Filippo Drive SE" + "Drive") —
    # nobody types "Drive Drive".
    if suffix and suffix.lower() in name.lower().split()[-2:]:
        suffix = ""
    # Same guard for directions the name already embeds ("N Harbor City" + "N").
    pre, post = g("StreetDirPrefix"), g("StreetDirSuffix")
    if pre and name.lower().split()[:1] == [pre.lower()]:
        pre = ""
    if post and name.lower().split()[-1:] == [post.lower()]:
        post = ""
    # Stored convention is "905 N Harbor City Blvd": number, then direction.
    parts = [g("StreetNumber"), pre, name, suffix, post]
    street = " ".join(p for p in parts if p)
    if not street:
        street = " ".join(g("UnparsedAddress").split(",")[0].split())
    # Bare unit value: "#2101" / "Unit 2101" / "Apt 2101" -> "2101", so the
    # stored "APT 2101" matches what the exact-address matcher compares.
    unit = re.sub(r"^(?:#|\b(?:apt|apartment|unit|ste|suite)\b[\s.#-]*)+", "",
                  g("UnitNumber"), flags=re.IGNORECASE).strip()
    # Only the UnparsedAddress fallback can already contain the unit. Whole-token
    # check past the house number: unit "A" is a substring of "Lane", "2" of
    # "2100", and "5 Elm St" unit 5 must still get its APT 5.
    present = [t.lstrip("#") for t in street.lower().split()[1:]]  # "#4" counts as 4
    if street and unit and unit.lower() not in present:
        street = f"{street} APT {unit}"
    return street


def _photos(d: dict) -> list[dict]:
    out = []
    photos = sorted(
        (p for p in (d.get("Photos") or []) if isinstance(p, dict) and p.get("IsActive", True)),
        key=lambda p: (p.get("DisplayOrder") is None, p.get("DisplayOrder", 0)),
    )
    for p in photos:
        seen: set[str] = set()
        jpeg = []
        for key, width in _PHOTO_URIS:
            url = p.get(key)
            if url and url not in seen:
                seen.add(url)
                jpeg.append({"url": url, "width": width})
        if jpeg:
            out.append({"caption": p.get("Caption") or "", "mixedSources": {"jpeg": jpeg}})
    return out


def _days_on_market(d: dict, home_status: str) -> int | None:
    for key in ("DaysOnMarket", "CumulativeDaysOnMarket"):
        v = d.get(key)
        if isinstance(v, (int, float)) and v >= 0:
            return int(v)
    if home_status != "FOR_SALE":
        return None
    start = d.get("OnMarketTimestamp") or d.get("ListingContractDate")
    if not start:
        return None
    try:
        s = str(start)[:10]
        begin = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return max((datetime.now(timezone.utc) - begin).days, 0)
    except ValueError:
        return None


def _standard_fields(d: dict) -> dict:
    try:
        sf = json.loads(d.get("StandardFieldsJson") or "{}")
        return sf if isinstance(sf, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _listing_terms(data: dict, sf: dict) -> str | None:
    """Comma-joined financing terms ('Cash,Conventional,VA Loan'). The flat
    ListingTerms field is null in the Spark feed; the real data is the
    StandardFieldsJson.ListingTerms {term: bool} map."""
    flat = data.get("ListingTerms")
    if isinstance(flat, str) and flat.strip():
        return flat
    lt = sf.get("ListingTerms")
    if isinstance(lt, dict):
        return ",".join(k for k, v in lt.items() if v) or None
    if isinstance(lt, list):
        return ",".join(str(x) for x in lt) or None
    return None


def transform_mls(data: dict, fallback_id: str = "") -> tuple[str, dict]:
    """MLS record -> (raw-row id, internal-shaped record)."""
    sf = _standard_fields(data)
    status = _STATUS_MAP.get(str(data.get("StandardStatus") or "").strip().lower(), "OTHER")
    listing_class = _listing_class(data, sf)
    if status == "FOR_SALE" and listing_class == "rental":
        status = "FOR_RENT"
    elif status == "FOR_SALE" and listing_class in ("land", "commercial"):
        status = "OTHER"
    home_type = _home_type(data.get("PropertySubType"), sf.get("PropertyTypeLabel"))
    if status == "FOR_SALE" and home_type is None:
        logger.warning(
            "MLS %s: unmapped PropertySubType %r (class=%s) -> home_type NULL",
            data.get("ListingId"), data.get("PropertySubType"), listing_class,
        )
    county = str(data.get("CountyOrParish") or "").strip()
    if county and not county.lower().endswith("county"):
        county = f"{county} County"

    # Identity: the wrapper's id is the FRONTEND's GUID (generated by the MLS
    # exporter and shared by both databases) — it wins so search responses
    # (propertyId = guid) match the frontend DB. A missing or zero-GUID wrapper
    # id falls back to the record's own SparkId/ListingKey.
    wrapper_id = (fallback_id or "").strip()
    if wrapper_id and not set(wrapper_id) <= set("0-"):
        item_id = wrapper_id
    else:
        item_id = str(data.get("SparkId") or data.get("ListingKey") or "").strip() or wrapper_id
    baths = data.get("BathroomsTotalInteger")
    if baths is None and (data.get("BathsFull") is not None or data.get("BathsHalf") is not None):
        baths = (data.get("BathsFull") or 0) + 0.5 * (data.get("BathsHalf") or 0)

    record: dict = {
        "zpid": data.get("ListingId"),
        "homeStatus": status,
        "address": {
            "streetAddress": _street(data, sf),
            "city": data.get("City") or "",
            "state": data.get("StateOrProvince") or "",
            "zipcode": str(data.get("PostalCode") or ""),
            "subdivision": data.get("SubdivisionName") or "",
        },
        "latitude": data.get("Latitude"),
        "longitude": data.get("Longitude"),
        "price": data.get("ListPrice"),
        "bedrooms": data.get("BedroomsTotal"),
        "bathrooms": baths,
        "livingArea": data.get("LivingArea"),
        "homeType": home_type,
        "listingClass": listing_class,
        "yearBuilt": data.get("YearBuilt"),
        "description": data.get("PublicRemarks"),
        "county": county or None,
        "currency": "USD",
        "daysOnZillow": _days_on_market(data, status),
        "resoFacts": {
            "stories": data.get("StoriesTotal") or sf.get("Stories"),
            "hasPrivatePool": bool(data.get("PoolYN")),
            "hasWaterfrontView": bool(data.get("WaterFrontYN")),
            "listingTerms": _listing_terms(data, sf),
            "garageParkingCapacity": data.get("GarageSpaces"),
            "hasGarage": bool(sf.get("GarageYN")) or bool(data.get("GarageSpaces")),
            "hasAttachedGarage": bool(sf.get("AttachedGarageYN")),
        },
        # Names only until the school-ratings reference lands: an empty list here
        # means school queries return EMPTY (never rows with fake distances).
        "schools": [],
        "mlsSchools": {
            "elementary": data.get("ElementarySchool"),
            "middle": data.get("MiddleOrJuniorSchool"),
            "high": data.get("HighSchool"),
        },
        "originalPhotos": _photos(data),
        # Everything the mapping does not consume, preserved for future features.
        "mls": {k: v for k, v in data.items() if k not in _MLS_KEEP_SKIP},
    }
    if data.get("LotSizeSquareFeet") is not None:
        record["lotSize"] = data.get("LotSizeSquareFeet")
    elif data.get("LotSizeAcres") is not None:
        record["lotAreaValue"] = data.get("LotSizeAcres")
        record["lotAreaUnits"] = "Acres"
    return item_id, record
