from enum import Enum

from pydantic import BaseModel


class CriterionType(str, Enum):
    ROOM_COUNT = "room_count"
    FEATURE = "feature"
    PRICE = "price"
    AREA = "area"
    LOCATION = "location"
    PROXIMITY = "proximity"
    PROPERTY = "property"
    COLOR_ROOM = "color_room"
    AREA_RELATION = "area_relation"


class RoomCountCriterion(BaseModel):
    type: CriterionType = CriterionType.ROOM_COUNT
    room_type: str
    min_count: int | None = None
    max_count: int | None = None
    exact_count: int | None = None


class FeatureCriterion(BaseModel):
    type: CriterionType = CriterionType.FEATURE
    feature: str
    room_context: str | None = None  # room type the feature must be in
    negated: bool = False  # True = must NOT have feature
    # "hard" = concrete, checkable requirement -> filters (may be auto-relaxed if it
    # starves the results); "soft" = subjective wish ("great kitchen", "feels
    # private") -> never filters, only ranks matching homes first.
    strength: str = "hard"


class PriceCriterion(BaseModel):
    type: CriterionType = CriterionType.PRICE
    min_price: int | None = None
    max_price: int | None = None


class AreaCriterion(BaseModel):
    type: CriterionType = CriterionType.AREA
    min_sqft: int | None = None
    max_sqft: int | None = None


class LocationCriterion(BaseModel):
    type: CriterionType = CriterionType.LOCATION
    city: str | None = None          # city/town OR any place/community (Melbourne, Rockledge, Viera, Suntree)
    state: str | None = None
    country: str | None = None
    district: str | None = None      # subdivision/plat
    county: str | None = None        # e.g. "Brevard County"
    neighborhood: str | None = None  # e.g. "Viera East"
    locality: str | None = None      # OSM place (e.g. "Viera")
    street: str | None = None        # street/road name (e.g. "Murrell Road")


class ProximityCriterion(BaseModel):
    type: CriterionType = CriterionType.PROXIMITY
    landmark_name: str
    max_distance_miles: float
    landmark_latitude: float | None = None
    landmark_longitude: float | None = None


class PropertyCriterion(BaseModel):
    type: CriterionType = CriterionType.PROPERTY
    home_type: str | None = None  # SINGLE_FAMILY, CONDO, TOWNHOUSE, MANUFACTURED, MULTI_FAMILY, LOT, COMMERCIAL
    listing_type: str | None = None  # "rent" when the user asks for rentals, "sale" when explicit, else None
    min_rent: int | None = None
    max_rent: int | None = None
    min_year_built: int | None = None
    max_year_built: int | None = None
    min_lot_sqft: int | None = None
    max_lot_sqft: int | None = None
    min_stories: int | None = None
    max_stories: int | None = None


class ColorRoomCriterion(BaseModel):
    """Property has a room of `room_type` with dominant color `color`."""
    type: CriterionType = CriterionType.COLOR_ROOM
    color: str                  # white, black, gray, brown, beige, blue, green, red, yellow, purple, pink, orange, gold
    room_type: str              # Kitchen, Bedroom, Bathroom, Living Room, Dining Room, Exterior, Pool, Garage
    negated: bool = False       # True = must NOT have such a room


class AreaRelationCriterion(BaseModel):
    """Spatial relation (near/between) of a property to one or two named areas, resolved to centroids and matched with PostGIS."""
    type: CriterionType = CriterionType.AREA_RELATION
    relation: str                       # "near" | "between"
    place_a: str
    place_b: str | None = None          # required for "between"
    radius_miles: float | None = None   # optional override of the default buffer


Criterion = (
    RoomCountCriterion
    | FeatureCriterion
    | PriceCriterion
    | AreaCriterion
    | LocationCriterion
    | ProximityCriterion
    | PropertyCriterion
    | ColorRoomCriterion
    | AreaRelationCriterion
)


class ParsedQuery(BaseModel):
    original_query: str
    criteria: list[Criterion]
    reconstructed_queries: list[str] = []  # query rebuilt from DB features
    understood_intent: str  # LLM's intent summary
