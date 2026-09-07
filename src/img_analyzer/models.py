"""Pydantic models for the image-analyzer ingest flow."""

from pydantic import BaseModel


class PhotoSource(BaseModel):
    # Defaults so a malformed source entry can't fail validation and wedge the worker in a retry loop.
    url: str = ""
    width: int = 0


class MixedSources(BaseModel):
    jpeg: list[PhotoSource] = []
    webp: list[PhotoSource] = []


class Photo(BaseModel):
    caption: str = ""
    # Default empty so photos missing this key don't fail validation (would wedge worker in retry loop)
    mixedSources: MixedSources = MixedSources()


class ZillowPropertyRecord(BaseModel):
    model_config = {"extra": "allow"}

    originalPhotos: list[Photo] = []


class PropertyItem(BaseModel):
    model_config = {"extra": "allow"}

    Id: str
    ZillowPropertyId: int = 0
    ZillowPropertyRecord: ZillowPropertyRecord


class PhotoResult(BaseModel):
    photo_url: str
    room_type: str
    color: str | None = None  # one of 13 palette colors, or None
    features: list[str]


class PropertyInput(BaseModel):
    """POST /process item.

    - `id`: the exporter-generated GUID for this listing — the SAME GUID the
      frontend database uses, byte-identical on every re-upload of the listing
      (updates, status changes). Search responses return it as `propertyId`.
    - `data`: the raw MLS/RESO record exactly as exported (ListingKey,
      StandardStatus, ListPrice, Photos, ...). MLS records are detected and
      mapped automatically. Re-uploads of the same GUID are treated as updates:
      only new photos are analyzed, and a non-Active StandardStatus removes the
      listing from search. (The legacy internal/Zillow record shape is also
      still accepted.)
    """
    id: str
    data: dict

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                "data": {
                    "ListingKey": "20231009143746930668000000",
                    "SparkId": "20231009143746930668000000",
                    "ListingId": "973477",
                    "StandardStatus": "Active",
                    "ListPrice": 255000,
                    "StreetNumber": "528",
                    "StreetName": "Clearview",
                    "StreetSuffix": "Drive",
                    "City": "Cocoa",
                    "StateOrProvince": "FL",
                    "PostalCode": "32927",
                    "CountyOrParish": "Brevard",
                    "SubdivisionName": "Sample Subdivision",
                    "Latitude": 28.4,
                    "Longitude": -80.8,
                    "BedroomsTotal": 3,
                    "BathroomsTotalInteger": 2,
                    "LivingArea": 1428,
                    "PropertySubType": "Manufactured Home",
                    "YearBuilt": 1985,
                    "PublicRemarks": "...",
                    "PoolYN": False,
                    "WaterFrontYN": False,
                    "GarageSpaces": 2,
                    "ElementarySchool": "Example Elementary",
                    "MiddleOrJuniorSchool": "Example Middle",
                    "HighSchool": "Example High",
                    "StandardFieldsJson": "{\"ListingTerms\": {\"Cash\": true, \"Conventional\": true, \"FHA\": true}}",
                    "Photos": [
                        {
                            "DisplayOrder": 0,
                            "Caption": "",
                            "Uri300": "https://.../300.jpg",
                            "Uri800": "https://.../800.jpg",
                            "Uri1024": "https://.../1024.jpg",
                            "Uri1600": "https://.../1600.jpg",
                            "Uri2048": "https://.../2048.jpg",
                            "UriLarge": "https://.../large.jpg",
                        }
                    ],
                },
            }
        }
    }
