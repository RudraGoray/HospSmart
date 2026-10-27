from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class Hospital(Strict):
    id: str
    name: str
    city: str | None
    hospital_type: str | None
    beds: float | None
    specialties: list[str]
    rating: float | None
    nabh: str

class Profile(Strict):
    hospital_type: str | None = Field(None, max_length=100)
    beds: int | None = Field(None, ge=1, le=100000)
    specialties: list[str] = Field(default_factory=list, max_length=50)

class RatingProfile(Profile):
    nabh: Literal['Accredited', 'Not accredited', 'Unknown'] = 'Unknown'

class RecommendationRequest(Strict):
    city: str | None = Field(None, max_length=100)
    specialties: list[str] = Field(default_factory=list, max_length=50)
    hospital_type: str | None = Field(None, max_length=100)
    min_beds: float | None = Field(None, ge=1, le=100000)
    min_rating: float | None = Field(None, ge=0)
    nabh: Literal['preferred', 'required', 'not_required', 'no_preference'] = 'no_preference'
    rating_weight: float = Field(.5, ge=0, le=1)

class Health(Strict):
    status: str
    dataset: str
    models: dict[str, str]

class Overview(Strict):
    dataset: dict[str, Any]
    total: int
    cities: int
    specialties: int
    distributions: dict[str, list[dict[str, Any]]]
    top_rated: list[Hospital]

class Filters(Strict):
    cities: list[str]
    specialties: list[str]
    hospital_types: list[str]
    nabh: list[str]
    available_columns: list[str]
    rating_range: list[float] | None

class Directory(Strict):
    items: list[Hospital]
    total: int
    page: int
    page_size: int

class Comparison(Strict):
    hospitals: list[Hospital]

class Match(Strict):
    hospital: Hospital
    match_score: float
    components: dict[str, float]
    weights: dict[str, float]
    reasons: list[str]
    ranking_explanation: str

class Recommendations(Strict):
    results: list[Match]
    message: str

class Prediction(Strict):
    prediction: float | str
    probabilities: dict[str, float] | None = None
    observed_range: list[float] | None = None
    explicit_denominator: float | None = None
    provided_features: list[str]
    unavailable_features: list[str]
    warnings: list[str]

class Metrics(Strict):
    status: str
    rating: dict[str, Any]
    nabh: dict[str, Any]
    recommendation: dict[str, Any]
    run: dict[str, Any] | None = None

class Methodology(Strict):
    dataset: dict[str, Any]
    training: dict[str, Any]
    preprocessing: list[str]
    limitations: list[str]
