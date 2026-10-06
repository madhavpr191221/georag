"""Typed contract for the first supported natural-language EO queries."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Intent = Literal[
    "flood", "no_flood", "visual_similarity",
    "spectral_water_signal", "spectral_water_increase", "spectral_water_decrease",
    "spectral_scene_similarity", "spectral_change_similarity", "unsupported",
]
PlaceLevel = Literal["country", "admin1"]
RequestedModality = Literal["rgb", "s2_12band", "unspecified"]


class EOQuery(BaseModel):
    """Inspectable parse result; this structure never ranks imagery by itself."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    start_date: date | None = None
    end_date: date | None = None
    place_name: str | None = Field(default=None, max_length=120)
    place_level: PlaceLevel | None = None
    place_country: str | None = Field(default=None, max_length=120)
    requested_modality: RequestedModality = "unspecified"
    example_tile_id: str | None = Field(default=None, max_length=200)
    unsupported_conditions: list[str] = Field(default_factory=list, max_length=8)
    interpretation: str = Field(max_length=400)
    original_query: str = Field(default="", max_length=1000)
    resolved_place_id: str | None = None

    @model_validator(mode="after")
    def validate_dates_and_place(self) -> "EOQuery":
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        if (self.place_name is None) != (self.place_level is None):
            raise ValueError("place_name and place_level must be supplied together")
        if self.place_country and self.place_level != "admin1":
            raise ValueError("place_country is a disambiguator for admin1 places")
        if self.resolved_place_id and not self.place_name:
            raise ValueError("resolved_place_id requires a place_name")
        return self


class ParseRequest(BaseModel):
    query: str = Field(min_length=4, max_length=1000)


class PlaceChoice(BaseModel):
    place_id: str
    name: str
    level: PlaceLevel
    country: str | None = None


class RetrievalRequest(BaseModel):
    query: EOQuery
    run_id: str = Field(min_length=1, max_length=160)
    k: int = Field(default=10, ge=1, le=30)


class SpectralRetrievalRequest(BaseModel):
    query: EOQuery
    k: int = Field(default=10, ge=1, le=30)
    before_example_tile_id: str | None = Field(default=None, max_length=200)
    after_example_tile_id: str | None = Field(default=None, max_length=200)


class EvidenceResult(BaseModel):
    rank: int
    tile_id: str
    score: float
    preview_url: str
    timestamp: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    sequence_id: str | None = None
    sensor: str | None = None
    bands: list[str]
    labels: list[str]
    bbox: list[float] | None = None


class RetrievalResponse(BaseModel):
    query: EOQuery
    run_id: str
    retrieval_method: str
    query_vector_method: str
    similarity_measure: str = "cosine similarity"
    candidate_count: int
    results: list[EvidenceResult]
    notice: str | None = None


class SpectralEvidenceResult(BaseModel):
    rank: int
    score: float
    score_description: str
    tile_id: str | None = None
    before_tile_id: str | None = None
    after_tile_id: str | None = None
    before_date: str | None = None
    after_date: str | None = None
    gap_days: int | None = None
    sequence_id: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    label: str | None = None
    preview_url: str | None = None
    before_preview_url: str | None = None
    after_preview_url: str | None = None
    index_url: str | None = None
    before_index_url: str | None = None
    after_index_url: str | None = None
    change_map_url: str | None = None
    index_summary: dict[str, dict[str, float]]
    change_summary: dict[str, dict[str, float]] | None = None


class SpectralRetrievalResponse(BaseModel):
    query: EOQuery
    retrieval_method: str
    similarity_measure: str
    candidate_count: int
    results: list[SpectralEvidenceResult]
    notice: str | None = None
