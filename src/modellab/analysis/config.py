from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FailureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    high_confidence_threshold: float = Field(0.9, gt=0.0, le=1.0)
    low_confidence_threshold: float = Field(0.6, gt=0.0, le=1.0)
    ambiguous_margin: float = Field(0.2, gt=0.0, le=1.0)

    min_support: int = Field(20, ge=2)
    alpha: float = Field(0.05, gt=0.0, lt=1.0)
    min_risk_difference: float = Field(0.05, ge=0.0, le=1.0)
    n_bins: int = Field(4, ge=2, le=10)
    max_categories: int = Field(20, ge=2)
    max_depth: int = Field(2, ge=1, le=3)
    beam_width: int = Field(20, ge=1)
    max_candidates: int = Field(20000, ge=100)
    rank_by: Literal["excess_errors", "risk_difference", "cohens_h", "q_value"] = "excess_errors"
    exclude_model_output_features: bool = False

    top_confusions: int = Field(5, ge=1)
    max_evidence: int = Field(20, ge=1)
    max_listed_slices: int = Field(100, ge=1)

    clusters: bool = False
    n_clusters: int = Field(4, ge=2, le=50)
    cluster_seed: int = Field(0, ge=0)

    @model_validator(mode="after")
    def _check(self):
        if self.low_confidence_threshold > self.high_confidence_threshold:
            raise ValueError("low_confidence_threshold must not exceed high_confidence_threshold")
        return self
