from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from modellab.analysis import FailureConfig
from modellab.hypotheses import HypothesisConfig


class InvestigationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypotheses: HypothesisConfig = Field(default_factory=HypothesisConfig)
    analysis: FailureConfig = Field(default_factory=FailureConfig)
    audit: dict[str, Any] = Field(default_factory=dict)
    batch_size: int = Field(32, ge=1)
    seed: int = Field(42, ge=0, lt=2**32)
    min_strength: float = Field(0.6, ge=0.0, le=1.0)
    min_margin: float = Field(0.2, ge=0.0, le=1.0)
    min_supporting_experiments: int = Field(1, ge=1)
    require_alternative_tested: bool = True
