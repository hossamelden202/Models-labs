from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Status = Literal["untested", "untestable", "supported", "weakened", "refuted", "conflicting", "inconclusive"]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(_M):
    kind: Literal["observation", "experiment"]
    stance: Literal["supports", "contradicts", "inconclusive"]
    summary: str
    strength: float = Field(ge=0.0, le=1.0)
    experiment_id: str | None = None
    test_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class ExperimentTest(_M):
    test_id: str
    description: str
    intervention: dict[str, Any]
    population: Literal["failure", "control"]
    expected_effect: Literal["improve", "degrade"]


class Hypothesis(_M):
    hypothesis_id: str
    failure_id: str
    rule: str
    category: str
    mechanism: str
    params: dict[str, Any] = Field(default_factory=dict)
    supporting_evidence: list[Evidence] = Field(default_factory=list)
    contradicting_evidence: list[Evidence] = Field(default_factory=list)
    inconclusive_evidence: list[Evidence] = Field(default_factory=list)
    evidence_strength: float = 0.0
    confidence_label: str = "very_weak"
    required_tests: list[ExperimentTest] = Field(default_factory=list)
    status: Status = "untested"
    status_history: list[dict[str, Any]] = Field(default_factory=list)
    competes_with: list[str] = Field(default_factory=list)
    note: str = ""


class FailureTarget(_M):
    failure_id: str
    kind: Literal["slice", "class", "confusion"]
    description: str
    slice_id: str | None = None
    class_name: str | None = None
    pair: list[str] | None = None
    conditions: list[list[str]] = Field(default_factory=list)
    support: int
    errors: int
    error_rate: float
    baseline_accuracy: float
    tier: str
    population_ref: dict[str, Any]
