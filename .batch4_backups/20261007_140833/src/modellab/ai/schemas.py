from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TargetMetric = Literal["recall", "precision", "f1", "mean_iou"]


class _S(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Change(_S):
    path: str
    value: Any


class Weakness(_S):
    class_name: str
    metric: TargetMetric
    value: float
    support: int
    note: str = ""


class ProblemAnalysis(_S):
    summary: str
    weaknesses: list[Weakness] = Field(default_factory=list)
    data_issues: list[str] = Field(default_factory=list)
    primary_target: TargetMetric = "recall"


class Candidate(_S):
    candidate_id: str
    title: str
    changes: list[Change]
    target_metric: TargetMetric
    rationale: str


class LLMChoice(_S):
    candidate_id: str
    claim: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class Hypothesis(_S):
    claim: str
    rationale: str
    evidence: list[str]
    expected_direction: Literal["improve"] = "improve"
    target_metric: TargetMetric
    confidence: float = Field(ge=0, le=1)


class ExperimentProposal(_S):
    name: str
    candidate_id: str
    changes: list[Change]
    expected_effect: str
    reason: str


class ValidationResult(_S):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    experiment_id: str | None = None
    spec: dict[str, Any] | None = None
