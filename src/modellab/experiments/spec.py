import hashlib
import itertools
import json
from copy import deepcopy
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from modellab.experiments.errors import ExperimentConfigError
from modellab.experiments.transforms import REGISTRY
from modellab.experiments.training_config import TrainingIntervention

MetricName = Literal[
    "accuracy", "macro_f1", "macro_precision", "macro_recall", "weighted_f1",
    "balanced_accuracy", "mean_confidence", "ece",
]


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Hypothesis(_Strict):
    claim: str = Field(min_length=1)
    expected_direction: Literal["improve", "degrade", "no_change"] | None = None


class ImageTransform(_Strict):
    kind: Literal["image_transform"]
    op: str
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self):
        if self.op not in REGISTRY:
            raise ValueError(f"unknown image op '{self.op}', available: {sorted(REGISTRY)}")
        self.params = REGISTRY[self.op].params(**self.params).model_dump()
        return self


class Preprocessing(_Strict):
    kind: Literal["preprocessing"]
    changes: dict[str, Any] = Field(min_length=1)


class Inference(_Strict):
    kind: Literal["inference"]
    temperature: float | None = Field(None, gt=0)
    confidence_threshold: float | None = Field(None, gt=0, lt=1)
    class_thresholds: dict[str, float] | None = None
    class_weights: dict[str, float] | None = None
    fallback_class: str | None = None

    @model_validator(mode="after")
    def _check(self):
        chosen = [
            n for n in ("temperature", "confidence_threshold", "class_thresholds", "class_weights")
            if getattr(self, n) is not None
        ]
        if len(chosen) != 1:
            raise ValueError(
                "set exactly one of temperature, confidence_threshold, class_thresholds, class_weights"
            )
        thresholded = self.confidence_threshold is not None or self.class_thresholds is not None
        if thresholded and not self.fallback_class:
            raise ValueError("threshold interventions need fallback_class")
        if not thresholded and self.fallback_class is not None:
            raise ValueError("fallback_class only applies to threshold interventions")
        if self.class_thresholds and any(not 0 < v < 1 for v in self.class_thresholds.values()):
            raise ValueError("class_thresholds values must lie in (0, 1)")
        if self.class_weights and any(v <= 0 for v in self.class_weights.values()):
            raise ValueError("class_weights values must be positive")
        return self


class DataIntervention(_Strict):
    kind: Literal["data"]
    op: Literal["reweighting", "filtering", "augmentation", "selection"]
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unsupported(self):
        raise ValueError(
            "data interventions (reweighting, filtering, augmentation, selection) need a training "
            "interface, which this version does not have; only fixed models can be evaluated"
        )


Intervention = Annotated[
    ImageTransform
    | Preprocessing
    | Inference
    | DataIntervention
    | TrainingIntervention,
    Field(discriminator="kind"),
]


class Population(_Strict):
    slice_ids: list[str] | None = None
    sample_ids: list[str] | None = None
    analysis_id: str | None = None

    @model_validator(mode="after")
    def _check(self):
        if self.slice_ids is not None and self.sample_ids is not None:
            raise ValueError("use slice_ids or sample_ids, not both")
        if self.slice_ids is not None and (not self.slice_ids or not self.analysis_id):
            raise ValueError("slice_ids need a non-empty list and an analysis_id")
        if self.sample_ids is not None and not self.sample_ids:
            raise ValueError("sample_ids must not be empty")
        return self


class ExperimentSpec(_Strict):
    name: str = Field(min_length=1)
    hypothesis: Hypothesis
    intervention: Intervention
    population: Population = Field(default_factory=Population)
    metrics: list[MetricName] = Field(default_factory=lambda: ["accuracy", "macro_f1"])
    seed: int = Field(42, ge=0, le=2**31)
    repetitions: int = Field(1, ge=1, le=50)
    n_bootstrap: int = Field(1000, ge=100)
    n_permutations: int = Field(5000, ge=100)
    alpha: float = Field(0.05, gt=0, lt=1)
    practical_threshold: float = Field(0.01, ge=0, le=1)

    @model_validator(mode="after")
    def _repetitions(self):
        iv = self.intervention
        stochastic = False
        if iv.kind == "image_transform":
            op = REGISTRY[iv.op]
            stochastic = op.stochastic(op.params(**iv.params))
        if self.repetitions > 1 and not stochastic:
            raise ValueError("repetitions above 1 only apply to stochastic interventions")
        return self


def parse_spec(obj) -> ExperimentSpec:
    try:
        return ExperimentSpec.model_validate(obj)
    except ValidationError as exc:
        raise ExperimentConfigError(f"invalid experiment spec: {exc}") from exc


def experiment_id(spec: ExperimentSpec) -> str:
    payload = canonical_json({"spec": spec.model_dump(mode="json")})
    return "exp_" + sha256_text(payload)[:12]


def expand_matrix(obj: dict) -> list[ExperimentSpec]:
    obj = dict(obj)
    grid = obj.pop("grid", None)
    base = deepcopy(obj.pop("intervention", None))
    name = obj.pop("name", None)
    if not grid or base is None or not name:
        raise ExperimentConfigError("a matrix needs name, intervention and a non-empty grid")
    keys = sorted(grid)
    specs = []
    for values in itertools.product(*(grid[k] for k in keys)):
        iv = deepcopy(base)
        if iv.get("kind") == "image_transform":
            target = iv.setdefault("params", {})
        elif iv.get("kind") == "preprocessing":
            target = iv.setdefault("changes", {})
        else:
            target = iv
        for key, value in zip(keys, values, strict=True):
            target[key] = value
        label = ",".join(f"{k}={v}" for k, v in zip(keys, values, strict=True))
        specs.append(parse_spec({**deepcopy(obj), "name": f"{name}[{label}]", "intervention": iv}))
    return specs
