from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


# ============================================================================
# Canonical helpers
# ============================================================================

def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)

    for key, value in update.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


# ============================================================================
# Candidate
# ============================================================================

@dataclass(frozen=True)
class ExperimentCandidate:
    """
    Immutable candidate configuration.

    The planner deliberately stores a plain configuration dictionary rather
    than depending on ExperimentSpec internals. This keeps Phase 5 orchestration
    independent from the existing experiment execution foundation.
    """

    config: dict[str, Any]
    stage: int
    parent_id: str | None = None
    reason: str = ""

    @property
    def experiment_id(self) -> str:
        return _fingerprint(self.config)

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "stage": self.stage,
            "parent_id": self.parent_id,
            "reason": self.reason,
            "config": self.config,
        }


# ============================================================================
# Search space
# ============================================================================

@dataclass
class SearchSpace:
    """
    Controlled search space.

    Each key maps to candidate values. Nested dictionaries are supported.

    Example:

        SearchSpace.from_mapping({
            "training": {
                "optimizer": ["adamw", "sgd"],
                "learning_rate": [1e-4, 3e-4],
            },
            "augmentation": {
                "horizontal_flip": [False, True],
            },
        })
    """

    dimensions: dict[str, list[Any]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SearchSpace":
        dimensions: dict[str, list[Any]] = {}

        def visit(prefix: str, value: Any) -> None:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    child_prefix = f"{prefix}.{key}" if prefix else str(key)
                    visit(child_prefix, child)
                return

            if isinstance(value, (list, tuple, set)):
                values = list(value)
            else:
                values = [value]

            if not values:
                return

            dimensions[prefix] = values

        visit("", mapping)

        return cls(dimensions=dimensions)

    def __len__(self) -> int:
        if not self.dimensions:
            return 0

        size = 1
        for values in self.dimensions.values():
            size *= len(values)

        return size

    def keys(self) -> list[str]:
        return list(self.dimensions)

    def values_for(self, key: str) -> list[Any]:
        return list(self.dimensions[key])


# ============================================================================
# Nested configuration writer
# ============================================================================

def _set_path(target: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    cursor = target

    for part in parts[:-1]:
        existing = cursor.get(part)

        if not isinstance(existing, dict):
            existing = {}
            cursor[part] = existing

        cursor = existing

    cursor[parts[-1]] = value


def _config_from_values(
    dimensions: Sequence[str],
    values: Sequence[Any],
) -> dict[str, Any]:
    config: dict[str, Any] = {}

    for key, value in zip(dimensions, values):
        _set_path(config, key, value)

    return config


# ============================================================================
# Planner
# ============================================================================

class ExperimentPlanner:
    """
    Wide-but-controlled Phase 5 experiment planner.

    Design principles:

    1. Never blindly Cartesian-product the complete search space.
    2. Stage 1 performs isolated screening.
    3. Stage 2 combines directions that performed well.
    4. Stage 3 performs local refinement.
    5. Every candidate receives a deterministic ID.
    6. Duplicate candidates are removed automatically.
    """

    def __init__(
        self,
        baseline_config: Mapping[str, Any] | None = None,
        *,
        max_stage1: int = 24,
        max_stage2: int = 16,
        max_stage3: int = 12,
        keep_top: int = 6,
        min_improvement: float = 0.0,
    ) -> None:
        self.baseline_config = dict(baseline_config or {})
        self.max_stage1 = max(1, int(max_stage1))
        self.max_stage2 = max(1, int(max_stage2))
        self.max_stage3 = max(1, int(max_stage3))
        self.keep_top = max(1, int(keep_top))
        self.min_improvement = float(min_improvement)

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    def stage1(
        self,
        search_space: SearchSpace,
    ) -> list[ExperimentCandidate]:
        """
        Broad screening.

        Each dimension is screened independently against the baseline.
        This avoids a giant Cartesian product.
        """

        candidates: list[ExperimentCandidate] = []

        dimensions = search_space.keys()

        for dimension in dimensions:
            for value in search_space.values_for(dimension):
                config = dict(self.baseline_config)
                _set_path(config, dimension, value)

                candidates.append(
                    ExperimentCandidate(
                        config=config,
                        stage=1,
                        reason=f"screen {dimension}",
                    )
                )

        return self._deduplicate(candidates)[: self.max_stage1]

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------

    def stage2(
        self,
        results: Sequence[Mapping[str, Any]],
        *,
        max_candidates: int | None = None,
    ) -> list[ExperimentCandidate]:
        """
        Combine the strongest independent directions.

        Expected result fields:

            experiment_id
            metric
            baseline_metric

        Optional:

            score
            improvement
            config
        """

        limit = self.max_stage2 if max_candidates is None else max(1, max_candidates)

        winners = self.rank_results(results)[: self.keep_top]

        if len(winners) < 2:
            return []

        candidates: list[ExperimentCandidate] = []

        for left, right in itertools.combinations(winners, 2):
            left_config = left.get("config")
            right_config = right.get("config")

            if not isinstance(left_config, Mapping):
                continue

            if not isinstance(right_config, Mapping):
                continue

            combined = _deep_merge(self.baseline_config, left_config)
            combined = _deep_merge(combined, right_config)

            candidates.append(
                ExperimentCandidate(
                    config=combined,
                    stage=2,
                    parent_id=left.get("experiment_id"),
                    reason=(
                        "combine promising directions: "
                        f"{left.get('experiment_id')} + "
                        f"{right.get('experiment_id')}"
                    ),
                )
            )

            if len(candidates) >= limit:
                break

        return self._deduplicate(candidates)[:limit]

    # ------------------------------------------------------------------
    # Stage 3
    # ------------------------------------------------------------------

    def stage3(
        self,
        results: Sequence[Mapping[str, Any]],
        *,
        max_candidates: int | None = None,
    ) -> list[ExperimentCandidate]:
        """
        Local refinement around the strongest configuration.

        Refinement is intentionally conservative. It only changes numerical
        parameters when a result exposes the corresponding configuration.
        """

        limit = self.max_stage3 if max_candidates is None else max(1, max_candidates)

        ranked = self.rank_results(results)

        if not ranked:
            return []

        best = ranked[0]
        config = best.get("config")

        if not isinstance(config, Mapping):
            return []

        candidates: list[ExperimentCandidate] = []

        numeric_paths = (
            "training.learning_rate",
            "training.weight_decay",
            "training.momentum",
            "training.warmup",
            "training.min_lr",
        )

        for path in numeric_paths:
            value = self._get_path(config, path)

            if not isinstance(value, (int, float)):
                continue

            if isinstance(value, bool):
                continue

            if value <= 0:
                continue

            factors = (0.5, 0.75, 1.25, 2.0)

            for factor in factors:
                refined = json.loads(_canonical(config))
                _set_path(refined, path, value * factor)

                candidates.append(
                    ExperimentCandidate(
                        config=refined,
                        stage=3,
                        parent_id=best.get("experiment_id"),
                        reason=f"local refinement of {path}",
                    )
                )

                if len(candidates) >= limit:
                    return self._deduplicate(candidates)[:limit]

        return self._deduplicate(candidates)[:limit]

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_results(
        self,
        results: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        """
        Rank results using improvement first.

        Supports both:

            improvement

        and:

            metric + baseline_metric
        """

        usable: list[Mapping[str, Any]] = []

        for result in results:
            if not isinstance(result, Mapping):
                continue

            if self._result_score(result) is None:
                continue

            usable.append(result)

        return sorted(
            usable,
            key=lambda item: self._result_score(item) or float("-inf"),
            reverse=True,
        )

    def _result_score(self, result: Mapping[str, Any]) -> float | None:
        if isinstance(result.get("improvement"), (int, float)):
            return float(result["improvement"])

        metric = result.get("metric")
        baseline_metric = result.get("baseline_metric")

        if isinstance(metric, (int, float)):
            if isinstance(baseline_metric, (int, float)):
                return float(metric) - float(baseline_metric)

            return float(metric)

        score = result.get("score")

        if isinstance(score, (int, float)):
            return float(score)

        return None

    # ------------------------------------------------------------------
    # Final recommendation
    # ------------------------------------------------------------------

    def final_recommendation(
        self,
        baseline_metric: float,
        results: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        ranked = self.rank_results(results)

        if not ranked:
            return {
                "baseline_metric": baseline_metric,
                "best_experiment": None,
                "improvement": 0.0,
                "recommendation": dict(self.baseline_config),
                "confidence": "LOW",
                "supported_changes": [],
                "rejected_changes": [],
            }

        best = ranked[0]
        best_metric = self._best_metric(best, baseline_metric)
        improvement = best_metric - baseline_metric

        supported: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for result in ranked:
            score = self._result_score(result)

            if score is None:
                continue

            item = {
                "experiment_id": result.get("experiment_id"),
                "improvement": score,
                "config": result.get("config"),
            }

            if score > self.min_improvement:
                supported.append(item)
            else:
                rejected.append(item)

        confidence = self._confidence(
            improvement=improvement,
            result_count=len(ranked),
        )

        recommendation = best.get("config")

        if not isinstance(recommendation, Mapping):
            recommendation = self.baseline_config

        return {
            "baseline_metric": baseline_metric,
            "best_experiment": best.get("experiment_id"),
            "best_metric": best_metric,
            "improvement": improvement,
            "recommendation": dict(recommendation),
            "confidence": confidence,
            "supported_changes": supported,
            "rejected_changes": rejected,
            "ranked_results": [
                {
                    "experiment_id": result.get("experiment_id"),
                    "improvement": self._result_score(result),
                    "metric": result.get("metric"),
                }
                for result in ranked
            ],
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _best_metric(
        result: Mapping[str, Any],
        baseline_metric: float,
    ) -> float:
        metric = result.get("metric")

        if isinstance(metric, (int, float)):
            return float(metric)

        improvement = result.get("improvement")

        if isinstance(improvement, (int, float)):
            return baseline_metric + float(improvement)

        score = result.get("score")

        if isinstance(score, (int, float)):
            return float(score)

        return baseline_metric

    @staticmethod
    def _confidence(
        *,
        improvement: float,
        result_count: int,
    ) -> str:
        """
        Estimate recommendation confidence from evidence strength.

        This is deliberately conservative:

        - no improvement -> LOW
        - only one observed experiment -> LOW
        - positive improvement with at least two results -> MEDIUM
        - >=5 percentage-point improvement with >=3 results -> HIGH

        The campaign layer can later replace this heuristic with a statistical
        confidence calculation once repeated measurements are available.
        """
        if improvement <= 0:
            return "LOW"

        if result_count < 2:
            return "LOW"

        if improvement >= 0.05 and result_count >= 3:
            return "HIGH"

        return "MEDIUM"

    @staticmethod
    def _get_path(
        mapping: Mapping[str, Any],
        dotted_path: str,
    ) -> Any:
        current: Any = mapping

        for part in dotted_path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return None

            current = current[part]

        return current

    @staticmethod
    def _deduplicate(
        candidates: Iterable[ExperimentCandidate],
    ) -> list[ExperimentCandidate]:
        seen: set[str] = set()
        result: list[ExperimentCandidate] = []

        for candidate in candidates:
            experiment_id = candidate.experiment_id

            if experiment_id in seen:
                continue

            seen.add(experiment_id)
            result.append(candidate)

        return result


# ============================================================================
# Report
# ============================================================================

@dataclass
class ExperimentCampaignReport:
    baseline_metric: float
    stage1: list[ExperimentCandidate]
    stage2: list[ExperimentCandidate]
    stage3: list[ExperimentCandidate]
    results: list[dict[str, Any]]
    recommendation: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_metric": self.baseline_metric,
            "stage1": [candidate.as_dict() for candidate in self.stage1],
            "stage2": [candidate.as_dict() for candidate in self.stage2],
            "stage3": [candidate.as_dict() for candidate in self.stage3],
            "results": self.results,
            "recommendation": self.recommendation,
        }


# ============================================================================
# Convenience API
# ============================================================================

def build_stage1(
    baseline_config: Mapping[str, Any],
    search_space: Mapping[str, Any],
    *,
    max_candidates: int = 24,
) -> list[ExperimentCandidate]:
    planner = ExperimentPlanner(
        baseline_config,
        max_stage1=max_candidates,
    )

    return planner.stage1(
        SearchSpace.from_mapping(search_space)
    )


def build_stage2(
    baseline_config: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    *,
    max_candidates: int = 16,
) -> list[ExperimentCandidate]:
    planner = ExperimentPlanner(
        baseline_config,
        max_stage2=max_candidates,
    )

    return planner.stage2(results)


def build_stage3(
    baseline_config: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    *,
    max_candidates: int = 12,
) -> list[ExperimentCandidate]:
    planner = ExperimentPlanner(
        baseline_config,
        max_stage3=max_candidates,
    )

    return planner.stage3(results)
