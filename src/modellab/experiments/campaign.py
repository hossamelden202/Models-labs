from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modellab.experiments.errors import ExperimentConfigError
from modellab.experiments.spec import (
    ExperimentSpec,
    canonical_json,
    experiment_id,
    parse_spec,
    sha256_text,
)
from modellab.experiments.training_config import TrainingConfig


@dataclass(frozen=True)
class CampaignBudget:
    max_experiments: int = 24
    max_stage_experiments: int = 8
    max_stages: int = 3

    def __post_init__(self):
        if self.max_experiments <= 0:
            raise ExperimentConfigError(
                "max_experiments must be greater than zero"
            )
        if self.max_stage_experiments <= 0:
            raise ExperimentConfigError(
                "max_stage_experiments must be greater than zero"
            )
        if self.max_stages <= 0:
            raise ExperimentConfigError(
                "max_stages must be greater than zero"
            )


@dataclass(frozen=True)
class TrainingSearchSpace:
    """
    Controlled training search space.

    Each key is a TrainingConfig dotted path and each value is a list
    of candidate values.

    Example:

        TrainingSearchSpace(
            values={
                "optimizer.learning_rate": [1e-3, 3e-4, 1e-4],
                "data.batch_size": [16, 32],
                "data.augmentation.horizontal_flip": [False, True],
            }
        )
    """

    values: dict[str, list[Any]]

    def __post_init__(self):
        if not self.values:
            raise ExperimentConfigError(
                "training search space must not be empty"
            )

        for path, values in self.values.items():
            if not isinstance(path, str) or not path.strip():
                raise ExperimentConfigError(
                    "training search-space paths must be non-empty strings"
                )

            if not values:
                raise ExperimentConfigError(
                    f"search-space path {path!r} has no candidate values"
                )


@dataclass(frozen=True)
class CampaignStage:
    name: str
    hypothesis: str
    search_space: TrainingSearchSpace
    expected_direction: str = "improve"
    repetitions: int = 1
    seed: int = 42
    metrics: tuple[str, ...] = ("accuracy", "macro_f1")
    max_experiments: int | None = None


@dataclass
class CampaignPlan:
    campaign_id: str
    family_id: str
    stages: list[dict[str, Any]]
    total_candidates: int
    budget: dict[str, int]


@dataclass
class CampaignResult:
    campaign_id: str
    family_id: str
    stages: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    best_experiment: dict[str, Any] | None
    final_recommendation: dict[str, Any]
    directory: Path | None = None


def _product_count(values: dict[str, list[Any]]) -> int:
    count = 1

    for candidates in values.values():
        count *= len(candidates)

        if count > 10_000_000:
            raise ExperimentConfigError(
                "search space exceeds the safe candidate limit"
            )

    return count


def _training_change_spec(
    *,
    name: str,
    hypothesis: str,
    changes: list[dict[str, Any]],
    expected_direction: str,
    repetitions: int,
    seed: int,
    metrics: tuple[str, ...],
) -> ExperimentSpec:
    return parse_spec(
        {
            "name": name,
            "hypothesis": {
                "claim": hypothesis,
                "expected_direction": expected_direction,
            },
            "intervention": {
                "kind": "training",
                "changes": changes,
            },
            "seed": seed,
            "repetitions": repetitions,
            "metrics": list(metrics),
        }
    )


def expand_training_search_space(
    stage: CampaignStage,
    *,
    limit: int | None = None,
) -> list[ExperimentSpec]:
    """
    Expand a training search space into valid ExperimentSpec objects.

    Unlike the older generic expand_matrix implementation, this correctly
    writes values into TrainingChange(path, value) entries.
    """

    paths = sorted(stage.search_space.values)

    total = _product_count(stage.search_space.values)

    if limit is not None:
        if limit <= 0:
            raise ExperimentConfigError(
                "search-space limit must be greater than zero"
            )
        total_limit = min(total, limit)
    else:
        total_limit = total

    specs: list[ExperimentSpec] = []

    # Deterministic mixed-radix expansion. This avoids importing or relying
    # on itertools.product ordering while making truncation deterministic.
    indices = [0] * len(paths)

    for _ in range(total_limit):
        changes = [
            {
                "path": path,
                "value": stage.search_space.values[path][indices[pos]],
            }
            for pos, path in enumerate(paths)
        ]

        label = ",".join(
            f"{path}={value}"
            for path, value in (
                (path, stage.search_space.values[path][indices[pos]])
                for pos, path in enumerate(paths)
            )
        )

        specs.append(
            _training_change_spec(
                name=f"{stage.name}[{label}]",
                hypothesis=stage.hypothesis,
                changes=changes,
                expected_direction=stage.expected_direction,
                repetitions=stage.repetitions,
                seed=stage.seed,
                metrics=stage.metrics,
            )
        )

        for pos in range(len(indices) - 1, -1, -1):
            indices[pos] += 1

            if indices[pos] < len(stage.search_space.values[paths[pos]]):
                break

            indices[pos] = 0

    return specs


def deduplicate_specs(
    specs: list[ExperimentSpec],
) -> list[ExperimentSpec]:
    seen: set[str] = set()
    result: list[ExperimentSpec] = []

    for spec in specs:
        eid = experiment_id(spec)

        if eid in seen:
            continue

        seen.add(eid)
        result.append(spec)

    return result


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)

        if not math.isfinite(result):
            return None

        return result
    except (TypeError, ValueError):
        return None


def score_outcome(outcome: Any) -> float | None:
    """
    Extract a primary improvement score from an ExperimentOutcome.

    Detection:
        comparison.delta.f1

    Classification / legacy:
        comparison.primary_delta

    Legacy fallback:
        statistics.affected.primary_p
    """

    result = getattr(outcome, "result", None)

    if not isinstance(result, dict):
        return None

    comparison = result.get("comparison")

    if isinstance(comparison, dict):
        delta = comparison.get("delta")

        if isinstance(delta, dict):
            f1 = _safe_float(delta.get("f1"))

            if f1 is not None:
                return f1

        primary = _safe_float(
            comparison.get("primary_delta")
        )

        if primary is not None:
            return primary

        differences = comparison.get("differences")

        if isinstance(differences, dict):
            row = differences.get("macro_f1") or differences.get("accuracy")

            if isinstance(row, dict):
                difference = _safe_float(row.get("difference"))

                if difference is not None:
                    return difference

    statistics = result.get("statistics")

    if isinstance(statistics, dict):
        affected = statistics.get("affected")

        if isinstance(affected, dict):
            difference = _safe_float(
                affected.get("accuracy_difference")
            )

            if difference is not None:
                return difference

    return None


def summarize_outcome(outcome: Any) -> dict[str, Any]:
    result = getattr(outcome, "result", {}) or {}

    return {
        "experiment_id": getattr(outcome, "experiment_id", None),
        "status": getattr(outcome, "status", None),
        "score": score_outcome(outcome),
        "directory": str(getattr(outcome, "directory", "")),
        "verdict": (
            result.get("verdict")
            if isinstance(result, dict)
            else None
        ),
    }


def rank_outcomes(outcomes: list[Any]) -> list[dict[str, Any]]:
    rows = [
        summarize_outcome(outcome)
        for outcome in outcomes
    ]

    rows = [
        row
        for row in rows
        if row["status"] == "completed"
        and row["score"] is not None
    ]

    rows.sort(
        key=lambda row: row["score"],
        reverse=True,
    )

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return rows


def select_promotions(
    outcomes: list[Any],
    *,
    limit: int,
) -> list[str]:
    """
    Select successful experiment IDs for the next stage.

    We deliberately select only completed experiments with a positive
    measurable score. Failed/inconclusive experiments are retained as
    findings but do not consume promotion slots.
    """

    ranked = rank_outcomes(outcomes)

    promoted = [
        row["experiment_id"]
        for row in ranked
        if row["score"] is not None and row["score"] > 0
    ]

    return promoted[:limit]



def _training_changes_from_spec(
    spec: ExperimentSpec,
) -> dict[str, Any]:
    """Return the training intervention as a path -> value mapping."""
    if getattr(spec.intervention, "kind", None) != "training":
        return {}

    return {
        change.path: change.value
        for change in spec.intervention.changes
    }


def _promoted_specs(
    outcomes: list[Any],
    candidate_specs: list[ExperimentSpec],
    *,
    limit: int,
) -> list[ExperimentSpec]:
    """
    Recover the actual ExperimentSpec objects that produced the best
    positive outcomes.

    The mapping is deliberately made from the specs passed to run_family()
    rather than depending on the shape of persisted result.json.
    """
    by_id = {
        experiment_id(spec): spec
        for spec in candidate_specs
    }

    ranked = rank_outcomes(outcomes)

    promoted: list[ExperimentSpec] = []

    for row in ranked:
        if row["score"] is None or row["score"] <= 0:
            continue

        spec = by_id.get(row["experiment_id"])

        if spec is None:
            continue

        promoted.append(spec)

        if len(promoted) >= limit:
            break

    return deduplicate_specs(promoted)


def expand_adaptive_training_search_space(
    stage: CampaignStage,
    parents: list[ExperimentSpec],
    *,
    limit: int | None = None,
) -> list[ExperimentSpec]:
    """
    Generate controlled refinements around promoted parent configurations.

    Later stages do NOT Cartesian-product every dimension.

    Instead, each candidate changes one search-space dimension while
    preserving the parent's other training values.

    Example:

        parent:
            learning_rate = 3e-4
            batch_size = 32

        refinement:
            learning_rate = [1e-4, 5e-4]
            weight_decay = [1e-5, 1e-4]

    produces bounded candidates around that parent rather than every
    possible combination of both dimensions.
    """
    if not parents:
        return []

    paths = sorted(stage.search_space.values)

    if not paths:
        return []

    candidates: list[ExperimentSpec] = []

    for parent in parents:
        parent_values = _training_changes_from_spec(parent)

        for path in paths:
            for value in stage.search_space.values[path]:
                changes = dict(parent_values)
                changes[path] = value

                ordered_changes = [
                    {
                        "path": change_path,
                        "value": changes[change_path],
                    }
                    for change_path in sorted(changes)
                ]

                label = ",".join(
                    f"{item['path']}={item['value']}"
                    for item in ordered_changes
                )

                candidates.append(
                    _training_change_spec(
                        name=f"{stage.name}[{label}]",
                        hypothesis=stage.hypothesis,
                        changes=ordered_changes,
                        expected_direction=stage.expected_direction,
                        repetitions=stage.repetitions,
                        seed=stage.seed,
                        metrics=stage.metrics,
                    )
                )

    candidates = deduplicate_specs(candidates)

    if limit is not None:
        if limit <= 0:
            raise ExperimentConfigError(
                "adaptive search-space limit must be greater than zero"
            )

        candidates = candidates[:limit]

    return candidates


def build_campaign_plan(
    *,
    family_id: str,
    stages: list[CampaignStage],
    budget: CampaignBudget,
) -> CampaignPlan:
    if not family_id:
        raise ExperimentConfigError(
            "family_id must not be empty"
        )

    if not stages:
        raise ExperimentConfigError(
            "campaign requires at least one stage"
        )

    if len(stages) > budget.max_stages:
        raise ExperimentConfigError(
            f"campaign defines {len(stages)} stages but budget allows "
            f"only {budget.max_stages}"
        )

    stage_payloads = []
    total = 0

    for stage in stages:
        requested = (
            stage.max_experiments
            if stage.max_experiments is not None
            else budget.max_stage_experiments
        )

        requested = min(
            requested,
            budget.max_stage_experiments,
        )

        count = min(
            _product_count(stage.search_space.values),
            requested,
        )

        total += count

        stage_payloads.append(
            {
                "name": stage.name,
                "hypothesis": stage.hypothesis,
                "candidate_count": count,
                "search_space": stage.search_space.values,
                "expected_direction": stage.expected_direction,
                "max_experiments": requested,
            }
        )

    total = min(
        total,
        budget.max_experiments,
    )

    payload = {
        "family_id": family_id,
        "stages": stage_payloads,
        "budget": {
            "max_experiments": budget.max_experiments,
            "max_stage_experiments": budget.max_stage_experiments,
            "max_stages": budget.max_stages,
        },
    }

    campaign_id = (
        "campaign_"
        + sha256_text(canonical_json(payload))[:12]
    )

    return CampaignPlan(
        campaign_id=campaign_id,
        family_id=family_id,
        stages=stage_payloads,
        total_candidates=total,
        budget=payload["budget"],
    )


def _recommendation(
    ranked: list[dict[str, Any]],
    *,
    campaign_id: str,
) -> dict[str, Any]:
    if not ranked:
        return {
            "campaign_id": campaign_id,
            "status": "no_successful_experiment",
            "recommendation": "retain_baseline",
            "reason": (
                "No completed experiment produced a positive measurable "
                "improvement over the baseline."
            ),
            "best_experiment_id": None,
            "score": None,
        }

    best = ranked[0]

    if best["score"] <= 0:
        return {
            "campaign_id": campaign_id,
            "status": "no_improvement",
            "recommendation": "retain_baseline",
            "reason": (
                "The best completed experiment did not improve the "
                "primary measured metric."
            ),
            "best_experiment_id": best["experiment_id"],
            "score": best["score"],
        }

    return {
        "campaign_id": campaign_id,
        "status": "improvement_found",
        "recommendation": "promote_best_experiment",
        "reason": (
            "The highest-scoring completed experiment produced a "
            "positive primary improvement."
        ),
        "best_experiment_id": best["experiment_id"],
        "score": best["score"],
    }


def persist_campaign(
    store,
    result: CampaignResult,
) -> Path:
    relative = (
        Path("experiments")
        / result.family_id
        / "campaigns"
        / result.campaign_id
    )

    payload = {
        "campaign_id": result.campaign_id,
        "family_id": result.family_id,
        "stages": result.stages,
        "findings": result.findings,
        "best_experiment": result.best_experiment,
        "final_recommendation": result.final_recommendation,
    }

    store.write_json(
        relative / "campaign.json",
        payload,
    )

    store.write_json(
        relative / "findings.json",
        {
            "campaign_id": result.campaign_id,
            "family_id": result.family_id,
            "findings": result.findings,
        },
    )

    store.write_json(
        relative / "final_report.json",
        {
            "campaign_id": result.campaign_id,
            "family_id": result.family_id,
            "best_experiment": result.best_experiment,
            "final_recommendation": result.final_recommendation,
        },
    )

    lines = [
        f"Campaign: {result.campaign_id}",
        f"Family: {result.family_id}",
        "",
        "Findings:",
    ]

    for finding in result.findings:
        lines.append(
            f"- {finding.get('experiment_id')}: "
            f"status={finding.get('status')}, "
            f"score={finding.get('score')}"
        )

    lines.extend(
        [
            "",
            "Final recommendation:",
            json.dumps(
                result.final_recommendation,
                indent=2,
                sort_keys=True,
            ),
        ]
    )

    store.write_text(
        relative / "final_report.txt",
        "\n".join(lines),
    )

    result.directory = store.root / relative

    return result.directory


def run_campaign(
    store,
    family_id: str,
    stages: list[CampaignStage],
    *,
    model=None,
    dataset=None,
    analysis=None,
    budget: CampaignBudget | None = None,
    force: bool = False,
    stop_on_failure: bool = False,
    alpha: float = 0.05,
) -> CampaignResult:
    """
    Execute a controlled multi-stage Phase 5 experiment campaign.

    Stage 1 performs bounded independent screening.

    Later stages are adaptive: they are generated from the actual
    ExperimentSpec objects that produced positive results in the previous
    stage.

    The existing run_family()/run_experiment() machinery remains
    responsible for experiment execution, evaluation, caching and
    persistence.
    """

    budget = budget or CampaignBudget()

    plan = build_campaign_plan(
        family_id=family_id,
        stages=stages,
        budget=budget,
    )

    # Lazy import preserves the existing runner import behavior and avoids
    # introducing a campaign -> runner -> campaign import cycle.
    from modellab.experiments.runner import run_family

    all_findings: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    stage_reports: list[dict[str, Any]] = []

    total_executed = 0
    promoted_specs: list[ExperimentSpec] = []

    for stage_index, stage in enumerate(stages, start=1):
        if stage_index > budget.max_stages:
            break

        remaining_budget = (
            budget.max_experiments - total_executed
        )

        if remaining_budget <= 0:
            stage_reports.append(
                {
                    "stage": stage.name,
                    "stage_index": stage_index,
                    "status": "skipped",
                    "reason": "campaign budget exhausted",
                    "num_specs": 0,
                    "num_outcomes": 0,
                    "promoted_experiment_ids": [],
                }
            )
            break

        stage_limit = min(
            budget.max_stage_experiments,
            remaining_budget,
            (
                stage.max_experiments
                if stage.max_experiments is not None
                else budget.max_stage_experiments
            ),
        )

        # -------------------------------------------------------------
        # Stage 1: broad independent screening.
        # -------------------------------------------------------------
        if stage_index == 1:
            specs = expand_training_search_space(
                stage,
                limit=stage_limit,
            )

            generation_mode = "independent_screening"

        # -------------------------------------------------------------
        # Stage 2+: adaptive refinement around actual winners.
        # -------------------------------------------------------------
        else:
            if not promoted_specs:
                stage_reports.append(
                    {
                        "stage": stage.name,
                        "stage_index": stage_index,
                        "status": "skipped",
                        "reason": "no positive experiments promoted",
                        "num_specs": 0,
                        "num_outcomes": 0,
                        "promoted_experiment_ids": [],
                    }
                )
                continue

            specs = expand_adaptive_training_search_space(
                stage,
                promoted_specs,
                limit=stage_limit,
            )

            generation_mode = "adaptive_refinement"

        specs = deduplicate_specs(specs)

        if not specs:
            stage_reports.append(
                {
                    "stage": stage.name,
                    "stage_index": stage_index,
                    "status": "skipped",
                    "reason": "candidate generation produced no valid specs",
                    "num_specs": 0,
                    "num_outcomes": 0,
                    "promoted_experiment_ids": [],
                }
            )
            continue

        # Detection training uses a dedicated execution layer because
        # the trusted 5976996 runner is intentionally classification-only.
        is_detection_training = (
            model is not None
            and getattr(
                getattr(model, "spec", None),
                "task",
                None,
            ) == "detection"
            and all(
                getattr(spec.intervention, "kind", None)
                == "training"
                for spec in specs
            )
        )

        if is_detection_training:
            from modellab.experiments.detection_runner import (
                run_detection_family,
            )

            family = run_detection_family(
                store,
                family_id,
                specs,
                model=model,
                dataset=dataset,
                analysis=analysis,
                force=force,
                stop_on_failure=stop_on_failure,
                alpha=alpha,
            )
        else:
            family = run_family(
                store,
                family_id,
                specs,
                model=model,
                dataset=dataset,
                analysis=analysis,
                force=force,
                stop_on_failure=stop_on_failure,
                alpha=alpha,
            )

        total_executed += len(family.outcomes)

        ranked = rank_outcomes(
            family.outcomes
        )

        all_findings.extend(
            summarize_outcome(outcome)
            for outcome in family.outcomes
        )

        best_rows.extend(ranked)

        # IMPORTANT:
        # Recover the exact specs that were submitted to run_family().
        # This avoids depending on result.json containing a serialized
        # ExperimentSpec.
        promoted_specs = _promoted_specs(
            family.outcomes,
            specs,
            limit=budget.max_stage_experiments,
        )

        promoted_ids = [
            experiment_id(spec)
            for spec in promoted_specs
        ]

        stage_reports.append(
            {
                "stage": stage.name,
                "stage_index": stage_index,
                "status": "completed",
                "generation_mode": generation_mode,
                "num_specs": len(specs),
                "num_outcomes": len(family.outcomes),
                "promoted_experiment_ids": promoted_ids,
                "ranking": ranked,
                "promoted_specs": [
                    spec.model_dump(mode="json")
                    for spec in promoted_specs
                ],
            }
        )

    best_rows.sort(
        key=lambda row: row["score"],
        reverse=True,
    )

    best = (
        best_rows[0]
        if best_rows
        else None
    )

    result = CampaignResult(
        campaign_id=plan.campaign_id,
        family_id=family_id,
        stages=stage_reports,
        findings=all_findings,
        best_experiment=best,
        final_recommendation=_recommendation(
            best_rows,
            campaign_id=plan.campaign_id,
        ),
    )

    persist_campaign(
        store,
        result,
    )

    return result


__all__ = [
    "CampaignBudget",
    "CampaignPlan",
    "CampaignResult",
    "CampaignStage",
    "TrainingSearchSpace",
    "build_campaign_plan",
    "deduplicate_specs",
    "expand_training_search_space",
    "expand_adaptive_training_search_space",
    "persist_campaign",
    "rank_outcomes",
    "run_campaign",
    "score_outcome",
    "select_promotions",
]
