from pathlib import Path

import pytest

from modellab.experiments.campaign import (
    CampaignBudget,
    CampaignStage,
    TrainingSearchSpace,
    build_campaign_plan,
    deduplicate_specs,
    expand_training_search_space,
    rank_outcomes,
    score_outcome,
)
from modellab.experiments.spec import experiment_id


def stage():
    return CampaignStage(
        name="lr-search",
        hypothesis="Learning rate affects validation performance.",
        search_space=TrainingSearchSpace(
            values={
                "optimizer.learning_rate": [
                    1e-3,
                    3e-4,
                    1e-4,
                ],
                "data.batch_size": [
                    16,
                    32,
                ],
            }
        ),
    )


def test_training_search_space_expands_to_valid_specs():
    specs = expand_training_search_space(stage())

    assert len(specs) == 6

    ids = [experiment_id(spec) for spec in specs]

    assert len(ids) == len(set(ids))

    for spec in specs:
        assert spec.intervention.kind == "training"
        assert len(spec.intervention.changes) == 2


def test_training_search_space_respects_limit():
    specs = expand_training_search_space(
        stage(),
        limit=3,
    )

    assert len(specs) == 3


def test_deduplication():
    specs = expand_training_search_space(
        stage(),
        limit=2,
    )

    result = deduplicate_specs(
        specs + specs
    )

    assert len(result) == 2


def test_campaign_budget():
    plan = build_campaign_plan(
        family_id="weapons-final-model",
        stages=[stage()],
        budget=CampaignBudget(
            max_experiments=5,
            max_stage_experiments=4,
            max_stages=2,
        ),
    )

    assert plan.family_id == "weapons-final-model"
    assert plan.total_candidates == 4
    assert plan.campaign_id.startswith("campaign_")


class FakeOutcome:
    def __init__(
        self,
        experiment_id,
        score,
        status="completed",
    ):
        self.experiment_id = experiment_id
        self.status = status
        self.directory = Path("/tmp") / experiment_id
        self.result = {
            "comparison": {
                "delta": {
                    "f1": score,
                }
            }
        }


def test_score_and_ranking():
    outcomes = [
        FakeOutcome("exp_a", 0.02),
        FakeOutcome("exp_b", -0.01),
        FakeOutcome("exp_c", 0.05),
    ]

    assert score_outcome(outcomes[0]) == pytest.approx(0.02)

    ranked = rank_outcomes(outcomes)

    assert [row["experiment_id"] for row in ranked] == [
        "exp_c",
        "exp_a",
        "exp_b",
    ]

    assert ranked[0]["rank"] == 1
