from pathlib import Path

import json

from modellab.experiments import campaign
from modellab.experiments.campaign import (
    CampaignBudget,
    CampaignStage,
    TrainingSearchSpace,
    run_campaign,
)
from modellab.experiments.spec import experiment_id
from modellab.core.results import ArtifactStore


class FakeOutcome:
    def __init__(self, spec, score):
        self.experiment_id = experiment_id(spec)
        self.status = "completed"
        self.directory = Path("/tmp") / self.experiment_id
        self.result = {
            "comparison": {
                "delta": {
                    "f1": score,
                }
            }
        }


class FakeFamily:
    def __init__(self, outcomes):
        self.outcomes = outcomes


def test_campaign_adapts_stage_two_from_stage_one(monkeypatch, tmp_path):
    calls = []

    scores = {}

    def fake_run_family(
        store,
        family_id,
        specs,
        **kwargs,
    ):
        calls.append(list(specs))

        # Stage 1:
        # Make only the first candidate positive.
        if len(calls) == 1:
            outcomes = [
                FakeOutcome(
                    spec,
                    0.10 if index == 0 else -0.01,
                )
                for index, spec in enumerate(specs)
            ]

        # Stage 2:
        # Make the first refined candidate positive.
        else:
            outcomes = [
                FakeOutcome(
                    spec,
                    0.20 if index == 0 else -0.01,
                )
                for index, spec in enumerate(specs)
            ]

        for outcome in outcomes:
            scores[outcome.experiment_id] = (
                outcome.result["comparison"]["delta"]["f1"]
            )

        return FakeFamily(outcomes)

    monkeypatch.setattr(
        campaign,
        "run_campaign",
        campaign.run_campaign,
    )

    # Patch the lazy runner import target used inside run_campaign().
    import modellab.experiments.runner as runner

    monkeypatch.setattr(
        runner,
        "run_family",
        fake_run_family,
    )

    stages = [
        CampaignStage(
            name="stage1",
            hypothesis="Find promising learning rate and batch size.",
            search_space=TrainingSearchSpace(
                values={
                    "optimizer.learning_rate": [
                        1e-3,
                        3e-4,
                    ],
                    "data.batch_size": [
                        16,
                        32,
                    ],
                }
            ),
            max_experiments=4,
        ),
        CampaignStage(
            name="stage2",
            hypothesis="Refine the promoted configuration.",
            search_space=TrainingSearchSpace(
                values={
                    "optimizer.learning_rate": [
                        1e-4,
                        5e-4,
                    ],
                    "optimizer.weight_decay": [
                        1e-5,
                        1e-4,
                    ],
                }
            ),
            max_experiments=4,
        ),
    ]

    store = ArtifactStore(tmp_path)

    result = run_campaign(
        store,
        "test-family",
        stages,
        budget=CampaignBudget(
            max_experiments=8,
            max_stage_experiments=4,
            max_stages=2,
        ),
    )

    assert len(calls) == 2

    stage1_specs = calls[0]
    stage2_specs = calls[1]

    assert len(stage1_specs) == 4
    assert len(stage2_specs) == 4

    # Every Stage 2 candidate must retain the promoted Stage 1
    # configuration except for the dimension being refined.
    promoted = stage1_specs[0]

    promoted_values = {
        change.path: change.value
        for change in promoted.intervention.changes
    }

    for spec in stage2_specs:
        values = {
            change.path: change.value
            for change in spec.intervention.changes
        }

        assert values["data.batch_size"] == promoted_values["data.batch_size"]

    assert result.final_recommendation["recommendation"] == (
        "promote_best_experiment"
    )


def test_campaign_stops_later_stages_when_nothing_is_promoted(
    monkeypatch,
    tmp_path,
):
    calls = []

    def fake_run_family(
        store,
        family_id,
        specs,
        **kwargs,
    ):
        calls.append(list(specs))

        return FakeFamily(
            [
                FakeOutcome(spec, -0.01)
                for spec in specs
            ]
        )

    import modellab.experiments.runner as runner

    monkeypatch.setattr(
        runner,
        "run_family",
        fake_run_family,
    )

    stages = [
        CampaignStage(
            name="stage1",
            hypothesis="Test a first intervention.",
            search_space=TrainingSearchSpace(
                values={
                    "optimizer.learning_rate": [
                        1e-3,
                        3e-4,
                    ],
                }
            ),
        ),
        CampaignStage(
            name="stage2",
            hypothesis="Should not execute without promotion.",
            search_space=TrainingSearchSpace(
                values={
                    "optimizer.weight_decay": [
                        1e-5,
                        1e-4,
                    ],
                }
            ),
        ),
    ]

    store = ArtifactStore(tmp_path)

    result = run_campaign(
        store,
        "test-family",
        stages,
        budget=CampaignBudget(
            max_experiments=8,
            max_stage_experiments=4,
            max_stages=2,
        ),
    )

    assert len(calls) == 1

    assert result.stages[0]["status"] == "completed"
    assert result.stages[1]["status"] == "skipped"
    assert result.stages[1]["reason"] == (
        "no positive experiments promoted"
    )

    assert result.final_recommendation["recommendation"] == (
        "retain_baseline"
    )


def test_campaign_respects_global_budget(
    monkeypatch,
    tmp_path,
):
    calls = []

    def fake_run_family(
        store,
        family_id,
        specs,
        **kwargs,
    ):
        calls.append(list(specs))

        return FakeFamily(
            [
                FakeOutcome(spec, 0.05)
                for spec in specs
            ]
        )

    import modellab.experiments.runner as runner

    monkeypatch.setattr(
        runner,
        "run_family",
        fake_run_family,
    )

    stages = [
        CampaignStage(
            name="stage1",
            hypothesis="First search.",
            search_space=TrainingSearchSpace(
                values={
                    "optimizer.learning_rate": [
                        1e-3,
                        3e-4,
                        1e-4,
                    ],
                }
            ),
            max_experiments=3,
        ),
        CampaignStage(
            name="stage2",
            hypothesis="Second search.",
            search_space=TrainingSearchSpace(
                values={
                    "optimizer.weight_decay": [
                        1e-5,
                        1e-4,
                        1e-3,
                    ],
                }
            ),
            max_experiments=3,
        ),
    ]

    store = ArtifactStore(tmp_path)

    result = run_campaign(
        store,
        "test-family",
        stages,
        budget=CampaignBudget(
            max_experiments=4,
            max_stage_experiments=3,
            max_stages=2,
        ),
    )

    total = sum(
        len(call)
        for call in calls
    )

    assert total <= 4

    assert result.campaign_id.startswith("campaign_")


def test_campaign_persists_report(
    monkeypatch,
    tmp_path,
):
    def fake_run_family(
        store,
        family_id,
        specs,
        **kwargs,
    ):
        return FakeFamily(
            [
                FakeOutcome(spec, 0.05)
                for spec in specs
            ]
        )

    import modellab.experiments.runner as runner

    monkeypatch.setattr(
        runner,
        "run_family",
        fake_run_family,
    )

    stage = CampaignStage(
        name="stage1",
        hypothesis="Persistence test.",
        search_space=TrainingSearchSpace(
            values={
                "optimizer.learning_rate": [
                    1e-3,
                ],
            }
        ),
    )

    store = ArtifactStore(tmp_path)

    result = run_campaign(
        store,
        "test-family",
        [stage],
    )

    assert result.directory is not None
    assert result.directory.is_dir()

    campaign_json = result.directory / "campaign.json"
    findings_json = result.directory / "findings.json"
    final_json = result.directory / "final_report.json"
    final_txt = result.directory / "final_report.txt"

    assert campaign_json.is_file()
    assert findings_json.is_file()
    assert final_json.is_file()
    assert final_txt.is_file()

    payload = json.loads(
        campaign_json.read_text()
    )

    assert payload["campaign_id"] == result.campaign_id
    assert payload["family_id"] == "test-family"
    assert payload["stages"]
    assert payload["findings"]
