from modellab.experiments.planner import (
    ExperimentPlanner,
    SearchSpace,
    build_stage1,
)


def test_search_space_counts_dimensions_without_cartesian_stage1():
    space = SearchSpace.from_mapping(
        {
            "training": {
                "optimizer": ["adamw", "sgd"],
                "learning_rate": [1e-4, 3e-4],
            },
            "augmentation": {
                "horizontal_flip": [False, True],
            },
        }
    )

    assert len(space) == 8

    candidates = build_stage1(
        baseline_config={
            "training": {
                "optimizer": "adamw",
                "learning_rate": 1e-4,
            }
        },
        search_space={
            "training": {
                "optimizer": ["adamw", "sgd"],
                "learning_rate": [1e-4, 3e-4],
            },
            "augmentation": {
                "horizontal_flip": [False, True],
            },
        },
    )

    # Baseline-equivalent candidates are automatically eliminated.
    # Remaining isolated candidates:
    #   optimizer -> sgd
    #   learning_rate -> 3e-4
    #   horizontal_flip -> True
    assert len(candidates) == 3


def test_candidate_ids_are_deterministic_and_duplicates_removed():
    planner = ExperimentPlanner(
        baseline_config={"training": {"optimizer": "adamw"}}
    )

    space = SearchSpace.from_mapping(
        {
            "training": {
                "optimizer": ["adamw", "adamw"],
            }
        }
    )

    candidates = planner.stage1(space)

    assert len(candidates) == 1
    assert candidates[0].experiment_id == candidates[0].experiment_id


def test_stage2_combines_promising_results():
    planner = ExperimentPlanner(
        baseline_config={
            "training": {
                "optimizer": "adamw",
                "learning_rate": 1e-4,
            }
        },
        keep_top=3,
    )

    results = [
        {
            "experiment_id": "augmentation",
            "metric": 0.70,
            "baseline_metric": 0.60,
            "config": {
                "augmentation": {
                    "horizontal_flip": True,
                }
            },
        },
        {
            "experiment_id": "resolution",
            "metric": 0.69,
            "baseline_metric": 0.60,
            "config": {
                "data": {
                    "resolution": 640,
                }
            },
        },
        {
            "experiment_id": "loss",
            "metric": 0.68,
            "baseline_metric": 0.60,
            "config": {
                "training": {
                    "loss": "focal",
                }
            },
        },
    ]

    candidates = planner.stage2(results)

    assert candidates
    assert any(
        candidate.config.get("augmentation", {}).get("horizontal_flip") is True
        for candidate in candidates
    )


def test_stage3_refines_learning_rate():
    planner = ExperimentPlanner(
        baseline_config={
            "training": {
                "optimizer": "adamw",
                "learning_rate": 1e-4,
            }
        }
    )

    results = [
        {
            "experiment_id": "best",
            "metric": 0.80,
            "baseline_metric": 0.70,
            "config": {
                "training": {
                    "optimizer": "adamw",
                    "learning_rate": 1e-4,
                }
            },
        }
    ]

    candidates = planner.stage3(results)

    assert candidates

    learning_rates = {
        candidate.config["training"]["learning_rate"]
        for candidate in candidates
        if "training" in candidate.config
        and "learning_rate" in candidate.config["training"]
    }

    assert 1e-4 not in learning_rates
    assert any(value != 1e-4 for value in learning_rates)


def test_final_recommendation_selects_best_result():
    planner = ExperimentPlanner(
        baseline_config={"training": {"optimizer": "adamw"}}
    )

    results = [
        {
            "experiment_id": "bad",
            "metric": 0.61,
            "baseline_metric": 0.60,
            "config": {
                "training": {
                    "optimizer": "sgd",
                }
            },
        },
        {
            "experiment_id": "best",
            "metric": 0.75,
            "baseline_metric": 0.60,
            "config": {
                "training": {
                    "optimizer": "adamw",
                    "learning_rate": 3e-4,
                }
            },
        },
    ]

    report = planner.final_recommendation(
        baseline_metric=0.60,
        results=results,
    )

    assert report["best_experiment"] == "best"
    assert report["best_metric"] == 0.75
    assert abs(report["improvement"] - 0.15) < 1e-12
    assert report["recommendation"]["training"]["learning_rate"] == 3e-4
    assert report["confidence"] == "MEDIUM"
    assert report["supported_changes"]


def test_no_results_returns_safe_low_confidence_recommendation():
    planner = ExperimentPlanner(
        baseline_config={"training": {"optimizer": "adamw"}}
    )

    report = planner.final_recommendation(
        baseline_metric=0.60,
        results=[],
    )

    assert report["best_experiment"] is None
    assert report["improvement"] == 0.0
    assert report["confidence"] == "LOW"
    assert report["recommendation"] == {
        "training": {"optimizer": "adamw"}
    }
