from modellab.ai.catalog import build_menu
from modellab.ai.ranking import rank_candidates


def exp(eid, changes, delta=None, status="completed"):
    return {"experiment_id": eid, "status": status, "changes": changes, "delta": delta or {}}


RECALL = {"primary_target": "recall"}
PRECISION = {"primary_target": "precision"}


def ids(ranked):
    return [r.candidate_id for r in ranked]


def by_id(ranked):
    return {r.candidate_id: r for r in ranked}


def test_every_candidate_gets_a_reason():
    ranked = rank_candidates(build_menu(), RECALL, [])
    assert len(ranked) == len(build_menu())
    assert all(r.reasons for r in ranked)


def test_target_match_lifts_candidates():
    r = by_id(rank_candidates(build_menu(), RECALL, []))
    assert r["res_416"].score == 1.0 and r["aug_geometry"].score == 1.0
    assert r["aug_color"].score == 0.0
    p = by_id(rank_candidates(build_menu(), PRECISION, []))
    assert p["aug_color"].score == 1.0 and p["res_416"].score == 0.0


def test_short_past_runs_favour_more_epochs():
    r = rank_candidates(build_menu(), RECALL, [exp("exp_a", {"epochs": 1})])
    assert ids(r)[0] == "epochs_30"
    assert "trained 1 epoch, this trains 30" in by_id(r)["epochs_30"].reasons[0]


def test_no_epoch_boost_without_history_or_after_long_runs():
    assert by_id(rank_candidates(build_menu(), RECALL, []))["epochs_30"].score == 0.0
    long_run = rank_candidates(build_menu(), RECALL, [exp("exp_a", {"epochs": 50})])
    assert by_id(long_run)["epochs_30"].score == 0.0
    default_run = by_id(rank_candidates(build_menu(), RECALL, [exp("exp_a", {})]))["epochs_30"]
    assert default_run.score == 1.5 and "trained 10 epochs" in default_run.reasons[0]


def test_failed_runs_do_not_count():
    r = rank_candidates(build_menu(), RECALL, [exp("exp_a", {"epochs": 1}, status="failed")])
    assert by_id(r)["epochs_30"].score == 0.0


def test_same_setting_that_helped_adds_evidence_and_level():
    past = [exp("exp_a", {"epochs": 1}, {"f1": 0.11})]
    r = by_id(rank_candidates(build_menu(), RECALL, past))
    assert r["epochs_30"].score == 2.0 and r["epochs_30"].level == "medium"
    assert any("exp_a" in reason for reason in r["epochs_30"].reasons)
    assert r["res_416"].level == "low"


def test_same_setting_that_hurt_adds_nothing():
    past = [exp("exp_a", {"epochs": 1}, {"f1": -0.05})]
    assert by_id(rank_candidates(build_menu(), RECALL, past))["epochs_30"].score == 1.5


def test_costlier_resolution_ranks_below_cheaper_one():
    r = rank_candidates(build_menu(), RECALL, [])
    assert ids(r).index("res_416") < ids(r).index("res_512")


def test_ties_keep_menu_order():
    r = ids(rank_candidates(build_menu(), RECALL, []))
    assert r.index("res_416") < r.index("aug_geometry")
