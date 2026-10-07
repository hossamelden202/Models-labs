import pytest

from modellab.ai.finding import interpret

HYP = {"target_metric": "recall"}


def result(delta, base=None, trained=None, status="completed", available=True):
    base = base or {"f1": 0.16, "precision": 0.165, "recall": 0.155, "mean_iou": 0.75}
    trained = trained or {k: v + delta.get(k, 0) for k, v in base.items()}
    return {"status": status, "comparison": {"available": available, "baseline": base, "trained": trained, "delta": delta}}


def test_improved():
    out = interpret(HYP, result({"recall": 0.155, "f1": 0.11, "precision": 0.07, "mean_iou": 0.1}))
    assert out["verdict"] == "improved"
    assert out["hypothesis_status"] == "consistent with the hypothesis"
    assert "0.155 to 0.310" in out["summary"] and "+0.155" in out["summary"]
    assert len(out["caveats"]) == 2


def test_worsened_and_other_drops_reported():
    out = interpret(HYP, result({"recall": -0.05, "f1": -0.03, "precision": 0.02, "mean_iou": 0.0}))
    assert out["verdict"] == "worsened"
    assert "f1 (-0.030)" in out["summary"] and "precision" not in out["summary"].split("Also dropped")[1]


def test_no_meaningful_change_inside_threshold():
    assert interpret(HYP, result({"recall": 0.005}))["verdict"] == "no_meaningful_change"
    assert interpret(HYP, result({"recall": 0.01}))["verdict"] == "improved"
    assert interpret(HYP, result({"recall": -0.01}))["verdict"] == "worsened"


def test_failed_and_unavailable():
    failed = interpret(HYP, {"status": "failed", "error": {"type": "OOM", "message": "cuda out of memory"}})
    assert failed["verdict"] == "failed" and "cuda out of memory" in failed["summary"]
    assert interpret(HYP, result({"recall": 0.1}, available=False))["verdict"] == "unavailable"
    assert interpret(HYP, result({"f1": 0.1}))["verdict"] == "unavailable"


def control(trained, status="completed", available=True):
    return {"status": status, "comparison": {"available": available, "trained": trained}}


def test_control_is_the_reference_when_present():
    res = result({"recall": 0.245, "f1": 0.14, "precision": 0.085, "mean_iou": 0.15},
                 trained={"f1": 0.30, "precision": 0.25, "recall": 0.40, "mean_iou": 0.90})
    out = interpret(HYP, res, control({"f1": 0.25, "precision": 0.20, "recall": 0.30, "mean_iou": 0.85}))
    assert out["basis"] == "control" and out["verdict"] == "improved"
    assert "0.300 to 0.400 (+0.100)" in out["summary"]
    assert "Against the baseline model the change is +0.245" in out["summary"]
    assert out["caveats"] == ["single run, no significance test"]
    assert out["delta"]["recall"] == pytest.approx(0.10)
    assert out["delta_vs_baseline"]["recall"] == 0.245


def test_gain_that_the_control_explains_is_not_called_an_improvement():
    res = result({"recall": 0.15}, trained={"f1": 0.3, "precision": 0.2, "recall": 0.305, "mean_iou": 0.8})
    out = interpret(HYP, res, control({"f1": 0.3, "precision": 0.2, "recall": 0.30, "mean_iou": 0.8}))
    assert out["verdict"] == "no_meaningful_change"
    assert interpret(HYP, res)["verdict"] == "improved"


def test_unusable_control_falls_back_to_baseline():
    res = result({"recall": 0.15})
    for bad in (None, control({}, status="failed"), control({}, available=False), control({"f1": 0.1})):
        out = interpret(HYP, res, bad)
        assert out["basis"] == "baseline" and len(out["caveats"]) == 2


def test_failed_candidate_ignores_control():
    out = interpret(HYP, {"status": "failed", "error": {"type": "OOM", "message": "x"}}, control({"recall": 0.3}))
    assert out["verdict"] == "failed"
