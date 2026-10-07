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
