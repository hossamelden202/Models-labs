THRESHOLD = 0.01
METRICS = ("f1", "precision", "recall", "mean_iou")
CAVEATS = [
    "single run, no significance test",
    "compared with the baseline model, not with a default-settings control, so part of any change may come from retraining itself",
]
STATUS = {
    "improved": "consistent with the hypothesis",
    "no_meaningful_change": "not supported by this run",
    "worsened": "contradicted by this run",
}


def _out(verdict, target, summary, **extra):
    return {
        "verdict": verdict,
        "hypothesis_status": STATUS.get(verdict, "inconclusive"),
        "target_metric": target,
        "summary": summary,
        "caveats": CAVEATS,
        **extra,
    }


def interpret(hypothesis, result, threshold=THRESHOLD):
    target = hypothesis["target_metric"]
    if result.get("status") != "completed":
        err = result.get("error") or {}
        return _out("failed", target, f"The experiment failed: {err.get('type')}: {err.get('message')}")

    comp = result.get("comparison") or {}
    delta = comp.get("delta") or {}
    if not comp.get("available") or delta.get(target) is None:
        return _out("unavailable", target, f"No baseline comparison for {target} was recorded.")

    base, trained = comp["baseline"], comp["trained"]
    d = delta[target]
    verdict = "improved" if d >= threshold else "worsened" if d <= -threshold else "no_meaningful_change"
    summary = f"{target} went from {base[target]:.3f} to {trained[target]:.3f} ({d:+.3f}) against the baseline model."
    dropped = [m for m in METRICS if m != target and (delta.get(m) or 0) <= -threshold]
    if dropped:
        summary += " Also dropped: " + ", ".join(f"{m} ({delta[m]:+.3f})" for m in dropped) + "."
    return _out(
        verdict, target, summary,
        baseline={m: base.get(m) for m in METRICS},
        trained={m: trained.get(m) for m in METRICS},
        delta={m: delta.get(m) for m in METRICS},
    )
