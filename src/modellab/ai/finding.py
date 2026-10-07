THRESHOLD = 0.01
METRICS = ("f1", "precision", "recall", "mean_iou")
SINGLE_RUN = "single run, no significance test"
NO_CONTROL = (
    "compared with the baseline model, not with a default-settings control, so part of any change "
    "may come from retraining itself"
)
STATUS = {
    "improved": "consistent with the hypothesis",
    "no_meaningful_change": "not supported by this run",
    "worsened": "contradicted by this run",
}


def _out(verdict, target, summary, caveats, **extra):
    return {
        "verdict": verdict,
        "hypothesis_status": STATUS.get(verdict, "inconclusive"),
        "target_metric": target,
        "summary": summary,
        "caveats": caveats,
        **extra,
    }


def _control_trained(control):
    if not control or control.get("status") != "completed":
        return None
    comp = control.get("comparison") or {}
    return comp.get("trained") if comp.get("available") else None


def interpret(hypothesis, result, control=None, threshold=THRESHOLD):
    target = hypothesis["target_metric"]
    if result.get("status") != "completed":
        err = result.get("error") or {}
        return _out("failed", target, f"The experiment failed: {err.get('type')}: {err.get('message')}", [SINGLE_RUN])

    comp = result.get("comparison") or {}
    delta = comp.get("delta") or {}
    if not comp.get("available") or delta.get(target) is None:
        return _out("unavailable", target, f"No baseline comparison for {target} was recorded.", [SINGLE_RUN])

    base, trained = comp["baseline"], comp["trained"]
    ctrl = _control_trained(control)
    if ctrl is not None and ctrl.get(target) is not None:
        basis = "control"
        ref = ctrl
        diffs = {m: trained[m] - ctrl[m] for m in METRICS if trained.get(m) is not None and ctrl.get(m) is not None}
        label = "the default-settings control"
        caveats = [SINGLE_RUN]
    else:
        basis = "baseline"
        ref = base
        diffs = {m: delta[m] for m in METRICS if delta.get(m) is not None}
        label = "the baseline model"
        caveats = [SINGLE_RUN, NO_CONTROL]

    d = diffs[target]
    verdict = "improved" if d >= threshold else "worsened" if d <= -threshold else "no_meaningful_change"
    summary = f"{target} went from {ref[target]:.3f} to {trained[target]:.3f} ({d:+.3f}) against {label}."
    if basis == "control":
        summary += f" Against the baseline model the change is {delta[target]:+.3f}."
    dropped = [m for m in METRICS if m != target and diffs.get(m, 0) <= -threshold]
    if dropped:
        summary += " Also dropped: " + ", ".join(f"{m} ({diffs[m]:+.3f})" for m in dropped) + "."
    return _out(
        verdict, target, summary, caveats,
        basis=basis,
        reference={m: ref.get(m) for m in METRICS},
        trained={m: trained.get(m) for m in METRICS},
        delta=diffs,
        delta_vs_baseline={m: delta.get(m) for m in METRICS},
    )
