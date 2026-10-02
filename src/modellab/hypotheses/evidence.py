from modellab.hypotheses.config import HypothesisConfig
from modellab.hypotheses.schema import Evidence, Hypothesis

DISCLAIMER = (
    "Hypotheses are candidate explanations. Only controlled experiments can support or weaken them, "
    "and no hypothesis is a proven cause."
)


def strength_label(value: float) -> str:
    for limit, label in ((0.7, "strong"), (0.45, "moderate"), (0.2, "weak")):
        if value >= limit:
            return label
    return "very_weak"


def _stance(label: str, expected: str):
    wanted, opposite = ("improved", "degraded") if expected == "improve" else ("degraded", "improved")
    if label == wanted:
        return "supports", None
    if label == opposite:
        return "contradicts", None
    if label == "no_practical_difference":
        return "contradicts", 0.75
    if label == "detectable_below_practical_threshold":
        return "inconclusive", 0.25
    return "inconclusive", 0.0


def experiment_evidence(result: dict, test: dict, row: dict | None, cfg: HypothesisConfig) -> Evidence:
    if result["status"] != "completed":
        error = result.get("error", {})
        return Evidence(kind="experiment", stance="inconclusive", summary=f"experiment failed: {error.get('type')}",
                        strength=0.0, experiment_id=result["experiment_id"], test_id=test["test_id"])
    stats, verdict = result["statistics"]["affected"], result["verdict"]
    corrected = row is not None
    label = row["family_label"] if corrected else verdict["label"]
    stance, fixed = _stance(label, test["expected_effect"])
    strength = fixed if fixed is not None else (1.0 if corrected else 0.5)
    p = row["primary_p"] if corrected else verdict["primary_p"]
    data = {
        "label": label, "accuracy_difference": stats["accuracy_difference"], "ci_low": stats["ci_low"],
        "ci_high": stats["ci_high"], "primary_p": p, "corrected": corrected,
        "p_adjusted": row["p_benjamini_hochberg"] if corrected else None,
        "global_accuracy_difference": result["comparison"]["global"]["differences"]["accuracy"]["difference"],
        "n_affected": result["population"]["num_affected"], "expected_effect": test["expected_effect"],
    }
    summary = (f"accuracy changed by {stats['accuracy_difference']:+.4f} on {data['n_affected']} samples "
               f"(p={p:.3g}, {'corrected' if corrected else 'uncorrected'}): {label}")
    return Evidence(kind="experiment", stance=stance, summary=summary, strength=strength,
                    experiment_id=result["experiment_id"], test_id=test["test_id"], data=data)


def _finalize(h: Hypothesis, untestable: set) -> None:
    observed = [e for e in h.supporting_evidence if e.kind == "observation"]
    o = 0.4 * max((e.strength for e in observed), default=0.0)
    experiments = [e for group in (h.supporting_evidence, h.contradicting_evidence, h.inconclusive_evidence) for e in group if e.kind == "experiment"]
    support = sum(e.strength for e in h.supporting_evidence if e.kind == "experiment")
    contra = sum(e.strength for e in h.contradicting_evidence if e.kind == "experiment")
    h.evidence_strength = round(min(1.0, max(0.0, o + 0.5 * (support - contra) / (support + contra + 1))), 4)
    h.confidence_label = strength_label(h.evidence_strength)
    if not experiments:
        status = "untestable" if (h.hypothesis_id in untestable or not h.required_tests) else "untested"
    elif support > 0 and contra > 0:
        status = "conflicting"
    elif support > 0:
        status = "supported"
    elif contra >= 0.75:
        status = "refuted"
    elif contra > 0:
        status = "weakened"
    else:
        status = "inconclusive"
    if not h.status_history or h.status_history[-1]["status"] != status:
        h.status_history.append({"status": status, "experiments": len(experiments)})
    h.status = status


def update_hypotheses(hypotheses, plan: dict, results: list, family_report: dict | None, cfg: HypothesisConfig) -> list:
    rows = {r["experiment_id"]: r for r in (family_report or {}).get("experiments", [])}
    by_name = {r["spec"]["name"]: r for r in results}
    untestable = {u["hypothesis_id"] for u in plan.get("untestable_hypotheses", [])}
    out = []
    for source in hypotheses:
        h = source.model_copy(deep=True)
        for field in ("supporting_evidence", "contradicting_evidence", "inconclusive_evidence"):
            setattr(h, field, [e for e in getattr(h, field) if e.kind == "observation"])
        buckets = {"supports": h.supporting_evidence, "contradicts": h.contradicting_evidence, "inconclusive": h.inconclusive_evidence}
        for entry in plan.get("selected", []):
            for test in entry["tests"]:
                if test["hypothesis_id"] != h.hypothesis_id or entry["name"] not in by_name:
                    continue
                result = by_name[entry["name"]]
                evidence = experiment_evidence(result, test, rows.get(result["experiment_id"]), cfg)
                buckets[evidence.stance].append(evidence)
        _finalize(h, untestable)
        out.append(h)
    return out
