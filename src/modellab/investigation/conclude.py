def _experimental(h) -> list:
    return [e for group in (h.supporting_evidence, h.contradicting_evidence, h.inconclusive_evidence) for e in group if e.kind == "experiment"]


def conclude(failure_id: str, hypotheses: list, cfg) -> dict:
    ranked = sorted(hypotheses, key=lambda h: (-h.evidence_strength, h.hypothesis_id))
    if not ranked:
        return {"failure_id": failure_id, "status": "insufficient_evidence", "hypothesis_id": None, "mechanism": None,
                "strength": None, "alternatives": [], "rule_checks": [], "statement": "No hypothesis was generated for this failure."}
    top, rest = ranked[0], ranked[1:]
    second = rest[0] if rest else None
    corrected = [e for e in top.supporting_evidence if e.kind == "experiment" and e.data.get("corrected") and e.strength >= 1.0]
    competitors = [h for h in rest if h.hypothesis_id in top.competes_with]
    checks = [
        ("top_hypothesis_supported", top.status == "supported", top.status),
        ("minimum_strength", top.evidence_strength >= cfg.min_strength, f"{top.evidence_strength:.2f} against {cfg.min_strength}"),
        ("corrected_experimental_support", len(corrected) >= cfg.min_supporting_experiments, f"{len(corrected)} corrected supporting experiments"),
        ("no_contradicting_experiments", not any(e.kind == "experiment" for e in top.contradicting_evidence), "contradicting experiments counted"),
        ("margin_over_runner_up", second is None or top.evidence_strength - second.evidence_strength >= cfg.min_margin,
         "no runner-up" if second is None else f"{top.evidence_strength - second.evidence_strength:.2f} against {cfg.min_margin}"),
        ("no_competing_supported_hypothesis", not any(h.status == "supported" for h in rest), "other supported hypotheses counted"),
        ("alternative_tested", (not cfg.require_alternative_tested) or not competitors or any(_experimental(h) for h in competitors),
         f"{len(competitors)} competing hypotheses"),
    ]
    rule_checks = [{"rule": n, "passed": bool(p), "detail": d} for n, p, d in checks]
    any_experiment = any(_experimental(h) for h in ranked)
    if all(c["passed"] for c in rule_checks):
        status = "supported_mechanism"
        statement = (f"Most supported mechanism: {top.mechanism}. This is supported by controlled experiments on this model "
                     "and dataset; it is not a proven cause.")
    elif not any_experiment:
        status = "insufficient_evidence"
        statement = "No experiment produced evidence for this failure; no mechanism is concluded."
    else:
        status = "inconclusive"
        failed = ", ".join(c["rule"] for c in rule_checks if not c["passed"])
        statement = f"The evidence does not meet the conclusion rules (failed: {failed}); no mechanism is concluded."
    return {
        "failure_id": failure_id, "status": status,
        "hypothesis_id": top.hypothesis_id if status == "supported_mechanism" else None,
        "mechanism": top.mechanism if status == "supported_mechanism" else None,
        "category": top.category if status == "supported_mechanism" else None,
        "strength": top.evidence_strength,
        "leading_hypothesis_id": top.hypothesis_id,
        "alternatives": [{"hypothesis_id": h.hypothesis_id, "category": h.category, "status": h.status, "strength": h.evidence_strength} for h in rest],
        "rule_checks": rule_checks, "statement": statement,
    }
