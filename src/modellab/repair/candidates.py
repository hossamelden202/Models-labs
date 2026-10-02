from modellab.experiments.spec import canonical_json, sha256_text
from modellab.repair.config import RepairConfig

PHOTOMETRIC_OPS = {"illumination": "brightness", "contrast": "contrast", "color_saturation": "saturation"}
RECOMMENDATION_RULES = {
    "detail_loss": ("targeted_data_selection", "augmentation_change", "hard_negative_selection"),
    "input_mode": ("targeted_data_selection", "augmentation_change"),
    "compression_format": ("targeted_data_selection", "augmentation_change"),
    "overprediction": ("class_balancing",),
    "class_bias": ("class_balancing", "hard_negative_selection"),
    "label_noise_or_shortcut": ("suspicious_sample_review",),
    "boundary_overlap": ("hard_negative_selection",),
}
SUGGESTIONS = {
    "augmentation_change": "add augmentation that reproduces the failing condition during training",
    "targeted_data_selection": "collect or upweight training samples that match the failing population",
    "hard_negative_selection": "mine hard negatives for the confused classes from the training data",
    "class_balancing": "rebalance or reweight the training data for the affected classes",
    "suspicious_sample_review": "review and relabel the listed high-confidence errors and flagged samples",
}


def target_population(failure: dict, ids: list, analysis_id: str | None) -> dict:
    ref = failure["population_ref"]
    if ref["kind"] == "slice" and analysis_id:
        return {"slice_ids": [ref["slice_id"]], "analysis_id": analysis_id}
    return {"sample_ids": ids}


def _cid(hypothesis_id, intervention, population_kind) -> str:
    return "cand_" + sha256_text(canonical_json({"h": hypothesis_id, "i": intervention, "p": population_kind}))[:10]


def build_candidates(inv: dict, analysis, ids_by_failure: dict, cfg: RepairConfig):
    hyps = {h["hypothesis_id"]: h for h in inv["hypotheses"]}
    failures = {f["failure_id"]: f for f in inv["failures"]}
    analysis_id = inv["inputs"]["analysis_id"]
    candidates, recommendations = [], []
    for conclusion in inv["conclusions"]:
        if conclusion["status"] != "supported_mechanism":
            continue
        h = hyps[conclusion["hypothesis_id"]]
        failure = failures[h["failure_id"]]
        ids = ids_by_failure[failure["failure_id"]]
        params, category = h["params"], h["category"]

        def add(kind, description, intervention, population_kind, index, rule, h=h, failure=failure, ids=ids):
            population = target_population(failure, ids, analysis_id) if population_kind == "target" else None
            candidates.append({
                "candidate_id": _cid(h["hypothesis_id"], intervention, population_kind), "hypothesis_id": h["hypothesis_id"],
                "failure_id": failure["failure_id"], "category": kind, "mechanism_category": h["category"], "grid_index": index,
                "description": description, "intervention": intervention, "population_kind": population_kind,
                "population": population, "deployment_rule": rule, "executable": True,
            })

        if category in PHOTOMETRIC_OPS:
            op = PHOTOMETRIC_OPS[category]
            factors = cfg.factors_low if params["direction"] == "low" else cfg.factors_high
            for i, factor in enumerate(factors):
                add("slice_input_transform", f"scale {op} by {factor} for inputs matching the failing slice",
                    {"kind": "image_transform", "op": op, "params": {"factor": factor}}, "target", i,
                    {"type": "slice_conditioned_input_transform", "condition": failure["conditions"], "op": op, "factor": factor,
                     "note": "the slice condition must be computable from the input at inference time"})
        elif category == "aspect_distortion" and params.get("change"):
            add("slice_preprocessing", "resize without cropping for inputs matching the failing slice",
                {"kind": "preprocessing", "changes": params["change"]}, "target", 0,
                {"type": "slice_conditioned_preprocessing", "condition": failure["conditions"], "changes": params["change"]})
        elif category == "overprediction":
            wrong, true = params["class"], params["dominant_true_class"]
            for i, weight in enumerate(cfg.weight_reduce):
                add("class_weighting", f"multiply the probability of {wrong} by {weight}",
                    {"kind": "inference", "class_weights": {wrong: weight}}, "all", i,
                    {"type": "class_weights", "weights": {wrong: weight}})
            for i, t in enumerate(cfg.thresholds):
                add("threshold", f"predict {true} instead of {wrong} when {wrong} has probability below {t}",
                    {"kind": "inference", "class_thresholds": {wrong: t}, "fallback_class": true}, "all", i,
                    {"type": "class_threshold", "thresholds": {wrong: t}, "fallback": true})
        elif category == "class_bias":
            true, wrong = params["class"], params["dominant_wrong_class"]
            for i, weight in enumerate(cfg.weight_boost):
                add("class_weighting", f"multiply the probability of {true} by {weight}",
                    {"kind": "inference", "class_weights": {true: weight}}, "all", i,
                    {"type": "class_weights", "weights": {true: weight}})
            for i, t in enumerate(cfg.thresholds):
                add("threshold", f"predict {true} instead of {wrong} when {wrong} has probability below {t}",
                    {"kind": "inference", "class_thresholds": {wrong: t}, "fallback_class": true}, "all", i,
                    {"type": "class_threshold", "thresholds": {wrong: t}, "fallback": true})

        samples = analysis.samples
        member = samples["sample_id"].isin(set(ids))
        errors = sorted(samples.loc[member & samples["is_error"].astype(bool), "sample_id"])
        confident = sorted(samples.loc[member & samples["is_high_confidence_error"].astype(bool), "sample_id"])
        for kind in RECOMMENDATION_RULES.get(category, ()):
            chosen = confident if kind == "suspicious_sample_review" else errors
            recommendations.append({
                "recommendation_id": "rec_" + sha256_text(canonical_json([h["hypothesis_id"], kind]))[:10],
                "hypothesis_id": h["hypothesis_id"], "failure_id": failure["failure_id"], "category": kind,
                "executable": False, "reason": "no training interface is available, so this is reported as a recommendation only",
                "suggestion": SUGGESTIONS[kind], "total_samples": len(chosen), "sample_ids": chosen[:50],
            })
        if category in ("detail_loss", "input_mode", "compression_format", "class_bias", "overprediction"):
            recommendations.append({
                "recommendation_id": "rec_" + sha256_text(canonical_json([h["hypothesis_id"], "fine_tuning"]))[:10],
                "hypothesis_id": h["hypothesis_id"], "failure_id": failure["failure_id"], "category": "limited_fine_tuning",
                "executable": False, "reason": "targeted fine-tuning is not supported by the available interfaces",
                "suggestion": "fine-tune on the failing population with a held-out regression suite", "total_samples": len(errors),
                "sample_ids": errors[:50],
            })
    candidates.sort(key=lambda c: (c["category"], c["grid_index"], c["candidate_id"]))
    return candidates, recommendations
