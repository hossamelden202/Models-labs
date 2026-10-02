import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from modellab.experiments.spec import canonical_json, sha256_text
from modellab.hypotheses.config import HypothesisConfig
from modellab.hypotheses.schema import Evidence, ExperimentTest, FailureTarget, Hypothesis

TIER_STRENGTH = {
    "statistically_supported": 1.0, "significant_small_effect": 0.5,
    "exploratory": 0.25, "descriptive": 0.5,
}
PHOTOMETRIC = {
    "brightness": ("illumination", "brightness"),
    "contrast": ("contrast", "contrast"),
    "saturation": ("color_saturation", "saturation"),
}
GEOMETRY = frozenset({"width", "height", "resolution", "file_size", "aspect_ratio"})
LABEL_FLAGS = (
    "label_integrity.conflicting_labels_identical_content",
    "label_integrity.duplicate_sample_ids",
)


@dataclass
class Failure:
    target: FailureTarget
    ids: list


@dataclass
class Context:
    samples: pd.DataFrame
    preprocess: dict | None
    flagged: dict


def audit_flag_ids(report) -> dict:
    out = {}
    for finding in (report or {}).get("findings", []):
        if finding.get("sample_ids") and not finding.get("evidence_truncated"):
            out[finding["finding_id"]] = set(finding["sample_ids"])
    return out


def _fid(kind, **parts) -> str:
    return "fail_" + sha256_text(canonical_json({"kind": kind, **parts}))[:10]


def _hid(failure_id, rule, params) -> str:
    return "hyp_" + sha256_text(canonical_json({"f": failure_id, "r": rule, "p": params}))[:10]


def _tid(failure_id, intervention, population) -> str:
    return "test_" + sha256_text(canonical_json({"f": failure_id, "i": intervention, "p": population}))[:10]


def population_ids(analysis, ref: dict) -> list:
    if ref["kind"] == "slice":
        return analysis.slice_sample_ids(ref["slice_id"])
    name = ref["class"] if ref["kind"] == "class" else ref["true_class"]
    return sorted(analysis.samples.loc[analysis.samples["true_class"] == name, "sample_id"])


def select_failures(analysis, cfg: HypothesisConfig) -> list:
    samples, report, slices = analysis.samples, analysis.report, analysis.slices
    chosen: list[Failure] = []
    sets: list[set] = []

    def add(kind, ids, description, tier, ref, slice_id=None, class_name=None, pair=None, conditions=None):
        if len(chosen) >= cfg.max_failures or not ids:
            return
        members = set(ids)
        if any(len(members & other) / len(members) >= cfg.overlap_skip for other in sets):
            return
        errors = int(samples.loc[samples["sample_id"].isin(members), "is_error"].sum())
        if errors < cfg.min_failure_errors:
            return
        rate = errors / len(members)
        target = FailureTarget(
            failure_id=_fid(kind, slice_id=slice_id, class_name=class_name, pair=pair),
            kind=kind, description=description, slice_id=slice_id, class_name=class_name,
            pair=pair, conditions=conditions or [], support=len(members), errors=errors,
            error_rate=round(rate, 6), baseline_accuracy=round(1 - rate, 6), tier=tier,
            population_ref=ref,
        )
        chosen.append(Failure(target, sorted(members)))
        sets.append(members)

    if len(slices):
        tested = slices[slices["tested"].astype(bool)]
        keep = (
            (tested["evidence"] == "statistically_supported")
            & ~tested["uses_model_output"].astype(bool)
            & (tested["risk_difference"] > 0)
            & (tested["n_conditions"] <= cfg.max_slice_conditions)
        )
        for row in tested[keep].sort_values("rank").itertuples():
            add(
                "slice", analysis.slice_sample_ids(row.slice_id),
                f"slice {row.slice_id}: error rate {row.error_rate:.3f} versus {row.complement_error_rate:.3f} elsewhere",
                row.evidence, {"kind": "slice", "slice_id": row.slice_id, "analysis_id": analysis.analysis_id},
                slice_id=row.slice_id, conditions=json.loads(row.conditions),
            )
    if cfg.include_classes:
        for item in report["confusion"]["class_failures"]:
            if item["evidence"] != "statistically_supported":
                continue
            ref = {"kind": "class", "class": item["class"]}
            add(
                "class", population_ids(analysis, ref),
                f"class {item['class']}: error rate {item['error_rate']:.3f} versus {item['rest_error_rate']:.3f} for other classes",
                item["evidence"], ref, class_name=item["class"],
            )
    if cfg.include_confusions:
        for item in report["confusion"]["top_confusions"]:
            if item["count"] < cfg.min_confusion_count:
                continue
            ref = {"kind": "confusion", "true_class": item["true_class"], "predicted_class": item["predicted_class"]}
            add(
                "confusion", population_ids(analysis, ref),
                f"{item['true_class']} predicted as {item['predicted_class']} ({item['count']} samples)",
                "descriptive", ref, pair=[item["true_class"], item["predicted_class"]],
            )
    return chosen


def _obs(summary, strength, **data):
    return Evidence(kind="observation", stance="supports", summary=summary, strength=float(min(1.0, max(0.0, strength))), data=data)


def generate_hypotheses(failure: Failure, ctx: Context, cfg: HypothesisConfig) -> list:
    t = failure.target
    samples = ctx.samples
    sub = samples[samples["sample_id"].isin(set(failure.ids))]
    errs = sub[sub["is_error"].astype(bool)]
    base = TIER_STRENGTH.get(t.tier, 0.25)
    out: dict[str, Hypothesis] = {}

    def add(rule, mechanism, params, tests, observations, note=""):
        hid = _hid(t.failure_id, rule, params)
        if hid not in out:
            out[hid] = Hypothesis(
                hypothesis_id=hid, failure_id=t.failure_id, rule=rule, category=rule,
                mechanism=mechanism, params=params, required_tests=tests,
                supporting_evidence=observations, note=note,
            )

    def test(intervention, population, expected, description):
        return ExperimentTest(
            test_id=_tid(t.failure_id, intervention, population), description=description,
            intervention=intervention, population=population, expected_effect=expected,
        )

    for feature, value in t.conditions:
        if feature in PHOTOMETRIC or feature in GEOMETRY:
            if feature not in samples.columns:
                continue
            mu_s, mu_p, sd = float(sub[feature].mean()), float(samples[feature].mean()), float(samples[feature].std())
            if not (np.isfinite(mu_s) and np.isfinite(sd) and sd > 0):
                continue
            effect = (mu_s - mu_p) / sd
            if abs(effect) < 0.25:
                continue
            direction = "low" if effect < 0 else "high"
            where = f"mean {feature} {mu_s:.4g} in the failing samples versus {mu_p:.4g} overall"
            observation = _obs(f"{where} ({effect:+.2f} standard deviations)", base, feature=feature, effect=round(effect, 4))
            if feature in PHOTOMETRIC:
                rule, op = PHOTOMETRIC[feature][0], PHOTOMETRIC[feature][1]
                factor = cfg.remedy_factor_low if direction == "low" else cfg.remedy_factor_high
                iv = {"kind": "image_transform", "op": op, "params": {"factor": factor}}
                add(
                    rule, f"{op} of the failing images is {direction} and the model may be sensitive to it",
                    {"feature": feature, "direction": direction, "factor": factor},
                    [test(iv, "failure", "improve", f"scale {op} by {factor} on the failing samples")], [observation],
                )
            elif feature == "aspect_ratio" and direction == "high":
                pre = ctx.preprocess or {}
                crop = pre.get("center_crop")
                tests = []
                if crop and pre.get("resize") != list(crop):
                    iv = {"kind": "preprocessing", "changes": {"resize": list(crop)}}
                    tests = [test(iv, "failure", "improve", f"resize to {list(crop)} without cropping on the failing samples")]
                add(
                    "aspect_distortion", "extreme aspect ratios are cropped or distorted by the preprocessing",
                    {"feature": feature, "direction": direction}, tests, [observation],
                    note="" if tests else "the baseline preprocessing has no center crop to compare against",
                )
            elif feature != "aspect_ratio" and direction == "low":
                blur = {"kind": "image_transform", "op": "blur", "params": {"radius": cfg.induce_blur_radius}}
                jpeg = {"kind": "image_transform", "op": "jpeg", "params": {"quality": cfg.induce_jpeg_quality}}
                add(
                    "detail_loss", "small or heavily compressed images carry less detail than the model needs",
                    {"feature": feature, "direction": direction},
                    [test(blur, "control", "degrade", f"blur correctly classified samples (radius {cfg.induce_blur_radius})"),
                     test(jpeg, "control", "degrade", f"recompress correctly classified samples (quality {cfg.induce_jpeg_quality})")],
                    [observation],
                )
        elif feature == "mode" and value != "RGB":
            tests = []
            if value == "L":
                iv = {"kind": "image_transform", "op": "grayscale", "params": {}}
                tests = [test(iv, "control", "degrade", "convert correctly classified RGB samples to grayscale")]
            add(
                "input_mode", f"{value} images differ from the RGB inputs the model expects", {"feature": "mode", "value": value},
                tests, [_obs(f"failing samples are in {value} mode", base, feature="mode", value=value)],
                note="" if tests else "no available intervention reproduces this input mode",
            )
        elif feature == "format" and value in ("JPEG", "WEBP"):
            iv = {"kind": "image_transform", "op": "jpeg", "params": {"quality": cfg.induce_jpeg_quality}}
            add(
                "compression_format", f"{value} compression artifacts reduce accuracy", {"feature": "format", "value": value},
                [test(iv, "control", "degrade", "recompress correctly classified samples")],
                [_obs(f"failing samples are {value} files", base, feature="format", value=value)],
            )
        elif feature == "split":
            add(
                "split_shift", f"the {value} split differs in distribution from the rest", {"feature": "split", "value": value},
                [], [_obs(f"failures concentrate in split {value}", base, feature="split", value=value)],
                note="no intervention can test a distribution difference between splits",
            )
        elif feature.startswith("audit_flag:") or feature == "has_exact_duplicate":
            add(
                "data_quality_flag", f"samples flagged by the audit ({feature}) are unreliable", {"feature": feature},
                [], [_obs(f"failures concentrate in samples with {feature}", base, feature=feature)],
                note="repairing data quality is outside the available interventions",
            )

    n_err = len(errs)
    if n_err:
        wrong = sorted(errs["predicted_class"].value_counts().items(), key=lambda kv: (-kv[1], kv[0]))
        true_counts = sorted(errs["true_class"].value_counts().items(), key=lambda kv: (-kv[1], kv[0]))
        top_wrong, share = wrong[0][0], wrong[0][1] / n_err
        top_true = true_counts[0][0]
        if share >= cfg.dominant_prediction_share:
            iv = {"kind": "inference", "class_weights": {top_wrong: cfg.overpredict_weight}}
            add(
                "overprediction", f"the model over-predicts {top_wrong} on this population",
                {"class": top_wrong, "share": round(share, 4), "dominant_true_class": top_true},
                [test(iv, "failure", "improve", f"down-weight {top_wrong} by {cfg.overpredict_weight} on the failing samples")],
                [_obs(f"{share:.0%} of the errors are predicted as {top_wrong}", share, wrong_class=top_wrong)],
            )
        if t.kind in ("class", "confusion"):
            true_class = t.class_name or t.pair[0]
            wrong_class = t.pair[1] if t.pair else top_wrong
            iv = {"kind": "inference", "class_weights": {true_class: cfg.bias_boost_weight}}
            add(
                "class_bias", f"the decision rule under-weights class {true_class}",
                {"class": true_class, "dominant_wrong_class": wrong_class},
                [test(iv, "failure", "improve", f"boost {true_class} by {cfg.bias_boost_weight} on its samples")],
                [_obs(f"class {true_class} is missed at error rate {t.error_rate:.3f}", base, true_class=true_class)],
            )
        ambiguous = float(errs["is_ambiguous"].astype(bool).mean())
        if ambiguous >= cfg.ambiguous_share:
            add(
                "boundary_overlap", "errors lie close to the decision boundary between classes", {"share": round(ambiguous, 4)},
                [], [_obs(f"{ambiguous:.0%} of the errors have a small top-2 margin", ambiguous)],
                note="no available intervention separates boundary overlap from other causes",
            )
        confident = float(errs["is_high_confidence_error"].astype(bool).mean())
        if confident >= cfg.high_confidence_share:
            evidence = [_obs(f"{confident:.0%} of the errors are high-confidence", confident)]
            flagged = set(errs["sample_id"]) & set().union(*(ctx.flagged.get(k, set()) for k in LABEL_FLAGS))
            if flagged:
                evidence.append(_obs(f"{len(flagged)} failing samples are flagged by the audit as label conflicts or duplicates", len(flagged) / n_err, flagged=len(flagged)))
            add(
                "label_noise_or_shortcut", "confidently wrong predictions suggest label noise or a shortcut feature",
                {"share": round(confident, 4)}, [], evidence,
                note="relabelling or shortcut removal needs a training interface",
            )

    hypotheses = sorted(out.values(), key=lambda h: h.hypothesis_id)
    for h in hypotheses:
        h.competes_with = sorted(o.hypothesis_id for o in hypotheses if o.hypothesis_id != h.hypothesis_id and o.category != h.category)
    return hypotheses
