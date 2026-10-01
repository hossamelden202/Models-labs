import hashlib
import io
import json
import shutil
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import modellab
from modellab.analysis.config import FailureConfig
from modellab.analysis.records import (
    audit_features,
    build_samples,
    combine_features,
    derive_slice_features,
    normalize_predictions,
)
from modellab.analysis.slices import discover_slices, empty_slices, evidence_for
from modellab.analysis.stats import benjamini_hochberg, fisher_p
from modellab.core.errors import ArtifactError, EvaluationError
from modellab.core.results import ArtifactStore
from modellab.utils import get_logger

log = get_logger("analysis")

CATEGORIES = (
    "high_confidence_error", "ambiguous_error", "error",
    "low_confidence_correct", "ambiguous_correct", "correct",
)
LIMITATIONS = [
    "Slices are observed associations in this evaluation set, not causes of the errors.",
    "Slice p-values are two-sided Fisher exact tests of a slice's error rate against the rest "
    "of the population, corrected with Benjamini-Hochberg across all tested slices.",
    "Slices using predicted class, confidence, entropy or margin describe model behaviour, "
    "not properties of the data.",
    "Slices smaller than min_support, or leaving fewer than min_support samples outside, are "
    "not generated or tested.",
    "Audit flags only cover findings whose evidence list was not truncated.",
]


def _clean(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else float(v)
    return v


def _parquet(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def load_evaluation(root, evaluation_id: str):
    base = Path(root) / "evaluations" / evaluation_id
    if not (base / "predictions.parquet").is_file():
        raise EvaluationError(f"evaluation '{evaluation_id}' not found under {root}")
    predictions = pd.read_parquet(base / "predictions.parquet")
    meta = json.loads((base / "metadata.json").read_text())
    return predictions, list(meta["class_names"]), meta


def load_audit(root, audit_id: str):
    base = Path(root) / "audits" / audit_id
    if not (base / "records.parquet").is_file():
        raise EvaluationError(f"audit '{audit_id}' not found under {root}")
    return pd.read_parquet(base / "records.parquet"), json.loads((base / "report.json").read_text())


@dataclass
class FailureAnalysis:
    analysis_id: str | None
    directory: Path | None
    samples: pd.DataFrame
    slices: pd.DataFrame
    report: dict

    def slice_mask(self, slice_id: str) -> np.ndarray:
        row = self.slices[self.slices["slice_id"] == slice_id]
        if row.empty:
            raise KeyError(f"unknown slice '{slice_id}'")
        mask = np.ones(len(self.samples), dtype=bool)
        for feature, value in json.loads(row.iloc[0]["conditions"]):
            column = f"f:{feature}" if f"f:{feature}" in self.samples.columns else feature
            mask &= (self.samples[column].astype(str) == value).to_numpy()
        return mask

    def slice_sample_ids(self, slice_id: str) -> list[str]:
        return sorted(self.samples.loc[self.slice_mask(slice_id), "sample_id"])


def load_failure_analysis(store: ArtifactStore, analysis_id: str) -> FailureAnalysis:
    base = store.root / "failure_analyses" / analysis_id
    if not (base / "report.json").is_file():
        raise EvaluationError(f"failure analysis '{analysis_id}' not found")
    return FailureAnalysis(
        analysis_id=analysis_id,
        directory=base,
        samples=pd.read_parquet(base / "samples.parquet"),
        slices=pd.read_parquet(base / "slices.parquet"),
        report=json.loads((base / "report.json").read_text()),
    )


def _category_summary(samples, cfg):
    out = {}
    for name in CATEGORIES:
        ids = sorted(samples.loc[samples["failure_type"] == name, "sample_id"])
        out[name] = {
            "count": len(ids),
            "sample_ids": ids[: cfg.max_evidence],
            "truncated": len(ids) > cfg.max_evidence,
        }
    out["flags"] = {
        "errors": int(samples["is_error"].sum()),
        "high_confidence_errors": int(samples["is_high_confidence_error"].sum()),
        "low_confidence_correct": int(samples["is_low_confidence_correct"].sum()),
        "ambiguous": int(samples["is_ambiguous"].sum()),
    }
    return out


def _confusion(samples, class_names, cfg):
    k, n = len(class_names), len(samples)
    y = samples["true_label"].to_numpy(dtype=np.int64)
    p = samples["predicted_label"].to_numpy(dtype=np.int64)
    ids = samples["sample_id"].to_numpy()
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k) if n else np.zeros((k, k), dtype=int)
    errors = int((y != p).sum())

    per_class, tests = [], []
    for c, name in enumerate(class_names):
        support = int(cm[c].sum())
        tp = int(cm[c, c])
        fn = support - tp
        fp = int(cm[:, c].sum()) - tp
        predicted = tp + fp
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        if precision is None or recall is None:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class.append(
            {
                "class": name, "support": support, "true_positives": tp,
                "false_negatives": fn, "false_positives": fp, "precision": precision,
                "recall": recall, "f1": f1,
                "false_negative_rate": fn / support if support else None,
            }
        )
        rest_n, rest_e = n - support, errors - fn
        if k >= 2 and support >= cfg.min_support and rest_n >= cfg.min_support:
            tests.append((c, fisher_p(fn, support - fn, rest_e, rest_n - rest_e), fn, support, rest_e, rest_n))

    class_failures = []
    if tests:
        q = benjamini_hochberg([t[1] for t in tests])
        for (c, pv, fn, support, rest_e, rest_n), qv in zip(tests, q, strict=True):
            rd = fn / support - rest_e / rest_n
            mask = y == c
            wrong = Counter(class_names[j] for j in p[mask & (p != c)].tolist())
            class_failures.append(
                {
                    "class": class_names[c], "support": support, "errors": fn,
                    "error_rate": fn / support, "rest_error_rate": rest_e / rest_n,
                    "risk_difference": rd, "p_value": float(pv), "q_value": float(qv),
                    "evidence": evidence_for(float(pv), float(qv), rd, cfg),
                    "most_common_wrong_predictions": [
                        {"class": name, "count": count}
                        for name, count in sorted(wrong.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
                    ],
                }
            )

    pairs = []
    for a in range(k):
        for b in range(k):
            if a != b and cm[a, b] > 0:
                pairs.append((int(cm[a, b]), a, b))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    top = []
    for count, a, b in pairs[: cfg.top_confusions]:
        members = sorted(ids[(y == a) & (p == b)].tolist())
        top.append(
            {
                "true_class": class_names[a], "predicted_class": class_names[b], "count": count,
                "rate_of_true_class": count / int(cm[a].sum()),
                "share_of_errors": count / errors if errors else None,
                "sample_ids": members[: cfg.max_evidence],
            }
        )
    return {
        "labels": list(class_names), "rows": "true", "columns": "predicted",
        "matrix": cm.tolist(), "per_class": per_class, "top_confusions": top,
        "class_failures": class_failures,
    }


def _clusters(samples, probs, cfg):
    base = {"enabled": True, "method": "kmeans on probability vectors of failures", "seed": cfg.cluster_seed}
    if probs is None:
        return {**base, "available": False, "reason": "class probabilities unavailable", "clusters": []}
    err = samples["is_error"].to_numpy(dtype=bool)
    if int(err.sum()) < 2:
        return {**base, "available": False, "reason": "fewer than two failures", "clusters": []}

    from sklearn.cluster import KMeans

    x_err = probs[err]
    distinct = len(np.unique(np.round(x_err, 9), axis=0))
    k = min(cfg.n_clusters, distinct)
    if k < 2:
        raw_err = np.zeros(len(x_err), dtype=int)
        centers = x_err.mean(axis=0, keepdims=True)
        raw_all = np.zeros(len(probs), dtype=int)
    else:
        model = KMeans(n_clusters=k, n_init=10, random_state=cfg.cluster_seed).fit(x_err)
        raw_err, centers = model.labels_, model.cluster_centers_
        raw_all = model.predict(probs)

    ids = samples["sample_id"].to_numpy()[err]
    true_class = samples["true_class"].to_numpy()[err]
    pred_class = samples["predicted_class"].to_numpy()[err]
    conf = samples["confidence"].to_numpy()[err]
    errors_all = err

    order = sorted(
        set(raw_err.tolist()),
        key=lambda r: (-int((raw_err == r).sum()), min(ids[raw_err == r].tolist())),
    )
    clusters = []
    for new_id, raw in enumerate(order):
        m = raw_err == raw
        region = raw_all == raw

        def dominant(values, mask):
            counts = Counter(values.tolist())
            name, count = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            return name, count / int(mask.sum())

        t_name, t_share = dominant(true_class[m], m)
        p_name, p_share = dominant(pred_class[m], m)
        dist = np.linalg.norm(x_err[m] - centers[raw], axis=1)
        member_ids = ids[m]
        reps = sorted(zip(dist.tolist(), member_ids.tolist(), strict=True))[:5]
        clusters.append(
            {
                "cluster_id": new_id,
                "size": int(m.sum()),
                "region_size": int(region.sum()),
                "region_error_rate": float(errors_all[region].mean()) if region.any() else None,
                "dominant_true_class": t_name, "dominant_true_class_share": t_share,
                "dominant_predicted_class": p_name, "dominant_predicted_class_share": p_share,
                "mean_confidence": float(conf[m].mean()),
                "representative_sample_ids": [s for _, s in reps],
            }
        )
    return {**base, "available": True, "n_clusters": len(clusters), "clusters": clusters}


def _fingerprint(samples, cfg) -> str:
    columns = ["sample_id", "true_label", "predicted_label"] + sorted(
        c for c in samples.columns if c.startswith("f:")
    )
    body = b""
    if len(samples):
        body = pd.util.hash_pandas_object(samples[columns], index=False).to_numpy().tobytes()
    conf = np.round(samples["confidence"].to_numpy(dtype=float), 6).tobytes()
    config = json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(config + body + conf).hexdigest()[:16]


def render_summary(report: dict) -> str:
    inputs, overall = report["inputs"], report["overall"]
    lines = [
        "ModelLab failure analysis",
        f"samples: {inputs['n_samples']}   classes: {', '.join(inputs['class_names'])}   "
        f"fingerprint: {inputs['fingerprint']}",
        f"accuracy: {overall['accuracy']}   errors: {overall['n_errors']}",
        "",
        "failure categories:",
    ]
    for name in CATEGORIES:
        lines.append(f"  {name}: {report['failure_categories'][name]['count']}")
    lines += ["", "top confusions:"]
    for item in report["confusion"]["top_confusions"]:
        lines.append(f"  {item['true_class']} -> {item['predicted_class']}: {item['count']}")
    flagged = [c for c in report["confusion"]["class_failures"] if c["evidence"] == "statistically_supported"]
    lines += ["", "classes with a statistically supported higher error rate:"]
    lines += [f"  {c['class']}: error rate {c['error_rate']:.3f} (q={c['q_value']:.2g})" for c in flagged] or ["  none"]
    lines += ["", "top slices (associations, not causes):"]
    for s in report["slices"]["top"][:10]:
        lines.append(
            f"  [{s['evidence']}] {s['slice_id']}: support {s['support']}, "
            f"error rate {s['error_rate']}, q={s['q_value']}"
        )
    if not report["slices"]["top"]:
        lines.append("  none")
    if report["clusters"] and report["clusters"].get("clusters"):
        lines += ["", "failure clusters:"]
        for c in report["clusters"]["clusters"]:
            lines.append(
                f"  #{c['cluster_id']}: {c['size']} failures, mostly {c['dominant_true_class']} "
                f"predicted as {c['dominant_predicted_class']}, region error rate {c['region_error_rate']}"
            )
    if report["warnings"]:
        lines += ["", "warnings:"] + [f"  {w}" for w in report["warnings"]]
    lines += ["", "limitations:"] + [f"  - {text}" for text in report["limitations"]] + [""]
    return "\n".join(lines)


def analyze_failures(
    predictions,
    class_names,
    config: FailureConfig | None = None,
    audit_records=None,
    audit_report=None,
    extra_features=None,
    store: ArtifactStore | None = None,
    analysis_id: str | None = None,
    evaluation_id: str | None = None,
    audit_id: str | None = None,
) -> FailureAnalysis:
    cfg = config or FailureConfig()
    names = [str(c) for c in class_names]
    started, clock = datetime.now(timezone.utc), time.monotonic()

    df, probs = normalize_predictions(predictions, names)
    frames = []
    if audit_records is not None:
        frames.append(audit_features(audit_records, audit_report))
    if extra_features is not None:
        frames.append(extra_features)
    features = combine_features(frames)
    samples = build_samples(df, probs, names, cfg, features)
    n = len(samples)

    warnings = []
    coverage = None
    if n == 0:
        warnings.append("no samples")
    elif probs is None:
        warnings.append("class probabilities unavailable; margin, entropy and clusters are not computed")
    if features is not None and n:
        coverage = float(samples["sample_id"].isin(features["sample_id"]).mean())
        if coverage < 1.0:
            warnings.append(f"features were found for {coverage:.1%} of samples; the rest are '<missing>'")

    labels, info, skipped = derive_slice_features(samples, cfg)
    for feature in labels.columns:
        samples[f"f:{feature}"] = labels[feature].to_numpy()

    if n and labels.shape[1]:
        slices, slice_info = discover_slices(samples, labels, names, cfg)
    else:
        slices = empty_slices()
        slice_info = {
            "num_candidates_generated": 0, "num_pruned_min_support": 0, "num_duplicates_dropped": 0,
            "num_tested": 0, "truncated": False, "correction": "benjamini_hochberg",
            "alpha": cfg.alpha, "max_depth": cfg.max_depth,
        }

    confusion = _confusion(samples, names, cfg)
    clusters = _clusters(samples, probs, cfg) if cfg.clusters else None
    errors = int(samples["is_error"].sum())
    tested = slices[slices["tested"].astype(bool)] if len(slices) else slices
    top = []
    for row in tested.sort_values("rank").head(cfg.max_listed_slices).to_dict("records"):
        top.append({k: _clean(row[k]) for k in (
            "slice_id", "slice_type", "support", "errors", "error_rate", "risk_difference",
            "p_value", "q_value", "evidence", "uses_model_output")})

    report = {
        "schema_version": 1,
        "config": cfg.model_dump(mode="json"),
        "inputs": {
            "evaluation_id": evaluation_id, "audit_id": audit_id, "n_samples": n,
            "n_classes": len(names), "class_names": names, "join_coverage": coverage,
            "has_probabilities": probs is not None, "fingerprint": _fingerprint(samples, cfg),
        },
        "overall": {
            "accuracy": 1.0 - errors / n if n else None,
            "n_errors": errors,
            "mean_confidence": float(samples["confidence"].mean()) if n else None,
        },
        "failure_categories": _category_summary(samples, cfg),
        "confusion": confusion,
        "slices": {
            "info": slice_info,
            "features": info,
            "skipped_features": skipped,
            "top": top,
        },
        "clusters": clusters,
        "warnings": warnings,
        "limitations": LIMITATIONS,
    }

    directory = None
    if store is not None:
        analysis_id = analysis_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
        if Path(analysis_id).name != analysis_id or analysis_id in (".", ".."):
            raise ArtifactError(f"invalid analysis id: {analysis_id!r}")
        relative = Path("failure_analyses") / analysis_id
        directory = store.resolve(relative)
        if directory.exists():
            raise ArtifactError(f"analysis '{analysis_id}' already exists in {store.root}")
        directory.mkdir(parents=True)
        metadata = {
            "analysis_id": analysis_id, "timestamp": started.isoformat(),
            "duration_seconds": round(time.monotonic() - clock, 3),
            "versions": {"modellab": modellab.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        }
        try:
            store.write_json(relative / "report.json", report)
            store.write_text(relative / "summary.txt", render_summary(report))
            store.write_json(relative / "confusion.json", confusion)
            if clusters is not None:
                store.write_json(relative / "clusters.json", clusters)
            store.write_bytes(relative / "samples.parquet", _parquet(samples))
            store.write_bytes(relative / "slices.parquet", _parquet(slices))
            store.write_json(relative / "metadata.json", metadata)
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
    return FailureAnalysis(analysis_id=analysis_id, directory=directory, samples=samples, slices=slices, report=report)
