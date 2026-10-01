import numpy as np
import pandas as pd

from modellab.analysis.config import FailureConfig
from modellab.core.errors import EvaluationError

REQUIRED = ("sample_id", "true_label", "predicted_label", "confidence")
MODEL_OUTPUT = frozenset({"predicted_class", "confidence", "entropy", "margin"})
TARGET_COLUMNS = frozenset(
    {
        "sample_id", "true_label", "predicted_label", "correct", "is_error",
        "is_high_confidence_error", "is_low_confidence_correct", "is_ambiguous",
        "failure_type", "false_negative_for", "false_positive_for",
    }
)
AUDIT_COLUMNS = (
    "split", "status", "format", "mode", "extension", "width", "height",
    "file_size", "brightness", "contrast", "saturation",
)
MAX_AUDIT_FLAGS = 30


def normalize_predictions(predictions: pd.DataFrame, class_names):
    k = len(class_names)
    if k < 1:
        raise EvaluationError("class_names must not be empty")
    missing = [c for c in REQUIRED if c not in predictions.columns]
    if missing:
        raise EvaluationError(f"predictions are missing columns: {missing}")

    df = predictions.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    if df["sample_id"].duplicated().any():
        raise EvaluationError("duplicate sample ids in predictions")
    df = df.sort_values("sample_id", kind="mergesort").reset_index(drop=True)

    for column in ("true_label", "predicted_label"):
        values = df[column].to_numpy(dtype=np.int64)
        if len(values) and (values.min() < 0 or values.max() >= k):
            raise EvaluationError(f"{column} has values outside [0, {k})")
        df[column] = values
    conf = df["confidence"].to_numpy(dtype=np.float64)
    if len(conf) and (np.isnan(conf).any() or conf.min() < 0 or conf.max() > 1 + 1e-6):
        raise EvaluationError("confidence must lie in [0, 1]")

    probs = None
    if "class_probabilities" in df.columns and len(df):
        probs = np.stack([np.asarray(v, dtype=np.float64) for v in df["class_probabilities"]])
        if probs.shape != (len(df), k):
            raise EvaluationError(f"class_probabilities have shape {probs.shape}, expected {(len(df), k)}")
    return df, probs


def audit_features(records: pd.DataFrame, report: dict | None = None) -> pd.DataFrame:
    df = records.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df = df.drop_duplicates("sample_id", keep="first").reset_index(drop=True)
    out = pd.DataFrame({"sample_id": df["sample_id"]})
    for column in AUDIT_COLUMNS:
        if column in df.columns:
            out[column] = df[column].to_numpy()
    if "width" in out.columns and "height" in out.columns:
        w = pd.to_numeric(out["width"], errors="coerce")
        h = pd.to_numeric(out["height"], errors="coerce")
        out["aspect_ratio"] = np.maximum(w, h) / np.minimum(w, h)
        out["resolution"] = w * h
    if "sha256" in df.columns:
        out["has_exact_duplicate"] = (df["sha256"].notna() & df.duplicated("sha256", keep=False)).to_numpy()
    if report:
        added = 0
        for finding in report.get("findings", []):
            if added >= MAX_AUDIT_FLAGS:
                break
            if finding.get("sample_ids") and not finding.get("evidence_truncated"):
                out[f"audit_flag:{finding['finding_id']}"] = out["sample_id"].isin(finding["sample_ids"]).to_numpy()
                added += 1
    return out


def combine_features(frames):
    out = None
    for frame in frames:
        frame = frame.copy()
        frame["sample_id"] = frame["sample_id"].astype(str)
        frame = frame.drop_duplicates("sample_id", keep="first")
        if out is None:
            out = frame
        else:
            extra = [c for c in frame.columns if c != "sample_id" and c not in out.columns]
            out = out.merge(frame[["sample_id", *extra]], on="sample_id", how="outer")
    return out


def build_samples(df, probs, class_names, cfg: FailureConfig, features=None) -> pd.DataFrame:
    k = len(class_names)
    n = len(df)
    names = np.asarray(class_names, dtype=object)
    y = df["true_label"].to_numpy(dtype=np.int64)
    p = df["predicted_label"].to_numpy(dtype=np.int64)
    conf = df["confidence"].to_numpy(dtype=np.float64)
    err = y != p

    margin = np.full(n, np.nan)
    entropy = np.full(n, np.nan)
    if probs is not None and n:
        clipped = np.clip(probs, 1e-12, 1.0)
        entropy = -(clipped * np.log(clipped)).sum(axis=1)
        if k >= 2:
            ordered = np.sort(probs, axis=1)
            margin = ordered[:, -1] - ordered[:, -2]

    ambiguous = margin < cfg.ambiguous_margin
    high_error = err & (conf >= cfg.high_confidence_threshold)
    low_correct = ~err & (conf < cfg.low_confidence_threshold)
    failure_type = np.select(
        [high_error, err & ambiguous, err, low_correct, ~err & ambiguous],
        ["high_confidence_error", "ambiguous_error", "error", "low_confidence_correct", "ambiguous_correct"],
        default="correct",
    ).astype(object)

    true_class = names[y]
    pred_class = names[p]
    out = pd.DataFrame(
        {
            "sample_id": df["sample_id"].to_numpy(),
            "true_label": y,
            "predicted_label": p,
            "true_class": true_class,
            "predicted_class": pred_class,
            "confidence": conf,
            "correct": ~err,
            "is_error": err,
            "margin": margin,
            "entropy": entropy,
            "is_high_confidence_error": high_error,
            "is_low_confidence_correct": low_correct,
            "is_ambiguous": ambiguous,
            "failure_type": failure_type,
            "false_negative_for": np.where(err, true_class, None),
            "false_positive_for": np.where(err, pred_class, None),
        }
    )
    if features is not None:
        keep = ["sample_id"] + [c for c in features.columns if c != "sample_id" and c not in out.columns]
        out = out.merge(features[keep], on="sample_id", how="left")
    return out


def _bin(series: pd.Series, n_bins: int):
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    ok = ~np.isnan(x)
    if ok.sum() == 0 or np.unique(x[ok]).size < 2:
        return None
    edges = np.unique(np.quantile(x[ok], np.linspace(0, 1, n_bins + 1)[1:-1]))
    idx = np.searchsorted(edges, x, side="right")
    present = np.unique(idx[ok])
    if present.size < 2:
        return None

    def fmt(v):
        return f"{v:.4g}"

    names = {}
    for i in present:
        low = "-inf" if i == 0 else fmt(edges[i - 1])
        high = "inf" if i == len(edges) else fmt(edges[i])
        names[int(i)] = f"[{low},{high})"
    labels = [names[int(i)] if o else "<missing>" for i, o in zip(idx, ok, strict=True)]
    return pd.Series(labels, index=series.index, dtype=object), [float(e) for e in edges]


def derive_slice_features(samples: pd.DataFrame, cfg: FailureConfig):
    columns, info, skipped = {}, {}, {}
    names = sorted(c for c in samples.columns if c not in TARGET_COLUMNS and not c.startswith("f:"))
    for name in names:
        if cfg.exclude_model_output_features and name in MODEL_OUTPUT:
            skipped[name] = "model output excluded by configuration"
            continue
        series = samples[name]
        if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
            labels = series.astype(object).where(series.notna(), "<missing>").astype(str)
            n_values = int(labels.nunique())
            if n_values < 2:
                skipped[name] = "single value"
                continue
            if n_values > cfg.max_categories:
                skipped[name] = f"more than {cfg.max_categories} categories"
                continue
            columns[name] = labels.astype(object)
            info[name] = {"kind": "categorical", "model_output": name in MODEL_OUTPUT, "n_values": n_values, "edges": None}
        else:
            binned = _bin(series, cfg.n_bins)
            if binned is None:
                skipped[name] = "no variation or no values"
                continue
            labels, edges = binned
            columns[name] = labels
            info[name] = {
                "kind": "numeric", "model_output": name in MODEL_OUTPUT,
                "n_values": int(labels.nunique()), "edges": edges,
            }
    return pd.DataFrame(columns, index=samples.index), info, skipped
