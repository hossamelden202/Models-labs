import json
from pathlib import Path

import numpy as np
import pandas as pd

NAMES = ["GORE", "BLOOD", "NEUTRAL"]


def _probs(pred, second, conf):
    v = np.zeros(3)
    v[pred] = conf
    v[second] = 1 - conf - 0.01
    v[3 - pred - second] = 0.01
    return v


def make_predictions():
    rows = []

    def add(true, k, pred, conf, second):
        rows.append(
            {
                "sample_id": f"{NAMES[true]}/{k:03d}.png",
                "true_label": true,
                "predicted_label": pred,
                "confidence": conf,
                "class_probabilities": _probs(pred, second, conf),
            }
        )

    for k in range(80):
        if k < 8:
            add(0, k, 1, 0.95, 0)
        elif k < 28:
            add(0, k, 0, 0.5, 1)
        else:
            add(0, k, 0, 0.85, 1)
    for k in range(80):
        if k < 10:
            add(1, k, 2, 0.55, 1)
        else:
            add(1, k, 1, 0.85, 2)
    for k in range(80):
        if k < 30:
            add(2, k, 0, 0.7, 2)
        else:
            add(2, k, 2, 0.85, 0)
    df = pd.DataFrame(rows)
    df["correct"] = df.true_label == df.predicted_label
    return df


def make_audit_records(pred):
    ids = pred.sample_id.tolist()
    n = len(ids)
    idx = np.arange(n)
    mode = ["RGB"] * n
    for i, sid in enumerate(ids):
        if sid.startswith("NEUTRAL/") and int(sid.split("/")[1][:3]) < 40:
            mode[i] = "L"
    return pd.DataFrame(
        {
            "sample_id": ids,
            "split": np.where(idx % 2 == 0, "train", "test"),
            "status": "ok",
            "mode": mode,
            "extension": ".png",
            "format": "PNG",
            "width": 64,
            "height": 64,
            "file_size": 1000 + (idx * 7) % 500,
            "brightness": (idx * 37 % 100) / 100,
            "contrast": 0.25,
            "saturation": (idx * 53 % 100) / 100,
            "sha256": [f"{i:064x}" for i in idx],
        }
    )


def write_evaluation(root, evaluation_id, predictions, names):
    out = Path(root) / "evaluations" / evaluation_id
    out.mkdir(parents=True)
    df = predictions.sort_values("sample_id").reset_index(drop=True)
    table = pd.DataFrame(
        {
            "sample_id": df.sample_id,
            "path": df.sample_id,
            "true_label": df.true_label.astype("int32"),
            "predicted_label": df.predicted_label.astype("int32"),
            "confidence": df.confidence.astype("float32"),
            "correct": df.correct,
            "class_probabilities": [np.asarray(v, dtype="float32") for v in df.class_probabilities],
        }
    )
    table.to_parquet(out / "predictions.parquet", index=False)
    meta = {
        "evaluation_id": evaluation_id, "model_id": "synthetic", "dataset_id": "synthetic",
        "seed": 42, "device": "cpu", "batch_size": 4, "preprocessing": None,
        "num_samples": len(df), "num_classes": len(names), "class_names": list(names),
        "model": {"source": "synthetic"},
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return out


def write_audit(root, audit_id, records):
    out = Path(root) / "audits" / audit_id
    out.mkdir(parents=True)
    records.to_parquet(out / "records.parquet", index=False)
    (out / "report.json").write_text(json.dumps({"findings": []}))
    return out
