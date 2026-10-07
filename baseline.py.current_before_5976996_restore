import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from modellab.core.results import ArtifactStore
from modellab.experiments.errors import ExperimentError
from modellab.experiments.spec import canonical_json, sha256_text

FILES = ("predictions.parquet", "metadata.json", "metrics.json", "confusion_matrix.json")
META_KEYS = (
    "model_id", "dataset_id", "class_names", "num_samples", "model", "preprocessing", "batch_size", "seed",
)


@dataclass(frozen=True)
class Baseline:
    family_id: str
    evaluation_id: str
    fingerprint: str
    class_names: list
    metadata: dict
    predictions: pd.DataFrame
    probs: np.ndarray


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_name(value: str, what: str) -> None:
    if not value or Path(value).name != value or value in (".", ".."):
        raise ExperimentError(f"invalid {what}: {value!r}")


def _fingerprint(files: dict, meta: dict) -> str:
    return sha256_text(canonical_json({"files": files, "meta": {k: meta.get(k) for k in META_KEYS}}))


def create_baseline(store: ArtifactStore, family_id: str, evaluation_id: str) -> Baseline:
    _check_name(family_id, "family id")
    _check_name(evaluation_id, "evaluation id")
    ev = store.root / "evaluations" / evaluation_id
    if not (ev / "predictions.parquet").is_file() or not (ev / "metadata.json").is_file():
        raise ExperimentError(f"evaluation '{evaluation_id}' is missing predictions or metadata")
    files = {
        name: _sha(ev / name)
        for name in FILES
        if name != "metadata.json" and (ev / name).is_file()
    }
    meta = json.loads((ev / "metadata.json").read_text())
    fingerprint = _fingerprint(files, meta)

    relative = Path("experiments") / family_id / "baseline.json"
    target = store.root / relative
    if target.is_file():
        existing = json.loads(target.read_text())
        if existing["evaluation_id"] != evaluation_id or existing["fingerprint"] != fingerprint:
            raise ExperimentError(f"family '{family_id}' already has a different baseline")
    else:
        store.write_json(
            relative,
            {
                "family_id": family_id, "evaluation_id": evaluation_id, "fingerprint": fingerprint,
                "files": files, "class_names": meta.get("class_names"),
                "num_samples": meta.get("num_samples"), "dataset_id": meta.get("dataset_id"),
                "model_id": meta.get("model_id"),
            },
        )
    return load_baseline(store, family_id)


def load_baseline(store: ArtifactStore, family_id: str) -> Baseline:
    _check_name(family_id, "family id")
    path = store.root / "experiments" / family_id / "baseline.json"
    if not path.is_file():
        raise ExperimentError(f"family '{family_id}' has no baseline, call create_baseline first")
    doc = json.loads(path.read_text())
    ev = store.root / "evaluations" / doc["evaluation_id"]
    for name, digest in doc["files"].items():
        if not (ev / name).is_file() or _sha(ev / name) != digest:
            raise ExperimentError(f"baseline artifact '{name}' was modified or removed")
    meta = json.loads((ev / "metadata.json").read_text())
    df = pd.read_parquet(ev / "predictions.parquet").sort_values("sample_id", kind="mergesort")
    df = df.reset_index(drop=True)
    if "class_probabilities" not in df.columns:
        raise ExperimentError("the baseline predictions have no class_probabilities")
    probs = np.stack([np.asarray(v, dtype=np.float64) for v in df["class_probabilities"]])
    return Baseline(
        family_id=family_id, evaluation_id=doc["evaluation_id"], fingerprint=doc["fingerprint"],
        class_names=list(meta["class_names"]), metadata=meta, predictions=df, probs=probs,
    )
