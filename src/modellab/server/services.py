import threading
from collections import Counter, OrderedDict
from pathlib import Path

from modellab.server.errors import ApiError
from modellab.server.workspace import Workspace, safe_join


class ModelCache:
    def __init__(self, size: int = 2):
        self._size = size
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, ws: Workspace, model_id: str, device: str):
        record = ws.get_model(model_id)
        key = (model_id, device, record["created_at"])
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        from modellab.evaluation.preprocessing import PreprocessConfig
        from modellab.evaluation.torch_model import TorchImageClassifier, TorchModelSpec

        spec = TorchModelSpec.model_validate(record["spec"])
        entry = (
            TorchImageClassifier(spec, device=device),
            PreprocessConfig.model_validate(record["preprocess"]),
            record,
        )
        with self._lock:
            self._items[key] = entry
            while len(self._items) > self._size:
                self._items.popitem(last=False)
        return entry

    def drop(self, model_id: str) -> None:
        with self._lock:
            for key in [k for k in self._items if k[0] == model_id]:
                del self._items[key]


def dataset_root(ws: Workspace, rec: dict, subpath: str | None = None) -> Path:
    root = Path(rec["root"])
    if not root.is_dir():
        raise ApiError(409, f"the directory of dataset '{rec['dataset_id']}' no longer exists")
    return safe_join(root, subpath)


def dataset_label(dataset_id: str, subpath: str | None) -> str:
    return dataset_id if not subpath else f"{dataset_id}:{subpath}"


def dataset_summary(ws: Workspace, rec: dict, subpath: str | None = None) -> dict:
    from modellab.audit import AuditConfig, scan_csv, scan_folder

    cfg = AuditConfig(path_column=rec.get("path_column") or "path", label_column=rec.get("label_column") or "label")
    if rec["kind"] == "csv":
        scan = scan_csv(rec["csv"], cfg, root=rec["root"])
    else:
        scan = scan_folder(dataset_root(ws, rec, subpath), cfg)
    labels = Counter(r.label if r.label is not None else "<missing>" for r in scan.records)
    splits = Counter(r.split for r in scan.records if r.split)
    return {
        "layout": scan.layout,
        "num_samples": len(scan.records),
        "classes": dict(sorted(labels.items())),
        "declared_classes": scan.declared_classes,
        "splits": dict(sorted(splits.items())),
        "num_excluded": len(scan.exclusions),
    }


def build_dataset(ws, dataset_id, subpath, class_names, preprocess):
    from modellab.evaluation.image_dataset import ImageClassificationDataset

    rec = ws.get_dataset(dataset_id)
    names = list(class_names) if class_names else None
    label = dataset_label(dataset_id, subpath)
    if rec["kind"] == "csv":
        if subpath:
            raise ApiError(422, "subpath is not supported for CSV datasets")
        return ImageClassificationDataset.from_csv(
            rec["csv"], root=rec["root"], path_column=rec.get("path_column") or "path",
            label_column=rec.get("label_column") or "label", class_names=names,
            dataset_id=label, preprocess=preprocess,
        )
    return ImageClassificationDataset.from_folder(
        dataset_root(ws, rec, subpath), class_names=names, dataset_id=label, preprocess=preprocess
    )


def evaluate(ws, cache, store, p):
    from modellab.evaluation.engine import EvalConfig, run_evaluation

    model, pre, _ = cache.get(ws, p["model_id"], p["device"])
    dataset = build_dataset(ws, p["dataset_id"], p.get("subpath"), model.spec.class_names, pre)
    config = EvalConfig(batch_size=p["batch_size"], num_workers=p["num_workers"], seed=p["seed"])
    result = run_evaluation(model, dataset, store, config, evaluation_id=p.get("evaluation_id"))
    m = result.metrics
    return {
        "evaluation_id": result.configuration["evaluation_id"],
        "num_samples": m["num_samples"],
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "weighted_f1": m["weighted_f1"],
        "device": result.configuration["device"],
    }


def audit(ws, store, p):
    from modellab.audit import AuditConfig, run_audit, scan_csv, scan_folder

    rec = ws.get_dataset(p["dataset_id"])
    raw = dict(p.get("config") or {})
    if rec["kind"] == "csv":
        raw = {"path_column": rec.get("path_column") or "path", "label_column": rec.get("label_column") or "label", **raw}
    cfg = AuditConfig.model_validate(raw)
    label = dataset_label(p["dataset_id"], p.get("subpath"))
    if rec["kind"] == "csv":
        scan = scan_csv(rec["csv"], cfg, root=rec["root"], dataset_id=label)
    else:
        scan = scan_folder(dataset_root(ws, rec, p.get("subpath")), cfg, dataset_id=label)
    run = run_audit(scan, store, cfg, audit_id=p.get("audit_id"))
    report = run.report
    return {
        "audit_id": run.audit_id,
        "fingerprint": report.fingerprint,
        "num_samples": report.inventory["total_samples"],
        "findings_by_severity": report.summary["findings_by_severity"],
    }


def analyze(ws, store, p):
    from modellab.analysis import FailureConfig, analyze_failures, load_audit, load_evaluation

    cfg = FailureConfig.model_validate(p.get("config") or {})
    predictions, names, _ = load_evaluation(store.root, p["evaluation_id"])
    records = report = None
    if p.get("audit_id"):
        records, report = load_audit(store.root, p["audit_id"])
    result = analyze_failures(
        predictions, names, cfg, audit_records=records, audit_report=report, store=store,
        analysis_id=p.get("analysis_id"), evaluation_id=p["evaluation_id"], audit_id=p.get("audit_id"),
    )
    rep = result.report
    return {
        "analysis_id": result.analysis_id,
        "n_samples": rep["inputs"]["n_samples"],
        "accuracy": rep["overall"]["accuracy"],
        "n_errors": rep["overall"]["n_errors"],
        "num_tested_slices": rep["slices"]["info"]["num_tested"],
        "top_slices": rep["slices"]["top"][:10],
    }


def experiments(ws, cache, store, p):
    from modellab.experiments import (
        ExperimentConfigError,
        create_baseline,
        expand_matrix,
        parse_spec,
        run_family,
    )

    family = p["family_id"]
    if p.get("baseline_evaluation_id"):
        create_baseline(store, family, p["baseline_evaluation_id"])
    specs = [parse_spec(s) for s in p["specs"]]
    for matrix in p["matrices"]:
        specs.extend(expand_matrix(matrix))
    model = dataset = None
    if any(s.intervention.kind != "inference" for s in specs):
        if not p.get("model_id") or not p.get("dataset_id"):
            raise ExperimentConfigError("image and preprocessing interventions need model_id and dataset_id")
        model, pre, _ = cache.get(ws, p["model_id"], p["device"])
        dataset = build_dataset(ws, p["dataset_id"], p.get("subpath"), model.spec.class_names, pre)
    run = run_family(
        store, family, specs, model=model, dataset=dataset, force=p["force"],
        stop_on_failure=p["stop_on_failure"], alpha=p["alpha"],
    )
    return {
        "family_id": family,
        "num_completed": run.report["num_completed"],
        "num_failed": run.report["num_failed"],
        "outcomes": [
            {"experiment_id": o.experiment_id, "name": o.result["spec"]["name"], "status": o.status}
            for o in run.outcomes
        ],
        "report": run.report,
    }
