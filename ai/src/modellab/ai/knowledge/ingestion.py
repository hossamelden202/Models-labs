import json
from pathlib import Path

from modellab.ai.analysis import normalize_per_class, per_class_lines
from modellab.ai.knowledge.documents import KnowledgeDocument
from modellab.experiments.training_config import TrainingConfig

METRICS = ("f1", "precision", "recall", "mean_iou")


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flat(v, key))
        else:
            out[key] = v
    return out


def config_changes(config):
    base = _flat(TrainingConfig().model_dump(mode="python"))
    missing = object()
    return {k: v for k, v in _flat(config).items() if base.get(k, missing) != v}


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def experiment_document(family_id, exp_id, result):
    training = result.get("training") or {}
    changes = config_changes(training.get("config") or {})
    change_text = ", ".join(f"{k}={v}" for k, v in sorted(changes.items())) or "defaults"
    lines = [f"Detection experiment {exp_id} in family {family_id}.", f"Changes from default config: {change_text}.", f"Status: {result.get('status')}."]

    if result.get("status") != "completed":
        err = result.get("error") or {}
        lines.append(f"Failed with {err.get('type')}: {err.get('message')}")
    else:
        if training:
            lines.append(f"Trained {training.get('total_epochs')} epochs, best epoch {training.get('best_epoch')}.")
        comp = result.get("comparison") or {}
        if comp.get("available"):
            for name in METRICS:
                lines.append(
                    f"{name}: baseline {_fmt(comp['baseline'].get(name))}, trained {_fmt(comp['trained'].get(name))}, "
                    f"delta {_fmt(comp['delta'].get(name))}"
                )
        per_class = ((result.get("evaluation") or {}).get("metrics") or {}).get("per_class")
        if per_class:
            lines.append("Per class after training:")
            lines += per_class_lines(normalize_per_class(per_class))

    return KnowledgeDocument(
        id=f"experiment:{family_id}:{exp_id}",
        source_type="experiment",
        source_id=exp_id,
        title=f"Experiment {exp_id}: {change_text}",
        content="\n".join(lines),
        metadata={"family_id": family_id, "experiment_id": exp_id, "task": "detection",
                  "status": result.get("status"), "changes": changes},
    )


def evaluation_document(eval_id, metrics, meta):
    lines = [f"Detection evaluation {eval_id}."]
    if meta.get("dataset_id"):
        lines.append(f"Dataset {meta['dataset_id']}, model {meta.get('model_id')}.")
    lines.append(
        f"Overall f1 {_fmt(metrics.get('f1'))}, precision {_fmt(metrics.get('precision'))}, "
        f"recall {_fmt(metrics.get('recall'))}, mean iou {_fmt(metrics.get('mean_iou'))}, "
        f"samples {metrics.get('num_samples')}."
    )
    lines += per_class_lines(normalize_per_class(metrics.get("per_class") or {}))
    return KnowledgeDocument(
        id=f"evaluation:{eval_id}",
        source_type="evaluation",
        source_id=eval_id,
        title=f"Evaluation {eval_id}",
        content="\n".join(lines),
        metadata={"evaluation_id": eval_id, "dataset_id": meta.get("dataset_id"),
                  "model_id": meta.get("model_id"), "task": "detection"},
    )


def failure_document(analysis_id, per_class, meta):
    rows = normalize_per_class(per_class)
    lines = [f"Detection failure analysis {analysis_id} of evaluation {meta.get('evaluation_id')}."]
    lines += per_class_lines(rows)
    for r in rows:
        if r["support"] == 0:
            lines.append(f"{r['class_name']} has no ground-truth objects in this evaluation set.")
    return KnowledgeDocument(
        id=f"failure_analysis:{analysis_id}",
        source_type="failure_analysis",
        source_id=analysis_id,
        title=f"Failure analysis {analysis_id}",
        content="\n".join(lines),
        metadata={"analysis_id": analysis_id, "evaluation_id": meta.get("evaluation_id"), "task": "detection"},
    )


def collect_documents(root):
    root = Path(root)
    docs, skipped = [], 0

    for path in sorted(root.glob("experiments/*/exp_*/result.json")):
        result = _read(path)
        if not result or result.get("task") != "detection":
            skipped += 1
            continue
        docs.append(experiment_document(path.parent.parent.name, path.parent.name, result))

    for path in sorted(root.glob("evaluations/*/metrics.json")):
        eval_id = path.parent.name
        metrics = _read(path)
        if not metrics or metrics.get("task") != "detection" or eval_id.startswith("exp_"):
            skipped += 1
            continue
        docs.append(evaluation_document(eval_id, metrics, _read(path.parent / "metadata.json") or {}))

    for path in sorted(root.glob("failure_analyses/*/per_class.json")):
        meta = _read(path.parent / "metadata.json") or {}
        per_class = _read(path)
        if not per_class or meta.get("task") != "detection":
            skipped += 1
            continue
        docs.append(failure_document(path.parent.name, per_class, meta))

    return docs, skipped


def ingest_all(root, store):
    docs, skipped = collect_documents(root)
    added = store.add(docs)
    store.save()
    return {"documents": len(docs), "added": added, "skipped": skipped}


def tried_changes(root, family_id):
    out = []
    for path in sorted((Path(root) / "experiments" / family_id).glob("exp_*/result.json")):
        result = _read(path)
        if result and result.get("status") == "completed":
            out.append(config_changes((result.get("training") or {}).get("config") or {}))
    return out
