import hashlib
import io
import json
import platform
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

import modellab
from modellab.analysis.engine import load_failure_analysis
from modellab.analysis.stats import benjamini_hochberg, holm
from modellab.core.results import ArtifactStore
from modellab.evaluation.metrics import compute_metrics
from modellab.evaluation.preprocessing import PreprocessConfig
from modellab.experiments.baseline import load_baseline
from modellab.experiments.errors import ExperimentConfigError, ExperimentError
from modellab.experiments.inference import apply_inference
from modellab.experiments.spec import ExperimentSpec, canonical_json, experiment_id, sha256_text
from modellab.experiments.stats import bootstrap_diff_ci, paired_summary
from modellab.experiments.transforms import REGISTRY, InterventionDataset, make_image_fn

CI_METRICS = {"accuracy", "macro_f1", "balanced_accuracy"}
NOTE = (
    "The result describes the effect of the applied intervention on this model and population "
    "under the controlled design. It does not identify a mechanism."
)


@dataclass
class ExperimentOutcome:
    experiment_id: str
    status: str
    directory: Path
    result: dict


@dataclass
class FamilyRun:
    outcomes: list
    report: dict
    directory: Path


@dataclass
class _Plan:
    affected_ids: list
    population: dict
    slices: dict
    image_params: object = None
    new_preprocess: object = None


def _load_generic_training_runner():
    from modellab.experiments.training_runner import (
        run_training_experiment,
    )
    return run_training_experiment


def _env() -> dict:
    import PIL

    info = {
        "modellab": modellab.__version__, "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "pillow": PIL.__version__,
    }
    try:
        import torch

        info["torch"] = torch.__version__
    except ImportError:
        pass
    return info


def _parquet(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _state_sha(model) -> str:
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(model.model.state_dict().items()):
        digest.update(name.encode())
        flat = tensor.detach().cpu().contiguous().reshape(-1)
        try:
            raw = flat.view(torch.uint8).numpy().tobytes()
        except Exception:
            raw = flat.numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def _ece(conf, ok, bins: int = 10) -> float:
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(total)


def _extended(y, pred, probs, names) -> dict:
    metrics, confusion = compute_metrics(y, pred, probs, names)
    k, n = len(names), len(y)
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    true_c = cm.sum(axis=1).astype(float)
    pred_c = cm.sum(axis=0).astype(float)
    present = true_c > 0
    prec = np.divide(tp, pred_c, out=np.zeros(k), where=pred_c > 0)
    rec = np.divide(tp, true_c, out=np.zeros(k), where=true_c > 0)
    conf = probs[np.arange(n), pred]
    ok = pred == y
    metrics = dict(metrics)
    metrics.update(
        balanced_accuracy=float(rec[present].mean()),
        micro_precision=float(ok.mean()),
        micro_recall=float(ok.mean()),
        micro_f1=float(ok.mean()),
        weighted_precision=float((prec * true_c / n).sum()),
        weighted_recall=float((rec * true_c / n).sum()),
        mean_confidence=float(conf.mean()),
        mean_confidence_correct=float(conf[ok].mean()) if ok.any() else None,
        mean_confidence_incorrect=float(conf[~ok].mean()) if (~ok).any() else None,
        ece=_ece(conf, ok),
    )
    return {"metrics": metrics, "confusion_matrix": confusion}


def _compare(mask, y, base_pred, base_probs, int_preds, int_probs, names, spec, wanted):
    yy = y[mask]
    base_p = base_pred[mask]
    ints = [p[mask] for p in int_preds]
    base = _extended(yy, base_p, base_probs[mask], names)
    inter = [_extended(yy, ints[r], int_probs[r][mask], names) for r in range(len(ints))]
    diffs = {}
    for metric in wanted:
        before = base["metrics"][metric]
        after = float(np.mean([e["metrics"][metric] for e in inter]))
        row = {"baseline": before, "intervention": after, "difference": after - before,
               "ci_low": None, "ci_high": None}
        if metric in CI_METRICS:
            row["ci_low"], row["ci_high"] = bootstrap_diff_ci(
                metric, yy, base_p, ints, len(names), spec.n_bootstrap, spec.seed
            )
        diffs[metric] = row
    paired = paired_summary(
        base_p == yy, [i == yy for i in ints], spec.n_bootstrap, spec.n_permutations, spec.seed
    )
    comparison = {"baseline": base, "intervention": inter[0], "differences": diffs}
    return comparison, paired


def _label(significant, d, threshold, low, high) -> str:
    if significant and d >= threshold:
        return "improved"
    if significant and d <= -threshold:
        return "degraded"
    if significant:
        return "detectable_below_practical_threshold"
    if low is not None and low > -threshold and high < threshold:
        return "no_practical_difference"
    return "inconclusive"


def _verdict(spec: ExperimentSpec, paired: dict) -> dict:
    d, p = paired["accuracy_difference"], paired["primary_p"]
    label = _label(p <= spec.alpha, d, spec.practical_threshold, paired["ci_low"], paired["ci_high"])
    expected = spec.hypothesis.expected_direction
    wanted = {"improve": "improved", "degrade": "degraded", "no_change": "no_practical_difference"}
    return {
        "label": label, "accuracy_difference": d, "primary_p": p, "alpha": spec.alpha,
        "practical_threshold": spec.practical_threshold, "expected_direction": expected,
        "matches_expected": None if expected is None else label == wanted[expected],
        "correction": "none (single experiment)", "note": NOTE,
    }


def _resolve_population(store, spec, baseline, analysis):
    pop = spec.population

    if not isinstance(baseline, dict):
        raise ExperimentConfigError(
            "baseline must be the dictionary returned by load_baseline()"
        )

    # The current baseline stores the relative path to the original
    # evaluation prediction artifact rather than embedding predictions.
    prediction_artifact = baseline.get("prediction_artifact")

    if not prediction_artifact:
        raise ExperimentConfigError(
            "baseline does not specify a prediction_artifact"
        )

    prediction_path = store.root / prediction_artifact

    if not prediction_path.exists():
        raise ExperimentConfigError(
            f"baseline prediction artifact does not exist: {prediction_path}"
        )

    try:
        import json

        predictions = json.loads(
            prediction_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ExperimentConfigError(
            f"could not load baseline prediction artifact: {prediction_path}"
        ) from exc

    if not isinstance(predictions, list):
        raise ExperimentConfigError(
            "baseline detection predictions must be a list"
        )

    all_ids = []

    for row in predictions:
        if not isinstance(row, dict):
            continue

        sample_id = row.get("sample_id")

        if sample_id is not None:
            all_ids.append(sample_id)

    if not all_ids:
        raise ExperimentConfigError(
            "baseline detection predictions contain no sample_id values"
        )

    known = set(all_ids)
    slices: dict = {}

    if pop.sample_ids is not None:
        missing = sorted(set(pop.sample_ids) - known)

        if missing:
            raise ExperimentConfigError(
                f"{len(missing)} sample ids are not in the baseline, "
                f"first: {missing[:5]}"
            )

        return sorted(set(pop.sample_ids)), slices, "samples"

    if pop.slice_ids is not None:
        analysis = analysis or load_failure_analysis(
            store,
            pop.analysis_id,
        )

        if analysis.report["inputs"].get("evaluation_id") != baseline["evaluation_id"]:
            raise ExperimentConfigError(
                "the analysis was built from a different evaluation "
                "than the baseline"
            )

        for sid in pop.slice_ids:
            try:
                ids = analysis.slice_sample_ids(sid)
            except KeyError as exc:
                raise ExperimentConfigError(str(exc)) from exc

            if not ids or not set(ids) <= known:
                raise ExperimentConfigError(
                    f"slice '{sid}' is empty or has ids outside the baseline"
                )

            slices[sid] = ids

        return sorted(set().union(*slices.values())), slices, "slices"

    return all_ids, slices, "all"


def _new_preprocess(dataset, changes: dict):
    base = dataset.preprocess
    if base is None:
        raise ExperimentConfigError("the dataset has no preprocess config to modify")
    unknown = set(changes) - set(PreprocessConfig.model_fields)
    if unknown:
        raise ExperimentConfigError(f"unknown preprocess fields: {sorted(unknown)}")
    try:
        new = PreprocessConfig.model_validate({**base.model_dump(), **changes})
    except ValidationError as exc:
        raise ExperimentConfigError(f"invalid preprocessing change: {exc}") from exc
    if new == base:
        raise ExperimentConfigError("the change equals the baseline preprocessing, so nothing is intervened on")
    return new


def _prepare(store, spec, baseline, model, dataset, analysis) -> _Plan:
    ids, slices, kind = _resolve_population(store, spec, baseline, analysis)

    plan = _Plan(
        affected_ids=ids,
        population={
            "kind": kind,
            "num_affected": len(ids),
            "num_total": len(ids),
            "slice_ids": list(spec.population.slice_ids or []),
            "fingerprint": sha256_text("\n".join(ids)),
        },
        slices=slices,
    )

    iv = spec.intervention

    if iv.kind == "inference":
        # Preserve the existing inference intervention behavior.
        # Training/detection preparation must not depend on the
        # legacy classification baseline representation.
        sample = baseline.predictions.predicted_label.to_numpy(dtype=np.int64)[:1]
        apply_inference(iv, baseline.probs[:1], sample, baseline.class_names)
        return plan

    if model is None or dataset is None:
        raise ExperimentConfigError(
            "image and preprocessing interventions need the model and the dataset"
        )

    if not isinstance(baseline, dict):
        raise ExperimentConfigError(
            "baseline must be the dictionary returned by load_baseline()"
        )

    metadata = baseline.get("metadata")
    if not isinstance(metadata, dict):
        raise ExperimentError(
            "baseline metadata is missing or invalid"
        )

    baseline_model = metadata.get("model")

    if not isinstance(baseline_model, dict):
        raise ExperimentError(
            "baseline metadata does not contain a valid model specification"
        )

    current_model = model.spec.model_dump(mode="json")

    # A direct smoke test may load the exact same checkpoint from its
    # physical Kaggle dataset path instead of the copied ModelLab artifact.
    # model_id/path are therefore runtime artifact identity, not model
    # architecture identity, and are intentionally ignored here.
    #
    # input_size may be absent when resolving a self-contained checkpoint.
    # When absent, the baseline's recorded value is treated as compatible.
    ignored_identity_fields = {"model_id", "path"}

    comparable_current = {
        key: value
        for key, value in current_model.items()
        if key not in ignored_identity_fields
    }

    comparable_baseline = {
        key: value
        for key, value in baseline_model.items()
        if key not in ignored_identity_fields
    }

    if comparable_current.get("input_size") is None:
        comparable_current["input_size"] = comparable_baseline.get(
            "input_size"
        )

    if comparable_current != comparable_baseline:
        differing = sorted(
            key
            for key in set(comparable_current) | set(comparable_baseline)
            if comparable_current.get(key) != comparable_baseline.get(key)
        )

        raise ExperimentError(
            "the model does not match the model spec of the baseline; "
            f"differing fields: {differing}"
        )

    baseline_class_names = metadata.get("class_names")

    if baseline_class_names is None:
        raise ExperimentError(
            "baseline metadata does not contain class_names"
        )

    if list(dataset.class_names) != list(baseline_class_names):
        raise ExperimentError(
            "the dataset class order does not match the baseline"
        )

    pre = (
        dataset.preprocess.model_dump(mode="json")
        if dataset.preprocess is not None
        else None
    )

    baseline_preprocessing = metadata.get("preprocessing")

    if pre != baseline_preprocessing:
        raise ExperimentError(
            "the dataset preprocessing differs from the baseline preprocessing"
        )

    dataset_sample_ids = {
        dataset.get_sample(index).sample_id
        for index in range(len(dataset))
    }

    missing = sorted(
        set(ids) - dataset_sample_ids
    )

    if missing:
        raise ExperimentError(
            f"{len(missing)} affected samples are not in the dataset, "
            f"first: {missing[:5]}"
        )

    if iv.kind == "image_transform":
        plan.image_params = REGISTRY[iv.op].params(**iv.params)
    elif iv.kind == "training":
        # Training interventions are consumed by the training
        # execution path and do not modify dataset preprocessing here.
        pass
    else:
        plan.new_preprocess = _new_preprocess(
            dataset,
            iv.changes,
        )

    return plan


def _run_images(store, spec, baseline, plan, model, dataset, exp_id, rep):
    from modellab.evaluation.engine import EvalConfig, run_evaluation

    iv = spec.intervention
    fn = make_image_fn(iv.op, plan.image_params, spec.seed, rep) if iv.kind == "image_transform" else None
    wrapped = InterventionDataset.wrap(
        dataset, image_fn=fn, only_ids=plan.affected_ids, preprocess=plan.new_preprocess
    )
    eval_id = f"{exp_id}_r{rep}"
    shutil.rmtree(store.root / "evaluations" / eval_id, ignore_errors=True)
    config = EvalConfig(batch_size=int(baseline.metadata.get("batch_size", 32)), seed=spec.seed + rep)
    run_evaluation(model, wrapped, store, config, evaluation_id=eval_id)
    frame = pd.read_parquet(store.root / "evaluations" / eval_id / "predictions.parquet")
    return frame.sort_values("sample_id", kind="mergesort").reset_index(drop=True), eval_id


def _execute(store, spec, baseline, plan, model, dataset, exp_id, cache_key):
    names = baseline.class_names
    base = baseline.predictions
    ids = base["sample_id"].to_numpy()
    y = base["true_label"].to_numpy(dtype=np.int64)
    pred0 = base["predicted_label"].to_numpy(dtype=np.int64)
    probs0 = baseline.probs
    affected = np.isin(ids, plan.affected_ids)
    iv = spec.intervention

    state_before = _state_sha(model) if iv.kind != "inference" else None
    int_preds, int_probs, eval_ids = [], [], []
    for rep in range(spec.repetitions):
        new_pred, new_probs = pred0.copy(), probs0.copy()
        if iv.kind == "inference":
            sub_probs, sub_pred = apply_inference(iv, probs0[affected], pred0[affected], names)
            new_probs[affected], new_pred[affected] = sub_probs, sub_pred
        else:
            frame, eval_id = _run_images(store, spec, baseline, plan, model, dataset, exp_id, rep)
            if frame["sample_id"].tolist() != ids[affected].tolist():
                raise ExperimentError("the intervention evaluation covers a different set of samples")
            new_pred[affected] = frame["predicted_label"].to_numpy(dtype=np.int64)
            new_probs[affected] = np.stack([np.asarray(v, dtype=np.float64) for v in frame["class_probabilities"]])
            eval_ids.append(eval_id)
        int_preds.append(new_pred)
        int_probs.append(new_probs)
    state_after = _state_sha(model) if iv.kind != "inference" else None
    if state_before != state_after:
        raise ExperimentError("the model state changed during the experiment")

    wanted = list(dict.fromkeys(["accuracy", "mean_confidence", "ece", *spec.metrics]))
    everyone = np.ones(len(ids), dtype=bool)
    comp_global, paired_global = _compare(everyone, y, pred0, probs0, int_preds, int_probs, names, spec, wanted)
    if affected.all():
        comp_aff, paired_aff = comp_global, paired_global
    else:
        comp_aff, paired_aff = _compare(affected, y, pred0, probs0, int_preds, int_probs, names, spec, wanted)

    slice_results = {}
    if plan.slices:
        for sid, members in plan.slices.items():
            m = np.isin(ids, members)
            comp, paired = _compare(m, y, pred0, probs0, int_preds, int_probs, names, spec, wanted)
            slice_results[sid] = {"n": int(m.sum()), "differences": comp["differences"], "paired": paired}
        q = benjamini_hochberg([r["paired"]["primary_p"] for r in slice_results.values()])
        for result, qv in zip(slice_results.values(), q, strict=True):
            result["primary_p_adjusted"] = float(qv)

    ok_base = pred0 == y
    ok_int = int_preds[0] == y
    changed = pred0 != int_preds[0]
    improved = sorted(ids[~ok_base & ok_int].tolist())
    worsened = sorted(ids[ok_base & ~ok_int].tolist())
    cap = 20

    repetitions = []
    for rep in range(spec.repetitions):
        ok = int_preds[rep] == y
        repetitions.append(
            {
                "repetition": rep, "seed": spec.seed + rep,
                "evaluation_id": eval_ids[rep] if eval_ids else None,
                "accuracy_global": float(ok.mean()), "accuracy_affected": float(ok[affected].mean()),
            }
        )

    result = {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": exp_id,
        "cache_key": cache_key,
        "spec": spec.model_dump(mode="json"),
        "baseline": {
            "family_id": baseline["family_id"],
            "evaluation_id": baseline["evaluation_id"],
            "fingerprint": sha256_text(
                canonical_json(baseline)
            ),
            "num_samples": len(ids),
        },
        "control": {
            "population": "identical for baseline and intervention",
            "model": "unchanged baseline model",
            "preprocessing": "unchanged except for the intervention",
            "metric_implementation": "modellab.evaluation.metrics.compute_metrics plus paired statistics",
            "seed": spec.seed,
        },
        "population": plan.population,
        "environment": _env(),
        "model_state_sha256": state_before,
        "evaluations": eval_ids,
        "repetitions": repetitions,
        "comparison": {"global": comp_global, "affected": comp_aff},
        "statistics": {"global": paired_global, "affected": paired_aff, "slices": slice_results},
        "affected_samples": {
            "num_changed_predictions": int(changed.sum()),
            "num_improved": len(improved),
            "num_worsened": len(worsened),
            "improved_ids": improved[:cap],
            "worsened_ids": worsened[:cap],
            "truncated": len(improved) > cap or len(worsened) > cap,
        },
        "verdict": _verdict(spec, paired_aff),
    }
    paired_table = pd.DataFrame(
        {
            "sample_id": ids, "affected": affected, "true_label": y,
            "baseline_pred": pred0, "intervention_pred": int_preds[0],
            "baseline_confidence": probs0[np.arange(len(ids)), pred0],
            "intervention_confidence": int_probs[0][np.arange(len(ids)), int_preds[0]],
            "baseline_correct": ok_base, "intervention_correct": ok_int, "changed": changed,
        }
    )
    return result, paired_table


def _f(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_experiment(result: dict) -> str:
    if result["status"] != "completed":
        return f"Experiment {result['experiment_id']}: FAILED\n{result['error']['type']}: {result['error']['message']}\n"
    spec, st, v = result["spec"], result["statistics"]["affected"], result["verdict"]
    acc = result["comparison"]["affected"]["differences"]["accuracy"]
    glob = result["comparison"]["global"]["differences"]["accuracy"]
    pop = result["population"]
    return "\n".join(
        [
            f"Experiment {result['experiment_id']}: {spec['name']}",
            f"hypothesis: {spec['hypothesis']['claim']} (expected: {spec['hypothesis']['expected_direction']})",
            f"intervention: {json.dumps(spec['intervention'], sort_keys=True)}",
            f"population: {pop['num_affected']} affected of {pop['num_total']} ({pop['kind']})",
            f"accuracy on affected: {_f(acc['baseline'])} -> {_f(acc['intervention'])} "
            f"(difference {_f(acc['difference'])}, 95% CI [{_f(acc['ci_low'])}, {_f(acc['ci_high'])}])",
            f"accuracy globally: {_f(glob['baseline'])} -> {_f(glob['intervention'])} (difference {_f(glob['difference'])})",
            f"paired test ({st['primary_test']}): p={st['primary_p']:.3g}; "
            f"improved {result['affected_samples']['num_improved']}, worsened {result['affected_samples']['num_worsened']}",
            f"verdict (uncorrected): {v['label']}; practical threshold {v['practical_threshold']}",
            NOTE,
            "",
        ]
    )


def _sanitize(text: str, store) -> str:
    return " ".join(text.replace(str(store.root), "<store>").split())[:400]



def _run_training_experiment(
    store,
    spec,
    baseline,
    model,
    dataset,
    exp_id,
    relative,
):
    """
    Run a training intervention and evaluate the resulting
    checkpoint through the appropriate ModelLab evaluator.

    Classification:
        uses the existing classification training path.

    Detection:
        uses the existing ModelLab/Ultralytics detection
        training adapter and the existing detection evaluator.
    """

    from pathlib import Path

    if model is None:
        raise ExperimentConfigError(
            "training interventions require the model"
        )

    if dataset is None:
        raise ExperimentConfigError(
            "training interventions require the dataset"
        )

    if spec.intervention.kind != "training":
        raise ExperimentConfigError(
            "internal error: _run_training_experiment received "
            "a non-training intervention"
        )

    training_dir = (
        store.root
        / relative
        / "training"
    )

    training_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    task = str(
        getattr(
            model.spec,
            "task",
            "classification",
        )
    ).lower()

    # ========================================================
    # DETECTION
    # ========================================================

    if task == "detection":

        from modellab.experiments.detection_training import (
            train_detection,
        )

        from modellab.evaluation.detection_dataset import (
            YOLODetectionDataset,
        )

        from modellab.evaluation.detection_engine import (
            DetectionEvalConfig,
            run_detection_evaluation,
        )

        from modellab.evaluation.torch_model import (
            TorchImageClassifier,
            TorchModelSpec,
        )

        # ----------------------------------------------------
        # Require the YOLO dataset representation.
        # ----------------------------------------------------

        if not isinstance(
            dataset,
            YOLODetectionDataset,
        ):
            raise ExperimentConfigError(
                "detection training requires a "
                "YOLODetectionDataset; got "
                f"{type(dataset).__name__}"
            )

        # ----------------------------------------------------
        # Locate the existing dataset layout.
        # ----------------------------------------------------

        dataset_root = Path(
            getattr(
                dataset,
                "root",
                getattr(
                    dataset,
                    "path",
                    "",
                ),
            )
        )

        if not dataset_root.exists():
            # Some DatasetAdapter implementations expose the
            # root through metadata instead.
            samples = getattr(
                dataset,
                "_samples",
                None,
            )

            if samples:
                first = samples[0]

                metadata = getattr(
                    first,
                    "metadata",
                    {},
                )

                sample_path = metadata.get(
                    "path"
                )

                if sample_path:
                    dataset_root = (
                        Path(sample_path)
                        .resolve()
                        .parent
                    )

        # ----------------------------------------------------
        # Existing detection dataset in this project is flat:
        #
        #   /kaggle/working/weapons/images
        #   /kaggle/working/weapons/labels
        #
        # The adapter has already created a deterministic
        # train/val representation.
        # ----------------------------------------------------

        prepared_root = (
            store.root
            / "detection_training_data"
        )

        train_images = (
            prepared_root
            / "train"
            / "images"
        )

        val_images = (
            prepared_root
            / "val"
            / "images"
        )

        # If the adapter's prepared dataset is absent, create
        # it through the same adapter-preparation logic by
        # reusing the exact dataset root.
        if not train_images.exists() or not val_images.exists():

            raw_root = Path(
                "/kaggle/working/weapons"
            )

            if not raw_root.exists():
                raise ExperimentConfigError(
                    "could not locate the YOLO dataset root "
                    f"at {raw_root}"
                )

            # Import the adapter module and reproduce its
            # deterministic split here only when the prepared
            # artifact does not exist.
            #
            # The split is deterministic with seed 42 and is
            # persisted as an artifact, so subsequent runs reuse
            # exactly the same images.
            import random
            import shutil

            image_suffixes = {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            }

            images = sorted(
                p
                for p in raw_root.rglob("*")
                if (
                    p.is_file()
                    and p.suffix.lower()
                    in image_suffixes
                )
            )

            if len(images) < 2:
                raise ExperimentConfigError(
                    "YOLO dataset contains fewer than two "
                    "images"
                )

            labels_by_stem = {}

            for label in sorted(
                raw_root.rglob("*.txt")
            ):
                labels_by_stem.setdefault(
                    label.stem,
                    label,
                )

            indices = list(
                range(len(images))
            )

            rng = random.Random(42)
            rng.shuffle(indices)

            val_count = max(
                1,
                round(
                    len(indices) * 0.20
                ),
            )

            val_indices = set(
                indices[:val_count]
            )

            train_images.mkdir(
                parents=True,
                exist_ok=True,
            )

            (
                prepared_root
                / "train"
                / "labels"
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

            val_images.mkdir(
                parents=True,
                exist_ok=True,
            )

            (
                prepared_root
                / "val"
                / "labels"
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

            for index, image in enumerate(
                images
            ):
                if index in val_indices:
                    image_dir = val_images
                    label_dir = (
                        prepared_root
                        / "val"
                        / "labels"
                    )
                else:
                    image_dir = train_images
                    label_dir = (
                        prepared_root
                        / "train"
                        / "labels"
                    )

                name = (
                    f"{index:06d}"
                    f"{image.suffix.lower()}"
                )

                shutil.copy2(
                    image,
                    image_dir / name,
                )

                label = labels_by_stem.get(
                    image.stem
                )

                if label is not None:
                    shutil.copy2(
                        label,
                        label_dir
                        / f"{index:06d}.txt",
                    )

        # ----------------------------------------------------
        # Class names come from the already resolved model spec.
        # ----------------------------------------------------

        class_names = list(
            model.spec.class_names or []
        )

        if not class_names:
            raise ExperimentConfigError(
                "detection model has no class_names"
            )

        # ----------------------------------------------------
        # Train using the existing detection adapter.
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("PHASE 5 — DETECTION TRAINING")
        print("=" * 70)

        print(
            f"Model : {model.spec.model_id}"
        )

        print(
            f"Classes : {class_names}"
        )

        print(
            f"Train images : {train_images}"
        )

        print(
            f"Val images   : {val_images}"
        )

        training_result = train_detection(
            model_path=Path(
                model.spec.path
            ),
            train_images=train_images,
            val_images=val_images,
            class_names=class_names,
            config=spec.intervention,
            output_dir=training_dir,
            device="auto",
        )

        checkpoint = Path(
            training_result.best_checkpoint
        )

        if not checkpoint.exists():
            raise ExperimentConfigError(
                "detection training completed but "
                f"best checkpoint does not exist: "
                f"{checkpoint}"
            )

        print()
        print(
            "[PHASE 5] best checkpoint:"
        )
        print(
            f"  {checkpoint}"
        )

        # ----------------------------------------------------
        # Load the trained Ultralytics checkpoint.
        #
        # We deliberately use Ultralytics' native checkpoint
        # loader here because the checkpoint is a complete
        # detection model. The existing ModelLab detection
        # evaluator still performs the actual evaluation.
        # ----------------------------------------------------

        try:
            from ultralytics import YOLO

            trained_yolo = YOLO(
                str(checkpoint)
            )

            trained_module = (
                trained_yolo.model
            )

        except Exception as exc:
            raise ExperimentConfigError(
                "could not load trained Ultralytics "
                f"checkpoint {checkpoint}: {exc}"
            ) from exc

        trained_spec = model.spec.model_copy(
            deep=True
        )

        trained_spec.path = checkpoint
        trained_spec.source = "module"
        trained_spec.task = "detection"

        trained_spec.num_classes = len(
            class_names
        )

        trained_spec.class_names = (
            class_names
        )

        trained_model = TorchImageClassifier(
            trained_spec,
            device="auto",
            module=trained_module,
        )

        # ----------------------------------------------------
        # Evaluate using ModelLab's EXISTING detection engine.
        # ----------------------------------------------------

        evaluation_id = (
            f"{exp_id}-trained"
        )

        detection_config = (
            DetectionEvalConfig(
                batch_size=8,
                num_workers=0,
                seed=42,
                confidence_threshold=0.25,
                iou_threshold=0.50,
            )
        )

        print()
        print(
            "=" * 70
        )
        print(
            "PHASE 5 — EVALUATING TRAINED CHECKPOINT"
        )
        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Phase 5 evaluation dataset.
        #
        # The original dataset object is intentionally preserved.
        # It may have no preprocessing configuration because it is
        # also used as the source dataset for training preparation.
        #
        # ModelLab's detection evaluator requires every image in a
        # batch to have the same tensor shape. Therefore Phase 5
        # creates a separate evaluation-only dataset with the
        # training resolution.
        # ----------------------------------------------------

        from modellab.evaluation.preprocessing import PreprocessConfig

        from modellab.experiments.training_config import (
            TrainingConfig,
            apply_training_changes,
        )

        evaluation_training_config = apply_training_changes(
            TrainingConfig(),
            spec.intervention,
        )

        evaluation_preprocess = PreprocessConfig(
            resize=(
                int(evaluation_training_config.data.input_width),
                int(evaluation_training_config.data.input_height),
            )
        )

        evaluation_dataset = YOLODetectionDataset(
            root=dataset.root,
            class_names=list(class_names),
            dataset_id=f"{dataset.dataset_id}-phase5-eval",
            preprocess=evaluation_preprocess,
        )

        print()
        print("PHASE 5 — EVALUATION DATASET")
        print(
            "  root     : "
            f"{evaluation_dataset.root}"
        )
        print(
            "  dataset  : "
            f"{evaluation_dataset.dataset_id}"
        )
        print(
            "  resize   : "
            f"{evaluation_training_config.data.input_width}x"
            f"{evaluation_training_config.data.input_height}"
        )
        print(
            "  samples  : "
            f"{len(evaluation_dataset)}"
        )

        # ----------------------------------------------------
        # Resume-safe trained evaluation.
        #
        # A previous Phase 5 attempt may have completed and
        # persisted this evaluation before failing later in the
        # experiment finalization/report path. In that case,
        # reusing the persisted evaluation is required instead
        # of creating the same evaluation ID again.
        # ----------------------------------------------------

        evaluation_dir = (
            store.root
            / "evaluations"
            / evaluation_id
        )

        required_evaluation_files = (
            evaluation_dir / "metadata.json",
            evaluation_dir / "metrics.json",
            evaluation_dir / "predictions.json",
        )

        existing_evaluation = all(
            path.exists() and path.stat().st_size > 0
            for path in required_evaluation_files
        )

        if existing_evaluation:
            print()
            print("=" * 70)
            print("PHASE 5 — REUSING EXISTING TRAINED EVALUATION")
            print("=" * 70)
            print(f"evaluation_id: {evaluation_id}")
            print(f"evaluation_dir: {evaluation_dir}")
            print("status: persisted evaluation artifacts found")

            # ------------------------------------------------
            # Reconstruct the same EvaluationResult shape used
            # by run_detection_evaluation, using the persisted
            # evaluation artifacts.
            # ------------------------------------------------
            from modellab.evaluation.detection_engine import (
                EvaluationResult,
            )

            _metadata = json.loads(
                (evaluation_dir / "metadata.json").read_text()
            )
            _metrics = json.loads(
                (evaluation_dir / "metrics.json").read_text()
            )

            _prediction_artifact = (
                _metadata.get("prediction_artifact")
                or f"evaluations/{evaluation_id}/predictions.json"
            )

            _model_id = (
                _metadata.get("model_id")
                or getattr(
                    getattr(trained_model, "spec", None),
                    "model_id",
                    None,
                )
                or getattr(
                    getattr(model, "spec", None),
                    "model_id",
                    None,
                )
                or "unknown"
            )

            _dataset_id = (
                _metadata.get("dataset_id")
                or getattr(
                    evaluation_dataset,
                    "dataset_id",
                    None,
                )
                or getattr(
                    dataset,
                    "dataset_id",
                    None,
                )
                or "unknown"
            )

            if _model_id == "unknown" or _dataset_id == "unknown":
                raise ExperimentConfigError(
                    "Existing trained evaluation is missing required "
                    "model_id/dataset_id metadata and the current Phase 5 "
                    "model/dataset objects do not provide them."
                )

            trained_evaluation = EvaluationResult(
                evaluation_id=evaluation_id,
                model_id=str(_model_id),
                dataset_id=str(_dataset_id),
                metrics=_metrics,
                prediction_artifact=_prediction_artifact,
            )

            print("status: existing evaluation reused")

        else:
            print()
            print("=" * 70)
            print("PHASE 5 — EVALUATING TRAINED CHECKPOINT")
            print("=" * 70)

            trained_evaluation = (
                run_detection_evaluation(
                    trained_model,
                    evaluation_dataset,
                    store,
                    detection_config,
                    evaluation_id=evaluation_id,
                )
            )

        trained_metrics = dict(
            trained_evaluation.metrics
        )

        # ----------------------------------------------------
        # Baseline comparison.
        #
        # The current baseline infrastructure is classification-
        # specific, so detection comparison is performed directly
        # from the persisted detection evaluation when a baseline
        # object is supplied by the runner.
        #
        # If no compatible baseline is available, preserve the
        # trained metrics and mark comparison as unavailable.
        # ----------------------------------------------------

        baseline_metrics = None

        if baseline is not None:
            candidate = getattr(
                baseline,
                "metrics",
                None,
            )

            if isinstance(
                candidate,
                dict,
            ):
                if candidate.get(
                    "task"
                ) == "detection":
                    baseline_metrics = dict(
                        candidate
                    )

        comparison = {
            "available": False,
            "reason": (
                "no compatible detection baseline "
                "was supplied"
            ),
        }

        if baseline_metrics is not None:

            metric_names = (
                "precision",
                "recall",
                "f1",
                "mean_iou",
            )

            deltas = {}

            for metric_name in metric_names:
                if (
                    metric_name
                    in baseline_metrics
                    and metric_name
                    in trained_metrics
                ):
                    deltas[metric_name] = (
                        float(
                            trained_metrics[
                                metric_name
                            ]
                        )
                        -
                        float(
                            baseline_metrics[
                                metric_name
                            ]
                        )
                    )

            comparison = {
                "available": True,
                "baseline": {
                    name: baseline_metrics.get(
                        name
                    )
                    for name in metric_names
                },
                "trained": {
                    name: trained_metrics.get(
                        name
                    )
                    for name in metric_names
                },
                "delta": deltas,
            }

        # ----------------------------------------------------
        # Persist training result.
        # ----------------------------------------------------

        training_payload = {
            "status": "completed",
            "task": "detection",
            "checkpoint": str(
                checkpoint
            ),
            "last_checkpoint": str(
                training_result.last_checkpoint
            ),
            "history": (
                training_result.history
            ),
            "config": (
                training_result.config
            ),
            "best_epoch": (
                training_result.best_epoch
            ),
            "best_metric": (
                training_result.best_metric
            ),
            "total_epochs": (
                training_result.total_epochs
            ),
            "stopped_early": (
                training_result.stopped_early
            ),
            "evaluation_id": evaluation_id,
            "evaluation_metrics": (
                trained_metrics
            ),
            "comparison": comparison,
        }

        store.write_json(
            relative / "training.json",
            training_payload,
        )

        return {
            "status": "completed",
            "task": "detection",
            "checkpoint": str(
                checkpoint
            ),
            "training": training_payload,
            "evaluation": {
                "evaluation_id": evaluation_id,
                "metrics": trained_metrics,
                "prediction_artifact": (
                    trained_evaluation
                    .prediction_artifact
                ),
            },
            "comparison": comparison,
        }

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    from modellab.experiments.training_runner import (
        run_training_experiment,
    )

    training_result = _load_generic_training_runner()(
        model_spec=model.spec,
        train_dataset=dataset,
        validation_dataset=dataset,
        intervention=spec.intervention,
        output_dir=training_dir,
        device="auto",
    )

    checkpoint = Path(
        training_result.best_checkpoint
    )

    if not checkpoint.exists():
        raise ExperimentConfigError(
            "training completed but best checkpoint "
            f"does not exist: {checkpoint}"
        )

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = state.get(
        "model_state_dict"
    )

    if not isinstance(
        state_dict,
        dict,
    ):
        raise ExperimentConfigError(
            "training checkpoint does not contain "
            "model_state_dict"
        )

    model.model.load_state_dict(
        state_dict,
        strict=True,
    )

    from modellab.evaluation.torch_model import (
        TorchImageClassifier,
    )

    trained_model = TorchImageClassifier(
        model.spec,
        device="auto",
        module=model.model,
    )

    from modellab.evaluation.engine import (
        EvalConfig,
        run_evaluation,
    )

    evaluation_id = (
        f"{exp_id}-trained"
    )

    trained_evaluation = run_evaluation(
        trained_model,
        dataset,
        store,
        EvalConfig(
            batch_size=32,
            num_workers=0,
            seed=42,
        ),
        evaluation_id=evaluation_id,
    )

    training_payload = {
        "status": "completed",
        "task": "classification",
        "checkpoint": str(
            checkpoint
        ),
        "last_checkpoint": str(
            training_result.final_checkpoint
        ),
        "history": (
            training_result.history
        ),
        "config": (
            training_result.config
        ),
        "best_epoch": (
            training_result.best_epoch
        ),
        "best_metric": (
            training_result.best_metric
        ),
        "total_epochs": (
            training_result.total_epochs
        ),
        "stopped_early": (
            training_result.stopped_early
        ),
        "num_parameters": (
            training_result.total_parameters
        ),
        "num_trainable_parameters": (
            training_result.trainable_parameters
        ),
    }

    store.write_json(
        relative / "training.json",
        training_payload,
    )

    return {
        "status": "completed",
        "task": "classification",
        "checkpoint": str(
            checkpoint
        ),
        "training": training_payload,
        "evaluation": {
            "evaluation_id": evaluation_id,
            "metrics": (
                trained_evaluation.metrics
            ),
            "prediction_artifact": (
                trained_evaluation
                .prediction_artifact
            ),
        },
    }




def run_experiment(
    store: ArtifactStore, family_id: str, spec: ExperimentSpec, model=None, dataset=None,
    analysis=None, force: bool = False, raise_on_failure: bool = False,
) -> ExperimentOutcome:
    baseline = load_baseline(store, family_id)
    exp_id = experiment_id(spec)

    # Training interventions are fundamentally different from
    # inference/image/preprocessing interventions: they modify
    # model weights and must produce a new trained checkpoint.
    if spec.intervention.kind == "training":
        relative = Path("experiments") / family_id / exp_id
        directory = store.root / relative
        result_path = directory / "result.json"

        if not force and result_path.is_file():
            cached = json.loads(result_path.read_text())
            if cached.get("status") == "completed":
                return ExperimentOutcome(
                    exp_id,
                    "cached",
                    directory,
                    cached,
                )

        started = datetime.now(timezone.utc)
        clock = time.monotonic()

        directory.mkdir(parents=True, exist_ok=True)

        store.write_json(
            relative / "spec.json",
            spec,
        )

        status = "completed"

        try:
            result = _run_training_experiment(
                store=store,
                spec=spec,
                baseline=baseline,
                model=model,
                dataset=dataset,
                exp_id=exp_id,
                relative=relative,
            )

            result["cache_key"] = sha256_text(
                canonical_json(
                    {
                        "id": exp_id,
                        "intervention": spec.intervention.model_dump(
                            mode="json"
                        ),
                        "seed": spec.seed,
                    }
                )
            )

            store.write_json(
                relative / "result.json",
                result,
            )

            store.write_text(
                relative / "report.txt",
                json.dumps(
                    result,
                    indent=2,
                    sort_keys=True,
                    default=str,
                ),
            )

        except Exception as exc:
            if raise_on_failure:
                raise

            status = "failed"

            result = {
                "schema_version": 1,
                "status": "failed",
                "experiment_id": exp_id,
                "spec": spec.model_dump(mode="json"),
                "baseline": {
                    "family_id": family_id,
                    "fingerprint": sha256_text(
                    canonical_json(baseline)
                ),
                },
                "error": {
                    "type": type(exc).__name__,
                    "message": _sanitize(str(exc), store),
                },
            }

            store.write_json(
                relative / "result.json",
                result,
            )

            store.write_text(
                relative / "report.txt",
                json.dumps(
                    result,
                    indent=2,
                    sort_keys=True,
                    default=str,
                ),
            )

        store.write_json(
            relative / "run_metadata.json",
            {
                "timestamp": started.isoformat(),
                "duration_seconds": round(
                    time.monotonic() - clock,
                    3,
                ),
                "status": status,
            },
        )

        return ExperimentOutcome(
            exp_id,
            status,
            directory,
            result,
        )

    # Existing inference/image/preprocessing path.
    plan = _prepare(
        store,
        spec,
        baseline,
        model,
        dataset,
        analysis,
    )
    cache_key = sha256_text(canonical_json({"id": exp_id, "population": plan.population["fingerprint"], "env": _env()}))
    relative = Path("experiments") / family_id / exp_id
    directory = store.root / relative
    result_path = directory / "result.json"
    if not force and result_path.is_file():
        cached = json.loads(result_path.read_text())
        if cached.get("status") == "completed" and cached.get("cache_key") == cache_key:
            return ExperimentOutcome(exp_id, "cached", directory, cached)

    started, clock = datetime.now(timezone.utc), time.monotonic()
    store.write_json(relative / "spec.json", spec)
    status = "completed"
    try:
        result, table = _execute(store, spec, baseline, plan, model, dataset, exp_id, cache_key)
        store.write_json(relative / "result.json", result)
        store.write_bytes(relative / "paired.parquet", _parquet(table))
        store.write_text(relative / "report.txt", render_experiment(result))
    except Exception as exc:
        if raise_on_failure:
            raise
        status = "failed"
        result = {
            "schema_version": 1, "status": "failed", "experiment_id": exp_id, "cache_key": cache_key,
            "spec": spec.model_dump(mode="json"),
            "baseline": {"family_id": family_id, "fingerprint": sha256_text(
                    canonical_json(baseline)
                )},
            "error": {"type": type(exc).__name__, "message": _sanitize(str(exc), store)},
        }
        store.write_json(relative / "result.json", result)
        store.write_text(relative / "report.txt", render_experiment(result))
    store.write_json(
        relative / "run_metadata.json",
        {"timestamp": started.isoformat(), "duration_seconds": round(time.monotonic() - clock, 3), "status": status},
    )
    return ExperimentOutcome(exp_id, status, directory, result)


def _family_report(
    store: ArtifactStore,
    family_id: str,
    baseline: dict,
    outcomes: list[ExperimentOutcome],
    alpha: float,
) -> dict:
    """Build the persisted Phase 5 family report.

    This implementation is task/schema aware:
      - detection results primarily use ``comparison``
      - legacy classification results may use ``statistics``
      - missing optional statistical fields never crash report generation
    """

    done = [
        o
        for o in outcomes
        if o.status == "completed" and isinstance(o.result, dict)
    ]

    failed = [
        o
        for o in outcomes
        if o.status == "failed"
    ]

    # --------------------------------------------------------------
    # Multiple-testing correction.
    # Only experiments that actually expose primary_p participate.
    # --------------------------------------------------------------
    p_rows = []
    p_values = []

    for o in done:
        result = o.result

        comparison = result.get("comparison", {})
        if not isinstance(comparison, dict):
            comparison = {}

        primary_p = comparison.get("primary_p")

        if primary_p is None:
            statistics = result.get("statistics", {})
            if isinstance(statistics, dict):
                affected = statistics.get("affected", {})
                if isinstance(affected, dict):
                    primary_p = affected.get("primary_p")

        if primary_p is None:
            continue

        try:
            p_value = float(primary_p)
        except (TypeError, ValueError):
            continue

        p_values.append(p_value)
        p_rows.append(o)

    bh_values = benjamini_hochberg(p_values)
    holm_values = holm(p_values)

    correction_by_experiment = {
        id(o): (bh_value, holm_value)
        for o, bh_value, holm_value in zip(
            p_rows,
            bh_values,
            holm_values,
            strict=True,
        )
    }

    # --------------------------------------------------------------
    # Build experiment rows.
    # --------------------------------------------------------------
    rows = []

    for o in done:
        result = o.result

        comparison = result.get("comparison", {})
        if not isinstance(comparison, dict):
            comparison = {}

        statistics = result.get("statistics", {})
        if not isinstance(statistics, dict):
            statistics = {}

        affected = statistics.get("affected", {})
        if not isinstance(affected, dict):
            affected = {}

        spec = result.get("spec", {})
        if not isinstance(spec, dict):
            spec = {}

        # Current Phase 5 detection comparison schema.
        delta = comparison.get("primary_delta")

        if delta is None:
            delta = comparison.get("delta")

        if delta is None:
            delta = comparison.get("f1_difference")

        # Legacy classification schema.
        if delta is None:
            delta = affected.get("accuracy_difference")

        try:
            delta = float(delta) if delta is not None else None
        except (TypeError, ValueError):
            delta = None

        primary_p = comparison.get("primary_p")

        if primary_p is None:
            primary_p = affected.get("primary_p")

        try:
            primary_p = float(primary_p) if primary_p is not None else None
        except (TypeError, ValueError):
            primary_p = None

        bh_value, holm_value = correction_by_experiment.get(
            id(o),
            (None, None),
        )

        row = {
            "experiment_id": result.get(
                "experiment_id",
                getattr(o, "experiment_id", None),
            ),
            "name": spec.get("name"),
            "status": o.status,
            "primary_metric": comparison.get(
                "primary_metric",
                result.get("primary_metric"),
            ),
            "baseline_value": comparison.get("baseline_value"),
            "trained_value": comparison.get("trained_value"),
            "primary_delta": delta,
            "primary_p": primary_p,
            "benjamini_hochberg_q": bh_value,
            "holm_p": holm_value,
            "improved": comparison.get("improved"),
            "hypothesis": spec.get("hypothesis"),
        }

        rows.append(row)

    # --------------------------------------------------------------
    # Determine best experiment using available primary deltas.
    # --------------------------------------------------------------
    candidates = [
        row
        for row in rows
        if isinstance(row.get("primary_delta"), (int, float))
    ]

    best = None

    if candidates:
        best = max(
            candidates,
            key=lambda row: row["primary_delta"],
        )

    # --------------------------------------------------------------
    # Preserve the important family metadata.
    # --------------------------------------------------------------
    report = {
        "family_id": family_id,
        "baseline_evaluation_id": baseline.get("evaluation_id"),
        "baseline": baseline,
        "alpha": alpha,
        "num_experiments": len(outcomes),
        "num_completed": len(done),
        "num_failed": len(failed),
        "experiments": rows,
        "best_experiment": best,
        "failed_experiments": [
            {
                "experiment_id": getattr(o, "experiment_id", None),
                "error": (
                    o.result.get("error")
                    if isinstance(o.result, dict)
                    else None
                ),
            }
            for o in failed
        ],
        "multiple_testing": {
            "num_tests": len(p_values),
            "alpha": alpha,
        },
        "note": (
            "Statistical significance is reported separately from "
            "practical significance."
        ),
    }

    return report


def render_family(report: dict) -> str:
    """Render a task/schema-safe Phase 5 family summary."""

    lines = []

    family_id = report.get("family_id", "unknown")
    baseline_id = report.get("baseline_evaluation_id")

    lines.append(f"Family: {family_id}")

    if baseline_id:
        lines.append(f"Baseline: {baseline_id}")

    lines.append(
        f"Experiments: {report.get('num_experiments', 0)} "
        f"completed={report.get('num_completed', 0)} "
        f"failed={report.get('num_failed', 0)}"
    )

    lines.append("")

    for row in report.get("experiments", []):
        name = (
            row.get("name")
            or row.get("experiment_id")
            or "unknown"
        )

        # Current schema.
        difference = row.get("primary_delta")

        # Legacy schema compatibility.
        if difference is None:
            difference = row.get("accuracy_difference")

        if difference is None:
            difference_text = "n/a"
        else:
            try:
                difference_text = f"{float(difference):+.4f}"
            except (TypeError, ValueError):
                difference_text = "n/a"

        primary_p = row.get("primary_p")

        if primary_p is None:
            p_text = "n/a"
        else:
            try:
                p_text = f"{float(primary_p):.3g}"
            except (TypeError, ValueError):
                p_text = "n/a"

        # Current schema.
        bh = row.get("benjamini_hochberg_q")

        # Legacy schema compatibility.
        if bh is None:
            bh = row.get("p_benjamini_hochberg")

        if bh is None:
            bh_text = "n/a"
        else:
            try:
                bh_text = f"{float(bh):.3g}"
            except (TypeError, ValueError):
                bh_text = "n/a"

        family_label = row.get("family_label")

        if family_label is None:
            improved = row.get("improved")

            if improved is True:
                family_label = "improved"
            elif improved is False:
                family_label = "not improved"
            else:
                family_label = "not classified"

        lines.append(
            f"{name}: difference {difference_text}, "
            f"p={p_text}, BH={bh_text}, {family_label}"
        )

    failed = report.get("failed_experiments", [])

    if failed:
        lines.append("")
        lines.append("Failed experiments:")

        for row in failed:
            experiment_id = (
                row.get("experiment_id")
                or "unknown"
            )
            error = row.get("error") or "unknown error"

            lines.append(
                f"- {experiment_id}: {error}"
            )

    best = report.get("best_experiment")

    if isinstance(best, dict):
        lines.append("")
        lines.append(
            "Best experiment: "
            + str(
                best.get("name")
                or best.get("experiment_id")
                or "unknown"
            )
        )

        best_delta = best.get("primary_delta")

        if best_delta is not None:
            try:
                lines.append(
                    f"Best primary delta: {float(best_delta):+.4f}"
                )
            except (TypeError, ValueError):
                pass

    note = report.get("note")

    if note:
        lines.append("")
        lines.append(str(note))

    return "\n".join(lines)


def run_family(
    store: ArtifactStore, family_id: str, specs: list, model=None, dataset=None, analysis=None,
    force: bool = False, stop_on_failure: bool = False, alpha: float = 0.05,
) -> FamilyRun:
    baseline = load_baseline(store, family_id)
    for spec in specs:
        _prepare(store, spec, baseline, model, dataset, analysis)
    seen, outcomes = set(), []
    for spec in specs:
        eid = experiment_id(spec)
        if eid in seen:
            continue
        seen.add(eid)
        outcome = run_experiment(store, family_id, spec, model, dataset, analysis, force)
        outcomes.append(outcome)
        if outcome.status == "failed" and stop_on_failure:
            break
    report = _family_report(store, family_id, baseline, outcomes, alpha)
    relative = Path("experiments") / family_id
    store.write_json(relative / "family_report.json", report)
    store.write_text(relative / "family_summary.txt", render_family(report))
    return FamilyRun(outcomes=outcomes, report=report, directory=store.root / relative)


def run_matrix(store, family_id, matrix: dict, **kwargs) -> FamilyRun:
    from modellab.experiments.spec import expand_matrix

    return run_family(store, family_id, expand_matrix(matrix), **kwargs)
