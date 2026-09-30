import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

import modellab
from modellab.config import load_config
from modellab.core.errors import ConfigurationError, ModelLabError
from modellab.utils import setup_logging

MIN_PYTHON = (3, 10)


def _cmd_version(args) -> int:
    print(modellab.__version__)
    return 0


def _cmd_doctor(args) -> int:
    healthy = True

    def report(passed: bool, message: str) -> None:
        nonlocal healthy
        healthy = healthy and passed
        print(f"[{'ok' if passed else 'fail'}] {message}")

    py = sys.version_info
    report(py >= MIN_PYTHON, f"python {py.major}.{py.minor}.{py.micro}")
    report(modellab.__version__ != "0.0.0", f"modellab {modellab.__version__}")

    try:
        cfg = load_config(args.config)
    except ConfigurationError as exc:
        report(False, f"config: {exc}")
        return 1
    report(True, f"config: {args.config or 'defaults'}")

    artifacts = cfg.paths.artifacts
    try:
        artifacts.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=artifacts):
            pass
    except OSError as exc:
        report(False, f"artifact directory not writable: {artifacts} ({exc})")
    else:
        report(True, f"artifact directory writable: {artifacts.resolve()}")

    return 0 if healthy else 1


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"{path}: top level must be a mapping")
    return data


def _load_model_file(path: Path):
    from modellab.evaluation.preprocessing import PreprocessConfig
    from modellab.evaluation.torch_model import TorchModelSpec

    raw = _read_mapping(path)
    if "model" not in raw or set(raw) - {"model", "preprocess"}:
        raise ConfigurationError(
            f"{path}: expected a 'model' section and an optional 'preprocess' section"
        )
    try:
        spec = TorchModelSpec.model_validate(raw["model"])
        preprocess = PreprocessConfig.model_validate(raw.get("preprocess") or {})
    except ValidationError as exc:
        raise ConfigurationError(f"invalid model file {path}: {exc}") from exc
    return spec, preprocess


def _cmd_evaluate(args) -> int:
    cfg = load_config(args.config)
    setup_logging(cfg.runtime.log_level)

    try:
        from modellab.core.results import ArtifactStore
        from modellab.evaluation.engine import EvalConfig, run_evaluation
        from modellab.evaluation.image_dataset import ImageClassificationDataset
        from modellab.evaluation.torch_model import TorchImageClassifier
    except ImportError as exc:
        raise ConfigurationError(
            f"evaluation needs the vision dependencies, pip install -e '.[vision]' ({exc})"
        ) from exc

    try:
        eval_cfg = EvalConfig(
            batch_size=args.batch_size, num_workers=args.num_workers, seed=cfg.project.seed
        )
    except ValidationError as exc:
        raise ConfigurationError(f"invalid evaluation settings: {exc}") from exc

    spec, preprocess = _load_model_file(args.model)

    source = args.dataset
    if source.is_dir():
        dataset = ImageClassificationDataset.from_folder(
            source, class_names=spec.class_names, preprocess=preprocess
        )
    elif source.suffix.lower() == ".csv":
        dataset = ImageClassificationDataset.from_csv(
            source,
            root=args.root,
            path_column=args.path_column,
            label_column=args.label_column,
            class_names=spec.class_names,
            preprocess=preprocess,
        )
    else:
        raise ConfigurationError(f"dataset must be an existing directory or a .csv file: {source}")

    model = TorchImageClassifier(spec, device=args.device or cfg.runtime.device)
    store = ArtifactStore(args.output or cfg.paths.artifacts)
    result = run_evaluation(model, dataset, store, eval_cfg, evaluation_id=args.evaluation_id)

    metrics = result.metrics
    location = store.root / "evaluations" / result.configuration["evaluation_id"]
    print(f"evaluation: {location}")
    print(
        f"samples: {metrics['num_samples']}  accuracy: {metrics['accuracy']:.4f}  "
        f"macro_f1: {metrics['macro_f1']:.4f}  weighted_f1: {metrics['weighted_f1']:.4f}"
    )
    return 0


def _cmd_audit(args) -> int:
    cfg = load_config(args.config)
    setup_logging(cfg.runtime.log_level)

    try:
        from modellab.audit import AuditConfig, run_audit, scan_csv, scan_folder
        from modellab.core.results import ArtifactStore
    except ImportError as exc:
        raise ConfigurationError(
            f"audit needs pillow, numpy, pandas and pyarrow ({exc})"
        ) from exc

    audit_cfg = AuditConfig()
    if args.audit_config is not None:
        try:
            audit_cfg = AuditConfig.model_validate(_read_mapping(args.audit_config))
        except ValidationError as exc:
            raise ConfigurationError(f"invalid audit config {args.audit_config}: {exc}") from exc

    source = args.dataset
    if source.is_dir():
        scan = scan_folder(source, audit_cfg, dataset_id=args.dataset_id)
    elif source.suffix.lower() == ".csv":
        scan = scan_csv(source, audit_cfg, root=args.root, dataset_id=args.dataset_id)
    else:
        raise ConfigurationError(f"dataset must be an existing directory or a .csv file: {source}")

    store = ArtifactStore(args.output or cfg.paths.artifacts)
    run = run_audit(scan, store, audit_cfg, audit_id=args.audit_id)
    counts = run.report.summary["findings_by_severity"]
    print(f"audit: {run.directory}")
    print(
        f"samples: {run.report.inventory['total_samples']}  "
        + "  ".join(f"{k}: {counts.get(k, 0)}" for k in ("high", "medium", "low", "info"))
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modellab", description="Investigate why image classifiers fail."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    version = sub.add_parser("version", help="print the installed version")
    version.set_defaults(func=_cmd_version)

    doctor = sub.add_parser("doctor", help="check the environment and configuration")
    doctor.add_argument("--config", type=Path, default=None, help="path to a YAML config")
    doctor.set_defaults(func=_cmd_doctor)

    ev = sub.add_parser("evaluate", help="evaluate an image classifier on a labelled dataset")
    ev.add_argument(
        "--model",
        type=Path,
        required=True,
        help="YAML file with a 'model' section and an optional 'preprocess' section",
    )
    ev.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="directory with one folder per class, or a CSV file",
    )
    ev.add_argument("--config", type=Path, default=None, help="path to a YAML config")
    ev.add_argument("--output", type=Path, default=None, help="artifact directory")
    ev.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    ev.add_argument("--batch-size", type=int, default=32)
    ev.add_argument("--num-workers", type=int, default=0)
    ev.add_argument("--root", type=Path, default=None, help="image root for CSV datasets")
    ev.add_argument("--path-column", default="path")
    ev.add_argument("--label-column", default="label")
    ev.add_argument("--evaluation-id", default=None)
    ev.set_defaults(func=_cmd_evaluate)

    au = sub.add_parser("audit", help="audit an image dataset without running a model")
    au.add_argument("--dataset", type=Path, required=True, help="directory or CSV file")
    au.add_argument("--audit-config", type=Path, default=None, help="YAML with audit settings")
    au.add_argument("--config", type=Path, default=None, help="path to a YAML config")
    au.add_argument("--output", type=Path, default=None, help="artifact directory")
    au.add_argument("--root", type=Path, default=None, help="image root for CSV datasets")
    au.add_argument("--audit-id", default=None)
    au.add_argument("--dataset-id", default=None)
    au.set_defaults(func=_cmd_audit)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ModelLabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
