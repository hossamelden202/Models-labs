import argparse
import sys
import tempfile
from pathlib import Path

import modellab
from modellab.config import load_config
from modellab.core.errors import ConfigurationError, ModelLabError

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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ModelLabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
