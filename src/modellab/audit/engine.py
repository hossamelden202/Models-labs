import hashlib
import io
import json
import platform
import shutil
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL

import modellab
from modellab.audit import detectors as det
from modellab.audit.config import AuditConfig
from modellab.audit.imaging import inspect_records
from modellab.audit.scan import Record, ScanResult
from modellab.audit.schema import SEVERITY_RANK, AuditReport, Category, Severity
from modellab.audit.summary import render_summary
from modellab.core.errors import ArtifactError, DatasetError
from modellab.core.results import ArtifactStore
from modellab.utils import get_logger

log = get_logger("audit")

COLUMNS = [
    "sample_id", "path", "split", "label", "label_name", "status", "extension", "format",
    "mode", "width", "height", "file_size", "sha256", "phash", "brightness", "contrast",
    "saturation", "error_stage", "error_type", "error_message", "warnings",
]
INT_COLUMNS = ["width", "height", "file_size"]


@dataclass(frozen=True)
class AuditRun:
    audit_id: str
    directory: Path
    report: AuditReport


def _records_parquet(records: list[Record]) -> bytes:
    import pandas as pd

    rows = [
        {
            "sample_id": r.sample_id, "path": r.path, "split": r.split, "label": r.label,
            "label_name": r.label_name, "status": r.status, "extension": r.extension,
            "format": r.format, "mode": r.mode, "width": r.width, "height": r.height,
            "file_size": r.file_size, "sha256": r.sha256,
            "phash": None if r.phash is None else f"{r.phash:016x}",
            "brightness": r.brightness, "contrast": r.contrast, "saturation": r.saturation,
            "error_stage": r.error_stage, "error_type": r.error_type,
            "error_message": r.error_message, "warnings": ";".join(r.warnings) or None,
        }
        for r in records
    ]
    frame = pd.DataFrame(rows, columns=COLUMNS)
    for column in INT_COLUMNS:
        frame[column] = frame[column].astype("Int64")
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _fingerprint(records: list[Record], cfg: AuditConfig) -> str:
    payload = {
        "config": cfg.model_dump(mode="json"),
        "records": [
            [r.sample_id, r.path, r.split, r.label, r.label_name, r.status, r.sha256]
            for r in records
        ],
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def run_audit(
    scan: ScanResult,
    store: ArtifactStore,
    config: AuditConfig | None = None,
    audit_id: str | None = None,
) -> AuditRun:
    cfg = config or AuditConfig()
    root = scan.root.resolve()
    if store.root == root or store.root.is_relative_to(root):
        raise DatasetError("the artifact directory must be outside the audited dataset")

    audit_id = audit_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
    if Path(audit_id).name != audit_id or audit_id in (".", ".."):
        raise ArtifactError(f"invalid audit id: {audit_id!r}")
    relative = Path("audits") / audit_id
    out_dir = store.resolve(relative)
    if out_dir.exists():
        raise ArtifactError(f"audit '{audit_id}' already exists in {store.root}")

    started = datetime.now(timezone.utc)
    clock = time.monotonic()
    records = sorted(scan.records, key=lambda r: (r.sample_id, r.path, r.row))
    log.info("auditing %d samples", len(records))
    inspect_records(records, scan.root, cfg)

    fs = det.Findings(cfg)
    if not records:
        fs.add(
            Category.INVENTORY, "no_samples", Severity.HIGH, "No samples found",
            "The scan produced no samples.", ids=[], count=0,
        )
    inventory = det.build_inventory(records, scan, cfg)
    labels = det.label_integrity(records, scan, cfg, fs)
    images = det.image_integrity(records, scan, cfg, fs)
    exact = det.exact_duplicates(records, cfg, fs)
    near = det.near_duplicates(records, cfg, fs)
    dist = det.distribution(records, scan, cfg, fs)
    stats = det.statistics(records, cfg, fs)
    has_splits = any(r.split is not None for r in records)
    overlap = det.filename_overlap(records, cfg, fs) if has_splits else {"num_groups": 0}

    findings = sorted(
        fs.items, key=lambda f: (SEVERITY_RANK[f.severity], f.category.value, f.code)
    )
    errors = Counter(f"{r.error_stage}:{r.error_type}" for r in records if r.error_type)
    scan_section = {
        "source_kind": scan.kind,
        "layout": scan.layout,
        "records_total": len(records),
        "records_by_status": inventory["by_status"],
        "excluded_total": len(scan.exclusions),
        "excluded_by_reason": dict(
            sorted(Counter(e["reason"] for e in scan.exclusions).items())
        ),
        "errors_by_stage_and_type": dict(sorted(errors.items())),
        "records_with_warnings": sum(1 for r in records if r.warnings),
        "declared_classes": scan.declared_classes,
        "declared_splits": scan.declared_splits,
    }
    leakage = {
        "applicable": has_splits,
        "exact_duplicate_cross_split_groups": exact["num_cross_split_groups"],
        "near_duplicate_cross_split_groups": near["num_cross_split_groups"],
        "label_conflict_groups": labels["conflicting_label_groups"],
        "filename_overlap": overlap,
        "note": (
            "Leakage findings report shared content or names between splits. File-name "
            "matches are leads only and are not evidence of related images."
        ),
    }
    summary = {
        "total_findings": len(findings),
        "findings_by_severity": {
            s.value: sum(1 for f in findings if f.severity is s) for s in Severity
        },
        "findings_by_category": dict(
            sorted(Counter(f.category.value for f in findings).items())
        ),
    }
    report = AuditReport(
        dataset_id=scan.dataset_id,
        fingerprint=_fingerprint(records, cfg),
        config=cfg.model_dump(mode="json"),
        scan=scan_section,
        inventory=inventory,
        label_integrity=labels,
        image_integrity=images,
        exact_duplicates=exact,
        near_duplicates=near,
        distribution=dist,
        leakage=leakage,
        statistics=stats,
        findings=findings,
        summary=summary,
    )

    metadata = {
        "audit_id": audit_id,
        "timestamp": started.isoformat(),
        "duration_seconds": round(time.monotonic() - clock, 3),
        "dataset_id": scan.dataset_id,
        "fingerprint": report.fingerprint,
        "source_kind": scan.kind,
        "num_records": len(records),
        "versions": {
            "modellab": modellab.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        },
    }
    exclusions = sorted(scan.exclusions, key=lambda e: (e["path"], e["reason"]))

    out_dir.mkdir(parents=True)
    try:
        store.write_json(relative / "report.json", report)
        store.write_text(relative / "summary.txt", render_summary(report))
        store.write_json(relative / "exclusions.json", exclusions)
        store.write_bytes(relative / "records.parquet", _records_parquet(records))
        store.write_json(relative / "metadata.json", metadata)
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return AuditRun(audit_id=audit_id, directory=out_dir, report=report)
