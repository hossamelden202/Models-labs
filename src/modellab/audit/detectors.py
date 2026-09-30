from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Any

import numpy as np

from modellab.audit.config import AuditConfig
from modellab.audit.scan import Record, ScanResult
from modellab.audit.schema import Category as C
from modellab.audit.schema import Finding
from modellab.audit.schema import Severity as S
from modellab.evaluation.image_dataset import IMAGE_EXTENSIONS

NO_LABEL = "<missing>"
NO_SPLIT = "<none>"

EXPECTED_FORMATS = {
    ".jpg": {"JPEG"},
    ".jpeg": {"JPEG"},
    ".png": {"PNG"},
    ".bmp": {"BMP"},
    ".gif": {"GIF"},
    ".webp": {"WEBP"},
    ".tif": {"TIFF"},
    ".tiff": {"TIFF"},
}

FAILURES = {
    "missing_files": ("Missing files", "Listed in the dataset but the file does not exist."),
    "invalid_paths": (
        "Invalid paths",
        "The listed path is absolute, escapes the dataset root or is malformed; not opened.",
    ),
    "invalid_dimensions": ("Invalid image dimensions", "Zero or negative width or height."),
    "truncated_images": ("Truncated images", "The decoder reported incomplete image data."),
    "decode_failures": ("Undecodable images", "The file exists but could not be decoded."),
    "unreadable_files": ("Unreadable files", "The file could not be read from disk."),
}


def _r(value, digits: int = 6) -> float:
    return round(float(value), digits)


def summarize(values) -> dict[str, Any]:
    if len(values) == 0:
        return {"n": 0}
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "min": _r(a.min()),
        "p05": _r(np.percentile(a, 5)),
        "median": _r(np.median(a)),
        "mean": _r(a.mean()),
        "p95": _r(np.percentile(a, 95)),
        "max": _r(a.max()),
        "std": _r(a.std()),
    }


def decoded(records: Sequence[Record]) -> list[Record]:
    return [r for r in records if r.status == "ok"]


def aspect(r: Record) -> float:
    return max(r.width, r.height) / min(r.width, r.height)


def known_classes(records: Sequence[Record], scan: ScanResult, cfg: AuditConfig) -> list[str]:
    if cfg.class_names is not None:
        return list(cfg.class_names)
    names = set(scan.declared_classes)
    names.update(r.label for r in records if r.label is not None)
    return sorted(names)


class Findings:
    def __init__(self, cfg: AuditConfig):
        self.cfg = cfg
        self.items: list[Finding] = []

    def add(
        self,
        category: C,
        code: str,
        severity: S,
        title: str,
        description: str,
        members: Sequence[Record] = (),
        ids: Sequence[str] | None = None,
        count: int | None = None,
        splits: Sequence[str] | None = None,
        classes: Sequence[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        members = list(members)
        ids = sorted({r.sample_id for r in members}) if ids is None else list(ids)
        if count is None:
            count = len(members) if members else len(ids)
        if splits is None:
            splits = {r.split for r in members if r.split is not None}
        if classes is None:
            classes = {r.label for r in members if r.label is not None}
        limit = self.cfg.max_evidence
        self.items.append(
            Finding(
                finding_id=f"{category.value}.{code}",
                category=category,
                code=code,
                severity=severity,
                title=title,
                description=description,
                count=count,
                sample_ids=ids[:limit],
                evidence_truncated=len(ids) > limit,
                affected_splits=sorted(splits),
                affected_classes=sorted(classes),
                details=details or {},
            )
        )


def build_inventory(records: list[Record], scan: ScanResult, cfg: AuditConfig) -> dict:
    total = len(records)
    ok = decoded(records)

    def pct(n: int) -> float:
        return _r(100.0 * n / total, 4) if total else 0.0

    class_counts = Counter(r.label if r.label is not None else NO_LABEL for r in records)
    for name in known_classes(records, scan, cfg):
        class_counts.setdefault(name, 0)
    split_counts = Counter(r.split if r.split is not None else NO_SPLIT for r in records)
    for name in scan.declared_splits or []:
        split_counts.setdefault(name, 0)

    by_path = defaultdict(list)
    for r in records:
        if r.sample_id and r.path != "<invalid>":
            by_path[r.path].append(r)
    dup_paths = sorted(p for p, rs in by_path.items() if len(rs) > 1)

    sizes = Counter(f"{r.width}x{r.height}" for r in ok)
    top_sizes = sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))[:10]

    return {
        "total_samples": total,
        "by_status": dict(sorted(Counter(r.status for r in records).items())),
        "classes": {
            k: {"count": v, "percent": pct(v)} for k, v in sorted(class_counts.items())
        },
        "splits": {k: {"count": v, "percent": pct(v)} for k, v in sorted(split_counts.items())},
        "extensions": dict(sorted(Counter(r.extension or "<none>" for r in records).items())),
        "formats": dict(sorted(Counter(r.format or "<unknown>" for r in ok).items())),
        "modes": dict(sorted(Counter(r.mode or "<unknown>" for r in ok).items())),
        "dimensions": {
            "width": summarize([r.width for r in ok]),
            "height": summarize([r.height for r in ok]),
            "aspect_ratio": summarize([aspect(r) for r in ok]),
            "most_common_sizes": [{"size": s, "count": c} for s, c in top_sizes],
        },
        "file_size_bytes": summarize([r.file_size for r in records if r.file_size is not None]),
        "duplicate_paths": {
            "num_groups": len(dup_paths),
            "paths": dup_paths[: cfg.max_listed_groups],
        },
        "missing_files": sum(1 for r in records if r.status == "missing"),
        "unreadable_files": sum(1 for r in records if r.status == "unreadable"),
    }


def label_integrity(records: list[Record], scan: ScanResult, cfg: AuditConfig, fs: Findings):
    known = known_classes(records, scan, cfg)
    counts = Counter(r.label for r in records if r.label is not None)
    split_counts = Counter(r.split for r in records if r.split is not None)
    out: dict[str, Any] = {}

    missing = [r for r in records if r.label is None]
    out["missing_labels"] = len(missing)
    if missing:
        fs.add(
            C.LABEL_INTEGRITY, "missing_labels", S.HIGH, "Samples without a label",
            "These samples have no label value.", members=missing,
        )

    unknown: list[Record] = []
    if cfg.class_names is not None:
        allowed = set(cfg.class_names)
        unknown = [r for r in records if r.label is not None and r.label not in allowed]
    out["unknown_labels"] = len(unknown)
    if unknown:
        fs.add(
            C.LABEL_INTEGRITY, "unknown_labels", S.HIGH, "Labels outside the declared classes",
            "These samples carry labels that are not in the configured class list.",
            members=unknown,
            details={
                "labels": sorted({r.label for r in unknown}),
                "allowed": list(cfg.class_names),
            },
        )

    empty = [c for c in known if counts.get(c, 0) == 0]
    out["empty_classes"] = empty
    if empty:
        fs.add(
            C.LABEL_INTEGRITY, "empty_classes", S.MEDIUM, "Classes without samples",
            "These classes are declared or expected but contain no samples.",
            ids=[], count=len(empty), classes=empty, details={"classes": empty},
        )

    empty_splits = [s for s in (scan.declared_splits or []) if split_counts.get(s, 0) == 0]
    out["empty_splits"] = empty_splits
    if empty_splits:
        fs.add(
            C.LABEL_INTEGRITY, "empty_splits", S.MEDIUM, "Splits without samples",
            "These split directories exist but contain no images.",
            ids=[], count=len(empty_splits), splits=empty_splits,
            details={"splits": empty_splits},
        )

    if split_counts and any(r.split is None for r in records):
        without = [r for r in records if r.split is None]
        fs.add(
            C.LABEL_INTEGRITY, "records_without_split", S.LOW,
            "Samples without a split value",
            "Some samples have a split while others do not.", members=without,
        )

    forms = defaultdict(set)
    for label in counts:
        forms[" ".join(label.split()).casefold()].add(label)
    variants = {k: sorted(v) for k, v in sorted(forms.items()) if len(v) > 1}
    out["label_variants"] = variants
    if variants:
        flagged = {lab for group in variants.values() for lab in group}
        fs.add(
            C.LABEL_INTEGRITY, "label_variants", S.MEDIUM,
            "Class labels that differ only by case or spacing",
            "These label strings would count as separate classes but match after "
            "normalising case and whitespace.",
            members=[r for r in records if r.label in flagged],
            details={"variants": variants},
        )

    id_to_names = defaultdict(set)
    name_to_ids = defaultdict(set)
    for r in records:
        if r.label is not None and r.label_name is not None:
            id_to_names[r.label].add(r.label_name)
            name_to_ids[r.label_name].add(r.label)
    bad_ids = {k: sorted(v) for k, v in sorted(id_to_names.items()) if len(v) > 1}
    bad_names = {k: sorted(v) for k, v in sorted(name_to_ids.items()) if len(v) > 1}
    out["inconsistent_mapping"] = {"label_to_names": bad_ids, "name_to_labels": bad_names}
    if bad_ids or bad_names:
        fs.add(
            C.LABEL_INTEGRITY, "inconsistent_class_mapping", S.HIGH,
            "Inconsistent label to class-name mapping",
            "A label maps to several class names, or a class name to several labels.",
            members=[r for r in records if r.label in bad_ids or r.label_name in bad_names],
            details={"label_to_names": bad_ids, "name_to_labels": bad_names},
        )

    id_counts = Counter(r.sample_id for r in records)
    dup_ids = sorted(i for i, c in id_counts.items() if c > 1)
    out["duplicate_sample_ids"] = len(dup_ids)
    if dup_ids:
        fs.add(
            C.LABEL_INTEGRITY, "duplicate_sample_ids", S.HIGH, "Duplicate sample IDs",
            "Several records share one sample ID.",
            members=[r for r in records if id_counts[r.sample_id] > 1],
            ids=dup_ids, count=len(dup_ids),
            details={"num_records": sum(id_counts[i] for i in dup_ids)},
        )

    id_splits = defaultdict(set)
    for r in records:
        if r.split is not None:
            id_splits[r.sample_id].add(r.split)
    overlap = sorted(i for i, s in id_splits.items() if len(s) > 1)
    out["split_overlap"] = len(overlap)
    if overlap:
        overlap_set = set(overlap)
        fs.add(
            C.LABEL_INTEGRITY, "split_overlap", S.HIGH, "Sample IDs present in several splits",
            "The same sample ID is assigned to more than one split.",
            members=[r for r in records if r.sample_id in overlap_set],
            ids=overlap, count=len(overlap),
        )

    by_hash = defaultdict(list)
    for r in records:
        if r.sha256:
            by_hash[r.sha256].append(r)
    conflicts = {
        h: rs for h, rs in by_hash.items() if len({r.label for r in rs if r.label is not None}) > 1
    }
    out["conflicting_label_groups"] = len(conflicts)
    if conflicts:
        ordered = sorted(conflicts.items())
        members = [r for _, rs in ordered for r in rs]
        fs.add(
            C.LABEL_INTEGRITY, "conflicting_labels_identical_content", S.HIGH,
            "Identical file content with different labels",
            "Byte-identical files carry more than one label.",
            members=members,
            details={
                "num_groups": len(conflicts),
                "groups": [
                    {
                        "sha256": h,
                        "labels": sorted({r.label for r in rs if r.label is not None}),
                        "sample_ids": sorted({r.sample_id for r in rs})[: cfg.max_evidence],
                    }
                    for h, rs in ordered[: cfg.max_listed_groups]
                ],
            },
        )

    if split_counts:
        per = Counter((r.split, r.label) for r in records if r.split and r.label)
        pairs = [
            {"split": s, "class": c}
            for s in sorted(split_counts)
            for c in known
            if per.get((s, c), 0) == 0
        ]
        out["classes_missing_from_splits"] = len(pairs)
        if pairs:
            fs.add(
                C.LABEL_INTEGRITY, "class_missing_from_split", S.MEDIUM,
                "Classes absent from a split",
                "Some split and class combinations contain no samples.",
                ids=[], count=len(pairs),
                splits=sorted({p["split"] for p in pairs}),
                classes=sorted({p["class"] for p in pairs}),
                details={"pairs": pairs[: cfg.max_listed_groups]},
            )
    return out


def failure_code(r: Record) -> str:
    if r.status == "missing":
        return "missing_files"
    if r.status == "invalid_path":
        return "invalid_paths"
    if r.error_type == "InvalidDimensions":
        return "invalid_dimensions"
    if r.error_stage == "decode":
        text = (r.error_message or "").lower()
        if "truncated" in text or "broken data stream" in text:
            return "truncated_images"
        return "decode_failures"
    return "unreadable_files"


def image_integrity(records: list[Record], scan: ScanResult, cfg: AuditConfig, fs: Findings):
    ok = decoded(records)
    groups = defaultdict(list)
    for r in records:
        if r.status != "ok":
            groups[failure_code(r)].append(r)
    out: dict[str, Any] = {"failures": {}}
    for code, members in sorted(groups.items()):
        title, text = FAILURES[code]
        out["failures"][code] = len(members)
        fs.add(
            C.IMAGE_INTEGRITY, code, S.HIGH, title, text, members=members,
            details={"error_types": dict(sorted(Counter(m.error_type for m in members).items()))},
        )

    unsupported = [
        r for r in records if r.path != "<invalid>" and r.extension not in IMAGE_EXTENSIONS
    ]
    out["unsupported_extension"] = len(unsupported)
    if unsupported:
        fs.add(
            C.IMAGE_INTEGRITY, "unsupported_extension", S.LOW,
            "Listed files with an unsupported extension",
            "These listed files do not have a recognised image extension.",
            members=unsupported,
            details={"extensions": dict(sorted(Counter(r.extension for r in unsupported).items()))},
        )

    excluded = [e for e in scan.exclusions if e["reason"] == "unsupported_extension"]
    out["excluded_non_image_files"] = len(excluded)
    if excluded:
        suffixes = Counter(PurePosixPath(e["path"]).suffix.lower() or "<none>" for e in excluded)
        fs.add(
            C.IMAGE_INTEGRITY, "excluded_non_image_files", S.INFO,
            "Non-image files skipped during the directory scan",
            "These files were not treated as samples because of their extension.",
            ids=[], count=len(excluded),
            details={
                "extensions": dict(sorted(suffixes.items())),
                "examples": sorted(e["path"] for e in excluded)[: cfg.max_evidence],
            },
        )

    mismatch = [
        r for r in ok if r.extension in EXPECTED_FORMATS and r.format not in EXPECTED_FORMATS[r.extension]
    ]
    out["format_extension_mismatch"] = len(mismatch)
    if mismatch:
        pairs = Counter(f"{r.extension}:{r.format}" for r in mismatch)
        fs.add(
            C.IMAGE_INTEGRITY, "format_extension_mismatch", S.LOW,
            "File extension does not match the decoded format",
            "The decoded image format differs from what the extension suggests.",
            members=mismatch, details={"pairs": dict(sorted(pairs.items()))},
        )

    by_mode = defaultdict(list)
    for r in ok:
        if r.mode != "RGB":
            by_mode[r.mode or "unknown"].append(r)
    out["non_rgb_modes"] = {m: len(rs) for m, rs in sorted(by_mode.items())}
    for mode, members in sorted(by_mode.items()):
        fs.add(
            C.IMAGE_INTEGRITY, f"mode_{mode}", S.INFO, f"Images in {mode} mode",
            f"These images are stored in {mode} mode, not 3-channel RGB.", members=members,
        )

    checks = [
        ("extreme_aspect_ratio", S.LOW, "Extreme aspect ratios",
         f"Longer side more than {cfg.max_aspect_ratio} times the shorter side.",
         [r for r in ok if aspect(r) > cfg.max_aspect_ratio]),
        ("tiny_images", S.LOW, "Very small images",
         f"Shorter side below {cfg.min_side} pixels.",
         [r for r in ok if min(r.width, r.height) < cfg.min_side]),
        ("huge_images", S.LOW, "Very large images",
         f"Longer side above {cfg.max_side} pixels.",
         [r for r in ok if max(r.width, r.height) > cfg.max_side]),
        ("tiny_files", S.LOW, "Very small files",
         f"File size below {cfg.min_file_bytes} bytes.",
         [r for r in records if r.file_size is not None and r.file_size < cfg.min_file_bytes]),
        ("huge_files", S.LOW, "Very large files",
         f"File size above {cfg.max_file_bytes} bytes.",
         [r for r in records if r.file_size is not None and r.file_size > cfg.max_file_bytes]),
    ]
    for code, severity, title, text, members in checks:
        out[code] = len(members)
        if members:
            fs.add(C.IMAGE_INTEGRITY, code, severity, title, text, members=members)
    return out


def exact_duplicates(records: list[Record], cfg: AuditConfig, fs: Findings):
    by_hash = defaultdict(list)
    for r in records:
        if r.sha256:
            by_hash[r.sha256].append(r)
    groups = []
    for digest, rs in by_hash.items():
        if len({r.path for r in rs}) < 2:
            continue
        rs = sorted(rs, key=lambda r: (r.sample_id, r.row))
        splits = sorted({r.split for r in rs if r.split is not None})
        labels = sorted({r.label for r in rs if r.label is not None})
        groups.append(
            {
                "sha256": digest,
                "count": len(rs),
                "sample_ids": [r.sample_id for r in rs][: cfg.max_evidence],
                "members_truncated": len(rs) > cfg.max_evidence,
                "splits": splits,
                "classes": labels,
                "cross_split": len(splits) > 1,
                "label_conflict": len(labels) > 1,
                "_members": rs,
            }
        )
    groups.sort(key=lambda g: (-g["count"], g["sha256"]))

    everyone = [r for g in groups for r in g["_members"]]
    cross = [r for g in groups if g["cross_split"] for r in g["_members"]]
    n_cross = sum(1 for g in groups if g["cross_split"])
    if groups:
        fs.add(
            C.EXACT_DUPLICATES, "duplicate_groups", S.MEDIUM, "Byte-identical files",
            "Several distinct files have identical content.",
            members=everyone,
            details={
                "num_groups": len(groups),
                "num_redundant_copies": sum(g["count"] - 1 for g in groups),
            },
        )
    if cross:
        fs.add(
            C.LEAKAGE, "exact_duplicate_leakage", S.HIGH,
            "Identical files in more than one split",
            "Byte-identical content is present in more than one split.",
            members=cross, details={"num_groups": n_cross},
        )
    public = [{k: v for k, v in g.items() if k != "_members"} for g in groups]
    return {
        "num_groups": len(groups),
        "num_affected_samples": len(everyone),
        "num_redundant_copies": sum(g["count"] - 1 for g in groups),
        "num_cross_split_groups": n_cross,
        "num_label_conflict_groups": sum(1 for g in groups if g["label_conflict"]),
        "groups": public[: cfg.max_listed_groups],
        "groups_truncated": len(public) > cfg.max_listed_groups,
    }


def near_duplicate_edges(values: list[int], threshold: int, max_bucket_size: int):
    chunks = threshold + 1
    bounds = [round(k * 64 / chunks) for k in range(chunks + 1)]
    checked: set[tuple[int, int]] = set()
    edges: list[tuple[int, int, int]] = []
    skipped = 0
    for c in range(chunks):
        low, high = bounds[c], bounds[c + 1]
        mask = (1 << (high - low)) - 1
        buckets: dict[int, list[int]] = defaultdict(list)
        for index, value in enumerate(values):
            buckets[(value >> low) & mask].append(index)
        for key in sorted(buckets):
            members = buckets[key]
            if len(members) < 2:
                continue
            if len(members) > max_bucket_size:
                skipped += 1
                continue
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    pair = (members[a], members[b])
                    if pair in checked:
                        continue
                    checked.add(pair)
                    dist = bin(values[pair[0]] ^ values[pair[1]]).count("1")
                    if dist <= threshold:
                        edges.append((pair[0], pair[1], dist))
    edges.sort()
    return edges, len(checked), skipped


def near_duplicates(records: list[Record], cfg: AuditConfig, fs: Findings):
    by_value = defaultdict(list)
    for r in records:
        if r.phash is not None:
            by_value[r.phash].append(r)
    values = sorted(by_value)
    edges, checked, skipped = near_duplicate_edges(
        values, cfg.near_duplicate_threshold, cfg.max_bucket_size
    )

    parent = list(range(len(values)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in edges:
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    components = defaultdict(list)
    for i in range(len(values)):
        components[find(i)].append(i)
    max_dist: dict[int, int] = defaultdict(int)
    n_links: dict[int, int] = defaultdict(int)
    for i, _, d in edges:
        root = find(i)
        max_dist[root] = max(max_dist[root], d)
        n_links[root] += 1

    groups = []
    for root, indexes in components.items():
        rs = sorted(
            (r for i in indexes for r in by_value[values[i]]), key=lambda r: (r.sample_id, r.row)
        )
        if len({r.sha256 for r in rs}) < 2:
            continue
        splits = sorted({r.split for r in rs if r.split is not None})
        labels = sorted({r.label for r in rs if r.label is not None})
        groups.append(
            {
                "count": len(rs),
                "num_distinct_files": len({r.sha256 for r in rs}),
                "sample_ids": [r.sample_id for r in rs][: cfg.max_evidence],
                "members_truncated": len(rs) > cfg.max_evidence,
                "splits": splits,
                "classes": labels,
                "cross_split": len(splits) > 1,
                "label_disagreement": len(labels) > 1,
                "max_link_distance": int(max_dist.get(root, 0)),
                "num_links": int(n_links.get(root, 0)),
                "_members": rs,
            }
        )
    groups.sort(key=lambda g: (-g["count"], g["sample_ids"][0]))

    everyone = [r for g in groups for r in g["_members"]]
    cross = [r for g in groups if g["cross_split"] for r in g["_members"]]
    disagree = [r for g in groups if g["label_disagreement"] for r in g["_members"]]
    n_cross = sum(1 for g in groups if g["cross_split"])
    note = (
        "Grouping uses a 64-bit difference hash. It links resized, re-encoded or lightly "
        "edited copies; similarity is not proof that two images share a source."
    )
    if groups:
        fs.add(
            C.NEAR_DUPLICATES, "near_duplicate_groups", S.LOW, "Near-duplicate image groups",
            note, members=everyone, details={"num_groups": len(groups)},
        )
    if cross:
        fs.add(
            C.LEAKAGE, "near_duplicate_leakage", S.MEDIUM,
            "Near-duplicate images in more than one split", note, members=cross,
            details={"num_groups": n_cross},
        )
    if disagree:
        fs.add(
            C.NEAR_DUPLICATES, "near_duplicate_label_disagreement", S.MEDIUM,
            "Near-duplicate images with different labels", note, members=disagree,
            details={"num_groups": sum(1 for g in groups if g["label_disagreement"])},
        )
    if skipped:
        fs.add(
            C.NEAR_DUPLICATES, "near_duplicate_search_incomplete", S.LOW,
            "Near-duplicate search skipped oversized buckets",
            f"{skipped} hash buckets exceeded max_bucket_size and were not compared, "
            "so some near-duplicate pairs may be missing.",
            ids=[], count=skipped,
        )
    public = [{k: v for k, v in g.items() if k != "_members"} for g in groups]
    return {
        "method": "dhash64",
        "threshold": cfg.near_duplicate_threshold,
        "num_hashed": sum(len(v) for v in by_value.values()),
        "num_unique_hashes": len(values),
        "num_candidate_pairs_checked": checked,
        "num_links": len(edges),
        "num_skipped_buckets": skipped,
        "num_groups": len(groups),
        "num_affected_samples": len(everyone),
        "num_cross_split_groups": n_cross,
        "groups": public[: cfg.max_listed_groups],
        "groups_truncated": len(public) > cfg.max_listed_groups,
    }


def robust_outliers(records: list[Record], getter, cfg: AuditConfig):
    pairs = [(getter(r), r) for r in records]
    pairs = [(v, r) for v, r in pairs if v is not None]
    if len(pairs) < cfg.min_samples_for_outliers:
        return [], None
    values = np.array([v for v, _ in pairs], dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0:
        return [], {"median": _r(median), "mad": 0.0}
    z = 0.6745 * (values - median) / mad
    flagged = [(abs(float(s)), r) for s, (_, r) in zip(z, pairs, strict=True) if abs(s) > cfg.outlier_z]
    flagged.sort(key=lambda t: (-t[0], t[1].sample_id))
    return flagged, {"median": _r(median), "mad": _r(mad)}


def distribution(records: list[Record], scan: ScanResult, cfg: AuditConfig, fs: Findings):
    ok = decoded(records)
    known = known_classes(records, scan, cfg)
    counts = Counter(r.label for r in records if r.label is not None)
    per_class = {name: counts.get(name, 0) for name in known}
    for name, n in counts.items():
        per_class.setdefault(name, n)
    non_empty = {k: v for k, v in per_class.items() if v > 0}

    imbalance: dict[str, Any] = {"ratio": None, "majority_class": None, "minority_class": None}
    if len(non_empty) >= 2:
        major = max(non_empty, key=lambda k: (non_empty[k], k))
        minor = min(non_empty, key=lambda k: (non_empty[k], k))
        ratio = non_empty[major] / non_empty[minor]
        imbalance = {"ratio": _r(ratio), "majority_class": major, "minority_class": minor}
        if ratio >= cfg.imbalance_ratio:
            severe = ratio >= cfg.severe_imbalance_ratio
            fs.add(
                C.DISTRIBUTION, "class_imbalance", S.HIGH if severe else S.MEDIUM,
                "Class imbalance",
                f"The largest class has {_r(ratio, 2)} times as many samples as the smallest "
                "non-empty class.",
                ids=[], count=len(non_empty), classes=[major, minor],
                details={"counts": dict(sorted(per_class.items())), "ratio": _r(ratio)},
            )

    labelled = [r for r in records if r.split is not None and r.label is not None]
    proportions: dict[str, dict[str, float]] = {}
    if len({r.split for r in labelled}) >= 2:
        total = Counter(r.label for r in labelled)
        n_all = len(labelled)
        per_split = defaultdict(Counter)
        for r in labelled:
            per_split[r.split][r.label] += 1
        worst = None
        for split in sorted(per_split):
            n = sum(per_split[split].values())
            proportions[split] = {c: _r(per_split[split].get(c, 0) / n, 4) for c in known}
            if n < cfg.min_split_size_for_gap:
                continue
            for c in known:
                gap = abs(per_split[split].get(c, 0) / n - total.get(c, 0) / n_all)
                if worst is None or gap > worst[0] + 1e-12:
                    worst = (gap, split, c)
        if worst is not None and worst[0] >= cfg.split_proportion_gap:
            fs.add(
                C.DISTRIBUTION, "split_class_proportion_gap", S.LOW,
                "Class proportions differ between splits",
                "A class makes up a noticeably different share of one split than of the "
                "whole dataset.",
                ids=[], count=1, splits=[worst[1]], classes=[worst[2]],
                details={"gap": _r(worst[0], 4), "split": worst[1], "class": worst[2]},
            )

    outliers: dict[str, Any] = {}
    metrics = (
        ("width", lambda r: r.width),
        ("height", lambda r: r.height),
        ("aspect_ratio", aspect),
        ("file_size", lambda r: r.file_size),
    )
    for name, getter in metrics:
        flagged, info = robust_outliers(ok, getter, cfg)
        outliers[name] = {"num_flagged": len(flagged), "robust_stats": info}
        if flagged:
            fs.add(
                C.DISTRIBUTION, f"outliers_{name}", S.LOW, f"Outliers in {name}",
                f"Values more than {cfg.outlier_z} robust z-scores from the median.",
                members=[r for _, r in flagged], ids=[r.sample_id for _, r in flagged],
                details=info,
            )
    return {
        "class_counts": dict(sorted(per_class.items())),
        "class_imbalance": imbalance,
        "split_class_proportions": proportions,
        "modes": dict(sorted(Counter(r.mode or "<unknown>" for r in ok).items())),
        "outliers": outliers,
    }


def statistics(records: list[Record], cfg: AuditConfig, fs: Findings):
    if not cfg.compute_stats:
        return {"enabled": False}
    ok = [r for r in decoded(records) if r.brightness is not None]
    names = ("brightness", "contrast", "saturation")

    def brief(values) -> dict:
        if not values:
            return {"n": 0}
        a = np.asarray(values, dtype=np.float64)
        return {"n": int(a.size), "mean": _r(a.mean()), "median": _r(np.median(a)), "std": _r(a.std())}

    per_class = defaultdict(list)
    for r in ok:
        per_class[r.label if r.label is not None else NO_LABEL].append(r)
    out: dict[str, Any] = {
        "enabled": True,
        "num_measured": len(ok),
        "overall": {n: summarize([getattr(r, n) for r in ok]) for n in names},
        "per_class": {
            label: {n: brief([getattr(r, n) for r in rs]) for n in names}
            for label, rs in sorted(per_class.items())
        },
        "outliers": {},
    }
    for name in names:
        flagged, info = robust_outliers(ok, lambda r, n=name: getattr(r, n), cfg)
        out["outliers"][name] = {"num_flagged": len(flagged), "robust_stats": info}
        if flagged:
            fs.add(
                C.STATISTICS, f"outliers_{name}", S.LOW, f"Outliers in {name}",
                f"Values more than {cfg.outlier_z} robust z-scores from the median.",
                members=[r for _, r in flagged], ids=[r.sample_id for _, r in flagged],
                details=info,
            )
    flat = [r for r in ok if r.contrast < cfg.min_contrast]
    out["near_uniform_images"] = len(flat)
    if flat:
        fs.add(
            C.STATISTICS, "near_uniform_images", S.LOW, "Near-uniform images",
            f"Luma standard deviation below {cfg.min_contrast}.", members=flat,
        )
    return out


def filename_overlap(records: list[Record], cfg: AuditConfig, fs: Findings):
    stems = defaultdict(list)
    for r in records:
        if r.split is None or r.path == "<invalid>":
            continue
        stem = PurePosixPath(r.path).stem.casefold()
        if len(stem) >= cfg.filename_overlap_min_stem:
            stems[stem].append(r)
    groups = []
    for stem, rs in sorted(stems.items()):
        splits = sorted({r.split for r in rs})
        if len(splits) < 2:
            continue
        groups.append(
            {
                "stem": stem,
                "splits": splits,
                "sample_ids": sorted(r.sample_id for r in rs)[: cfg.max_evidence],
                "distinct_content_hashes": len({r.sha256 for r in rs if r.sha256}),
                "_members": rs,
            }
        )
    if groups:
        fs.add(
            C.LEAKAGE, "filename_overlap", S.INFO,
            "File names shared between splits",
            "The same file name appears in more than one split. A matching name is only a "
            "lead to inspect; it does not show that the images are related.",
            members=[r for g in groups for r in g["_members"]],
            details={"num_groups": len(groups)},
        )
    public = [{k: v for k, v in g.items() if k != "_members"} for g in groups]
    return {"num_groups": len(groups), "groups": public[: cfg.max_listed_groups]}
