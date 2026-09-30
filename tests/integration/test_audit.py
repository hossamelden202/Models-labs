import hashlib
import importlib.util
import json
import random
import shutil
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("PIL")

from modellab.audit import AuditConfig, run_audit, scan_adapter, scan_csv, scan_folder  # noqa: E402
from modellab.audit.detectors import near_duplicate_edges  # noqa: E402
from modellab.audit.scan import canonical_path  # noqa: E402
from modellab.cli.main import main  # noqa: E402
from modellab.core.errors import ArtifactError, DatasetError  # noqa: E402
from modellab.core.results import ArtifactStore  # noqa: E402
from modellab.evaluation.image_dataset import ImageClassificationDataset  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "audit_fixtures.py"
_spec = importlib.util.spec_from_file_location("audit_fixtures_for_tests", FIXTURES)
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)

CFG = AuditConfig(min_side=16, imbalance_ratio=2.5)


@pytest.fixture
def dataset(tmp_path):
    return fixtures.build_synthetic_dataset(tmp_path / "ds")


def snapshot(root):
    state = {}
    for path in sorted(root.rglob("*")):
        key = path.relative_to(root).as_posix()
        if path.is_file():
            info = path.stat()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            state[key] = (info.st_size, info.st_mtime_ns, digest)
        else:
            state[key] = None
    return state


def audit(root, out, audit_id, cfg=CFG):
    scan = scan_folder(root, cfg, dataset_id="synthetic")
    return run_audit(scan, ArtifactStore(out), cfg, audit_id=audit_id)


def by_id(report):
    return {f.finding_id: f for f in report.findings}


def ids(finding):
    return set(finding.sample_ids)


def test_folder_audit_finds_every_planted_problem(dataset, tmp_path):
    before = snapshot(dataset)
    result = audit(dataset, tmp_path / "out", "a")
    assert snapshot(dataset) == before

    report = result.report
    f = by_id(report)
    inv = report.inventory
    assert report.scan["layout"] == "split_class"
    assert inv["total_samples"] == 85
    assert inv["by_status"] == {"ok": 83, "unreadable": 2}
    assert {k: v["count"] for k, v in inv["classes"].items()} == {
        "GORE": 17, "BLOOD": 17, "NEUTRAL": 51, "EMPTY": 0,
    }
    assert {k: v["count"] for k, v in inv["splits"].items()} == {"train": 62, "val": 12, "test": 11}
    assert inv["extensions"][".png"] == 84 and inv["extensions"][".jpg"] == 1
    assert inv["modes"]["L"] == 1 and inv["modes"]["RGBA"] == 1
    assert inv["duplicate_paths"]["num_groups"] == 0
    assert report.scan["excluded_by_reason"] == {
        "hidden_directory": 1, "hidden_file": 1, "unsupported_extension": 1,
    }
    assert f["image_integrity.excluded_non_image_files"].count == 1

    failed = set()
    for code in ("decode_failures", "truncated_images"):
        if f"image_integrity.{code}" in f:
            failed |= ids(f[f"image_integrity.{code}"])
    assert failed == {"train/GORE/corrupt.png", "val/BLOOD/truncated.jpg"}
    assert "train/GORE/corrupt.png" in ids(f["image_integrity.decode_failures"])
    assert ids(f["image_integrity.mode_L"]) == {"train/NEUTRAL/gray.png"}
    assert ids(f["image_integrity.mode_RGBA"]) == {"val/GORE/rgba.png"}
    assert ids(f["image_integrity.extreme_aspect_ratio"]) == {"train/BLOOD/wide.png"}

    assert report.exact_duplicates["num_groups"] == 3
    assert report.exact_duplicates["num_affected_samples"] == 6
    assert report.exact_duplicates["num_cross_split_groups"] == 1
    assert report.exact_duplicates["num_label_conflict_groups"] == 1
    assert f["exact_duplicates.duplicate_groups"].details["num_groups"] == 3
    leak = f["leakage.exact_duplicate_leakage"]
    assert ids(leak) == {"train/BLOOD/b01.png", "test/BLOOD/leak_exact_b01.png"}
    assert leak.affected_splits == ["test", "train"] and leak.affected_classes == ["BLOOD"]
    assert leak.severity.value == "high"

    conflict = f["label_integrity.conflicting_labels_identical_content"]
    assert ids(conflict) == {"train/GORE/g02.png", "train/NEUTRAL/conflict_g2.png"}
    assert conflict.affected_classes == ["GORE", "NEUTRAL"]

    near_ids = {"train/NEUTRAL/n05.png", "val/NEUTRAL/near_n05.png"}
    assert report.near_duplicates["num_groups"] == 1
    assert report.near_duplicates["groups"][0]["max_link_distance"] <= 4
    assert ids(f["near_duplicates.near_duplicate_groups"]) == near_ids
    assert ids(f["leakage.near_duplicate_leakage"]) == near_ids
    assert f["leakage.near_duplicate_leakage"].affected_splits == ["train", "val"]

    overlap = f["leakage.filename_overlap"]
    assert ids(overlap) == {"train/NEUTRAL/scene_alpha_001.png", "test/NEUTRAL/scene_alpha_001.png"}
    assert overlap.severity.value == "info"
    assert report.leakage["filename_overlap"]["groups"][0]["distinct_content_hashes"] == 2

    assert f["label_integrity.empty_classes"].details["classes"] == ["EMPTY"]
    assert "label_integrity.class_missing_from_split" in f
    imbalance = f["distribution.class_imbalance"]
    assert imbalance.severity.value == "medium"
    assert imbalance.details["ratio"] == pytest.approx(3.0)
    for code in ("missing_labels", "unknown_labels", "duplicate_sample_ids", "split_overlap"):
        assert f"label_integrity.{code}" not in f
    assert report.summary["findings_by_severity"]["high"] >= 4
    assert report.statistics["enabled"] and report.statistics["num_measured"] == 83

    files = sorted(p.name for p in result.directory.iterdir())
    assert files == ["exclusions.json", "metadata.json", "records.parquet", "report.json", "summary.txt"]
    assert len(json.loads((result.directory / "exclusions.json").read_text())) == 3
    assert json.loads((result.directory / "report.json").read_text())["schema_version"] == 1

    frame = pd.read_parquet(result.directory / "records.parquet")
    assert len(frame) == 85 and frame.sha256.notna().all()
    assert not frame.path.str.startswith("/").any()
    corrupt = frame[frame.sample_id == "train/GORE/corrupt.png"].iloc[0]
    assert corrupt.status == "unreadable" and corrupt.error_type == "UnidentifiedImageError"
    assert "<root>" in corrupt.error_message

    summary = (result.directory / "summary.txt").read_text()
    assert "leakage.exact_duplicate_leakage" in summary and "[HIGH]" in summary
    for name in ("report.json", "summary.txt", "exclusions.json", "metadata.json"):
        assert str(dataset) not in (result.directory / name).read_text()


def test_runs_are_deterministic_and_path_independent(dataset, tmp_path):
    moved = tmp_path / "elsewhere" / "ds_copy"
    shutil.copytree(dataset, moved)
    runs = [
        audit(dataset, tmp_path / "o1", "r1"),
        audit(dataset, tmp_path / "o2", "r2"),
        audit(moved, tmp_path / "o3", "r3"),
    ]
    for name in ("report.json", "summary.txt", "exclusions.json"):
        blobs = [(run.directory / name).read_bytes() for run in runs]
        assert blobs[0] == blobs[1] == blobs[2], name
    frames = [pd.read_parquet(run.directory / "records.parquet") for run in runs]
    pd.testing.assert_frame_equal(frames[0], frames[1])
    pd.testing.assert_frame_equal(frames[0], frames[2])
    for run in runs[2:]:
        assert str(moved) not in (run.directory / "report.json").read_text()

    with pytest.raises(ArtifactError, match="already exists"):
        audit(dataset, tmp_path / "o1", "r1")
    with pytest.raises(DatasetError, match="outside"):
        audit(moved, moved / "out", "inside")
    with pytest.raises(DatasetError, match="not found"):
        scan_folder(tmp_path / "nope", CFG)


def test_cli_audit_matches_the_api(dataset, tmp_path):
    api = audit(dataset, tmp_path / "api_out", "same")
    audit_config = tmp_path / "audit.yaml"
    audit_config.write_text("min_side: 16\nimbalance_ratio: 2.5\n")
    code = main([
        "audit", "--dataset", str(dataset), "--audit-config", str(audit_config),
        "--output", str(tmp_path / "cli_out"), "--audit-id", "same", "--dataset-id", "synthetic",
    ])
    assert code == 0
    cli_report = tmp_path / "cli_out" / "audits" / "same" / "report.json"
    assert cli_report.read_bytes() == (api.directory / "report.json").read_bytes()

    bad = tmp_path / "bad.yaml"
    bad.write_text("no_such_option: 1\n")
    assert main(["audit", "--dataset", str(dataset), "--audit-config", str(bad),
                 "--output", str(tmp_path / "x")]) == 1
    assert main(["audit", "--dataset", str(tmp_path / "missing"),
                 "--output", str(tmp_path / "x")]) == 1


def test_csv_audit_survives_bad_rows(dataset, tmp_path):
    rows = [
        "path,label,split",
        "train/GORE/g00.png,GORE,train",
        "train/GORE/g01.png,GORE,train",
        "train/GORE/g01.png,BLOOD,val",
        "train/BLOOD/does_not_exist.png,BLOOD,train",
        "/etc/passwd,GORE,train",
        "../outside.png,GORE,train",
        "train/GORE/g03.png,,train",
        "train/GORE/g04.png,gore,train",
        "train/BLOOD/b00.png,ALIEN,test",
        ",GORE,train",
        "train/GORE/corrupt.png,GORE,train",
    ]
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("\n".join(rows) + "\n")
    cfg = AuditConfig(class_names=("GORE", "BLOOD", "NEUTRAL"), min_side=16)
    before = snapshot(dataset)
    scan = scan_csv(csv_path, cfg, root=dataset, dataset_id="csv-synthetic")
    result = run_audit(scan, ArtifactStore(tmp_path / "out"), cfg, audit_id="csv")
    assert snapshot(dataset) == before

    report = result.report
    f = by_id(report)
    assert report.inventory["total_samples"] == 10
    assert report.inventory["by_status"] == {"invalid_path": 2, "missing": 1, "ok": 6, "unreadable": 1}
    assert report.scan["excluded_by_reason"] == {"empty_path": 1}
    assert ids(f["image_integrity.missing_files"]) == {"train/BLOOD/does_not_exist.png"}
    assert ids(f["image_integrity.invalid_paths"]) == {"row6", "row7"}
    assert ids(f["image_integrity.decode_failures"]) == {"train/GORE/corrupt.png"}
    assert ids(f["label_integrity.missing_labels"]) == {"train/GORE/g03.png"}
    unknown = f["label_integrity.unknown_labels"]
    assert ids(unknown) == {"train/GORE/g04.png", "train/BLOOD/b00.png"}
    assert unknown.details["labels"] == ["ALIEN", "gore"]
    assert f["label_integrity.label_variants"].details["variants"] == {"gore": ["GORE", "gore"]}
    assert ids(f["label_integrity.duplicate_sample_ids"]) == {"train/GORE/g01.png"}
    assert ids(f["label_integrity.split_overlap"]) == {"train/GORE/g01.png"}
    assert ids(f["label_integrity.conflicting_labels_identical_content"]) == {"train/GORE/g01.png"}
    assert report.inventory["duplicate_paths"]["num_groups"] == 1
    assert report.exact_duplicates["num_groups"] == 0
    assert report.near_duplicates["num_groups"] == 0
    assert (result.directory / "records.parquet").is_file()

    tiny = tmp_path / "tiny"
    tiny.mkdir()
    for name, seed in (("a.png", 1), ("b.png", 2), ("c.png", 3)):
        fixtures.base_image(seed).save(tiny / name)
    mapping = tiny / "map.csv"
    mapping.write_text("path,label,label_name\na.png,0,cat\nb.png,0,dog\nc.png,1,cat\n")
    cfg2 = AuditConfig(label_name_column="label_name", split_column=None, min_side=16)
    run2 = run_audit(scan_csv(mapping, cfg2), ArtifactStore(tmp_path / "out2"), cfg2, audit_id="map")
    f2 = by_id(run2.report)
    assert f2["label_integrity.inconsistent_class_mapping"].severity.value == "high"
    assert run2.report.leakage["applicable"] is False

    no_label = tmp_path / "nolabel.csv"
    no_label.write_text("path,other\na.png,1\n")
    with pytest.raises(DatasetError, match="column 'label'"):
        scan_csv(no_label, cfg2)
    with pytest.raises(DatasetError, match="not found"):
        scan_csv(tmp_path / "nope.csv", cfg2)


def test_adapter_bridge_and_path_normalisation(tmp_path):
    root = tmp_path / "small"
    for cls, seeds in (("a", (1, 2)), ("b", (3, 4))):
        for i, seed in enumerate(seeds):
            path = root / cls / f"x{i}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fixtures.base_image(seed).save(path)
    cfg = AuditConfig(min_side=16)
    adapter = ImageClassificationDataset.from_folder(root)
    via_adapter = scan_adapter(adapter, root, cfg)
    via_folder = scan_folder(root, cfg)

    def view(scan):
        return sorted((r.sample_id, r.path, r.label, r.split) for r in scan.records)

    assert view(via_adapter) == view(via_folder)
    assert via_adapter.declared_classes == ["a", "b"]
    result = run_audit(via_adapter, ArtifactStore(tmp_path / "out"), cfg, audit_id="adapter")
    assert result.report.inventory["total_samples"] == 4
    assert result.report.inventory["by_status"] == {"ok": 4}

    assert canonical_path("a\\b/./c.png") == "a/b/c.png"
    assert canonical_path("e\u0301.png") == "\u00e9.png"
    for bad in ("", "/abs/x.png", "C:\\x.png", "../x.png", "a/../b.png", "./"):
        assert canonical_path(bad) is None


def test_near_duplicate_search_matches_brute_force():
    rng = random.Random(7)
    base = [rng.getrandbits(64) for _ in range(200)]
    for threshold in (0, 2, 4, 9, 16):
        values = list(base)
        for value in base[:60]:
            variant = value
            for bit in rng.sample(range(64), rng.randint(1, max(1, threshold))):
                variant ^= 1 << bit
            values.append(variant)
        values = sorted(set(values))
        edges, checked, skipped = near_duplicate_edges(values, threshold, 10_000)
        expected = set()
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                dist = bin(values[i] ^ values[j]).count("1")
                if dist <= threshold:
                    expected.add((i, j, dist))
        assert set(edges) == expected and len(edges) == len(expected)
        assert skipped == 0
        if threshold:
            assert expected

    edges, checked, skipped = near_duplicate_edges([1, 3, 7, 15, 31], 4, 2)
    assert skipped > 0
