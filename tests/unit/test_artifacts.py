import json
import math

import pytest
from pydantic import BaseModel

from modellab.core.errors import ArtifactError
from modellab.core.results import ArtifactStore


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "art")


def test_root_is_created(tmp_path):
    root = tmp_path / "a" / "b"
    ArtifactStore(root)
    assert root.is_dir()


def test_write_json_returns_location(store):
    path = store.write_json("evaluations/e1/metrics.json", {"acc": 0.5})
    assert path.is_file()
    assert path.is_relative_to(store.root)
    assert json.loads(path.read_text()) == {"acc": 0.5}


def test_write_text_bytes_and_overwrite(store):
    store.write_text("a.txt", "one")
    path = store.write_text("a.txt", "two")
    assert path.read_text() == "two"
    assert store.write_bytes("b.bin", b"\x00\x01").read_bytes() == b"\x00\x01"
    assert sorted(p.name for p in store.root.iterdir()) == ["a.txt", "b.bin"]


def test_pydantic_model_is_serialized(store):
    class M(BaseModel):
        x: int

    path = store.write_json("m.json", M(x=3))
    assert json.loads(path.read_text()) == {"x": 3}


@pytest.mark.parametrize("bad", ["../x.json", "a/../../x.json", "", "."])
def test_escaping_paths_rejected(store, bad):
    with pytest.raises(ArtifactError):
        store.write_text(bad, "x")


def test_absolute_path_rejected(store, tmp_path):
    with pytest.raises(ArtifactError):
        store.write_text(str(tmp_path / "abs.txt"), "x")


def test_unserializable_and_nan_rejected(store):
    with pytest.raises(ArtifactError):
        store.write_json("x.json", {"a": object()})
    with pytest.raises(ArtifactError):
        store.write_json("x.json", {"a": math.nan})
    assert not (store.root / "x.json").exists()


def test_directory_target_and_file_parent(store):
    (store.root / "d.json").mkdir()
    with pytest.raises(ArtifactError):
        store.write_text("d.json", "x")
    store.write_text("f.txt", "x")
    with pytest.raises(ArtifactError):
        store.write_text("f.txt/inner.txt", "y")
    assert not list(store.root.glob("*.tmp"))


def test_root_under_a_file(tmp_path):
    f = tmp_path / "f"
    f.write_text("x")
    with pytest.raises(ArtifactError):
        ArtifactStore(f / "sub")
