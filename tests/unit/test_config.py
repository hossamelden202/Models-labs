from pathlib import Path

import pytest

from modellab.config import Config, load_config
from modellab.core.errors import ConfigurationError


def test_defaults():
    cfg = Config()
    assert cfg.project.seed == 42
    assert cfg.runtime.device == "auto"
    assert cfg.paths.artifacts == Path("artifacts")


def test_no_path_returns_defaults():
    assert load_config(None) == Config()


def test_empty_file_returns_defaults(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("")
    assert load_config(f) == Config()


def test_valid_partial_config(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("project:\n  seed: 7\nruntime:\n  device: cpu\npaths:\n  artifacts: out\n")
    cfg = load_config(f)
    assert cfg.project.seed == 7
    assert cfg.runtime.device == "cpu"
    assert cfg.paths.artifacts == Path("out")
    assert cfg.paths.data == Path("data")


def test_shipped_default_file_is_valid():
    root = Path(__file__).resolve().parents[2]
    assert load_config(root / "configs" / "default.yaml") == Config()


@pytest.mark.parametrize(
    "text",
    [
        "runtime:\n  device: tpu\n",
        "runtime:\n  log_level: LOUD\n",
        "project:\n  seed: -1\n",
        "project:\n  seed: abc\n",
        "unknown: 1\n",
        "- a\n- b\n",
        "a: [unclosed\n",
    ],
)
def test_invalid_config_raises(tmp_path, text):
    f = tmp_path / "c.yaml"
    f.write_text(text)
    with pytest.raises(ConfigurationError):
        load_config(f)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigurationError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_error_names_the_bad_field(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("runtime:\n  device: tpu\n")
    with pytest.raises(ConfigurationError, match=r"runtime\.device"):
        load_config(f)
