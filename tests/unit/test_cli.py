import subprocess
import sys

import pytest

import modellab
from modellab.cli.main import main


def test_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "version" in out and "doctor" in out


def test_version(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == modellab.__version__


def test_no_command_is_an_error():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_doctor_ok(tmp_path, capsys):
    art = tmp_path / "art"
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"paths:\n  artifacts: '{art}'\n")
    assert main(["doctor", "--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "[ok]" in out and "[fail]" not in out
    assert art.is_dir()


def test_doctor_bad_config(tmp_path, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("runtime:\n  device: tpu\n")
    assert main(["doctor", "--config", str(cfg)]) == 1
    assert "[fail]" in capsys.readouterr().out


def test_doctor_unwritable_artifacts(tmp_path, capsys):
    blocker = tmp_path / "f"
    blocker.write_text("x")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"paths:\n  artifacts: '{blocker / 'sub'}'\n")
    assert main(["doctor", "--config", str(cfg)]) == 1
    assert "not writable" in capsys.readouterr().out


def test_module_entry_point():
    res = subprocess.run(
        [sys.executable, "-m", "modellab", "version"], capture_output=True, text=True
    )
    assert res.returncode == 0
    assert res.stdout.strip() == modellab.__version__
