import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "apply_phase6_patch.py"
ANCHOR = "    add_pipeline_routes(api, ws, jobs, cache, store, enqueue, art, read)\n"
APP = (
    "def create_app():\n"
    "    from modellab.server.pipeline_routes import add_pipeline_routes\n\n"
    + ANCHOR +
    "    from modellab.server.model_routes import add_model_routes\n"
    "    add_model_routes(api, ws, settings, cache, require_local)\n"
    "    return app\n"
)


def run(path):
    return subprocess.run([sys.executable, str(SCRIPT), str(path)], capture_output=True, text=True)


def test_patch_inserts_once_and_backs_up(tmp_path):
    app = tmp_path / "app.py"
    app.write_text(APP)
    first = run(app)
    assert first.returncode == 0 and "patched" in first.stdout
    text = app.read_text()
    compile(text, "app.py", "exec")
    assert text.count("add_ai_routes(api, ws, jobs, cache, store)") == 1
    assert text.index("add_pipeline_routes(api") < text.index("add_ai_routes(api") < text.index("add_model_routes(api")
    assert (tmp_path / "app.py.bak_phase6").read_text() == APP
    again = run(app)
    assert again.returncode == 0 and "already" in again.stdout
    assert app.read_text() == text


def test_patch_refuses_when_anchor_missing(tmp_path):
    app = tmp_path / "app.py"
    app.write_text("def create_app():\n    return None\n")
    result = run(app)
    assert result.returncode != 0
    assert app.read_text() == "def create_app():\n    return None\n"
    assert not (tmp_path / "app.py.bak_phase6").exists()
