from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.ai_fixtures import make_env  # noqa: E402

PANEL = str(Path(__file__).resolve().parents[2] / "ai_panel.py")


def button(at, label):
    return next(b for b in at.button if b.label == label)


@pytest.fixture
def app(tmp_path):
    env = make_env(tmp_path)
    with patch("requests.Session", lambda: env.http):
        at = AppTest.from_file(PANEL, default_timeout=30)
        at.run()
        yield env, at


def test_panel_full_flow(app):
    env, at = app
    assert not at.exception
    assert any("AI research" in h.value for h in at.header)

    button(at, "Run research").click().run()
    assert not at.exception, at.exception
    assert any("Validation passed" in s.value for s in at.success)
    assert any(s.value == "Hypothesis" for s in at.subheader)
    assert len(env.client.list_research()) == 1
    tables = [t.value.to_dict("records") for t in at.table]
    assert any(r.get("path") == "data.input_width" and r.get("value") == 416 for t in tables for r in t)

    assert not any("confidence" in c.value for c in at.caption)
    assert any(c.value.startswith("Chosen by") and "evidence low" in c.value for c in at.caption)
    assert any("Why this experiment" in e.label for e in at.expander)
    assert at.checkbox[0].value is True

    button(at, "Approve and run").click().run()
    assert not at.exception, at.exception
    assert env.services.calls and env.services.calls[0]["model_id"] == "weapons-final-model"
    assert [s["name"] for s in env.services.calls[0]["specs"]][0] == "advisor_control_default"
    text = " ".join(m.value for m in at.markdown)
    assert "improved" in text and "consistent with the hypothesis" in text
    assert any("Compared against: control" in c.value for c in at.caption)
    assert not [b for b in at.button if b.label == "Approve and run"]


def test_panel_reject(app):
    env, at = app
    button(at, "Run research").click().run()
    button(at, "Reject").click().run()
    assert not at.exception
    assert env.client.list_research()[0]["status"] == "rejected"
    assert not [b for b in at.button if b.label == "Approve and run"]


def test_panel_shows_backend_errors(tmp_path):
    env = make_env(tmp_path)
    with patch("requests.Session", lambda: env.http):
        at = AppTest.from_file(PANEL, default_timeout=30)
        at.run()
        next(t for t in at.text_input if t.label == "Family id").set_value("no-such-family").run()
        button(at, "Run research").click().run()
        assert not at.exception
        assert any("422" in e.value and "no-such-family" in e.value for e in at.error)
        assert env.client.list_research() == []


def test_panel_can_skip_the_control(app):
    env, at = app
    button(at, "Run research").click().run()
    at.checkbox[0].uncheck().run()
    button(at, "Approve and run").click().run()
    assert not at.exception, at.exception
    assert len(env.services.calls[0]["specs"]) == 1
    assert any("Compared against: baseline" in c.value for c in at.caption)
