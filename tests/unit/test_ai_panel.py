from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.ai_fixtures import GOOD, add_context, make_env  # noqa: E402

PANEL = str(Path(__file__).resolve().parents[2] / "ai_panel.py")


def button(at, label):
    return next(b for b in at.button if b.label == label)


def has_button(at, label):
    return any(b.label == label for b in at.button)


def all_text(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [e.value for e in at.error]
    return " ".join(parts)


@pytest.fixture
def app(tmp_path):
    env = make_env(tmp_path, [GOOD, "Handgun recall is 0.013.", GOOD])
    add_context(tmp_path)
    patcher = patch("requests.Session", lambda: env.http)
    patcher.start()
    at = AppTest.from_file(PANEL, default_timeout=30)
    at.run()
    yield env, at
    patcher.stop()


def test_landing_page_and_sidebar(app):
    env, at = app
    assert not at.exception
    assert "Why is my detector failing" in all_text(at)
    assert at.sidebar.selectbox[0].label == "Experiment family"
    assert at.sidebar.selectbox[0].options == ["weapons-final-model (0 runs)"]
    assert [s.label for s in at.sidebar.selectbox] == [
        "Experiment family", "Dataset audit (optional)", "Failure analysis (optional)"]
    assert at.sidebar.selectbox[1].options == ["None", "aud1"]
    assert at.sidebar.selectbox[2].options[:2] == ["Latest for the baseline", "fa1"]
    assert [b.label for b in at.sidebar.button] == ["Start chat"]
    assert not at.chat_input


def test_a_whole_conversation(app):
    env, at = app
    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    assert at.chat_input and "family weapons-final-model" in all_text(at)
    assert [b.label for b in at.button if b.label.startswith(("Why", "Propose", "How"))] == [
        "Why is my detector weak?", "Propose the next experiment", "How are the classes doing?"]

    button(at, "Why is my detector weak?").click().run()
    assert not at.exception, at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assert [m.label for m in at.metric] == ["Precision", "Recall", "F1"]
    assert has_button(at, "Approve and run") and has_button(at, "Reject")
    assert "validation passed" in all_text(at)

    button(at, "Approve and run").click().run()
    assert not at.exception, at.exception
    assert env.services.calls and not has_button(at, "Approve and run")
    text = all_text(at)
    assert "improved" in text and "vs control" in text

    at.chat_input[0].set_value("How is handgun doing?").run()
    assert not at.exception, at.exception
    assert [m.name for m in at.chat_message][-2:] == ["user", "assistant"]
    assert any("Handgun recall is 0.013." in m.value for m in at.chat_message[-1].markdown)
    assert len(env.client.list_chats()) == 1


def test_reject_then_ask_for_another(app):
    env, at = app
    at.sidebar.button[0].click().run()
    button(at, "Why is my detector weak?").click().run()
    button(at, "Reject").click().run()
    assert not at.exception and "Rejected" in all_text(at)
    assert not has_button(at, "Approve and run")
    at.chat_input[0].set_value("propose another experiment").run()
    assert not at.exception, at.exception
    assert has_button(at, "Approve and run")
    chat = env.client.list_chats()[0]
    ids = [m["research"]["result"]["proposal"]["candidate_id"]
           for m in env.client.get_chat(chat["id"])["messages"] if m["kind"] == "research"]
    assert len(ids) == 2 and ids[0] != ids[1]


def test_report_request_creates_a_report_message(app):
    env, at = app
    at.sidebar.button[0].click().run()
    button(at, "Why is my detector weak?").click().run()
    at.chat_input[0].set_value("export a report").run()
    assert not at.exception, at.exception
    msgs = env.client.get_chat(env.client.list_chats()[0]["id"])["messages"]
    assert msgs[-1]["kind"] == "report" and "# ModelLab AI research report" in msgs[-1]["report"]
    assert "Preview" in [e.label for e in at.expander]


def test_prepare_report_button_on_the_card(app):
    env, at = app
    at.sidebar.button[0].click().run()
    button(at, "Why is my detector weak?").click().run()
    button(at, "Prepare report").click().run()
    assert not at.exception, at.exception
    assert not has_button(at, "Prepare report")


def test_context_is_chosen_at_start_and_changeable(app):
    env, at = app
    at.sidebar.selectbox[1].select("aud1")
    at.sidebar.selectbox[2].select("fa1")
    at.sidebar.button[0].click().run()
    text = all_text(at)
    assert "audit aud1" in text and "failure analysis fa1" in text

    at.selectbox(key="ctx_audit").select("None")
    button(at, "Apply").click().run()
    assert not at.exception, at.exception
    assert "audit none" in all_text(at)
    assert env.client.list_chats()[0]["messages"] == 1


def test_context_can_be_attached_after_the_chat_started(app):
    env, at = app
    at.sidebar.button[0].click().run()
    assert "audit none" in all_text(at)
    at.selectbox(key="ctx_audit").select("aud1")
    at.selectbox(key="ctx_analysis").select("fa1")
    button(at, "Apply").click().run()
    assert not at.exception, at.exception
    text = all_text(at)
    assert "audit aud1" in text and "failure analysis fa1" in text
    chat = env.client.get_chat(env.client.list_chats()[0]["id"])
    assert (chat["audit_id"], chat["analysis_id"]) == ("aud1", "fa1")


def test_server_errors_are_shown_not_raised(app):
    env, at = app
    at.sidebar.selectbox[1].select("aud1")
    at.sidebar.button[0].click().run()
    import shutil
    shutil.rmtree(env.root / "audits" / "aud1")
    button(at, "Why is my detector weak?").click().run()
    assert not at.exception, at.exception
    assert any("422" in e.value for e in at.error)
    assert at.chat_input


def test_switching_between_chats(app):
    env, at = app
    at.sidebar.button[0].click().run()
    button(at, "Why is my detector weak?").click().run()
    first_title = env.client.list_chats()[0]["title"]
    at.sidebar.button[0].click().run()
    assert not at.chat_message
    open_buttons = [b for b in at.sidebar.button if b.label == first_title]
    assert open_buttons
    open_buttons[0].click().run()
    assert not at.exception and len(at.chat_message) == 2


def test_backend_down_shows_one_clear_error():
    class Resp:
        status_code = 502
        text = "bad gateway"

        def json(self):
            raise ValueError

    class Dead:
        headers = {}

        def __init__(self):
            self.headers = type("H", (), {"update": lambda self, d: None})()

        def get(self, *a, **k):
            return Resp()

        post = get

    with patch("requests.Session", Dead):
        at = AppTest.from_file(PANEL, default_timeout=30)
        at.run()
    assert not at.exception
    assert any("502" in e.value for e in at.sidebar.error)
