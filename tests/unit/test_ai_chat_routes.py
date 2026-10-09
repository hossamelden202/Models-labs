import shutil

import pytest

from ai_client import AiApiError
from modellab.ai.llm.provider import LLMError
from tests.ai_fixtures import FAMILY, GOOD, Env, add_context, make_env, put, BASE_METRICS


def last(chat):
    return chat["messages"][-1]


def user_prompt(env, i):
    return env.llm.calls[i][-1]["content"]


def test_create_chat_validates_family_and_context(tmp_path):
    env = make_env(tmp_path)
    add_context(tmp_path)
    for kwargs in ({"family_id": "nope"}, {"family_id": "../x"},
                   {"family_id": FAMILY, "audit_id": "missing"},
                   {"family_id": FAMILY, "analysis_id": "missing"}):
        with pytest.raises(AiApiError) as e:
            env.client.create_chat(**kwargs)
        assert e.value.status == 422
    chat = env.client.create_chat(FAMILY, "aud1", "fa1")
    assert (chat["audit_id"], chat["analysis_id"], chat["messages"]) == ("aud1", "fa1", [])


def test_context_options_lists_choices_and_prefers_matching_analysis(tmp_path):
    env = make_env(tmp_path)
    add_context(tmp_path)
    opts = env.client.options(FAMILY)
    assert opts["families"][0]["family_id"] == FAMILY and opts["families"][0]["model_id"] == "weapons-final-model"
    assert [a["audit_id"] for a in opts["audits"]] == ["aud1"]
    assert [(a["analysis_id"], a["matches_baseline"]) for a in opts["analyses"]] == [("fa1", True), ("other", False)]
    assert all(a["matches_baseline"] is False for a in env.client.options()["analyses"])
    with pytest.raises(AiApiError):
        env.client.options("nope")


def test_ask_gets_context_and_remembers_the_conversation(tmp_path):
    env = make_env(tmp_path, ["Handgun recall is 0.013, almost every handgun is missed.", "Knife recall is 0.202."])
    add_context(tmp_path)
    chat = env.client.create_chat(FAMILY, "aud1", "fa1")

    chat = env.client.send(chat["id"], "Why is handgun recall so low?")
    assert [m["role"] for m in chat["messages"]] == ["user", "assistant"]
    assert last(chat)["kind"] == "text" and last(chat)["warning"] is None
    prompt = user_prompt(env, 0)
    assert "failure_analysis:fa1" in prompt and "DATASET AUDIT" in prompt and "dup_pairs" in prompt
    assert "500 items" in prompt and "h499" not in prompt
    assert "FAILURE ANALYSIS REPORT" in prompt

    env.client.send(chat["id"], "And the knife?")
    assert [m["role"] for m in env.llm.calls[1]] == ["system", "user", "assistant", "user"]
    assert env.llm.calls[1][1]["content"] == "Why is handgun recall so low?"


def test_numbers_not_in_the_data_are_flagged(tmp_path):
    env = make_env(tmp_path, ["Handgun recall is 0.77.", "Handgun recall is 0.013."])
    chat = env.client.create_chat(FAMILY)
    first = last(env.client.send(chat["id"], "How is handgun doing?"))
    assert "0.77" in first["warning"]
    assert last(env.client.send(chat["id"], "And again?"))["warning"] is None


def test_ask_falls_back_when_backend_is_missing_or_fails(tmp_path):
    put(tmp_path, f"experiments/{FAMILY}/baseline.json",
        {"task": "detection", "evaluation_id": "ev1", "metrics": BASE_METRICS, "metadata": {}})
    env = Env(tmp_path, LLMError("model file missing"))
    chat = env.client.create_chat(FAMILY)
    reply = last(env.client.send(chat["id"], "how is handgun doing?"))
    assert reply["text"].startswith("The language model is not available.") and "handgun" in reply["text"]

    env2 = make_env(tmp_path / "second", [])
    chat2 = env2.client.create_chat(FAMILY)
    assert "could not answer" in last(env2.client.send(chat2["id"], "how is handgun doing?"))["text"]


def test_propose_creates_a_research_card(tmp_path):
    env = make_env(tmp_path, [GOOD])
    chat = env.client.create_chat(FAMILY)
    chat = env.client.send(chat["id"], "Why is my weapon detector weak?")
    msg = last(chat)
    assert msg["kind"] == "research" and msg["research"]["status"] == "proposed"
    assert msg["research"]["result"]["validation"]["valid"] is True
    assert msg["research"]["result"]["proposal"]["candidate_id"] == "res_416"
    assert chat["title"] == "Why is my weapon detector weak?"


def test_propose_without_backend_still_works_and_says_so(tmp_path):
    put(tmp_path, f"experiments/{FAMILY}/baseline.json",
        {"task": "detection", "evaluation_id": "ev1", "metrics": BASE_METRICS, "metadata": {}})
    env = Env(tmp_path, LLMError("model file missing"))
    chat = env.client.create_chat(FAMILY)
    res = last(env.client.send(chat["id"], "Why is my weapon detector weak?"))["research"]["result"]
    assert res["llm_used"] is False and res["validation"]["valid"] is True
    assert any("model file missing" in n for n in res["notes"])


def test_asking_again_gives_a_different_experiment_until_options_run_out(tmp_path):
    env = make_env(tmp_path, [GOOD] * 60)
    chat = env.client.create_chat(FAMILY)
    seen = []
    for _ in range(12):
        chat = env.client.send(chat["id"], "propose another experiment")
        msg = last(chat)
        if msg["kind"] != "research":
            assert msg["text"].startswith("I have no further experiment")
            break
        seen.append(msg["research"]["result"]["proposal"]["candidate_id"])
    else:
        pytest.fail("options never ran out")
    assert len(seen) == len(set(seen)) == 8


def test_audit_digest_reaches_the_hypothesis_prompt(tmp_path):
    env = make_env(tmp_path, [GOOD])
    add_context(tmp_path)
    chat = env.client.create_chat(FAMILY, audit_id="aud1")
    env.client.send(chat["id"], "Why is my weapon detector weak?")
    prompt = user_prompt(env, 0)
    assert "Dataset audit (raw digest)" in prompt and "dup_pairs" in prompt


def test_report_before_and_after_research(tmp_path):
    env = make_env(tmp_path, [GOOD])
    chat = env.client.create_chat(FAMILY)
    chat = env.client.send(chat["id"], "export a report")
    assert last(chat)["kind"] == "text" and "nothing to report" in last(chat)["text"]

    chat = env.client.send(chat["id"], "Why is my weapon detector weak?")
    rid = last(chat)["research_id"]
    chat = env.client.send(chat["id"], "export a report")
    msg = last(chat)
    assert msg["kind"] == "report" and msg["research_id"] == rid
    md = msg["report"]
    for needle in ("# ModelLab AI research report", FAMILY, "## Hypothesis", "## Proposed experiment",
                   "## Options considered", "## Conversation", "Why is my weapon detector weak?"):
        assert needle in md
    plain = env.client.report(rid)
    assert plain.startswith("# ModelLab AI research report") and "## Conversation" not in plain
    assert env.http.get(f"/ai/research/{rid}/report").headers["content-type"].startswith("text/markdown")


def test_chat_shows_the_finding_after_approval_and_report_includes_it(tmp_path):
    env = make_env(tmp_path, [GOOD])
    chat = env.client.create_chat(FAMILY)
    chat = env.client.send(chat["id"], "Why is my weapon detector weak?")
    env.client.approve(last(chat)["research_id"])
    card = last(env.client.get_chat(chat["id"]))["research"]
    assert card["status"] == "approved" and card["finding"]["verdict"] == "improved"
    md = env.client.report(card["id"])
    for needle in ("## Execution", "## Finding", "**improved**", "control", "| recall |"):
        assert needle in md


def test_report_escapes_table_cells(tmp_path):
    env = make_env(tmp_path, [GOOD])
    chat = env.client.create_chat(FAMILY)
    chat = env.client.send(chat["id"], "Why is my weapon detector weak?")
    rid = last(chat)["research_id"]
    path = tmp_path / "ai_research" / rid / "research.json"
    import json
    doc = json.loads(path.read_text())
    doc["result"]["ranking"][0]["reasons"] = ["a | b\nc"]
    path.write_text(json.dumps(doc))
    assert "a \\| b c" in env.client.report(rid)


def test_set_context_changes_what_later_messages_see(tmp_path):
    env = make_env(tmp_path, [GOOD])
    add_context(tmp_path)
    chat = env.client.create_chat(FAMILY)
    chat = env.client.set_context(chat["id"], "aud1", "fa1")
    assert (chat["audit_id"], chat["analysis_id"]) == ("aud1", "fa1")
    assert last(chat)["text"].startswith("Context updated")
    env.client.send(chat["id"], "Why is my weapon detector weak?")
    assert "Dataset audit (raw digest)" in user_prompt(env, 0)
    with pytest.raises(AiApiError) as e:
        env.client.set_context(chat["id"], "missing", None)
    assert e.value.status == 422
    cleared = env.client.set_context(chat["id"], None, None)
    assert (cleared["audit_id"], cleared["analysis_id"]) == (None, None)


def test_chats_are_persisted_and_listed(tmp_path):
    env = make_env(tmp_path, ["Some answer 0.155."])
    chat = env.client.create_chat(FAMILY)
    env.client.send(chat["id"], "How is handgun doing overall in this evaluation set today please?")
    rows = env.client.list_chats()
    assert rows[0]["id"] == chat["id"] and rows[0]["messages"] == 2
    assert len(rows[0]["title"]) <= 60 and rows[0]["title"].startswith("How is handgun")
    assert env.client.get_chat(chat["id"])["messages"][0]["text"].startswith("How is handgun")
    assert (tmp_path / "ai_chats" / chat["id"] / "chat.json").is_file()


def test_chat_errors(tmp_path):
    env = make_env(tmp_path)
    with pytest.raises(AiApiError) as e:
        env.client.get_chat("chat_nope")
    assert e.value.status == 404
    chat = env.client.create_chat(FAMILY)
    for text in ("", "x" * 2001):
        assert env.http.post(f"/ai/chats/{chat['id']}/messages", json={"text": text}).status_code == 422
    assert env.http.post(f"/ai/chats/{chat['id']}/messages", json={"text": "hi", "extra": 1}).status_code == 422


def test_failed_message_is_not_saved(tmp_path):
    env = make_env(tmp_path)
    add_context(tmp_path)
    chat = env.client.create_chat(FAMILY, audit_id="aud1")
    shutil.rmtree(tmp_path / "audits" / "aud1")
    with pytest.raises(AiApiError) as e:
        env.client.send(chat["id"], "Why is my weapon detector weak?")
    assert e.value.status == 422
    assert env.client.get_chat(chat["id"])["messages"] == []
