import pytest
from pydantic import BaseModel

from modellab.ai.llm.provider import LLMError, ScriptedLLM, extract_json, merge_system, provider_from_env


class Pick(BaseModel):
    name: str
    score: float


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('Sure! {"a": {"b": 2}} hope that helps') == '{"a": {"b": 2}}'


def test_valid_reply_first_try():
    llm = ScriptedLLM(['{"name": "x", "score": 0.5}'])
    assert llm.generate_structured("sys", "user", Pick).name == "x"
    assert len(llm.calls) == 1


def test_bad_json_then_good_retries_with_feedback():
    llm = ScriptedLLM(["not json", '{"name": "x", "score": 1}'])
    assert llm.generate_structured("sys", "user", Pick).score == 1
    last = llm.calls[1][-1]
    assert last["role"] == "user" and "rejected" in last["content"]


def test_check_failure_is_retried():
    llm = ScriptedLLM(['{"name": "bad", "score": 1}', '{"name": "good", "score": 1}'])
    out = llm.generate_structured("s", "u", Pick, check=lambda o: None if o.name == "good" else "name must be good")
    assert out.name == "good"
    assert "name must be good" in llm.calls[1][-1]["content"]


def test_gives_up_after_retries():
    llm = ScriptedLLM(["x", "y", "z"], max_retries=2)
    with pytest.raises(LLMError):
        llm.generate_structured("s", "u", Pick)
    assert len(llm.calls) == 3


def test_extra_fields_rejected_by_schema_strictness():
    class Strict(BaseModel):
        model_config = {"extra": "forbid"}
        name: str

    llm = ScriptedLLM(['{"name": "a", "other": 1}', '{"name": "a"}'])
    assert llm.generate_structured("s", "u", Strict).name == "a"
    assert len(llm.calls) == 2


def test_merge_system_folds_into_first_user_message():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"}, {"role": "user", "content": "U2"}]
    merged = merge_system(msgs)
    assert [m["role"] for m in merged] == ["user", "assistant", "user"]
    assert merged[0]["content"] == "S\n\nU"
    assert msgs[0]["role"] == "system"


def test_factory_errors():
    with pytest.raises(LLMError):
        provider_from_env({"MODELLAB_LLM_BACKEND": "nope"})
    with pytest.raises(LLMError):
        provider_from_env({"MODELLAB_LLM_BACKEND": "openai_compat"})
