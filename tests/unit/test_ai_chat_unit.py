import pytest

from modellab.ai.chat import route_intent, ungrounded_numbers
from modellab.ai.digest import digest_json
from modellab.ai.llm.provider import LLMError, NoLLM, ScriptedLLM

NAMES = ["handgun", "knife", "sword"]


@pytest.mark.parametrize("text,expected", [
    ("Why is my weapon detector weak?", "propose"),
    ("why is the model so bad", "propose"),
    ("Propose the next experiment", "propose"),
    ("try something else", "propose"),
    ("give me another one", "propose"),
    ("What should I change?", "propose"),
    ("export a report", "report"),
    ("Can I download the write-up?", "report"),
    ("why is handgun recall so low?", "ask"),
    ("what does the audit say about duplicates", "ask"),
    ("how many images are in the evaluation set", "ask"),
    ("explain the last result", "ask"),
    ("hello", "ask"),
])
def test_route_intent(text, expected):
    assert route_intent(text, NAMES) == expected


def test_report_wins_over_propose():
    assert route_intent("write a report and suggest next steps", NAMES) == "report"


CONTEXT = "Overall precision 0.165, recall 0.155, f1 0.160 over 472 samples. knife recall 0.202. 190 of 238 missed."


@pytest.mark.parametrize("answer,bad", [
    ("Recall is 0.155 and precision is 0.165.", []),
    ("Recall is about 15.5% and knife recall is 20.2%.", []),
    ("Recall is 0.42 which is poor.", ["0.42"]),
    ("Recall is 0.155 but f1 is 0.31 and 40% of objects are missed.", ["0.31", "40%"]),
    ("190 of 238 knives were missed.", []),
    ("Version 2 of the model had 999 images.", []),
    ("Recall is 0.1551.", []),
    ("Recall is 0.155. Knife recall is 20.2%.", []),
    ("Missed 190 (79.8%).", ["79.8%"]),
    ("It went to version 1.2.3 of the tool.", []),
    ("About 42.5%.", ["42.5%"]),
])
def test_ungrounded_numbers(answer, bad):
    assert ungrounded_numbers(answer, CONTEXT) == bad


def test_integers_in_the_data_do_not_ground_look_alike_decimals():
    context = "77 of 78 handgun objects were missed. Recall 0.013."
    assert ungrounded_numbers("Recall is 0.77.", context) == ["0.77"]
    assert ungrounded_numbers("Recall is 0.013.", context) == []


def test_percent_in_the_data_grounds_the_decimal_form():
    assert ungrounded_numbers("Recall is 0.155.", "Recall is 15.5% on this set.") == []
    assert ungrounded_numbers("Recall is 15.5%.", "Recall is 0.155 on this set.") == []


def test_ungrounded_numbers_are_deduplicated_and_capped():
    answer = " ".join(f"{i}.77" for i in range(1, 10)) + " 1.77"
    found = ungrounded_numbers(answer, "nothing here")
    assert len(found) == 5 and len(set(found)) == 5


def test_digest_is_bounded_and_keeps_the_head():
    doc = {"summary": {"duplicates": 12, "ratio": 0.123456789},
           "rows": [{"sample": i, "hash": "x" * 300} for i in range(500)],
           "big": {f"k{i}": i for i in range(100)}}
    text = digest_json(doc, 600)
    assert len(text) <= 600
    assert '"duplicates":12' in text and "0.1235" in text
    assert "500 items" in digest_json(doc, 2000)
    assert digest_json(doc, 600) == text


def test_digest_handles_scalars_and_tiny_budgets():
    assert digest_json(5) == "5"
    assert digest_json("x" * 500, 50).endswith("...") or len(digest_json("x" * 500, 50)) <= 50
    assert len(digest_json({"a": list(range(1000))}, 20)) <= 20


def test_generate_text_passes_history_and_returns_reply():
    llm = ScriptedLLM(["  The answer.  "])
    out = llm.generate_text("sys", "question", [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "reply"}])
    assert out == "The answer."
    roles = [m["role"] for m in llm.calls[0]]
    assert roles == ["system", "user", "assistant", "user"]
    assert llm.calls[0][-1]["content"] == "question"


def test_generate_text_rejects_empty_and_no_backend():
    with pytest.raises(LLMError):
        ScriptedLLM(["   "]).generate_text("s", "q")
    with pytest.raises(LLMError):
        NoLLM().generate_text("s", "q")
