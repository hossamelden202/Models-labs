import pytest

from modellab.ai.catalog import build_menu, check_value
from modellab.ai.schemas import Change
from modellab.ai.validation import validate_changes
from modellab.experiments.training_config import TrainingChange, TrainingConfig, apply_training_changes


def changes(**pairs):
    return [Change(path=p.replace("__", "."), value=v) for p, v in pairs.items()]


def check(cs):
    return validate_changes(cs, "t", "claim")


def test_resolution_change_is_valid():
    res = check(changes(data__input_width=416, data__input_height=416))
    assert res.valid, res.errors
    assert res.experiment_id.startswith("exp_")
    assert res.spec["intervention"]["kind"] == "training"


def test_same_changes_same_id():
    a = check(changes(epochs=30))
    b = check(changes(epochs=30))
    assert a.experiment_id == b.experiment_id
    assert a.experiment_id != check(changes(epochs=31)).experiment_id


@pytest.mark.parametrize("path,value", [
    ("freeze.mode", "all_but_head"),
    ("loss.label_smoothing", 0.1),
    ("max_grad_norm", 1.0),
    ("data.sampler", "weighted"),
    ("data.augmentation.blur_radius", 2.0),
    ("gradient_accumulation_steps", 2),
    ("made_up.path", 1),
])
def test_unsupported_paths_rejected(path, value):
    res = check([Change(path=path, value=value)])
    assert not res.valid
    assert res.errors[0].startswith(path)


@pytest.mark.parametrize("path,value", [
    ("freeze.mode", "all_but_head"),
    ("loss.label_smoothing", 0.1),
    ("max_grad_norm", 1.0),
    ("data.sampler", "weighted"),
    ("data.augmentation.blur_radius", 2.0),
    ("gradient_accumulation_steps", 2),
])
def test_unsupported_paths_are_valid_training_config(path, value):
    apply_training_changes(TrainingConfig(), [TrainingChange(path=path, value=value)])


@pytest.mark.parametrize("path,value", [
    ("optimizer.learning_rate", 5.0),
    ("optimizer.learning_rate", 0.0),
    ("epochs", True),
    ("epochs", 2.5),
    ("epochs", 0),
    ("data.batch_size", 7),
    ("scheduler.name", "one_cycle"),
    ("data.augmentation.horizontal_flip", 1),
    ("optimizer.name", "lion"),
])
def test_bad_values_rejected(path, value):
    assert check_value(path, value) is not None
    assert not check([Change(path=path, value=value)]).valid


def test_width_and_height_must_match():
    assert not check(changes(data__input_width=416)).valid
    assert not check(changes(data__input_width=416, data__input_height=320)).valid


def test_duplicates_empty_and_too_many():
    assert not check([]).valid
    assert not check(changes(epochs=5) + changes(epochs=6)).valid
    many = changes(epochs=5, optimizer__learning_rate=1e-4, scheduler__name="cosine",
                   data__batch_size=16, mixed_precision=False)
    assert not check(many).valid


def test_empty_claim_rejected():
    assert not validate_changes(changes(epochs=5), "t", "").valid


def test_every_menu_candidate_validates():
    for cand in build_menu():
        res = validate_changes(cand.changes, cand.candidate_id, cand.rationale)
        assert res.valid, (cand.candidate_id, res.errors)


def test_menu_drops_tried_candidates():
    tried = [{"data.input_width": 416, "data.input_height": 416, "epochs": 10}]
    ids = [c.candidate_id for c in build_menu(tried)]
    assert "res_416" not in ids
    assert "res_512" in ids
    assert len(build_menu([])) == len(ids) + 1
