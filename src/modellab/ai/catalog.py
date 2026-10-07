from modellab.ai.schemas import Candidate, Change

MAX_CHANGES = 4

RULES = {
    "epochs": {"kind": "int", "min": 1, "max": 300},
    "data.input_width": {"kind": "choice", "choices": (224, 320, 416, 512, 640)},
    "data.input_height": {"kind": "choice", "choices": (224, 320, 416, 512, 640)},
    "data.batch_size": {"kind": "choice", "choices": (8, 16, 32)},
    "optimizer.name": {"kind": "choice", "choices": ("adam", "adamw", "sgd", "rmsprop")},
    "optimizer.learning_rate": {"kind": "float", "min": 1e-5, "max": 1e-1},
    "optimizer.weight_decay": {"kind": "float", "min": 0.0, "max": 0.1},
    "scheduler.name": {"kind": "choice", "choices": ("none", "cosine")},
    "early_stopping_patience": {"kind": "int", "min": 1, "max": 100},
    "mixed_precision": {"kind": "bool"},
    "data.augmentation.enabled": {"kind": "bool"},
    "data.augmentation.probability": {"kind": "float", "min": 0.0, "max": 1.0},
    "data.augmentation.rotation_degrees": {"kind": "float", "min": 1.0, "max": 45.0},
    "data.augmentation.saturation_factor": {"kind": "float", "min": 0.0, "max": 1.0},
    "data.augmentation.brightness_factor": {"kind": "float", "min": 0.05, "max": 1.0},
    "data.augmentation.horizontal_flip": {"kind": "bool"},
}

UNSUPPORTED = {
    "freeze": "freeze settings are not applied by the detection trainer",
    "loss": "loss settings are not passed to the detection trainer",
    "data.sampler": "sampler settings are not passed to the detection trainer",
    "gradient_accumulation_steps": "not passed to the detection trainer",
    "max_grad_norm": "would be forwarded to Ultralytics as an argument it likely rejects",
    "data.augmentation": "this augmentation option is not mapped to the detection trainer",
}


def unsupported_reason(path):
    for key, reason in UNSUPPORTED.items():
        if path == key or path.startswith(key + "."):
            return f"{path}: {reason}"
    return f"{path}: not a path the detection trainer supports"


def check_value(path, value):
    rule = RULES.get(path)
    if rule is None:
        return unsupported_reason(path)
    kind = rule["kind"]
    if kind == "bool":
        return None if isinstance(value, bool) else f"{path}: expected true or false"
    if kind == "choice":
        return None if value in rule["choices"] and not isinstance(value, bool) else (
            f"{path}: {value!r} not in {list(rule['choices'])}"
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{path}: expected a number"
    if kind == "int" and (isinstance(value, float) and not value.is_integer()):
        return f"{path}: expected an integer"
    if not rule["min"] <= value <= rule["max"]:
        return f"{path}: {value} outside [{rule['min']}, {rule['max']}]"
    return None


def _cand(cid, title, pairs, target, rationale):
    return Candidate(
        candidate_id=cid, title=title, target_metric=target, rationale=rationale,
        changes=[Change(path=p, value=v) for p, v in pairs],
    )


def _all_candidates():
    return [
        _cand("res_416", "Raise input resolution from 224 to 416",
              [("data.input_width", 416), ("data.input_height", 416)], "recall",
              "Small or thin objects such as knives and handguns keep more pixels at a higher input size."),
        _cand("res_512", "Raise input resolution from 224 to 512",
              [("data.input_width", 512), ("data.input_height", 512)], "recall",
              "A larger step in input size, at higher training cost."),
        _cand("epochs_30", "Train for 30 epochs",
              [("epochs", 30)], "f1",
              "Short runs are likely undertrained, so recall and precision may still be rising."),
        _cand("lr_1e-4", "Lower the learning rate to 1e-4",
              [("optimizer.learning_rate", 1e-4)], "f1",
              "A smaller step size can stabilise fine-tuning of a pretrained detector."),
        _cand("cosine", "Use a cosine learning-rate schedule",
              [("scheduler.name", "cosine")], "f1",
              "Decaying the rate late in training often improves the final checkpoint."),
        _cand("aug_geometry", "Add rotation and horizontal flip augmentation",
              [("data.augmentation.rotation_degrees", 10.0), ("data.augmentation.horizontal_flip", True)],
              "recall", "More pose variety can help when the training set is small."),
        _cand("aug_color", "Add colour jitter augmentation",
              [("data.augmentation.saturation_factor", 0.5), ("data.augmentation.brightness_factor", 0.4)],
              "precision", "Colour variation can reduce false positives caused by lighting and background."),
        _cand("batch_16", "Use batch size 16",
              [("data.batch_size", 16)], "f1",
              "Smaller batches add gradient noise and more updates per epoch."),
    ]


def build_menu(tried=()):
    tried = [dict(t) for t in tried]
    menu = []
    for cand in _all_candidates():
        wanted = {c.path: c.value for c in cand.changes}
        done = any(all(t.get(p) == v for p, v in wanted.items()) for t in tried)
        if not done:
            menu.append(cand)
    return menu


def get_candidate(menu, candidate_id):
    return next((c for c in menu if c.candidate_id == candidate_id), None)
