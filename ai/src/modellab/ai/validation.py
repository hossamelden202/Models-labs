from modellab.ai.catalog import MAX_CHANGES, check_value
from modellab.ai.schemas import ValidationResult
from modellab.experiments.errors import ExperimentConfigError
from modellab.experiments.spec import experiment_id, parse_spec
from modellab.experiments.training_config import (
    TrainingChange,
    TrainingConfig,
    TrainingIntervention,
    apply_training_changes,
)


def check_changes(changes):
    if not changes:
        return ["proposal has no changes"]
    errors = []
    if len(changes) > MAX_CHANGES:
        errors.append(f"too many changes ({len(changes)}), at most {MAX_CHANGES}")
    seen = set()
    for c in changes:
        if c.path in seen:
            errors.append(f"{c.path}: listed twice")
        seen.add(c.path)
        msg = check_value(c.path, c.value)
        if msg:
            errors.append(msg)
    values = {c.path: c.value for c in changes}
    w, h = values.get("data.input_width"), values.get("data.input_height")
    if (w is None) != (h is None) or (w is not None and w != h):
        errors.append("data.input_width and data.input_height must be set together to the same value")
    return errors


def validate_changes(changes, name, claim, expected_direction="improve"):
    errors = check_changes(changes)
    if errors:
        return ValidationResult(valid=False, errors=errors)
    raw = [{"path": c.path, "value": c.value} for c in changes]
    try:
        intervention = TrainingIntervention(changes=[TrainingChange(**r) for r in raw])
        apply_training_changes(TrainingConfig(), intervention)
        spec = parse_spec({
            "name": name,
            "hypothesis": {"claim": claim, "expected_direction": expected_direction},
            "intervention": {"kind": "training", "changes": raw},
        })
    except (ValueError, ExperimentConfigError) as exc:
        return ValidationResult(valid=False, errors=[str(exc)[:400]])
    return ValidationResult(
        valid=True,
        experiment_id=experiment_id(spec),
        spec=spec.model_dump(mode="json"),
    )
