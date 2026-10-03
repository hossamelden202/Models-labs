import torch

from modellab.core.errors import ModelLoadError

CAP = 50
TOLERATED = "num_batches_tracked"


class CheckpointMismatchError(ModelLoadError):
    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = diagnostics


def tensor_shapes(state) -> dict:
    return {k: tuple(v.shape) for k, v in state.items() if isinstance(v, torch.Tensor)}


def model_shapes_of(model) -> dict:
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def diff_shapes(model_shapes: dict, ckpt_shapes: dict, non_tensor=()) -> dict:
    missing = sorted(set(model_shapes) - set(ckpt_shapes))
    unexpected = sorted(set(ckpt_shapes) - set(model_shapes))
    common = sorted(set(model_shapes) & set(ckpt_shapes))
    mismatched = [
        {"key": k, "expected": list(model_shapes[k]), "found": list(ckpt_shapes[k])}
        for k in common
        if model_shapes[k] != ckpt_shapes[k]
    ]
    tolerated = [k for k in missing if k.endswith(TOLERATED)]
    hard = [k for k in missing if k not in tolerated]
    return {
        "compatible": not hard and not unexpected and not mismatched and not non_tensor,
        "num_model_keys": len(model_shapes),
        "num_checkpoint_keys": len(ckpt_shapes),
        "num_matched": len(common) - len(mismatched),
        "num_missing_keys": len(hard),
        "missing_keys": hard[:CAP],
        "num_unexpected_keys": len(unexpected),
        "unexpected_keys": unexpected[:CAP],
        "num_shape_mismatches": len(mismatched),
        "shape_mismatches": mismatched[:CAP],
        "non_tensor_entries": sorted(non_tensor)[:CAP],
        "tolerated_missing": tolerated[:CAP],
    }


def describe(diag: dict) -> str:
    parts = []
    if diag["num_missing_keys"]:
        parts.append(f"{diag['num_missing_keys']} missing keys, e.g. {', '.join(diag['missing_keys'][:3])}")
    if diag["num_unexpected_keys"]:
        parts.append(f"{diag['num_unexpected_keys']} unexpected keys, e.g. {', '.join(diag['unexpected_keys'][:3])}")
    if diag["num_shape_mismatches"]:
        first = diag["shape_mismatches"][0]
        parts.append(
            f"{diag['num_shape_mismatches']} shape mismatches, e.g. {first['key']}: "
            f"expected {first['expected']}, found {first['found']}"
        )
    if diag["non_tensor_entries"]:
        parts.append(f"non-tensor entries: {', '.join(diag['non_tensor_entries'][:3])}")
    return "; ".join(parts)


def check_and_load(model, state, label: str) -> dict:
    non_tensor = [k for k, v in state.items() if not isinstance(v, torch.Tensor)]
    diag = diff_shapes(model_shapes_of(model), tensor_shapes(state), non_tensor)
    if not diag["compatible"]:
        raise CheckpointMismatchError(
            f"could not load model '{label}': checkpoint does not match the architecture ({describe(diag)})", diag
        )
    model.load_state_dict(state, strict=not diag["tolerated_missing"])
    return diag
