import importlib
import importlib.util
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from modellab.core.datasets import DatasetSample
from modellab.core.errors import ConfigurationError, EvaluationError, ModelLoadError
from modellab.core.models import ModelAdapter, ModelMetadata
from modellab.core.results import Prediction
from modellab.utils import get_logger

log = get_logger("model")


class TorchModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str
    source: Literal["torchscript", "module", "state_dict", "factory"]
    num_classes: int = Field(gt=0)
    path: Path | None = None
    factory: str | None = None
    factory_kwargs: dict[str, Any] = Field(default_factory=dict)
    state_dict_key: str | None = None
    strip_prefix: str | None = None
    class_names: list[str] | None = None
    input_size: tuple[int, int] | None = None
    output_key: str | None = None
    output_index: int = Field(0, ge=0)
    single_logit_binary: bool = False
    mixed_precision: bool = False

    @model_validator(mode="after")
    def _check(self):
        uses_path = self.source != "factory"
        uses_factory = self.source in ("state_dict", "factory")
        if uses_path and self.path is None:
            raise ValueError(f"source '{self.source}' requires path")
        if not uses_path and self.path is not None:
            raise ValueError("source 'factory' does not take a path")
        if uses_factory and not self.factory:
            raise ValueError(f"source '{self.source}' requires factory")
        if not uses_factory and (self.factory or self.factory_kwargs):
            raise ValueError(f"source '{self.source}' does not take factory or factory_kwargs")
        if self.source != "state_dict" and (self.state_dict_key or self.strip_prefix):
            raise ValueError("state_dict_key and strip_prefix only apply to source 'state_dict'")
        if self.class_names is not None and len(self.class_names) != self.num_classes:
            raise ValueError("class_names length must equal num_classes")
        if self.single_logit_binary and self.num_classes != 2:
            raise ValueError("single_logit_binary requires num_classes == 2")
        return self


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise ConfigurationError("device 'cuda' requested but CUDA is not available")
        return torch.device("cuda")
    raise ConfigurationError(f"unknown device '{name}', expected auto, cpu or cuda")


def _import_file(path: Path):
    if not path.is_file():
        raise ModelLoadError(f"factory file not found: {path}")
    name = f"_modellab_factory_{path.stem}_{abs(hash(str(path.resolve()))) % 10**8}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ModelLoadError(f"cannot load factory file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _import_callable(ref: str):
    target, sep, name = ref.rpartition(":")
    if not sep or not target or not name:
        raise ModelLoadError(
            f"factory must look like 'module:callable' or 'file.py:callable', got '{ref}'"
        )
    if target.endswith(".py"):
        module = _import_file(Path(target))
    else:
        try:
            module = importlib.import_module(target)
        except ImportError as exc:
            raise ModelLoadError(f"cannot import '{target}': {exc}") from exc

    obj = module
    for part in name.split("."):
        if not hasattr(obj, part):
            raise ModelLoadError(f"'{target}' has no attribute '{name}'")
        obj = getattr(obj, part)
    if not callable(obj):
        raise ModelLoadError(f"'{ref}' is not callable")
    return obj


def _require_module(obj: Any) -> torch.nn.Module:
    if not isinstance(obj, torch.nn.Module):
        raise ModelLoadError(f"loaded object is {type(obj).__name__}, not an nn.Module")
    return obj


def _existing(path: Path | None) -> Path:
    if path is None or not path.is_file():
        raise ModelLoadError(f"model file not found: {path}")
    return path


def _build_from_factory(spec: TorchModelSpec) -> torch.nn.Module:
    fn = _import_callable(spec.factory)
    try:
        model = fn(**spec.factory_kwargs)
    except Exception as exc:
        raise ModelLoadError(f"factory '{spec.factory}' failed: {exc}") from exc
    return _require_module(model)


def _apply_state_dict(model: torch.nn.Module, spec: TorchModelSpec) -> None:
    state = torch.load(_existing(spec.path), map_location="cpu", weights_only=True)
    if spec.state_dict_key is not None:
        if not isinstance(state, Mapping) or spec.state_dict_key not in state:
            raise ModelLoadError(f"checkpoint has no key '{spec.state_dict_key}'")
        state = state[spec.state_dict_key]
    if not isinstance(state, Mapping):
        raise ModelLoadError("checkpoint is not a state dict, set state_dict_key if it is nested")
    if spec.strip_prefix:
        prefix = spec.strip_prefix
        state = {
            (k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in state.items()
        }
    model.load_state_dict(state, strict=True)


def _load_model(spec: TorchModelSpec) -> torch.nn.Module:
    try:
        if spec.source == "torchscript":
            model = torch.jit.load(str(_existing(spec.path)), map_location="cpu")
        elif spec.source == "module":
            path = _existing(spec.path)
            log.warning("unpickling a full model from %s, only do this for files you trust", path)
            model = _require_module(torch.load(path, map_location="cpu", weights_only=False))
        elif spec.source == "state_dict":
            model = _build_from_factory(spec)
            _apply_state_dict(model, spec)
        else:
            model = _build_from_factory(spec)
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(f"could not load model '{spec.model_id}': {exc}") from exc
    return model


def _extract_logits(output: Any, key: str | None, index: int) -> Any:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping):
        name = key or "logits"
        if name not in output:
            raise EvaluationError(
                f"model output has no '{name}' key, available: {sorted(map(str, output))}"
            )
        return output[name]
    if key is not None or hasattr(output, "logits"):
        name = key or "logits"
        if not hasattr(output, name):
            raise EvaluationError(f"model output has no attribute '{name}'")
        return getattr(output, name)
    if isinstance(output, tuple | list):
        if index >= len(output):
            raise EvaluationError(f"output_index {index} out of range for {len(output)} outputs")
        return output[index]
    raise EvaluationError(f"cannot find logits in model output of type {type(output).__name__}")


def _shape(value: Any) -> Any:
    return tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__


class TorchImageClassifier(ModelAdapter):
    def __init__(self, spec: TorchModelSpec, device: str = "auto"):
        self.spec = spec
        self.device = resolve_device(device)
        self.model = _load_model(spec).to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self._amp = spec.mixed_precision and self.device.type == "cuda"
        if spec.mixed_precision and not self._amp:
            log.warning("mixed_precision ignored because the device is %s", self.device)

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self.spec.model_id,
            framework="pytorch",
            num_classes=self.spec.num_classes,
            class_names=self.spec.class_names,
            device=str(self.device),
            input_size=self.spec.input_size,
            details={"source": self.spec.source, "mixed_precision": self._amp},
        )

    def _logits(self, output: Any, batch_size: int) -> torch.Tensor:
        logits = _extract_logits(output, self.spec.output_key, self.spec.output_index)
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 2
            or logits.shape[0] != batch_size
        ):
            raise EvaluationError(
                f"expected logits of shape (batch, classes), got {_shape(logits)}"
            )
        if self.spec.single_logit_binary:
            if logits.shape[1] != 1:
                raise EvaluationError(
                    f"single_logit_binary expects one logit per sample, got {logits.shape[1]}"
                )
            # softmax([0, z]) equals [1 - sigmoid(z), sigmoid(z)]
            return torch.cat([torch.zeros_like(logits), logits], dim=1)
        if logits.shape[1] != self.spec.num_classes:
            raise EvaluationError(
                f"model outputs {logits.shape[1]} classes but the spec says {self.spec.num_classes}"
            )
        return logits

    def predict_probabilities(self, batch: torch.Tensor) -> np.ndarray:
        if batch.ndim != 4:
            raise EvaluationError(f"batch must have shape (N, C, H, W), got {tuple(batch.shape)}")
        batch = batch.to(self.device)
        try:
            with torch.no_grad():
                if self._amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = self.model(batch)
                else:
                    output = self.model(batch)
        except Exception as exc:
            raise EvaluationError(
                f"model forward failed on input of shape {tuple(batch.shape)}: {exc}"
            ) from exc

        probs = torch.softmax(self._logits(output, batch.shape[0]).float(), dim=1)
        if not torch.isfinite(probs).all():
            raise EvaluationError("model produced non-finite outputs")
        return probs.cpu().numpy()

    def predict_batch(self, samples: Sequence[DatasetSample]) -> list[Prediction]:
        if not samples:
            return []
        tensors = []
        for sample in samples:
            if not isinstance(sample.input, torch.Tensor):
                raise EvaluationError(
                    f"sample {sample.sample_id}: input must be a tensor, "
                    "build the dataset with a preprocess config"
                )
            tensors.append(sample.input)
        try:
            batch = torch.stack(tensors)
        except RuntimeError as exc:
            raise EvaluationError("samples in one batch must have identical shapes") from exc

        probs = self.predict_probabilities(batch)
        predictions = []
        for sample, row in zip(samples, probs, strict=True):
            cls = int(row.argmax())
            predictions.append(
                Prediction(
                    sample_id=sample.sample_id,
                    predicted_class=cls,
                    confidence=float(row[cls]),
                    class_probabilities=row.tolist(),
                )
            )
        return predictions

    def predict(self, sample: DatasetSample) -> Prediction:
        return self.predict_batch([sample])[0]
