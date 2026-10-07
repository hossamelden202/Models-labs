import importlib
import importlib.util
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from modellab.core.datasets import DatasetSample
from modellab.core.errors import ConfigurationError, EvaluationError, ModelLoadError
from modellab.core.models import ModelAdapter, ModelMetadata
from modellab.core.results import Prediction
from modellab.loading.diagnostics import check_and_load
from modellab.utils import get_logger

log = get_logger("model")


class TorchModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str
    source: Literal["torchscript", "module", "state_dict", "factory"]
    task: Literal["classification", "detection"] = "classification"
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
    architecture: str | None = None
    architecture_config: dict[str, Any] | None = None
    provider_id: str | None = None

    @model_validator(mode="after")
    def _check(self):
        uses_path = self.source != "factory"
        uses_factory = self.source in ("state_dict", "factory")
        if uses_path and self.path is None:
            raise ValueError(f"source '{self.source}' requires path")
        if not uses_path and self.path is not None:
            raise ValueError("source 'factory' does not take a path")
        if self.architecture is not None and self.factory:
            raise ValueError("use either architecture or factory, not both")
        if self.architecture is None and self.architecture_config is not None:
            raise ValueError("architecture_config needs architecture")
        if self.architecture is not None:
            if self.source != "state_dict":
                raise ValueError("architecture only applies to source 'state_dict'")
            if self.factory_kwargs:
                raise ValueError("factory_kwargs do not apply to a registered architecture")
        elif uses_factory and not self.factory:
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

    @model_serializer(mode="wrap")
    def _omit_unset_architecture(self, handler):
        data = handler(self)
        for key in ("architecture", "architecture_config"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


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


def _build_registered(spec: TorchModelSpec) -> torch.nn.Module:
    from modellab.loading.architectures import build_architecture

    return _require_module(build_architecture(spec.architecture, spec.architecture_config, spec.num_classes))


def _build_huggingface(spec: TorchModelSpec) -> torch.nn.Module:
    """
    Build a Hugging Face image-classification architecture from the
    local artifact directory.

    This constructs the architecture only. Checkpoint weights are loaded
    separately by _apply_state_dict(), preserving ModelLab's existing
    diagnostics and compatibility checks.
    """
    if spec.path is None:
        raise ModelLoadError("Hugging Face provider requires a checkpoint path")

    model_dir = Path(spec.path).parent
    config_path = model_dir / "config.json"

    if not config_path.is_file():
        raise ModelLoadError(
            f"Hugging Face model directory has no config.json: {model_dir}"
        )

    try:
        from transformers import AutoConfig, AutoModelForImageClassification
    except ImportError as exc:
        raise ModelLoadError(
            "Hugging Face Transformers is required for provider architecture loading"
        ) from exc

    try:
        config = AutoConfig.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )

        # ModelLab's resolved class count remains authoritative.
        if spec.num_classes is not None:
            config.num_labels = spec.num_classes

        if spec.class_names:
            config.id2label = {
                i: name for i, name in enumerate(spec.class_names)
            }
            config.label2id = {
                name: i for i, name in enumerate(spec.class_names)
            }

        model = AutoModelForImageClassification.from_config(config)
    except Exception as exc:
        raise ModelLoadError(
            f"could not construct Hugging Face model from {model_dir}: {exc}"
        ) from exc

    return _require_module(model)


def _read_state(path: Path):
    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ModelLoadError("reading .safetensors files needs the safetensors package") from exc
        return load_file(str(path))
    return torch.load(path, map_location="cpu", weights_only=True)




def _normalize_hf_vit_state_dict(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    """
    Normalize legacy Hugging Face ViT checkpoint keys to the
    parameter names used by newer Transformers ViT models.

    This is intentionally narrow:
      - activates only when the checkpoint contains legacy
        `vit.encoder.layer.*` keys
      - activates only when the target model contains
        `vit.layers.*` keys
      - leaves all other architectures untouched
    """

    state_keys = tuple(state.keys())
    model_keys = frozenset(model.state_dict().keys())

    legacy_prefix = "vit.encoder.layer."
    current_prefix = "vit.layers."

    has_legacy_vit = any(
        key.startswith(legacy_prefix)
        for key in state_keys
    )

    has_current_vit = any(
        key.startswith(current_prefix)
        for key in model_keys
    )

    if not has_legacy_vit or not has_current_vit:
        return state

    replacements = (
        (
            ".attention.attention.query.",
            ".attention.q_proj.",
        ),
        (
            ".attention.attention.key.",
            ".attention.k_proj.",
        ),
        (
            ".attention.attention.value.",
            ".attention.v_proj.",
        ),
        (
            ".attention.output.dense.",
            ".attention.o_proj.",
        ),
        (
            ".intermediate.dense.",
            ".mlp.fc1.",
        ),
        (
            ".output.dense.",
            ".mlp.fc2.",
        ),
    )

    normalized: dict[str, torch.Tensor] = {}

    for key, value in state.items():
        new_key = key

        if new_key.startswith(legacy_prefix):
            new_key = new_key.replace(
                legacy_prefix,
                current_prefix,
                1,
            )

            for old, new in replacements:
                if old in new_key:
                    new_key = new_key.replace(old, new, 1)
                    break

        normalized[new_key] = value

    return normalized


def _apply_state_dict(model: torch.nn.Module, spec: TorchModelSpec) -> None:
    state = _read_state(_existing(spec.path))
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

    state = _normalize_hf_vit_state_dict(model, state)

    check_and_load(model, state, spec.model_id)


def _load_model(spec: TorchModelSpec) -> torch.nn.Module:
    try:
        if spec.source == "torchscript":
            model = torch.jit.load(str(_existing(spec.path)), map_location="cpu")
        elif spec.source == "module":
            path = _existing(spec.path)
            log.warning(
                "unpickling a full model from %s, only do this for files you trust",
                path,
            )

            model = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            # Ultralytics checkpoints commonly serialize the actual
            # inference model inside a dictionary under "ema" or
            # "model". Prefer EMA for inference and fall back to
            # model. Do not accept arbitrary dictionaries as models.
            if isinstance(model, dict):
                ema = model.get("ema")
                checkpoint_model = model.get("model")

                if isinstance(ema, torch.nn.Module):
                    model = ema
                elif isinstance(checkpoint_model, torch.nn.Module):
                    model = checkpoint_model

            model = _require_module(model)
        elif spec.source == "state_dict":
            is_huggingface = spec.provider_id == "huggingface"

            # Runtime fallback for resolved local Hugging Face artifacts.
            # Some API/runtime paths reconstruct TorchModelSpec without
            # preserving provider_id, while architecture_config still
            # contains the Hugging Face model configuration.
            if not is_huggingface:
                config = spec.architecture_config
                is_huggingface = (
                    isinstance(config, dict)
                    and (
                        config.get("model_type")
                        or config.get("architectures")
                    )
                )

            if is_huggingface:
                model = _build_huggingface(spec)
            else:
                model = (
                    _build_registered(spec)
                    if spec.architecture
                    else _build_from_factory(spec)
                )
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


def _is_ultralytics_detector(module: torch.nn.Module) -> bool:
    """
    Detect an Ultralytics detection model without importing Ultralytics.

    The controlled runtime may contain Ultralytics while the host runtime
    does not. Detection therefore relies only on stable characteristics
    exposed by the loaded nn.Module.
    """
    cls = type(module)
    module_name = cls.__module__.lower()
    class_name = cls.__name__.lower()

    if "ultralytics" not in module_name:
        return False

    if "detectionmodel" in class_name:
        return True

    task = getattr(module, "task", None)
    if isinstance(task, str) and task.lower() == "detect":
        return True

    args = getattr(module, "args", None)
    if isinstance(args, Mapping):
        task = args.get("task")
        if isinstance(task, str) and task.lower() == "detect":
            return True

    return False


def _extract_detection_output(
    output: Any,
    num_classes: int,
) -> torch.Tensor:
    """
    Extract and validate the raw YOLO detection tensor.

    Ultralytics detection inference normally produces:
        (batch, 4 + num_classes, num_candidates)

    This is intentionally separate from classification logits.
    """
    value = output

    if isinstance(value, Mapping):
        for key in ("preds", "predictions", "output"):
            if key in value:
                value = value[key]
                break

    if isinstance(value, (tuple, list)):
        tensor_values = [item for item in value if isinstance(item, torch.Tensor)]
        if len(tensor_values) == 1:
            value = tensor_values[0]
        elif tensor_values:
            value = tensor_values[0]

    if not isinstance(value, torch.Tensor):
        raise EvaluationError(
            "detection model output is not a tensor: "
            f"{type(value).__name__}"
        )

    if value.ndim != 3:
        raise EvaluationError(
            "expected detection output of shape "
            f"(batch, 4 + classes, candidates), got {tuple(value.shape)}"
        )

    if value.shape[0] <= 0:
        raise EvaluationError("detection output has an empty batch dimension")

    expected_channels = 4 + num_classes

    if value.shape[1] != expected_channels:
        raise EvaluationError(
            "detection model output has "
            f"{value.shape[1]} channels, expected "
            f"4 + {num_classes} = {expected_channels}"
        )

    if value.shape[2] <= 0:
        raise EvaluationError("detection output has no candidate predictions")

    return value


class TorchImageClassifier(ModelAdapter):
    def __init__(self, spec: TorchModelSpec, device: str = "auto", module: torch.nn.Module | None = None):
        self.spec = spec
        self.device = resolve_device(device)
        self.model = (module if module is not None else _load_model(spec)).to(self.device).eval()

        # CPU inference should use FP32 inputs and FP32 model weights.
        #
        # Some checkpoints, including Ultralytics checkpoints, are
        # serialized with FP16 weights. CPU validation creates FP32
        # tensors, while many CPU operators do not reliably support
        # FP16. Normalize the loaded model to FP32 on CPU instead of
        # forcing validation/inference inputs to FP16.
        if self.device.type == "cpu":
            self.model.float()

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
            details={
                "source": self.spec.source,
                "task": self.spec.task,
                "mixed_precision": self._amp,
                **({"architecture": self.spec.architecture} if self.spec.architecture else {}),
            },
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

    def _prepare_model_input(self, batch: torch.Tensor) -> torch.Tensor:
        """Move an inference batch to the model device and dtype.

        ModelLab preserves the model's weights and adapts the input
        tensor to the model's floating-point execution dtype. This
        keeps FP32, FP16, and BF16 models architecture-independent.

        Mixed-precision autocast remains responsible for its own
        operator-level casting and is not changed here.
        """
        if not isinstance(batch, torch.Tensor):
            raise EvaluationError(
                f"model input must be a torch.Tensor, got {type(batch).__name__}"
            )

        dtype = batch.dtype

        if batch.is_floating_point():
            for parameter in self.model.parameters():
                if parameter.is_floating_point():
                    dtype = parameter.dtype
                    break
            else:
                for buffer in self.model.buffers():
                    if buffer.is_floating_point():
                        dtype = buffer.dtype
                        break

        return batch.to(
            device=self.device,
            dtype=dtype,
        )

    def predict_probabilities(self, batch: torch.Tensor) -> np.ndarray:
        if batch.ndim != 4:
            raise EvaluationError(f"batch must have shape (N, C, H, W), got {tuple(batch.shape)}")
        batch = self._prepare_model_input(batch)
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
