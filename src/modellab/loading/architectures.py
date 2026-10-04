import importlib.util
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from torch import nn

from modellab.core.errors import ConfigurationError

Scalar = int | float | bool | str | None
ArgValue = Scalar | list[Scalar]


class ArchitectureError(ConfigurationError):
    pass


class ArchitectureConfigError(ArchitectureError):
    def __init__(self, message: str, missing=(), errors=()):
        super().__init__(message)
        self.missing = list(missing)
        self.errors = list(errors)


ALLOWED_LAYERS: dict[str, type[nn.Module]] = {
    "Conv2d": nn.Conv2d, "BatchNorm2d": nn.BatchNorm2d, "GroupNorm": nn.GroupNorm, "LayerNorm": nn.LayerNorm,
    "ReLU": nn.ReLU, "LeakyReLU": nn.LeakyReLU, "GELU": nn.GELU, "SiLU": nn.SiLU, "ELU": nn.ELU,
    "Hardswish": nn.Hardswish, "Sigmoid": nn.Sigmoid, "Tanh": nn.Tanh, "Dropout": nn.Dropout,
    "MaxPool2d": nn.MaxPool2d, "AvgPool2d": nn.AvgPool2d, "AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d,
    "AdaptiveMaxPool2d": nn.AdaptiveMaxPool2d, "Flatten": nn.Flatten, "Linear": nn.Linear, "Identity": nn.Identity,
}


class LayerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    args: dict[str, ArgValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self):
        cls = ALLOWED_LAYERS.get(self.type)
        if cls is None:
            raise ValueError(f"unknown layer type '{self.type}', allowed: {sorted(ALLOWED_LAYERS)}")
        try:
            inspect.signature(cls).bind(**self.args)
        except TypeError as exc:
            raise ValueError(f"invalid arguments for {self.type}: {exc}") from exc
        return self


class SequentialConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layers: list[LayerSpec] = Field(min_length=1)


class LibraryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kwargs: dict[str, ArgValue] = Field(default_factory=dict)


TorchvisionConfig = LibraryConfig
TimmConfig = LibraryConfig


def _always() -> bool:
    return True


@dataclass(frozen=True)
class ArchitectureDef:
    name: str
    description: str
    config_model: type[BaseModel]
    build: Callable[[Any, int | None], nn.Module]
    probe: Callable[[dict, int | None, float, list | None], list] | None = None
    available: Callable[[], bool] = _always
    needs_num_classes: bool = True


ARCHITECTURES: dict[str, ArchitectureDef] = {}


def register_architecture(defn: ArchitectureDef, replace: bool = False) -> None:
    if defn.name in ARCHITECTURES and not replace:
        raise ArchitectureError(f"architecture '{defn.name}' is already registered")
    ARCHITECTURES[defn.name] = defn


def unregister_architecture(name: str) -> None:
    ARCHITECTURES.pop(name, None)


def available_names() -> list[str]:
    return [n for n, d in sorted(ARCHITECTURES.items()) if d.available()]


def get_architecture(name: str) -> ArchitectureDef:
    defn = ARCHITECTURES.get(name)
    if defn is None:
        raise ArchitectureError(f"unknown architecture '{name}', registered: {sorted(ARCHITECTURES)}")
    if not defn.available():
        raise ArchitectureError(f"architecture '{name}' needs a package that is not installed")
    return defn


def parse_config(defn: ArchitectureDef, config: dict | None):
    try:
        return defn.config_model.model_validate(config or {})
    except ValidationError as exc:
        where = lambda e: ".".join(str(p) for p in e["loc"])  # noqa: E731
        missing = [where(e) for e in exc.errors() if e["type"] == "missing"]
        errors = [f"{where(e)}: {e['msg']}" for e in exc.errors()]
        raise ArchitectureConfigError("invalid architecture_config: " + "; ".join(errors), missing, errors) from exc


def build_architecture(name: str, config: dict | None, num_classes: int | None) -> nn.Module:
    defn = get_architecture(name)
    return defn.build(parse_config(defn, config), num_classes)


def describe_architectures() -> list[dict]:
    return [
        {
            "name": d.name, "description": d.description, "available": d.available(),
            "can_probe": d.probe is not None, "needs_num_classes": d.needs_num_classes,
            "config_schema": d.config_model.model_json_schema(),
        }
        for _, d in sorted(ARCHITECTURES.items())
    ]


def list_architectures() -> list[dict]:
    """Return the architecture registry in the shape expected by the server API."""
    return describe_architectures()


def probe_all(shapes: dict, num_classes: int | None, seconds: float = 20.0, models: list | None = None) -> list[dict]:
    deadline = time.monotonic() + seconds
    out = []
    for name in sorted(ARCHITECTURES):
        defn = ARCHITECTURES[name]
        if defn.probe is None or not defn.available():
            continue
        for config in defn.probe(shapes, num_classes, deadline, models):
            out.append({"architecture": name, "config": config})
    return out


def _build_sequential(config: SequentialConfig, num_classes: int | None) -> nn.Module:
    try:
        layers = [ALLOWED_LAYERS[spec.type](**spec.args) for spec in config.layers]
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ArchitectureConfigError(f"a layer could not be built: {exc}", errors=[str(exc)]) from exc
    if num_classes is not None:
        for layer in reversed(layers):
            out = getattr(layer, "out_features", None) or getattr(layer, "out_channels", None)
            if out is not None:
                if out != num_classes:
                    raise ArchitectureConfigError(
                        f"the last layer has {out} outputs but num_classes is {num_classes}",
                        errors=[f"last layer outputs {out}, expected {num_classes}"],
                    )
                break
    return nn.Sequential(*layers)


def _library_build(module_name: str, make: Callable) -> Callable:
    def build(config: LibraryConfig, num_classes: int | None) -> nn.Module:
        kwargs = dict(config.kwargs)
        if num_classes is not None:
            kwargs["num_classes"] = num_classes
        try:
            return make(config.name, kwargs)
        except Exception as exc:
            raise ArchitectureConfigError(f"{module_name} rejected '{config.name}': {exc}", errors=[str(exc)]) from exc

    return build


def _torchvision_make(name: str, kwargs: dict):
    import torchvision

    if name not in torchvision.models.list_models(module=torchvision.models):
        raise ValueError(f"unknown torchvision model '{name}'")
    return torchvision.models.get_model(name, weights=None, **kwargs)


def _timm_make(name: str, kwargs: dict):
    import timm

    return timm.create_model(name, pretrained=False, **kwargs)


def _probe_torchvision(shapes: dict, num_classes: int | None, deadline: float, names: list | None) -> list[dict]:
    import torchvision

    if num_classes is None:
        return []
    target = {k: tuple(v) for k, v in shapes.items() if not k.endswith("num_batches_tracked")}
    pool = sorted(torchvision.models.list_models(module=torchvision.models))
    if names:
        pool = [n for n in pool if n in names]
    found = []
    for name in pool:
        if time.monotonic() > deadline:
            break
        try:
            with torch.device("meta"):
                module = torchvision.models.get_model(name, weights=None, num_classes=num_classes)
        except Exception:
            continue
        have = {k: tuple(v.shape) for k, v in module.state_dict().items() if not k.endswith("num_batches_tracked")}
        if have == target:
            found.append({"name": name})
    return found


def _has(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


register_architecture(
    ArchitectureDef(
        name="sequential",
        description="Declarative stack of standard torch.nn layers (convolutions, norms, activations, pooling, linear).",
        config_model=SequentialConfig, build=_build_sequential,
    )
)
register_architecture(
    ArchitectureDef(
        name="torchvision",
        description="Any torchvision classification architecture by name, built without pretrained weights.",
        config_model=LibraryConfig, build=_library_build("torchvision", _torchvision_make),
        probe=_probe_torchvision, available=lambda: _has("torchvision"),
    )
)
register_architecture(
    ArchitectureDef(
        name="timm",
        description="Any timm model by name, built without pretrained weights.",
        config_model=LibraryConfig, build=_library_build("timm", _timm_make),
        available=lambda: _has("timm"),
    )
)
