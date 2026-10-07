from modellab.evaluation.torch_model import _load_model
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from modellab.core.errors import ModelLabError
from modellab.evaluation.preprocessing import PreprocessConfig
from modellab.evaluation.torch_model import (
    TorchImageClassifier,
    TorchModelSpec,
    _build_from_factory,
    _build_registered,
    _normalize_hf_vit_state_dict,
    _read_state,
)
from modellab.loading import architectures as arch
from modellab.loading.artifact import ArtifactInfo, inspect_artifact
from modellab.loading.diagnostics import (
    CheckpointMismatchError,
    check_and_load,
    describe,
    diff_shapes,
    model_shapes_of,
    tensor_shapes,
)

from modellab.loading.providers.base import ArchitectureBuildContext, ArchitectureSource
from modellab.loading.providers.orchestrator import resolve_provider


class ModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str | None = None
    artifact: str | None = None
    num_classes: int | None = Field(None, gt=0)
    class_names: list[str] | None = None
    class_to_id: dict[str, int] | None = None
    input_size: tuple[int, int] | None = None
    architecture: str | None = None
    architecture_config: dict[str, Any] | None = None
    # Optional external source for the architecture implementation.
    #
    # Examples:
    #   architecture_source = "local"
    #   architecture_path = "/kaggle/input/.../architecture"
    #
    # These are deliberately separate from `architecture`.
    architecture_source: str | None = None
    architecture_path: str | None = None
    architecture_identifier: str | None = None
    factory: str | None = None
    factory_kwargs: dict[str, Any] = Field(default_factory=dict)
    state_dict_key: str | None = None
    strip_prefix: str | None = None
    output_key: str | None = None
    output_index: int = Field(0, ge=0)
    single_logit_binary: bool = False
    mixed_precision: bool = False
    preprocess: dict[str, Any] = Field(default_factory=dict)
    trust_executable: bool = False
    use_inferred: bool = True
    validate_load: bool = True
    probe_architectures: bool = True
    probe_models: list[str] | None = None


class Inferred(BaseModel):
    field: str
    value: Any
    source: str
    applied: bool = True


class Missing(BaseModel):
    field: str
    message: str
    options: list[Any] | None = None
    json_schema: dict[str, Any] | None = None


class Candidate(BaseModel):
    architecture: str
    config: dict[str, Any]
    fits: bool = True


class Resolution(BaseModel):
    status: Literal["ready", "needs_input", "invalid"]
    route: Literal["self_contained", "registered_architecture", "provider_architecture", "custom_factory"] | None = None
    artifact: ArtifactInfo | None = None
    spec: dict[str, Any] | None = None
    resolved: dict[str, Any] = Field(default_factory=dict)
    preprocess: dict[str, Any] = Field(default_factory=dict)
    inferred: list[Inferred] = Field(default_factory=list)
    missing: list[Missing] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    diagnostics: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ResolveSettings:
    allow_custom_code: bool = True
    allow_pickle: bool = True
    probe_seconds: float = 20.0


def _msg(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())


def _strip_keys(state: dict, prefix: str) -> dict:
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state.items()}


def _build_huggingface(
    spec: TorchModelSpec,
    provider: dict[str, Any],
) -> torch.nn.Module:
    """
    Build a standard Hugging Face image-classification architecture
    from config only.

    IMPORTANT:
    This constructs the architecture but does NOT load checkpoint
    weights. ModelLab's existing _state(), _diagnose(), and
    check_and_load() pipeline remains responsible for the weights.
    """
    try:
        from transformers import AutoConfig, AutoModelForImageClassification
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face Transformers is required for provider "
            "architecture construction"
        ) from exc

    source = provider.get("source") or {}
    provider_config = dict(provider.get("config") or {})

    provider_path = source.get("path")
    provider_identifier = source.get("identifier")

    # MVP deliberately prefers a local HF directory. This prevents an
    # automatic registration from unexpectedly downloading arbitrary
    # model code or weights from the network.
    config_source = provider_path

    if not config_source:
        config_source = provider_identifier

    if not config_source:
        raise RuntimeError(
            "Hugging Face provider has neither a local path nor "
            "an architecture identifier"
        )

    trust_remote_code = bool(
        provider.get("trust_executable", False)
    )

    try:
        if provider_path:
            config = AutoConfig.from_pretrained(
                str(config_source),
                local_files_only=True,
                trust_remote_code=trust_remote_code,
            )
        else:
            config = AutoConfig.from_pretrained(
                str(config_source),
                trust_remote_code=trust_remote_code,
            )

        # Keep the already-validated ModelLab class count authoritative.
        if spec.num_classes is not None:
            if hasattr(config, "num_labels"):
                config.num_labels = spec.num_classes

            # Preserve the corrected label metadata from config.json.
            if spec.class_names:
                config.id2label = {
                    index: name
                    for index, name in enumerate(spec.class_names)
                }
                config.label2id = {
                    name: index
                    for index, name in enumerate(spec.class_names)
                }

        model = AutoModelForImageClassification.from_config(
            config,
            trust_remote_code=trust_remote_code,
        )

    except Exception as exc:
        raise RuntimeError(
            f"could not construct Hugging Face architecture "
            f"from {config_source!r}: {exc}"
        ) from exc

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "Hugging Face AutoModelForImageClassification did not "
            f"produce a torch.nn.Module: {type(model).__name__}"
        )

    return model


class _Resolver:
    def __init__(self, req: ModelRequest, settings: ResolveSettings, artifact_path, factory_file):
        self.req = req
        self.s = settings
        self.path = Path(artifact_path) if artifact_path is not None else (Path(req.artifact) if req.artifact else None)
        self.factory_file = Path(factory_file) if factory_file is not None else None
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.missing: list[Missing] = []
        self.inferred: list[Inferred] = []
        self.candidates: list[Candidate] = []
        self.info = self.diagnostics = self.validation = self.spec = self.route = self.summary = None
        self.names = None
        self.num_classes = req.num_classes
        self.input_size = req.input_size
        self.pre = dict(req.preprocess)
        self.state_key = None
        self.strip = req.strip_prefix
        self.fields: dict[str, Any] = {}
        self.provider_resolution: dict[str, Any] | None = None
        self._state_cache = None

    def err(self, text):
        self.errors.append(text)

    def need(self, field, message, options=None, json_schema=None):
        self.missing.append(Missing(field=field, message=message, options=options, json_schema=json_schema))

    def infer(self, field, value, source, applied=True):
        self.inferred.append(Inferred(field=field, value=value, source=source, applied=applied))

    def run(self) -> Resolution:
        try:
            self._steps()
        except Exception as exc:
            self.err(f"unexpected failure while resolving: {type(exc).__name__}: {exc}")
        if self.errors:
            status = "invalid"
        elif self.missing:
            status = "needs_input"
        elif self.spec is None:
            status = "invalid"
            self.err("the model definition could not be completed")
        else:
            status = "ready"
        try:
            pre = PreprocessConfig.model_validate(self.pre).model_dump(mode="json")
        except ValidationError:
            pre = dict(self.pre)
        resolved = {
            "num_classes": self.num_classes,
            "class_names": self.names,
            "input_size": list(self.input_size) if self.input_size else None,
            "state_dict_key": self.state_key,
            "strip_prefix": self.strip,
        }

        # Architecture source information is metadata for the UI and
        # downstream model creation. It is NOT the architecture name.
        if self.req.architecture_source is not None:
            resolved["architecture_source"] = self.req.architecture_source

        if self.req.architecture_path is not None:
            resolved["architecture_path"] = self.req.architecture_path
        return Resolution(
            status=status, route=self.route, artifact=self.info,
            spec=(
                {
                    **self.spec.model_dump(mode="json"),
                    "provider_id": self.provider_resolution.get("provider_id"),
                }
                if status == "ready"
                and self.provider_resolution is not None
                and self.provider_resolution.get("provider_id")
                else (
                    self.spec.model_dump(mode="json")
                    if status == "ready"
                    else None
                )
            ),
            resolved={k: v for k, v in resolved.items() if v is not None}, preprocess=pre,
            inferred=self.inferred, missing=self.missing, candidates=self.candidates,
            diagnostics=self.diagnostics, validation=self.validation, errors=self.errors, warnings=self.warnings,
        )

    def _steps(self):
        r = self.req

        # ----------------------------------------------------
        # Validate optional architecture source/path.
        # ----------------------------------------------------
        if r.architecture_source is not None:
            if r.architecture_source not in {
                "local",
                "huggingface",
                "hf",
                "transformers",
                "registered",
            }:
                self.err(
                    "architecture_source must be one of: "
                    "'local', 'huggingface', 'hf', "
                    "'transformers', 'registered'"
                )
                return

        if r.architecture_path:
            source = r.architecture_source or "local"

            if source == "local":
                architecture_path = Path(
                    r.architecture_path
                ).expanduser()

                if not architecture_path.exists():
                    self.err(
                        f"architecture_path does not exist: "
                        f"{architecture_path}"
                    )
                    return

                if not architecture_path.is_dir():
                    self.err(
                        f"architecture_path must be a directory: "
                        f"{architecture_path}"
                    )
                    return

        if not self._classes() or not self._check_pre():
            return
        if self.path is None:
            if r.architecture:
                self.err("an architecture needs weights; provide the model file")
            elif not r.factory:
                self.need("artifact", "provide a model file, or a factory that builds the whole model")
            else:
                self._factory_only()
            return
        self._artifact()

    def _classes(self) -> bool:
        r = self.req
        names = list(r.class_names) if r.class_names is not None else None
        if r.class_to_id is not None:
            if sorted(r.class_to_id.values()) != list(range(len(r.class_to_id))):
                self.err("class_to_id must use consecutive ids starting at 0")
                return False
            ordered = [n for n, _ in sorted(r.class_to_id.items(), key=lambda kv: kv[1])]
            if names is not None and names != ordered:
                self.err("class_names and class_to_id disagree")
                return False
            names = ordered
        if names is not None:
            if len(set(names)) != len(names):
                self.err("class names must be unique")
                return False
            if self.num_classes is not None and self.num_classes != len(names):
                self.err(f"num_classes is {self.num_classes} but {len(names)} class names were given")
                return False
            if self.num_classes is None:
                self.num_classes = len(names)
                self.infer("num_classes", len(names), "length of the class names")
        self.names = names
        return True

    def _check_pre(self) -> bool:
        try:
            PreprocessConfig.model_validate(self.pre)
        except ValidationError as exc:
            self.err("invalid preprocess: " + _msg(exc))
            return False
        return True

    def _hints(self):
        if not self.req.use_inferred:
            return
        seen, pending = set(), {}
        for hint in self.info.hints:
            field, value, source = hint["field"], hint["value"], hint["source"]
            if field in seen:
                continue
            seen.add(field)
            if field == "class_names":
                if self.names is None:
                    if self.num_classes is not None and self.num_classes != len(value):
                        self.warnings.append(f"{source} lists {len(value)} classes; ignored because num_classes is {self.num_classes}")
                        continue
                    self.names = list(value)
                    self.infer("class_names", self.names, source)
                    if self.num_classes is None:
                        self.num_classes = len(value)
                        self.infer("num_classes", len(value), f"length of the class names from {source}")
                elif list(value) != self.names:
                    self.warnings.append(f"{source} differs from the class names in the request; the request wins")
            elif field == "num_classes":
                if self.num_classes is None:
                    self.num_classes = int(value)
                    self.infer("num_classes", int(value), source)
                elif self.num_classes != value:
                    self.warnings.append(f"{source} says {value} classes; the request says {self.num_classes}")
            elif field == "input_size":
                if self.input_size is None:
                    self.input_size = (value[0], value[1])
                    self.infer("input_size", value, source)
            elif field in ("preprocess.mean", "preprocess.std"):
                pending[field] = (value, source)
            elif field == "architecture_hint":
                # IMPORTANT:
                #
                # A checkpoint's architecture_hint commonly comes from
                # metadata such as `model_name`.
                #
                # Example:
                #   facebook/dinov3-vits16-pretrain-lvd1689m
                #
                # This is a model/backbone identifier, NOT necessarily
                # a ModelLab registered architecture.
                #
                # Never promote it to `architecture`.
                self.inferred.append(
                    Inferred(
                        field="architecture_hint",
                        value=value,
                        source=source,
                        applied=False,
                    )
                )
        if "preprocess.mean" in pending and "preprocess.std" in pending and "mean" not in self.pre and "std" not in self.pre:
            trial = dict(self.pre)
            for key in ("mean", "std"):
                trial[key] = pending[f"preprocess.{key}"][0]
            try:
                PreprocessConfig.model_validate(trial)
            except ValidationError:
                self.warnings.append("the mean and std stored in the checkpoint do not fit the preprocessing; they were ignored")
                return
            self.pre = trial
            for key in ("mean", "std"):
                value, source = pending[f"preprocess.{key}"]
                self.infer(f"preprocess.{key}", value, source)

    def _artifact(self):
        info = inspect_artifact(self.path)
        self.info = info
        fmt = info.format
        if fmt == "missing":
            self.err("the model file was not found on the server")
        elif fmt == "onnx":
            self.err("ONNX models are not supported by the PyTorch evaluator; export to TorchScript, or provide weights plus an architecture")
        elif fmt == "unknown":
            self.err("the file was not recognised as a PyTorch artifact" + (f" ({info.warnings[0]})" if info.warnings else ""))
        else:
            self._hints()
            if not self._check_pre():
                return
            if fmt == "torchscript":
                self._self_contained("torchscript")
            elif fmt == "pickle_object":
                self._route_pickle()
            else:
                self._route_state_dict()

    def _route_pickle(self):
        r = self.req
        if r.architecture or r.factory or r.architecture_config:
            self.err(
                "this file is a pickled Python object, not a plain checkpoint; re-save the weights as a state_dict "
                "(torch.save(model.state_dict(), path)) or safetensors to use an architecture or factory"
            )
            return
        if not self.s.allow_pickle:
            self.err("loading pickled models is disabled on this server")
            return
        if not r.trust_executable:
            self.need("trust_executable", "this file is a pickled Python object; loading it executes code stored in the file. Confirm that you trust it.")
            return
        self._self_contained("module")

    def _dummy_size(self):
        if self.input_size:
            return tuple(self.input_size)
        crop = self.pre.get("center_crop")
        if crop:
            return tuple(crop)
        resize = self.pre.get("resize")
        if isinstance(resize, (list, tuple)):
            return tuple(resize)
        return None

    def _channels(self) -> int:
        return 1 if self.pre.get("color_mode") == "L" else 3

    def _measure(self, source):
        size = self._dummy_size()
        if size is None:
            return None
        try:
            if source == "torchscript":
                module = torch.jit.load(str(self.path), map_location="cpu")
            else:
                module = torch.load(self.path, map_location="cpu", weights_only=False)
            with torch.no_grad():
                out = module.eval()(torch.zeros(1, self._channels(), size[0], size[1]))
        except Exception as exc:
            self.warnings.append(f"could not measure the output size: {exc}")
            return None
        if isinstance(out, (tuple, list)) and out:
            out = out[0]
        if isinstance(out, torch.Tensor) and out.ndim == 2:
            self.infer("num_classes", int(out.shape[1]), f"forward pass on a zero input of size {size[0]}x{size[1]}")
            return int(out.shape[1])
        self.warnings.append("the model output is not a (batch, classes) tensor, so the class count was not measured")
        return None

    def _self_contained(self, source):
        r = self.req
        self.route = "self_contained"
        if r.architecture or r.factory or r.architecture_config or r.state_dict_key or r.strip_prefix:
            self.err("this file is self-contained; architecture, factory, state_dict_key and strip_prefix do not apply")
            return
        if self.num_classes is None:
            measured = self._measure(source)
            if measured is None:
                self.need("num_classes", "the class count cannot be read from this file; give num_classes or class_names, or input_size so it can be measured with a forward pass")
                return
            self.num_classes = measured
        self.fields = {"source": source, "path": self.path}
        if self._build_spec():
            self._validate(None)

    def _code_allowed(self) -> bool:
        if not self.s.allow_custom_code:
            self.err("custom factories are disabled on this server")
            return False
        if not self.req.trust_executable:
            self.need("trust_executable", "a custom factory runs Python code on the server; confirm that you trust it")
            return False
        return True

    def _factory_ref(self):
        ref = self.req.factory
        if self.factory_file is not None:
            if ":" in ref:
                self.err("with a factory file, factory must be the callable name only")
                return None
            return f"{self.factory_file}:{ref}"
        if ":" not in ref:
            self.err("factory must look like 'module:callable' or 'file.py:callable', or give a factory file")
            return None
        return ref

    def _factory_only(self):
        self.route = "custom_factory"
        if not self._code_allowed():
            return
        if self.num_classes is None:
            self.need("num_classes", "give num_classes or class_names; a factory model has no checkpoint to read it from")
            return
        ref = self._factory_ref()
        if ref is None:
            return
        self.fields = {"source": "factory", "factory": ref, "factory_kwargs": self.req.factory_kwargs}
        if not self._build_spec():
            return
        module = self._construct(meta=False)
        if module is not None:
            self._validate(module)


    def _resolve_provider_architecture(self) -> bool:
        r = self.req
        source_type = r.architecture_source
        source_path = r.architecture_path
        source_identifier = r.architecture_identifier

        if source_type in {"hf", "transformers"}:
            source_type = "huggingface"

        if source_path and not source_type:
            source_type = "local"

        if source_identifier and not source_type:
            source_type = "huggingface"

        if (
            not source_type
            and r.architecture
            and "/" in r.architecture
            and not r.factory
        ):
            source_type = "huggingface"
            source_identifier = r.architecture

        # -------------------------------------------------------------
        # Automatic local Hugging Face detection.
        #
        # A normal state-dict checkpoint becomes a HF candidate only
        # when its sibling directory contains a real Transformers
        # config.json with model_type/architectures metadata.
        #
        # This does not affect YOLO or ordinary PyTorch checkpoints.
        # -------------------------------------------------------------
        if not source_type and self.path is not None:
            artifact_dir = self.path.parent

            config_path = artifact_dir / "config.json"

            if config_path.is_file():
                try:
                    import json

                    config_data = json.loads(
                        config_path.read_text(encoding="utf-8")
                    )

                    if isinstance(config_data, dict) and (
                        config_data.get("model_type")
                        or config_data.get("architectures")
                    ):
                        source_type = "huggingface"
                        source_path = str(artifact_dir)

                        self.infer(
                            "architecture_source",
                            "huggingface",
                            "local checkpoint directory contains a Transformers config.json",
                        )
                        self.infer(
                            "architecture_path",
                            str(artifact_dir),
                            "local checkpoint directory contains a Transformers config.json",
                        )
                except Exception:
                    # Do not turn an unrelated/broken config.json into
                    # a resolver failure. The normal probe path below
                    # remains available.
                    pass

        if not source_type:
            return False

        source = ArchitectureSource(
            source_type=source_type,
            identifier=source_identifier,
            path=source_path,
            config=dict(r.architecture_config or {}),
        )

        provider_architecture = r.architecture

        try:
            result = resolve_provider(
                architecture=provider_architecture,
                source=source,
                artifact=self.path,
                context={
                    "request": r,
                    "resolver": self,
                },
            )
        except Exception as exc:
            self.err(
                "architecture provider resolution failed: "
                f"{exc}"
            )
            return True

        if result.status == "resolved":
            architecture = result.architecture

            if not architecture:
                self.err(
                    "architecture provider resolved successfully "
                    "but returned no architecture"
                )
                return True

            provider_source = result.source or source

            self.provider_resolution = {
                "provider_id": result.provider_id,
                "architecture": architecture,
                "confidence": result.confidence,
                "reason": result.reason,
                "source": {
                    "source_type": provider_source.source_type,
                    "identifier": provider_source.identifier,
                    "path": provider_source.path,
                    "config": dict(provider_source.config),
                },
                "config": dict(result.config or {}),
                "trust_executable": bool(r.trust_executable),
            }

            self.infer(
                "architecture",
                architecture,
                (
                    "architecture provider "
                    f"'{result.provider_id}'"
                ),
            )

            return True

        if result.status in {"ambiguous", "needs_input"}:
            candidates = [
                candidate.architecture
                for candidate in result.candidates
            ]

            self.need(
                "architecture",
                result.reason or (
                    "the architecture provider found multiple "
                    "possible architectures"
                ),
                options=candidates or None,
            )
            return True

        if result.status == "error":
            reason = result.reason or "provider resolution failed"

            if result.errors:
                reason += ": " + "; ".join(result.errors)

            self.err(
                "architecture provider resolution failed: "
                + reason
            )
            return True

        self.err(
            "architecture provider returned an unknown resolution "
            f"status: {result.status!r}"
        )
        return True

    def _route_state_dict(self):
        r, info = self.req, self.info

        if not info.state_dicts:
            self.err(
                "no state dict (a mapping of tensors) was found "
                "in this checkpoint"
            )
            return

        keys = [s.key_path for s in info.state_dicts]
        key = r.state_dict_key

        if key is None:
            if len(keys) == 1:
                key = keys[0]

                if key is not None:
                    self.infer(
                        "state_dict_key",
                        key,
                        "the only state dict in the checkpoint",
                    )
            else:
                self.need(
                    "state_dict_key",
                    (
                        "the checkpoint contains several state dicts; "
                        "choose one"
                    ),
                    options=keys,
                )
                return

        elif key not in keys:
            self.err(
                f"state_dict_key '{key}' is not in the checkpoint, "
                f"available: {keys}"
            )
            return

        self.state_key = key

        self.summary = next(
            s for s in info.state_dicts
            if s.key_path == key
        )

        if r.factory:
            if r.architecture or r.architecture_config:
                self.err(
                    "use either a factory or a registered architecture, "
                    "not both"
                )
                return

            self._route_factory()
            return

        if (
            r.architecture_config is not None
            and not r.architecture
        ):
            self.err(
                "architecture_config needs architecture"
            )
            return

        # -------------------------------------------------------------
        # Provider architectures are resolved independently from
        # ModelLab's registered architecture registry.
        #
        # IMPORTANT:
        # Automatic Hugging Face detection must happen BEFORE the
        # generic architecture probe. A Transformers checkpoint can
        # contain internal dimensions such as 768 that are NOT the
        # number of classification labels.
        # -------------------------------------------------------------
        if (
            r.architecture_source
            or r.architecture_path
            or r.architecture_identifier
        ):
            handled = self._resolve_provider_architecture()

            if self.provider_resolution is not None:
                self._route_provider()
                return

            if handled:
                return

        # -------------------------------------------------------------
        # Automatic provider detection for local model directories.
        #
        # This runs only when the request did not explicitly provide
        # a provider source.
        # -------------------------------------------------------------
        if not (
            r.architecture_source
            or r.architecture_path
            or r.architecture_identifier
        ):
            handled = self._resolve_provider_architecture()

            if self.provider_resolution is not None:
                self._route_provider()
                return

            if handled:
                return
        if r.architecture:
            self._route_architecture(r.architecture)
            return

        self._probe()

        self.need(
            "architecture",
            (
                "this checkpoint holds weights only; choose the "
                "network structure, or provide a custom factory"
            ),
            options=[
                "auto",
                *arch.available_names(),
            ],
        )

    def _route_provider(self):
        """Route a provider-backed architecture."""

        if self.provider_resolution is None:
            if not self._resolve_provider_architecture():
                self.err(
                    "provider architecture routing was requested "
                    "but provider resolution is unavailable"
                )
                return

        provider = self.provider_resolution

        if provider is None:
            self.err(
                "provider architecture resolution produced no result"
            )
            return

        self.route = "provider_architecture"

        # Provider metadata is authoritative when it declares the
        # classification label count. This is especially important
        # for Hugging Face models: generic checkpoint heuristics can
        # mistake an internal hidden dimension (for example 768 in
        # ViT-Base) for the number of classes.
        if (
            self.num_classes is None
            and provider.get("provider_id") == "huggingface"
        ):
            provider_config = provider.get("config") or {}
            provider_num_labels = provider_config.get("num_labels")

            if isinstance(provider_num_labels, int) and provider_num_labels > 0:
                self.num_classes = provider_num_labels
                self.infer(
                    "num_classes",
                    provider_num_labels,
                    "Hugging Face config.json num_labels",
                )

                # A generic checkpoint heuristic may already have
                # requested num_classes before provider metadata was read.
                # HF config.json is authoritative, so remove that stale
                # missing-input request.
                self.missing = [
                    item
                    for item in self.missing
                    if item.field != "num_classes"
                ]

        if self.num_classes is None:
            self._need_classes()
            return

        provider_id = provider.get("provider_id")
        provider_source = provider.get("source") or {}

        # Hugging Face uses the controlled Transformers runtime directly.
        # No custom factory is required.
        if provider_id == "huggingface":
            architecture = provider.get("architecture")

            if not architecture:
                self.err(
                    "Hugging Face provider resolved without an "
                    "architecture identifier"
                )
                return

            self.fields = {
                "source": "state_dict",
                "path": self.path,
                "state_dict_key": self.state_key,
                "strip_prefix": self.strip,
                "architecture": architecture,
                "architecture_config": dict(
                    provider.get("config") or {}
                ),
            }

            self._finish_state(meta=False)
            return

        # Existing provider behavior remains unchanged for providers
        # that require executable/custom construction.
        if not self._code_allowed():
            return

        factory = self.req.factory

        if not factory:
            self.need(
                "factory",
                (
                    "this provider architecture requires a "
                    "factory for model construction"
                ),
            )
            return

        self.fields = {
            "source": "state_dict",
            "path": self.path,
            "state_dict_key": self.state_key,
            "strip_prefix": self.strip,
            "factory": factory,
            "factory_kwargs": dict(self.req.factory_kwargs),
        }

        self._finish_state(meta=False)

    def _route_factory(self):
        self.route = "custom_factory"
        if not self._code_allowed():
            return
        ref = self._factory_ref()
        if ref is None:
            return
        if self.num_classes is None:
            self._need_classes()
            return
        self.fields = {
            "source": "state_dict", "path": self.path, "state_dict_key": self.state_key,
            "strip_prefix": self.strip, "factory": ref, "factory_kwargs": self.req.factory_kwargs,
        }
        self._finish_state(meta=False)

    def _route_architecture(self, name):
        if name == "auto":
            self._route_auto()
            return
        try:
            defn = arch.get_architecture(name)
        except arch.ArchitectureError as exc:
            self.err(str(exc))
            return
        try:
            parsed = arch.parse_config(defn, self.req.architecture_config)
        except arch.ArchitectureConfigError as exc:
            if exc.missing and len(exc.missing) == len(exc.errors):
                schema = defn.config_model.model_json_schema()
                for field in exc.missing:
                    self.need(f"architecture_config.{field}", f"the architecture '{name}' requires '{field}'", json_schema=schema)
            else:
                self.err(str(exc))
            return
        self._use_architecture(name, defn, parsed.model_dump(mode="json"))

    def _use_architecture(self, name, defn, config):
        self.route = "registered_architecture"
        if defn.needs_num_classes and self.num_classes is None:
            self._need_classes()
            return
        self.fields = {
            "source": "state_dict", "path": self.path, "state_dict_key": self.state_key,
            "strip_prefix": self.strip, "architecture": name, "architecture_config": config,
        }
        self._finish_state(meta=True)

    def _route_auto(self):
        if self.num_classes is None:
            self._need_classes("architectures can only be tested against the checkpoint when the class count is known.")
            return
        self._probe()
        if len(self.candidates) == 1:
            found = self.candidates[0]
            defn = arch.get_architecture(found.architecture)
            self.infer("architecture", found.architecture, "the only registered architecture whose keys and shapes match the checkpoint")
            config = arch.parse_config(defn, found.config).model_dump(mode="json")
            self.infer("architecture_config", config, "from the matching architecture")
            self._use_architecture(found.architecture, defn, config)
        elif self.candidates:
            self.need("architecture", "several registered architectures fit this checkpoint; choose one", options=[c.model_dump() for c in self.candidates])
        else:
            self.need("architecture", "no registered architecture fits this checkpoint; give an architecture with its configuration, or a custom factory", options=arch.available_names())

    def _probe(self):
        r = self.req
        if self.num_classes is None or not r.probe_architectures:
            return
        try:
            state = self._state()
        except Exception:
            return
        prefix = self.strip or (self.summary.common_prefix if self.summary is not None else None)
        if prefix:
            state = _strip_keys(state, prefix)
        found = arch.probe_all(tensor_shapes(state), self.num_classes, self.s.probe_seconds, r.probe_models)
        self.candidates = [Candidate(architecture=f["architecture"], config=f["config"]) for f in found]

    def _need_classes(self, why=None):
        hint = ""
        if self.summary is not None and self.summary.last_matrix:
            n = self.summary.last_matrix["shape"][0]
            self.infer("num_classes", n, f"output size of the last weight matrix '{self.summary.last_matrix['key']}'", applied=False)
            hint = f" The last weight matrix suggests {n}."
        self.need("num_classes", (why or "give num_classes or class_names.") + hint)

    def _state(self):
        if self._state_cache is None:
            state = _read_state(self.path)
            if self.state_key is not None:
                state = state[self.state_key]
            self._state_cache = dict(state)
        return self._state_cache

    def _build_spec(self) -> bool:
        r = self.req
        base = {
            "model_id": r.model_id or "pending",
            "num_classes": self.num_classes,
            "class_names": self.names,
            "input_size": self.input_size,
            "output_key": r.output_key,
            "output_index": r.output_index,
            "single_logit_binary": r.single_logit_binary,
            "mixed_precision": r.mixed_precision,
            "provider_id": (
                self.provider_resolution.get("provider_id")
                if self.provider_resolution is not None
                else None
            ),
        }
        try:
            self.spec = TorchModelSpec.model_validate({**base, **self.fields})
        except ValidationError as exc:
            self.err("invalid model definition: " + _msg(exc))
            return False
        return True

    def _construct(self, meta: bool):
        def build():
            if self.route == "custom_factory":
                return _build_from_factory(self.spec)

            if (
                self.route == "provider_architecture"
                and self.provider_resolution is not None
            ):
                provider_id = self.provider_resolution.get("provider_id")

                if provider_id == "huggingface":
                    return _build_huggingface(
                        self.spec,
                        self.provider_resolution,
                    )

            # Existing registered-architecture path.
            return _build_registered(self.spec)

        try:
            if meta:
                try:
                    with torch.device("meta"):
                        return build()
                except Exception:
                    pass
            return build()
        except Exception as exc:
            self.err(f"could not construct the model: {exc}")
            return None

    def _diagnose(self, module, state):
        # Normalize known legacy Hugging Face ViT checkpoint keys
        # before comparing them with the currently constructed
        # Transformers model.
        #
        # This is intentionally narrow and only activates when the
        # model/checkpoint key families match the legacy/current ViT
        # layouts. Other architectures, including YOLO, are untouched.
        state = _normalize_hf_vit_state_dict(module, state)

        shapes = model_shapes_of(module)

        def diff(candidate):
            extras = [k for k, v in candidate.items() if not isinstance(v, torch.Tensor)]
            return diff_shapes(shapes, tensor_shapes(candidate), extras)

        if self.strip:
            state = _strip_keys(state, self.strip)
        diag = diff(state)
        prefix = self.summary.common_prefix if self.summary is not None else None
        if not diag["compatible"] and not self.strip and prefix:
            alt_state = _strip_keys(state, prefix)
            alt = diff(alt_state)
            if alt["compatible"]:
                self.strip = prefix
                self.fields["strip_prefix"] = prefix
                self.infer("strip_prefix", prefix, "every checkpoint key starts with this prefix and the weights only fit without it")
                if not self._build_spec():
                    return None
                state, diag = alt_state, alt
        self.diagnostics = diag
        if not diag["compatible"]:
            self.err("the checkpoint does not match the architecture: " + describe(diag))
            return None
        if diag["tolerated_missing"]:
            self.warnings.append(f"{len(diag['tolerated_missing'])} BatchNorm 'num_batches_tracked' counters are missing from the checkpoint and keep their default value")
        return state

    def _finish_state(self, meta: bool):
        if not self._build_spec():
            return
        try:
            state = self._state()
        except Exception as exc:
            self.err(f"could not read the checkpoint: {exc}")
            return
        module = self._construct(meta)
        if module is None:
            return
        state = self._diagnose(module, state)
        if state is None or self.spec is None:
            return
        if not self.req.validate_load:
            self.validation = {"skipped": True}
            self.warnings.append("the model was not loaded; weights and the forward pass were not verified")
            return
        if meta:
            module = self._construct(False)
            if module is None:
                return
        try:
            check_and_load(module, state, self.spec.model_id)
        except CheckpointMismatchError as exc:
            self.diagnostics = exc.diagnostics
            self.err(str(exc))
            return
        except ModelLabError as exc:
            self.err(str(exc))
            return
        self._validate(module)

    def _validate(self, module):
        if not self.req.validate_load:
            self.validation = {"skipped": True}
            self.warnings.append(
                "the model was not loaded; weights and the forward pass were not verified"
            )
            return

        if module is None:
            try:
                module = _load_model(self.spec)
            except ModelLabError as exc:
                self.err(f"could not load model for validation: {exc}")
                return

        # Self-contained Ultralytics checkpoints are actual detection
        # modules rather than image classifiers. Detect that before
        # constructing TorchImageClassifier, because a YOLO output has
        # shape (batch, 4 + classes, candidates), not (batch, classes).
        if (
            self.spec.task == "classification"
            and type(module).__module__.lower().startswith("ultralytics.")
        ):
            from modellab.evaluation.torch_model import _is_ultralytics_detector

            if _is_ultralytics_detector(module):
                self.spec.task = "detection"
                self.infer(
                    "task",
                    "detection",
                    "loaded Ultralytics DetectionModel",
                )

        checks = [{"check": "model_loaded", "passed": True}]
        size = self._dummy_size()

        if size is None:
            self.warnings.append(
                "the forward pass was not verified: give input_size "
                "(or a center_crop in preprocess) to check the output shape"
            )
        else:
            shape = (1, self._channels(), size[0], size[1])
            dummy = torch.zeros(*shape)

            try:
                if self.spec.task == "detection":
                    # Ultralytics DetectionModel checkpoints contain
                    # authoritative detection metadata. Reconcile the
                    # request with the loaded model before validating the
                    # raw detection output.
                    model_num_classes = getattr(module, "nc", None)

                    if isinstance(model_num_classes, torch.Tensor):
                        if model_num_classes.numel() == 1:
                            model_num_classes = int(
                                model_num_classes.detach().cpu().item()
                            )
                        else:
                            model_num_classes = None
                    elif isinstance(model_num_classes, int):
                        model_num_classes = int(model_num_classes)
                    else:
                        model_num_classes = None

                    if (
                        model_num_classes is not None
                        and model_num_classes > 0
                    ):
                        requested_num_classes = self.spec.num_classes

                        if model_num_classes != requested_num_classes:
                            self.infer(
                                "num_classes",
                                model_num_classes,
                                (
                                    "loaded Ultralytics "
                                    "DetectionModel.nc "
                                    f"(request specified "
                                    f"{requested_num_classes})"
                                ),
                            )

                        self.spec.num_classes = model_num_classes

                    model_names = getattr(module, "names", None)

                    if isinstance(model_names, dict):
                        ordered_names = [
                            str(model_names[index])
                            for index in range(self.spec.num_classes)
                            if index in model_names
                        ]

                        if len(ordered_names) == self.spec.num_classes:
                            if self.spec.class_names != ordered_names:
                                self.infer(
                                    "class_names",
                                    ordered_names,
                                    (
                                        "loaded Ultralytics "
                                        "DetectionModel.names"
                                    ),
                                )

                            self.spec.class_names = ordered_names

                    with torch.no_grad():
                        output = module.float().eval()(dummy)

                    from modellab.evaluation.torch_model import (
                        _extract_detection_output,
                    )

                    detections = _extract_detection_output(
                        output,
                        self.spec.num_classes,
                    )

                    checks.append(
                        {
                            "check": "forward_pass",
                            "passed": True,
                            "task": "detection",
                            "input_shape": list(shape),
                            "output_shape": list(detections.shape),
                            "output_classes": self.spec.num_classes,
                        }
                    )
                else:
                    clf = TorchImageClassifier(
                        self.spec,
                        device="cpu",
                        module=module,
                    )
                    probs = clf.predict_probabilities(dummy)

                    if (
                        tuple(probs.shape) != (1, self.spec.num_classes)
                        or abs(float(probs.sum()) - 1.0) > 1e-3
                    ):
                        self.err(
                            f"forward pass returned {tuple(probs.shape)}, "
                            f"expected (1, {self.spec.num_classes})"
                        )
                        return

                    checks.append(
                        {
                            "check": "forward_pass",
                            "passed": True,
                            "task": "classification",
                            "input_shape": list(shape),
                            "output_classes": int(probs.shape[1]),
                        }
                    )

            except ModelLabError as exc:
                self.err(f"forward pass failed: {exc}")
                return
            except Exception as exc:
                self.err(f"forward pass failed: {exc}")
                return

        self.validation = {"checks": checks}


def resolve_model(req: ModelRequest, settings: ResolveSettings | None = None, artifact_path=None, factory_file=None) -> Resolution:
    return _Resolver(req, settings or ResolveSettings(), artifact_path, factory_file).run()
