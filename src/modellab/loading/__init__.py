import importlib

_EXPORTS = {
    "ArtifactInfo": "modellab.loading.artifact",
    "inspect_artifact": "modellab.loading.artifact",
    "ModelRequest": "modellab.loading.resolver",
    "Resolution": "modellab.loading.resolver",
    "ResolveSettings": "modellab.loading.resolver",
    "resolve_model": "modellab.loading.resolver",
    "ArchitectureDef": "modellab.loading.architectures",
    "register_architecture": "modellab.loading.architectures",
    "describe_architectures": "modellab.loading.architectures",
    "build_architecture": "modellab.loading.architectures",
    "CheckpointMismatchError": "modellab.loading.diagnostics",
}


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(name)


__all__ = list(_EXPORTS)
