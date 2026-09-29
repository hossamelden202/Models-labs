class ModelLabError(Exception):
    """Base class for all ModelLab errors."""


class ConfigurationError(ModelLabError):
    pass


class ModelLoadError(ModelLabError):
    pass


class DatasetError(ModelLabError):
    pass


class EvaluationError(ModelLabError):
    pass


class ArtifactError(ModelLabError):
    pass


__all__ = [
    "ArtifactError",
    "ConfigurationError",
    "DatasetError",
    "EvaluationError",
    "ModelLabError",
    "ModelLoadError",
]
