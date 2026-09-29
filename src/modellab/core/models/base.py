from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from modellab.core.datasets import DatasetSample
from modellab.core.results import Prediction


class ModelMetadata(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    model_id: str
    framework: str
    num_classes: int = Field(gt=0)
    class_names: list[str] | None = None
    device: str | None = None
    input_size: tuple[int, int] | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_class_names(self):
        if self.class_names is not None and len(self.class_names) != self.num_classes:
            raise ValueError("class_names length must equal num_classes")
        return self


class ModelAdapter(ABC):
    @abstractmethod
    def metadata(self) -> ModelMetadata: ...

    @abstractmethod
    def predict(self, sample: DatasetSample) -> Prediction: ...

    def predict_batch(self, samples: Sequence[DatasetSample]) -> list[Prediction]:
        return [self.predict(s) for s in samples]
