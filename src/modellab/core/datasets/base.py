from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    input: Any
    label: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(ABC):
    @property
    @abstractmethod
    def dataset_id(self) -> str: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get_sample(self, index: int) -> DatasetSample: ...

    @abstractmethod
    def get_label(self, index: int) -> int | None: ...

    def get_metadata(self, index: int) -> dict[str, Any]:
        return self.get_sample(index).metadata

    def iterate(self) -> Iterator[DatasetSample]:
        for i in range(len(self)):
            yield self.get_sample(i)
