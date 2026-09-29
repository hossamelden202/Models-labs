from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator


class PreprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    color_mode: Literal["RGB", "L"] = "RGB"
    resize: tuple[int, int] | None = None
    center_crop: tuple[int, int] | None = None
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def _validate(self):
        if (self.mean is None) != (self.std is None):
            raise ValueError("mean and std must be given together")
        if self.mean is not None:
            channels = 3 if self.color_mode == "RGB" else 1
            if len(self.mean) != channels or len(self.std) != channels:
                raise ValueError(
                    f"mean and std need {channels} values for {self.color_mode}"
                )
            if any(s <= 0 for s in self.std):
                raise ValueError("std values must be positive")
        for name in ("resize", "center_crop"):
            size = getattr(self, name)
            if size is not None and min(size) <= 0:
                raise ValueError(f"{name} must be positive, got {size}")
        return self


def build_transform(cfg: PreprocessConfig) -> Callable[[Any], Any]:
    from torchvision import transforms as T

    steps = []
    if cfg.resize is not None:
        steps.append(T.Resize(list(cfg.resize)))
    if cfg.center_crop is not None:
        steps.append(T.CenterCrop(list(cfg.center_crop)))
    steps.append(T.ToTensor())
    if cfg.mean is not None:
        steps.append(T.Normalize(list(cfg.mean), list(cfg.std)))
    return T.Compose(steps)
