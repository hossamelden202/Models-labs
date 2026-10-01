import hashlib
import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from pydantic import BaseModel, ConfigDict, Field, model_validator

from modellab.evaluation.image_dataset import ImageClassificationDataset
from modellab.evaluation.preprocessing import PreprocessConfig, build_transform
from modellab.experiments.errors import ExperimentError


class _P(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FactorP(_P):
    factor: float = Field(gt=0)


class SaturationP(_P):
    factor: float = Field(ge=0)


class HueP(_P):
    shift: float = Field(ge=-0.5, le=0.5)


class BlurP(_P):
    radius: float = Field(gt=0, le=50)


class NoiseP(_P):
    std: float = Field(gt=0, le=255)


class JpegP(_P):
    quality: int = Field(ge=1, le=95)


class RotationP(_P):
    angle: float = Field(gt=-360, lt=360)


class OcclusionP(_P):
    fraction: float = Field(gt=0, lt=1)
    position: Literal["center", "random"] = "center"
    fill: int = Field(0, ge=0, le=255)


class CropP(_P):
    fraction: float = Field(gt=0, lt=1)


class FlipP(_P):
    semantically_valid: bool

    @model_validator(mode="after")
    def _ack(self):
        if not self.semantically_valid:
            raise ValueError("horizontal flip must be declared semantically valid for this task")
        return self


class EmptyP(_P):
    pass


class SwapP(_P):
    order: list[int]

    @model_validator(mode="after")
    def _perm(self):
        if sorted(self.order) != [0, 1, 2]:
            raise ValueError("order must be a permutation of [0, 1, 2]")
        return self


class DropP(_P):
    channel: int = Field(ge=0, le=2)


def sample_rng(seed: int, repetition: int, sample_id: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{repetition}:{sample_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _color(img, value):
    return (value,) * 3 if img.mode == "RGB" else value


def _box(img, fraction):
    w, h = img.size
    side = fraction**0.5
    cw, ch = max(1, round(w * side)), max(1, round(h * side))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return x0, y0, cw, ch


def _brightness(img, p, rng):
    return ImageEnhance.Brightness(img).enhance(p.factor)


def _contrast(img, p, rng):
    return ImageEnhance.Contrast(img).enhance(p.factor)


def _saturation(img, p, rng):
    return ImageEnhance.Color(img).enhance(p.factor)


def _hue(img, p, rng):
    if img.mode != "RGB":
        raise ExperimentError("hue shift needs RGB images")
    hsv = np.array(img.convert("HSV"))
    hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + int(round(p.shift * 255))) % 256).astype(np.uint8)
    bands = [Image.fromarray(np.ascontiguousarray(hsv[..., i])) for i in range(3)]
    return Image.merge("HSV", bands).convert("RGB")


def _blur(img, p, rng):
    return img.filter(ImageFilter.GaussianBlur(p.radius))


def _noise(img, p, rng):
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + rng.normal(0.0, p.std, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))


def _jpeg(img, p, rng):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=p.quality)
    buffer.seek(0)
    with Image.open(buffer) as out:
        return out.convert(img.mode)


def _rotation(img, p, rng):
    return img.rotate(p.angle, resample=Image.Resampling.BICUBIC, fillcolor=_color(img, 0))


def _occlusion(img, p, rng):
    w, h = img.size
    x0, y0, cw, ch = _box(img, p.fraction)
    if p.position == "random":
        x0 = int(rng.integers(0, w - cw + 1))
        y0 = int(rng.integers(0, h - ch + 1))
    out = img.copy()
    ImageDraw.Draw(out).rectangle([x0, y0, x0 + cw - 1, y0 + ch - 1], fill=_color(img, p.fill))
    return out


def _crop(img, p, rng):
    x0, y0, cw, ch = _box(img, p.fraction)
    return img.crop((x0, y0, x0 + cw, y0 + ch)).resize(img.size, Image.Resampling.BICUBIC)


def _hflip(img, p, rng):
    return ImageOps.mirror(img)


def _grayscale(img, p, rng):
    return img.convert("L").convert(img.mode)


def _swap(img, p, rng):
    if img.mode != "RGB":
        raise ExperimentError("channel swap needs RGB images")
    bands = img.split()
    return Image.merge("RGB", [bands[i] for i in p.order])


def _drop(img, p, rng):
    if img.mode != "RGB":
        raise ExperimentError("channel drop needs RGB images")
    bands = list(img.split())
    bands[p.channel] = Image.new("L", img.size, 0)
    return Image.merge("RGB", bands)


@dataclass(frozen=True)
class Op:
    params: type[BaseModel]
    apply: Callable
    stochastic: Callable[[BaseModel], bool] = lambda p: False  # noqa: E731


REGISTRY: dict[str, Op] = {
    "brightness": Op(FactorP, _brightness),
    "contrast": Op(FactorP, _contrast),
    "saturation": Op(SaturationP, _saturation),
    "hue": Op(HueP, _hue),
    "blur": Op(BlurP, _blur),
    "noise": Op(NoiseP, _noise, lambda p: True),
    "jpeg": Op(JpegP, _jpeg),
    "rotation": Op(RotationP, _rotation),
    "occlusion": Op(OcclusionP, _occlusion, lambda p: p.position == "random"),
    "crop": Op(CropP, _crop),
    "hflip": Op(FlipP, _hflip),
    "grayscale": Op(EmptyP, _grayscale),
    "channel_swap": Op(SwapP, _swap),
    "channel_drop": Op(DropP, _drop),
}


def make_image_fn(op_name: str, params: BaseModel, seed: int, repetition: int):
    op = REGISTRY[op_name]
    stochastic = op.stochastic(params)

    def fn(sample_id, image):
        rng = sample_rng(seed, repetition, sample_id) if stochastic else None
        out = op.apply(image, params, rng)
        if out.size != image.size or out.mode != image.mode:
            raise ExperimentError(f"op '{op_name}' changed the image size or mode")
        return out

    return fn


class InterventionDataset(ImageClassificationDataset):
    @classmethod
    def wrap(cls, base, image_fn=None, only_ids=None, preprocess: PreprocessConfig | None = None):
        new = cls.__new__(cls)
        new.__dict__.update(base.__dict__)
        records = base._records
        if only_ids is not None:
            keep = set(only_ids)
            records = [r for r in records if r.sample_id in keep]
        new._records = list(records)
        new._image_fn = image_fn
        if preprocess is not None:
            new.preprocess = preprocess
            new._color_mode = preprocess.color_mode
            new._transform = build_transform(preprocess)
        return new

    def _load(self, record):
        image = super()._load(record)
        if self._image_fn is None:
            return image
        return self._image_fn(record.sample_id, image)
