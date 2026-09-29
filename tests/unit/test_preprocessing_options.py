import numpy as np
import pytest
from pydantic import ValidationError

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from modellab.evaluation.preprocessing import PreprocessConfig, build_transform  # noqa: E402


def test_int_resize_scales_the_shortest_side():
    tensor = build_transform(PreprocessConfig(resize=3))(Image.new("RGB", (12, 6)))
    assert tuple(tensor.shape) == (3, 3, 6)


def test_int_resize_then_center_crop():
    cfg = PreprocessConfig(resize=8, center_crop=(8, 8))
    tensor = build_transform(cfg)(Image.new("RGB", (20, 10)))
    assert tuple(tensor.shape) == (3, 8, 8)


def test_tuple_resize_still_forces_exact_size():
    tensor = build_transform(PreprocessConfig(resize=(4, 6)))(Image.new("RGB", (12, 6)))
    assert tuple(tensor.shape) == (3, 4, 6)


def test_interpolation_is_passed_to_the_transform():
    bicubic = build_transform(PreprocessConfig(resize=(4, 4), interpolation="bicubic"))
    default = build_transform(PreprocessConfig(resize=(4, 4)))
    assert bicubic.transforms[0].interpolation == T.InterpolationMode.BICUBIC
    assert default.transforms[0].interpolation == T.InterpolationMode.BILINEAR


def test_bicubic_and_bilinear_give_different_pixels():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8))
    a = build_transform(PreprocessConfig(resize=(5, 5), interpolation="bicubic"))(img)
    b = build_transform(PreprocessConfig(resize=(5, 5)))(img)
    assert not torch.equal(a, b)


@pytest.mark.parametrize(
    "kwargs",
    [{"interpolation": "lanczos"}, {"resize": 0}, {"resize": -3}, {"center_crop": (0, 4)}],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValidationError):
        PreprocessConfig(**kwargs)


def test_json_roundtrip_with_int_resize():
    cfg = PreprocessConfig(
        resize=256,
        center_crop=(224, 224),
        interpolation="bicubic",
        mean=(0.1, 0.2, 0.3),
        std=(0.4, 0.5, 0.6),
    )
    assert PreprocessConfig.model_validate_json(cfg.model_dump_json()) == cfg
    assert cfg.resize == 256
