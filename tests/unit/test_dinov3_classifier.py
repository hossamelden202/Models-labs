import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

FACTORY = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "dinov3_gore"
)

_spec = importlib.util.spec_from_file_location(
    "dinov3_classifier_for_tests",
    FACTORY / "dinov3_classifier.py",
)

mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


class FakeBackbone(nn.Module):
    def __init__(self, dim=384, registers=4):
        super().__init__()

        self.patches = nn.Conv2d(
            3,
            dim,
            kernel_size=16,
            stride=16,
        )

        self.special = nn.Parameter(
            torch.randn(1, 1 + registers, dim)
        )

    def forward(self, pixel_values):
        tokens = (
            self.patches(pixel_values)
            .flatten(2)
            .transpose(1, 2)
        )

        special = self.special.expand(
            tokens.shape[0],
            -1,
            -1,
        )

        return SimpleNamespace(
            last_hidden_state=torch.cat(
                [special, tokens],
                dim=1,
            )
        )


def make(seed=0, **kwargs):
    torch.manual_seed(seed)

    return mod.DinoV3Classifier(
        FakeBackbone(),
        **kwargs,
    ).eval()


def expected_head_shapes():
    d = 384

    shapes = {}

    for name in ("cls_norm", "patch_norm"):
        shapes[
            f"attention_pool.{name}.weight"
        ] = (d,)
        shapes[
            f"attention_pool.{name}.bias"
        ] = (d,)

    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        shapes[
            f"attention_pool.{name}.weight"
        ] = (d, d)
        shapes[
            f"attention_pool.{name}.bias"
        ] = (d,)

    shapes.update(
        {
            "attention_pool.dropout.p": (),
            "classifier.0.weight": (768,),
            "classifier.0.bias": (768,),
            "classifier.1.weight": (384, 768),
            "classifier.1.bias": (384,),
            "classifier.4.weight": (128, 384),
            "classifier.4.bias": (128,),
            "classifier.7.weight": (3, 128),
            "classifier.7.bias": (3,),
        }
    )

    return shapes


def test_head_parameter_names_and_shapes_match_the_checkpoint():
    state = make().state_dict()

    head = {
        key: tuple(value.shape)
        for key, value in state.items()
        if not key.startswith("backbone.")
    }

    expected = {
        key: shape
        for key, shape in expected_head_shapes().items()
        if not key.endswith(".p")
    }

    assert head == expected

    assert make().attention_pool.dropout.p == 0.10
    assert make().classifier[3].p == 0.10
    assert make().classifier[6].p == 0.10


def test_attention_pool_matches_training_implementation():
    torch.manual_seed(1)

    pool = mod.AttentionPool(
        384,
        heads=6,
        dropout=0.10,
    ).eval()

    cls = torch.randn(2, 384)
    tokens = torch.randn(2, 10, 384)

    batch_size, num_tokens, dim = tokens.shape
    heads = 6
    head_dim = 64

    cls_normed = pool.cls_norm(cls)
    patches_normed = pool.patch_norm(tokens)

    q = pool.q_proj(cls_normed).view(
        batch_size,
        1,
        heads,
        head_dim,
    ).transpose(1, 2)

    k = pool.k_proj(patches_normed).view(
        batch_size,
        num_tokens,
        heads,
        head_dim,
    ).transpose(1, 2)

    v = pool.v_proj(patches_normed).view(
        batch_size,
        num_tokens,
        heads,
        head_dim,
    ).transpose(1, 2)

    scale = head_dim ** -0.5

    attention = (q @ k.transpose(-2, -1)) * scale
    attention = attention.softmax(dim=-1)

    expected = (
        attention @ v
    ).transpose(1, 2).reshape(batch_size, dim)

    expected = pool.out_proj(expected)

    actual = pool(cls, tokens)

    assert torch.allclose(
        actual,
        expected,
        atol=1e-5,
        rtol=1e-5,
    )


def test_forward_shape_and_determinism():
    model = make()

    x = torch.randn(
        2,
        3,
        224,
        224,
    )

    with torch.no_grad():
        first = model(x)
        second = model(x)

    assert tuple(first.shape) == (2, 3)
    assert torch.equal(first, second)


def test_feature_dimension_is_768():
    model = make()

    x = torch.randn(
        2,
        3,
        224,
        224,
    )

    with torch.no_grad():
        features = model.extract_features(x)

    assert tuple(features.shape) == (2, 768)


def test_cls_feature_is_raw_cls_token():
    model = make()

    x = torch.randn(
        2,
        3,
        224,
        224,
    )

    with torch.no_grad():
        hidden = model.backbone(
            pixel_values=x
        ).last_hidden_state

        features = model.extract_features(x)

    raw_cls = hidden[:, 0, :]

    assert torch.allclose(
        features[:, :384],
        raw_cls,
        atol=1e-6,
        rtol=1e-6,
    )


def test_register_tokens_are_excluded_from_attention_pool():
    model = make()

    x = torch.randn(
        2,
        3,
        224,
        224,
    )

    with torch.no_grad():
        hidden = model.backbone(
            pixel_values=x
        ).last_hidden_state

        captured = {}

        original_forward = model.attention_pool.forward

        def wrapped(cls_token, patch_tokens):
            captured["tokens"] = patch_tokens.detach().clone()
            return original_forward(
                cls_token,
                patch_tokens,
            )

        model.attention_pool.forward = wrapped

        model(x)

    expected_patches = hidden[:, 1 + 4 :, :]

    assert torch.equal(
        captured["tokens"],
        expected_patches,
    )


def test_token_layout_mismatch_is_rejected():
    model = mod.DinoV3Classifier(
        FakeBackbone(registers=3),
        num_register_tokens=4,
    ).eval()

    with pytest.raises(ValueError, match="tokens"):
        model(
            torch.randn(
                1,
                3,
                224,
                224,
            )
        )


def test_invalid_hidden_size_is_rejected():
    with pytest.raises(ValueError, match="not divisible"):
        mod.DinoV3Classifier(
            FakeBackbone(dim=100),
            hidden_size=100,
        )


def test_attention_pool_requires_divisible_heads():
    with pytest.raises(ValueError, match="not divisible"):
        mod.AttentionPool(
            dim=385,
            heads=6,
        )


def test_dropout_configuration_matches_training():
    model = make(dropout=0.10)

    assert model.attention_pool.dropout.p == 0.10
    assert model.classifier[3].p == 0.10
    assert model.classifier[6].p == 0.10


def test_production_classifier_has_no_experimental_pooling_modes():
    model = make()

    assert not hasattr(model, "cls_feature")
    assert not hasattr(model, "kv_tokens")


def test_classifier_output_has_three_classes():
    model = make(num_classes=3)

    x = torch.randn(
        2,
        3,
        224,
        224,
    )

    with torch.no_grad():
        logits = model(x)

    assert tuple(logits.shape) == (2, 3)
