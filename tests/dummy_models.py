from types import SimpleNamespace

import torch
from torch import nn


def identity_classifier(scale=10.0):
    fc = nn.Linear(3, 3)
    with torch.no_grad():
        fc.weight.copy_(torch.eye(3) * scale)
        fc.bias.zero_()
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), fc)


def random_classifier(num_classes=3, channels=3, seed=0):
    gen = torch.Generator().manual_seed(seed)
    fc = nn.Linear(channels, num_classes)
    with torch.no_grad():
        fc.weight.copy_(torch.randn(num_classes, channels, generator=gen))
        fc.bias.copy_(torch.randn(num_classes, generator=gen))
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), fc)


def binary_single_logit():
    fc = nn.Linear(3, 1)
    with torch.no_grad():
        fc.weight.copy_(torch.tensor([[10.0, 0.0, 0.0]]))
        fc.bias.copy_(torch.tensor([-5.0]))
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), fc)


class Wrapped(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.inner = identity_classifier()
        self.mode = mode

    def forward(self, x):
        y = self.inner(x)
        if self.mode == "dict":
            return {"logits": y}
        if self.mode == "custom_key":
            return {"scores": y}
        if self.mode == "tuple":
            return (torch.zeros(1), y)
        if self.mode == "attr":
            return SimpleNamespace(logits=y)
        if self.mode == "bad_shape":
            return y.unsqueeze(1)
        if self.mode == "nan":
            return y * float("nan")
        raise ValueError(self.mode)


def wrapped(mode="dict"):
    return Wrapped(mode)


def raises_error():
    raise RuntimeError("boom")


def not_a_model():
    return 5


def identity_two_class(scale=10.0):
    fc = nn.Linear(3, 2)
    with torch.no_grad():
        fc.weight.copy_(torch.tensor([[scale, 0.0, 0.0], [0.0, scale, 0.0]]))
        fc.bias.zero_()
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), fc)


def dark_sensitive_classifier(scale=30.0, bias=0.24):
    fc = nn.Linear(3, 3)
    with torch.no_grad():
        fc.weight.copy_(torch.eye(3) * scale)
        fc.bias.copy_(torch.tensor([0.0, 0.0, bias * scale]))
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), fc)
