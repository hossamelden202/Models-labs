import pytest

torch = pytest.importorskip("torch")

from modellab.utils import set_seed  # noqa: E402


def test_torch_is_seeded():
    set_seed(5)
    a = torch.rand(3)
    set_seed(5)
    b = torch.rand(3)
    set_seed(6)
    c = torch.rand(3)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
