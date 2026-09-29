import random

import numpy as np
import pytest

from modellab.utils import set_seed


def draw():
    return random.random(), np.random.rand(3).tolist()


def test_same_seed_same_values():
    set_seed(123)
    a = draw()
    set_seed(123)
    b = draw()
    assert a == b


def test_different_seed_different_values():
    set_seed(1)
    a = draw()
    set_seed(2)
    b = draw()
    assert a != b


@pytest.mark.parametrize("bad", [-1, 2**32])
def test_out_of_range_seed(bad):
    with pytest.raises(ValueError):
        set_seed(bad)
