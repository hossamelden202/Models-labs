import random

import numpy as np


def set_seed(seed: int) -> None:
    if not 0 <= seed < 2**32:
        raise ValueError(f"seed must be in [0, 2**32), got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
