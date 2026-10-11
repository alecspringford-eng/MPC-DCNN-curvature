from __future__ import annotations

import random

import numpy as np
import torch

from dcnn_curvature.utils.seed import set_seed


def test_set_seed_makes_random_numpy_torch_reproducible():
    set_seed(123)
    a = (random.random(), np.random.rand(), torch.rand(1).item())

    set_seed(123)
    b = (random.random(), np.random.rand(), torch.rand(1).item())

    assert a == b
