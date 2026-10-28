"""Test for random fourier features"""

import pytest
import torch
from mr2.nn import FourierFeatures
from mr2.utils import RandomGenerator


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param(
            'cuda',
            marks=(pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')),
            id='cuda',
        ),
    ],
)
def test_fourierfeatures(device: str) -> None:
    """Test FourierFeatures."""
    n_features_in = 1
    n_features_out = 16
    std = 1.0
    rng = RandomGenerator(444)
    x = rng.float32_tensor((1, n_features_in)).to(device)
    ff = FourierFeatures(n_features_in, n_features_out, std).to(device)
    y = ff(x)
    assert y.shape == (1, n_features_out)
