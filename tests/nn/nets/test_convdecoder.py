"""Tests for ConvDecoder."""

from copy import deepcopy

import pytest
import torch
from mr2.nn.nets import ConvDecoder


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('n_dim', [1, 2, 3, 4, 5])
def test_forward_backward(n_dim: int, device: str) -> None:
    """Test ND output shape, gradients and restored weights."""
    model = ConvDecoder(n_dim, 2, 3, (4,) * n_dim, n_features=2, n_levels=2).to(device)
    latent = torch.rand((1, 2, *((2,) * n_dim)), device=device, requires_grad=True)
    output = model(latent)
    assert output.device == latent.device
    assert output.shape == (1, 3, *((4,) * n_dim))
    output.square().mean().backward()
    assert latent.grad is not None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    restored = ConvDecoder(n_dim, 2, 3, (4,) * n_dim, n_features=2, n_levels=2).to(device)
    restored.load_state_dict(deepcopy(model.state_dict()))
    torch.testing.assert_close(restored(latent), output)
