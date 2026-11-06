"""Tests for PermutedBlock module."""

import pytest
import torch
from mr2.nn import FiLM, PermutedBlock
from mr2.utils import RandomGenerator


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    ('features_last', 'input_shape', 'apply_along_dim'),
    [
        (False, (2, 4, 3, 5), (-1,)),
        (True, (2, 3, 5, 4), (-2,)),
    ],
)
def test_permutedblock_conditioning(
    features_last: bool, input_shape: tuple[int, ...], apply_along_dim: tuple[int, ...], device: str
) -> None:
    """Folding spatial slices into the batch preserves per-example conditioning and gradients."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor(input_shape).to(device)
    cond = rng.float32_tensor((2, 8)).to(device).requires_grad_(True)
    film = FiLM(channels=4, cond_dim=8, features_last=features_last).to(device)
    block = PermutedBlock(apply_along_dim, film, features_last=features_last)

    output = block(x, cond=cond)
    expected = film(x, cond=cond)
    torch.testing.assert_close(output, expected)

    expected_grad = torch.autograd.grad(expected.square().mean(), cond)[0]
    output.square().mean().backward()
    assert cond.grad is not None, 'No gradient computed for conditioning'
    torch.testing.assert_close(cond.grad, expected_grad)
    assert torch.isfinite(cond.grad).all(), 'Non-finite conditioning gradients'
    assert cond.grad.abs().max() > 0, 'Conditioning gradients are zero'
