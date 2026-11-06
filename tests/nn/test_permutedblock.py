"""Tests for conditioning along selected spatial dimensions."""

import pytest
import torch
from mr2.nn import FiLM, PermutedBlock, Sequential
from mr2.nn.attention.SpatialTransformerBlock import BasicTransformerBlock


@pytest.mark.parametrize('transformer', [True, False])
@pytest.mark.parametrize('features_last', [True, False])
@pytest.mark.parametrize('cond_batch', [1, 2])
def test_permutedblock_conditioning(features_last: bool, cond_batch: int, transformer: bool) -> None:
    """Each folded spatial slice receives its original example's conditioning."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 3, 5, 6)
    cond = torch.randn(cond_batch, 8, requires_grad=True)
    module = (
        BasicTransformerBlock(4, 1, cond_dim=8, features_last=features_last)
        if transformer
        else FiLM(4, 8, features_last=features_last)
    )
    if features_last:
        x = x.movedim(1, -1)
        dims = (-3, -2)
    else:
        dims = (-2, -1)
    wrapped = Sequential(PermutedBlock(dims, module, features_last=features_last))
    output = wrapped(x, cond=cond)
    slice_dim = 1 if features_last else 2
    expected = torch.cat(
        [
            torch.stack(
                [
                    module(x[i : i + 1].select(slice_dim, j), cond=cond[i : i + 1] if cond_batch > 1 else cond)
                    for j in range(3)
                ],
                dim=slice_dim,
            )
            for i in range(2)
        ]
    )
    torch.testing.assert_close(output, expected)
    assert not torch.allclose(output, wrapped(x, cond=cond + 1))
    actual_grad = torch.autograd.grad(output.square().mean(), cond)[0]
    expected_grad = torch.autograd.grad(expected.square().mean(), cond)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.isfinite(actual_grad).all()
    assert actual_grad.abs().max() > 0
