"""Tests for conditioning dispatch through module wrappers."""

import pytest
import torch
from mr2.nn import ComplexAsChannel, PermutedBlock, Residual, SeparableResBlock, Sequential
from mr2.nn.nets import UNet
from torch.nn import Module


@pytest.mark.parametrize('module_type', [UNet, SeparableResBlock])
@pytest.mark.parametrize('wrapper', ['complex', 'sequential', 'permuted', 'residual', 'nested'])
def test_conditioning_dispatch(module_type: type[Module], wrapper: str) -> None:
    """Wrappers preserve conditioning outputs and gradients."""
    torch.manual_seed(0)
    module: Module
    if module_type is UNet:
        module = UNet(
            n_dim=2,
            n_channels_in=2,
            n_channels_out=2,
            n_features=(8, 16),
            attention_depths=(),
            cond_dim=16,
        )
    else:
        module = SeparableResBlock(dim_groups=((-2, -1),), n_channels_in=2, n_channels_out=2, cond_dim=16)
    module.eval()
    x = torch.randn(1, 2, 16, 16)
    cond = torch.randn(1, 16, requires_grad=True)
    other_cond = torch.randn(1, 16)

    wrapped: Module
    if wrapper in ('complex', 'nested'):
        wrapped = ComplexAsChannel(module)
        if wrapper == 'nested':
            wrapped = Sequential(wrapped)
        x_wrapped = torch.complex(x[:, :1], x[:, 1:])
    else:
        x_wrapped = x
        if wrapper == 'sequential':
            wrapped = Sequential(module)
        elif wrapper == 'permuted':
            wrapped = PermutedBlock((-2, -1), module)
        else:
            wrapped = Residual(module)

    output = wrapped(x_wrapped, cond=cond)
    expected = module(x, cond=cond)
    if wrapper in ('complex', 'nested'):
        expected = torch.complex(expected[:, :1], expected[:, 1:])
    elif wrapper == 'residual':
        expected = expected + x
    torch.testing.assert_close(output, expected)
    assert not torch.allclose(output, wrapped(x_wrapped, cond=other_cond))

    output.abs().square().mean().backward()
    assert cond.grad is not None, 'No gradient computed for conditioning'
    assert torch.isfinite(cond.grad).all(), 'Non-finite conditioning gradients'
    assert cond.grad.abs().max() > 0, 'Conditioning gradients are zero'
