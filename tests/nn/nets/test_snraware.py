"""Checks for SNRAware networks and official weight conversion."""

from typing import cast

import pytest
import torch
from mr2.nn.nets import SNRAware
from mr2.utils import RandomGenerator


@pytest.mark.parametrize('torch_compile', [True, False], ids=['compiled', 'uncompiled'])
@pytest.mark.parametrize(
    'device',
    [pytest.param('cpu', id='cpu'), pytest.param('cuda', marks=pytest.mark.cuda, id='cuda')],
)
def test_snraware_forward(torch_compile: bool, device: str) -> None:
    """Test compiled and uncompiled forward passes on CPU and CUDA."""
    model = SNRAware(n_features=4, n_heads=2, image_size=16, p_dropout=0).to(device).eval()
    x = RandomGenerator(seed=4).float32_tensor((1, 3, 2, 16, 16), device=device)
    if torch_compile:
        # Avoid automatic dynamic-shape recompilation in older PyTorch versions.
        model = cast(SNRAware, torch.compile(model, dynamic=False))
    y = model(x)
    assert y.shape == (1, 2, 2, 16, 16)
    if torch_compile:
        # Older PyTorch shares shape-specialization state with later network tests.
        torch.compiler.reset()


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_snraware_backward(device: str) -> None:
    """Every branch participates in a differentiable compact model."""
    model = SNRAware(n_features=4, n_heads=2, image_size=16, p_dropout=0).to(device)
    x = RandomGenerator(seed=4).float32_tensor((1, 3, 2, 16, 16), device=device).requires_grad_()
    y = model(x)
    assert y.shape == (1, 2, 2, 16, 16)
    y.square().mean().backward()
    assert x.grad is not None, 'x.grad is None'
    assert torch.isfinite(x.grad).all(), 'x.grad is not finite'
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f'{name}.grad is None'
        assert torch.isfinite(parameter.grad).all(), f'{name}.grad is not finite'


def test_snraware_downsample_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convert upstream offset-major merging weights, rejecting unknown or missing keys."""
    model = SNRAware(n_features=4, n_heads=2, image_size=16).double()
    weight = torch.arange(8 * 16).reshape(8, 16, 1, 1).double()
    # Only the merging parameter is expected here, so the test isolates its ordering.
    monkeypatch.setattr(model, 'state_dict', lambda: {'down_initial.module.1.weight': torch.empty_like(weight)})
    converted = model.convert_state_dict({'bk.down_00_11.block.D2_0.conv.conv.weight': weight})
    x = RandomGenerator(seed=4).float64_tensor((1, 4, 2, 16, 16))
    official = torch.cat([x[..., 0::2, 0::2], x[..., 1::2, 0::2], x[..., 0::2, 1::2], x[..., 1::2, 1::2]], dim=1)
    expected = torch.einsum('oc,bcthw->bothw', weight[:, :, 0, 0], official)
    with torch.no_grad():
        dict(model.named_parameters())['down_initial.module.1.weight'].copy_(converted['down_initial.module.1.weight'])
    torch.testing.assert_close(model.down_initial(x), expected)
    with pytest.raises(ValueError, match='Unrecognized official key'):
        model.convert_state_dict({'unknown': weight})
    with pytest.raises(ValueError, match='Missing official parameters'):
        model.convert_state_dict({})
