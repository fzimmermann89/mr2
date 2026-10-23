"""Checks for SNRAware networks and official weight conversion."""

import pytest
import torch
from mr2.nn.nets import SNRAware


def test_snraware_backward() -> None:
    """Every branch participates in a differentiable compact model."""
    model = SNRAware(n_features=4, n_heads=2, image_size=16, p_dropout=0)
    x = torch.randn(1, 3, 2, 16, 16, requires_grad=True)
    y = model(x)
    assert y.shape == (1, 2, 2, 16, 16)
    y.square().mean().backward()
    for parameter in (x, *model.parameters()):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_snraware_downsample_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convert upstream offset-major merging weights, rejecting unknown or missing keys."""
    model = SNRAware(n_features=4, n_heads=2, image_size=16).double()
    weight = torch.arange(8 * 16).reshape(8, 16, 1, 1).double()
    # Only the merging parameter is expected here, so the test isolates its ordering.
    monkeypatch.setattr(model, 'state_dict', lambda: {'down_initial.module.1.weight': torch.empty_like(weight)})
    converted = model.convert_state_dict({'bk.down_00_11.block.D2_0.conv.conv.weight': weight})
    x = torch.randn(1, 4, 2, 16, 16, dtype=torch.float64)
    official = torch.cat([x[..., 0::2, 0::2], x[..., 1::2, 0::2], x[..., 0::2, 1::2], x[..., 1::2, 1::2]], dim=1)
    expected = torch.einsum('oc,bcthw->bothw', weight[:, :, 0, 0], official)
    with torch.no_grad():
        dict(model.named_parameters())['down_initial.module.1.weight'].copy_(converted['down_initial.module.1.weight'])
    torch.testing.assert_close(model.down_initial(x), expected)
    with pytest.raises(ValueError, match='Unrecognized official key'):
        model.convert_state_dict({'unknown': weight})
    with pytest.raises(ValueError, match='Missing official parameters'):
        model.convert_state_dict({})
