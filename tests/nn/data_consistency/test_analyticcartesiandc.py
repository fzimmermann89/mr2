"""Tests for AnalyticCartesianDC module."""

import pytest
import torch
from mr2.data.KData import KData
from mr2.nn.data_consistency.AnalyticCartesianDC import AnalyticCartesianDC


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_analytic_cartesian_dc(
    image_noisy: torch.Tensor, kdata_us: KData, image: torch.Tensor, image_us: torch.Tensor, device: str
) -> None:
    kdata_us = kdata_us.to(device)
    image = image.to(device)
    image_us = image_us.to(device)
    image_noisy = image_noisy.to(device).detach().requires_grad_(True)
    dc = AnalyticCartesianDC(initial_regularization_weight=1e-6).to(device)
    result = dc(image_noisy, kdata_us)
    assert result.device.type == device
    loss = (result - image).abs().mean()
    assert loss < (image_noisy - image).abs().mean()
    assert loss < (image_us - image).abs().mean()
    loss.backward()
    assert image_noisy.grad is not None
    assert dc.log_weight.grad is not None
    assert dc.log_weight.grad.device.type == device
    assert torch.isfinite(dc.log_weight.grad)
    assert image_noisy.grad.device.type == device
    assert torch.isfinite(image_noisy.grad).all()
