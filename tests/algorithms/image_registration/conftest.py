"""Fixtures for image registration tests."""

import pytest
import torch


@pytest.fixture
def blob_image() -> torch.Tensor:
    """A smooth image with three distinct blobs, shaped as batch, channel, z, y, x."""
    y, x = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing='ij')
    image = (
        torch.exp(-((y + 0.3) ** 2 / 0.07 + (x + 0.2) ** 2 / 0.11))
        + 0.8 * torch.exp(-((y - 0.25) ** 2 / 0.05 + (x - 0.3) ** 2 / 0.06))
        + 0.5 * torch.exp(-((y + 0.1) ** 2 / 0.03 + (x - 0.4) ** 2 / 0.04))
    )
    return image[None, None, None]
