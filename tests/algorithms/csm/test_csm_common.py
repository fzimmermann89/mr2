"""Tests shared by coil sensitivity estimation methods."""

from collections.abc import Callable

import pytest
import torch
from mr2.algorithms.csm import inati, walsh


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('csm_method', [walsh, inati])
def test_csm_zero_signal_and_phase(csm_method: Callable, dtype: torch.dtype, device: str) -> None:
    """Preserve coil phases, dtype, and fallback maps in empty regions."""
    weights = torch.tensor([1 + 2j, 2 - 1j, -1 + 1j], dtype=dtype, device=device)
    images = weights[None, :, None, None, None].expand(2, 3, 1, 4, 4).clone()
    images[0] = 0
    images[..., -1, :] = 0

    actual = csm_method(images, smoothing_width=1)

    expected = (weights / weights.norm())[None, :, None, None, None].expand_as(images).clone()
    expected[0] = 0
    expected[0, 0] = 1
    expected[..., -1, :] = 0
    expected[:, 0, :, -1, :] = 1
    torch.testing.assert_close(actual, expected)
