"""Tests for Cartesian GRAPPA calibration and application helpers."""

from itertools import product

import pytest
import torch
from mr2.algorithms import grappa_calibration, grappa_interpolation
from torch.nn.functional import conv3d


@pytest.mark.parametrize('regularization', [0.0, 0.03])
@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_grappa_calibration_explicit(regularization: float, dtype: torch.dtype) -> None:
    """Fit the acquired neighbors of a missing center, matching an explicit solve."""
    acs = torch.randn(2, 1, 5, 9, generator=torch.Generator().manual_seed(13), dtype=dtype)
    pattern = torch.zeros((1, 3, 3), dtype=torch.bool)
    pattern[..., ::2] = True
    offsets = [(dy, dx) for dy, dx in product(range(-1, 2), repeat=2) if dx != 0]
    sources = torch.stack(
        [
            torch.stack([acs[:, 0, y + dy, x + dx] for dy, dx in offsets], dim=-1).flatten()
            for y, x in product(range(1, 4), range(1, 8))
        ]
    )
    targets = torch.stack([acs[:, 0, y, x] for y, x in product(range(1, 4), range(1, 8))])
    if regularization:
        ridge = regularization * sources.abs().square().sum() / sources.shape[-1]
        weights = torch.linalg.solve(sources.mH @ sources + ridge * torch.eye(sources.shape[-1]), sources.mH @ targets)
    else:
        weights = torch.linalg.lstsq(sources, targets, driver='gelsd').solution
    kernel = grappa_calibration(acs, pattern, regularization=regularization)
    assert kernel.shape == (2, 2, 1, 3, 3)
    actual = kernel.flatten(start_dim=-3)[..., pattern.flatten()].flatten(start_dim=-2).mT
    torch.testing.assert_close(actual, weights, atol=2e-5, rtol=2e-5)
    assert torch.count_nonzero(kernel[..., 0, 1, 1]) == 0


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_grappa_interpolation(dtype: torch.dtype) -> None:
    """Match complex cross-correlation, including zero boundaries."""
    generator = torch.Generator().manual_seed(42)
    data = torch.randn(2, 3, 1, 48, 50, generator=generator, dtype=dtype)
    pattern = torch.zeros((1, 3, 3), dtype=torch.bool)
    pattern[..., 0] = True
    kernel = grappa_calibration(data[0, :, :, 10:30, 10:30], pattern)
    expected = conv3d(data, kernel, padding=(0, 1, 1))
    actual = grappa_interpolation(data, kernel)
    torch.testing.assert_close(actual, expected)
    locations = torch.arange(1, 48 * 50, 2)
    selected = grappa_interpolation(data, kernel, locations)
    torch.testing.assert_close(selected.flatten(-3)[..., locations], expected.flatten(-3)[..., locations])
    assert torch.equal(selected.flatten(-3)[..., ::2], data.flatten(-3)[..., ::2])


def test_grappa_shared_calibration_and_scale() -> None:
    """Broadcast fitted kernels over batches without repeating calibration."""
    generator = torch.Generator().manual_seed(42)
    acs = torch.randn(1, 3, 2, 1, 12, 13, generator=generator, dtype=torch.complex128)
    data = torch.randn(2, 3, 2, 1, 8, 9, generator=generator, dtype=acs.dtype)
    pattern = torch.zeros((1, 3, 3), dtype=torch.bool)
    pattern[..., ::2] = True
    kernel = grappa_calibration(acs, pattern)
    assert kernel.shape[:2] == (1, 3)
    output = grappa_interpolation(data, kernel)
    for i, j in product(range(2), range(3)):
        torch.testing.assert_close(output[i, j], grappa_interpolation(data[i, j], kernel[0, j]))
    scale = 3.1 - 2.7j
    torch.testing.assert_close(grappa_calibration(scale * acs, pattern), kernel)
    torch.testing.assert_close(grappa_interpolation(scale * data, kernel), scale * output)


@pytest.mark.parametrize('regularization', [-1.0, float('nan')])
def test_grappa_invalid_regularization(regularization: float) -> None:
    """Reject invalid ridge strengths before solving."""
    pattern = torch.tensor([[[True, False, True]]])
    with pytest.raises(ValueError, match='finite and nonnegative'):
        grappa_calibration(torch.ones(2, 1, 4, 5, dtype=torch.complex64), pattern, regularization=regularization)


@pytest.mark.parametrize('pattern', [torch.ones(1, 3, 3, dtype=torch.bool), torch.zeros(1, 3, 3, dtype=torch.bool)])
def test_grappa_invalid_sources(pattern: torch.Tensor) -> None:
    """A kernel needs acquired sources and cannot regress its own center."""
    with pytest.raises(ValueError, match='excluding the center'):
        grappa_calibration(torch.ones(2, 1, 4, 5, dtype=torch.complex64), pattern)


def test_grappa_insufficient_acs() -> None:
    """Require enough calibration rows for the coil-neighborhood features."""
    pattern = torch.tensor([[[True, False, True]]])
    with pytest.raises(ValueError, match='Insufficient ACS'):
        grappa_calibration(torch.ones(4, 1, 1, 3, dtype=torch.complex64), pattern)
