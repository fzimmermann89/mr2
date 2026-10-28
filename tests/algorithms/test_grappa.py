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
@pytest.mark.parametrize(('shape', 'kernel_size'), [((1, 48, 50), (1, 3, 3)), ((7, 8, 9), (3, 3, 3))])
def test_grappa_interpolation(
    dtype: torch.dtype, shape: tuple[int, int, int], kernel_size: tuple[int, int, int]
) -> None:
    """Match complex cross-correlation, including zero boundaries."""
    generator = torch.Generator().manual_seed(42)
    data = torch.randn(2, 3, *shape, generator=generator, dtype=dtype)
    pattern = torch.zeros(kernel_size, dtype=torch.bool)
    pattern[..., 0] = True
    kernel = grappa_calibration(data[0], pattern)
    expected = conv3d(data, kernel, padding=tuple(k // 2 for k in kernel_size))
    actual = grappa_interpolation(data, kernel)
    torch.testing.assert_close(actual, expected)
    locations = torch.arange(1, data[0, 0].numel(), 2)
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


@pytest.mark.parametrize(
    ('pattern', 'message'),
    [
        (torch.ones(1, 3, 3, dtype=torch.bool), 'exclude the center'),
        (torch.zeros(1, 3, 3, dtype=torch.bool), 'No acquired neighbor within kernel_size'),
    ],
)
def test_grappa_invalid_sources(pattern: torch.Tensor, message: str) -> None:
    """A kernel needs acquired sources and cannot regress its own center."""
    with pytest.raises(ValueError, match=message):
        grappa_calibration(torch.ones(2, 1, 4, 5, dtype=torch.complex64), pattern)


def test_grappa_insufficient_acs() -> None:
    """Require at least one complete calibration window."""
    pattern = torch.tensor([[[True, False, True]]])
    with pytest.raises(ValueError, match='too small'):
        grappa_calibration(torch.ones(4, 1, 1, 2, dtype=torch.complex64), pattern)


@pytest.mark.parametrize('regularization', [0.0, 0.03])
def test_grappa_underdetermined_calibration(regularization: float) -> None:
    """Allow more source features than rows and match an augmented least-squares solve."""
    acs = torch.randn(4, 1, 1, 5, dtype=torch.complex128, generator=torch.Generator().manual_seed(27))
    pattern = torch.tensor([[[True, False, True]]])
    sources = torch.stack((acs[:, 0, 0, :-2].mT, acs[:, 0, 0, 2:].mT), dim=-1).flatten(-2)
    targets = acs[:, 0, 0, 1:-1].mT
    ridge = regularization * sources.abs().square().sum() / sources.shape[-1]
    augmented = torch.cat((sources, ridge.sqrt() * torch.eye(sources.shape[-1], dtype=acs.dtype)))
    right = torch.cat((targets, targets.new_zeros(sources.shape[-1], targets.shape[-1])))
    expected = torch.linalg.lstsq(augmented, right, driver='gelsd').solution
    kernel = grappa_calibration(acs, pattern, regularization=regularization)
    actual = kernel.flatten(-3)[..., pattern.flatten()].flatten(-2).mT
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('regularization', [0.0, 0.01])
def test_grappa_zero_calibration_batch(regularization: float) -> None:
    """An empty-signal batch yields zero kernels without affecting neighboring batches."""
    acs = torch.randn(2, 2, 1, 4, 5, dtype=torch.complex128, generator=torch.Generator().manual_seed(28))
    acs[0] = 0
    pattern = torch.tensor([[[True, False, True]]])
    kernels = grappa_calibration(acs, pattern, regularization=regularization)
    assert torch.isfinite(kernels).all()
    assert torch.count_nonzero(kernels[0]) == 0
    torch.testing.assert_close(kernels[1], grappa_calibration(acs[1], pattern, regularization=regularization))
