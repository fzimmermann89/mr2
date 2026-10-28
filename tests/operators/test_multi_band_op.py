"""Tests for simultaneous slice encoding."""

from collections.abc import Callable

import pytest
import torch
from mr2.data import KTrajectory, SpatialDimension
from mr2.operators import FastFourierOp, FourierOp, MultiBandOp, SensitivityOp
from mr2.utils import RandomGenerator

from tests import dotproduct_adjointness_test, gradient_of_linear_operator_test


@pytest.mark.parametrize('axis', [-3, -2, -1])
def test_multi_band_encoding(axis: int) -> None:
    """Sample coordinates can vary along any Fourier axis and across batches."""
    rng = RandomGenerator(seed=0)
    shape = [2, 1, 1, 1, 1]
    shape[axis] = 4
    k = torch.tensor([[-6, -2, 0, 4], [-4, 0, 2, 8]], dtype=torch.float64).reshape(shape)
    shifts = [0.0, 1 / 3, -0.5]
    op = MultiBandOp(k, shifts)
    x = rng.rand_tensor(size=(2, 3, 4, 4, 4, 4), dtype=torch.complex128)
    expected = torch.stack([torch.exp(-2j * torch.pi * k * shift) * x[:, s] for s, shift in enumerate(shifts)]).sum(0)
    torch.testing.assert_close(op(x)[0], expected, rtol=1e-12, atol=1e-12)
    y = rng.rand_like(expected)
    dotproduct_adjointness_test(op, x, y)
    gradient_of_linear_operator_test(op, x, y)


def test_multi_band_caipi_native_coordinates() -> None:
    """The factor is a full-FOV denominator and selection preserves native phases."""
    rng = RandomGenerator(seed=0)
    k = torch.arange(-8, 9, 2).reshape(1, 1, -1, 1)
    op = MultiBandOp.from_caipi(k, multiband_factor=3, fov_shift_factor=6)
    x = rng.rand_tensor(size=(3, 2, 1, 9, 5), dtype=torch.complex64)
    expected = torch.stack([torch.exp(-2j * torch.pi * k * s / 6) * x[s] for s in range(3)]).sum(0)
    torch.testing.assert_close(op(x)[0], expected)
    indices = torch.tensor([7, 1, 4])
    selected = MultiBandOp.from_caipi(k.index_select(-2, indices), 3, 6)
    torch.testing.assert_close(selected(x.index_select(-2, indices))[0], expected.index_select(-2, indices))


def test_multi_band_caipi_no_shift() -> None:
    """A factor of one sums without phases, also for noninteger coordinates."""
    rng = RandomGenerator(seed=0)
    k = torch.tensor([0.25, 0.5, 1.5]).reshape(1, 1, 3, 1)
    x = rng.rand_tensor(size=(3, 2, 1, 3, 4), dtype=torch.complex64)
    torch.testing.assert_close(MultiBandOp.from_caipi(k, 3, 1)(x)[0], x.sum(0))


@pytest.mark.parametrize('slices', [2, 3])
def test_multi_band_batched_fourier_coordinates(slices: int) -> None:
    """Batch-dependent Fourier coordinates broadcast over the slice axis."""
    rng = RandomGenerator(seed=0)
    ky = torch.tensor([[-3.0, -1.0, 1.0], [-2.0, 0.0, 2.0]]).reshape(2, 1, 1, 3, 1)
    trajectory = KTrajectory(torch.zeros(1, 1, 1, 1, 1), ky, torch.arange(-4.0, 4.0).reshape(1, 1, 1, 1, 8))
    matrix = SpatialDimension(1, 6, 8)
    csm = rng.rand_tensor(size=(slices, 4, 1, 6, 8), dtype=torch.complex64)
    x = rng.rand_tensor(size=(2, slices, 1, 1, 6, 8), dtype=torch.complex64)
    encoding = (
        MultiBandOp.from_caipi(trajectory.ky, slices, 3)
        @ FourierOp(matrix, matrix, trajectory.unsqueeze(-5))
        @ SensitivityOp(csm)
    )
    fft = FastFourierOp(dim=(-2, -1))
    expected = torch.stack(
        [
            torch.stack(
                [
                    torch.exp(-2j * torch.pi * ky[b] * s / 3)
                    * fft(csm[s] * x[b, s])[0].index_select(-2, (ky[b].flatten() + 3).long())
                    for s in range(slices)
                ]
            ).sum(0)
            for b in range(2)
        ]
    )
    torch.testing.assert_close(encoding(x)[0], expected)
    dotproduct_adjointness_test(encoding, x, rng.rand_like(expected))
    single_slice = torch.zeros_like(x)
    single_slice[:, 0] = x[:, 0]
    assert encoding.gram(single_slice)[0][:, 1:].abs().sum() > 0


@pytest.mark.parametrize(
    ('constructor', 'kwargs', 'match'),
    [
        (MultiBandOp, {'shifts': []}, 'shifts'),
        (MultiBandOp, {'shifts': [[0.0, 1.0]]}, 'shifts'),
        (MultiBandOp, {'k': torch.zeros(3), 'shifts': [0.0]}, 'k must'),
        (MultiBandOp, {'k': torch.zeros(2, 1, 3, 4), 'shifts': [0.0]}, 'k must'),
        (MultiBandOp, {'k': torch.ones(1, 1, 1, 1, dtype=torch.complex64), 'shifts': [0.0]}, 'k must'),
        (MultiBandOp.from_caipi, {'multiband_factor': 0, 'fov_shift_factor': 3}, 'multiband_factor'),
        (MultiBandOp.from_caipi, {'multiband_factor': 3, 'fov_shift_factor': 0.5}, 'fov_shift_factor'),
        (MultiBandOp.from_caipi, {'multiband_factor': 3, 'fov_shift_factor': float('inf')}, 'fov_shift_factor'),
        (MultiBandOp.from_caipi, {'multiband_factor': 3, 'fov_shift_factor': float('nan')}, 'fov_shift_factor'),
    ],
)
def test_multi_band_invalid_inputs(constructor: Callable[..., MultiBandOp], kwargs: dict, match: str) -> None:
    """Reject invalid coordinate shapes, slice shifts and CAIPI factors."""
    with pytest.raises(ValueError, match=match):
        constructor(**({'k': torch.zeros(1, 1, 1, 1)} | kwargs))
