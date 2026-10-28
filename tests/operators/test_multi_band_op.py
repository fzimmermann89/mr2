"""Tests for simultaneous slice encoding."""

import pytest
import torch
from mr2.algorithms.optimizers import cg
from mr2.operators import FastFourierOp, MultiBandOp, SensitivityOp

from tests import dotproduct_adjointness_test, gradient_of_linear_operator_test


@pytest.mark.parametrize('axis', [-3, -2, -1])
def test_multi_band_encoding(axis: int) -> None:
    """Sample coordinates can vary along any Fourier axis and across batches."""
    shape = [2, 1, 1, 1, 1]
    shape[axis] = 4
    k = torch.tensor([[-6, -2, 0, 4], [-4, 0, 2, 8]], dtype=torch.float64).reshape(shape)
    shifts = torch.tensor([0.0, 0.25, -0.5], dtype=torch.float64)
    op = MultiBandOp(k, shifts)
    x = torch.randn(2, 3, 4, 4, 4, 4, dtype=torch.complex128)
    expected = sum(torch.exp(-2j * torch.pi * k * shift) * x[:, s] for s, shift in enumerate(shifts))
    torch.testing.assert_close(op(x)[0], expected)
    y = torch.randn_like(expected)
    dotproduct_adjointness_test(op, x, y)
    gradient_of_linear_operator_test(op, x, y)
    torch.testing.assert_close(op.gram(x)[0], op.H(op(x)[0])[0])


def test_multi_band_caipi_native_coordinates() -> None:
    """The factor is a full-FOV denominator and selection preserves native phases."""
    k = torch.arange(-8, 9, 2).reshape(1, 1, -1, 1)
    op = MultiBandOp.from_caipi(k, multiband_factor=3, fov_shift_factor=6)
    x = torch.randn(3, 2, 1, 9, 5, dtype=torch.complex64)
    expected = sum(torch.exp(-2j * torch.pi * k * s / 6) * x[s] for s in range(3))
    torch.testing.assert_close(op(x)[0], expected)
    indices = torch.tensor([7, 1, 4])
    selected = MultiBandOp.from_caipi(k.index_select(-2, indices), 3, 6)
    torch.testing.assert_close(selected(x.index_select(-2, indices))[0], expected.index_select(-2, indices))
    assert op.matrix.dtype == torch.complex64


def test_multi_band_caipi_no_shift() -> None:
    """A factor of one sums without phases, also for noninteger coordinates."""
    k = torch.tensor([0.25, 0.5, 1.5]).reshape(1, 1, 3, 1)
    x = torch.randn(3, 2, 1, 3, 4, dtype=torch.complex64)
    torch.testing.assert_close(MultiBandOp.from_caipi(k, 3, 1)(x)[0], x.sum(0))


def test_multi_band_fourier_sensitivity_composition() -> None:
    """Joint encoding uses per-slice maps and its adjoint preserves slice coupling."""
    k = torch.arange(-3, 3).reshape(1, 1, 6, 1)
    collapse = MultiBandOp.from_caipi(k, 3, 3)
    fourier = FastFourierOp(dim=(-2, -1))
    csm = torch.randn(3, 4, 1, 6, 8, dtype=torch.complex64)
    encoding = collapse @ fourier @ SensitivityOp(csm)
    x = torch.randn(3, 1, 1, 6, 8, dtype=torch.complex64)
    expected = sum(torch.exp(-2j * torch.pi * k * s / 3) * fourier(csm[s] * x[s])[0] for s in range(3))
    torch.testing.assert_close(encoding(x)[0], expected)
    dotproduct_adjointness_test(encoding, x, torch.randn_like(expected))
    single_slice = torch.zeros_like(x)
    single_slice[0] = x[0]
    assert encoding.gram(single_slice)[0][1:].abs().sum() > 0

    normal = encoding.gram + 2.0
    rhs = normal(x)[0]
    solution = cg(normal, rhs, max_iterations=100, tolerance=1e-5)[0]
    torch.testing.assert_close(solution, x, rtol=2e-5, atol=2e-5)


def test_multi_band_sequence_shift_precision() -> None:
    """Python shifts retain double precision when coordinates use float64."""
    k = torch.tensor([1.0, 100.0], dtype=torch.float64).reshape(1, 1, 2, 1)
    op = MultiBandOp(k, [0.0, 1 / 3])
    torch.testing.assert_close(op.matrix[1], torch.exp(-2j * torch.pi * k[0] / 3), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('factor', [0, -1, 1.5, True])
def test_multi_band_invalid_multiband_factor(factor: int) -> None:
    """Require an explicit positive slice count."""
    with pytest.raises(ValueError, match='multiband_factor'):
        MultiBandOp.from_caipi(torch.zeros(1, 1, 1, 1), factor, 3)


@pytest.mark.parametrize('factor', [0, -1, 0.5, float('inf'), float('nan')])
def test_multi_band_invalid_shift_factor(factor: float) -> None:
    """Reject invalid FOV-shift denominators."""
    with pytest.raises(ValueError, match='fov_shift_factor'):
        MultiBandOp.from_caipi(torch.zeros(1, 1, 1, 1), 3, factor)


@pytest.mark.parametrize('shifts', [[], [[0.0, 1.0]], [float('nan')], [1j]])
def test_multi_band_invalid_shifts(shifts: list) -> None:
    """Reject ambiguous or nonfinite slice shifts."""
    with pytest.raises(ValueError, match='shifts'):
        MultiBandOp(torch.zeros(1, 1, 1, 1), shifts)


@pytest.mark.parametrize('k', [torch.zeros(3), torch.zeros(2, 1, 3, 4), torch.full((1, 1, 1, 1), float('nan'))])
def test_multi_band_invalid_coordinates(k: torch.Tensor) -> None:
    """Coordinates require the singleton coil axis and finite values."""
    with pytest.raises(ValueError, match='k must'):
        MultiBandOp(k, [0.0, 0.5])
