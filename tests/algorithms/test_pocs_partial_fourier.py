"""Tests for Cartesian partial Fourier completion."""

import pytest
import torch
from mr2.algorithms.partial_fourier import pocs
from mr2.algorithms.partial_fourier.homodyne import _partial_fourier_center
from mr2.operators import FastFourierOp


@pytest.mark.parametrize(('size', 'side', 'axis'), [(31, 'low', 0), (31, 'high', -1), (32, 'low', -1), (32, 'high', 0)])
def test_pocs_recovery(size: int, side: str, axis: int) -> None:
    """Both sides of odd/even grids recover smooth complex phase better than zero fill."""
    coordinate = torch.linspace(-1, 1, size)
    y, x = torch.meshgrid(coordinate, coordinate, indexing='ij')
    magnitude = ((x.square() + 1.4 * y.square()) < 0.55).float()
    magnitude += 0.4 * (((x - 0.2).square() + (y + 0.1).square()) < 0.08)
    image = (magnitude * torch.exp(0.3j * (x + y.square()))).to(torch.complex64)
    fft = FastFourierOp((0, 1))
    (kspace,) = fft(image)
    support = torch.arange(size) >= size // 4 if side == 'high' else torch.arange(size) < size - size // 4
    mask = support[:, None] if axis == 0 else support[None, :]
    measured = torch.where(mask, kspace, 0)

    completed = pocs(measured, mask, (0, -1), axis, n_iterations=20)
    (recovered,) = fft.H(completed)
    (zero_fill,) = fft.H(measured)

    assert torch.linalg.vector_norm(recovered - image) < 0.5 * torch.linalg.vector_norm(zero_fill - image)
    assert torch.equal(completed[mask.expand_as(kspace)], kspace[mask.expand_as(kspace)])
    assert completed.dtype == kspace.dtype
    assert completed.device == kspace.device


def test_pocs_preserves_measurements() -> None:
    """Acquired outer samples and zeros survive every iteration; missing values are ignored."""
    kspace = torch.ones(2, 3, 16, 12, dtype=torch.complex64)
    kspace[..., 4, :] = 0
    mask = (torch.arange(16) >= 4)[:, None].expand(16, 12)
    original = kspace.clone()
    kspace[..., :4, :] = torch.tensor(complex(float('nan'), 0))
    for iterations in (0, 1, 4):
        completed = pocs(kspace, mask, (-2, -1), -2, iterations)
        assert torch.equal(completed[..., mask], original[..., mask])
        assert completed.isfinite().all()
        assert completed.device == kspace.device
        assert completed.dtype == kspace.dtype
        if iterations == 0:
            assert torch.equal(completed[..., ~mask], torch.zeros_like(completed[..., ~mask]))
    assert torch.equal(kspace[..., mask], original[..., mask])


def test_pocs_full_sampling() -> None:
    """Full sampling preserves arbitrary complex data exactly, even for singleton axes."""
    kspace = torch.tensor([[-1 + 2j, 0, 3j]], dtype=torch.complex128)
    completed = pocs(kspace, torch.tensor(True), (0, 1), 0)
    assert torch.equal(completed, kspace)


@pytest.mark.parametrize('size', [15, 16])
@pytest.mark.parametrize('side', ['low', 'high'])
def test_partial_fourier_calibration(size: int, side: str) -> None:
    """The calibration taper is symmetric and acquired on odd/even grids and either side."""
    support = torch.arange(size) >= 3 if side == 'high' else torch.arange(size) < size - 3
    center, _, _ = _partial_fourier_center(support, 0, torch.get_default_dtype())
    conjugate_indices = (2 * (size // 2) - torch.arange(size)) % size
    assert torch.equal(center, center[conjugate_indices])
    assert torch.all(center[~support] == 0)
    assert center[size // 2] == 1
    if size % 2 == 0:
        assert center[0] == 0
    assert torch.all(center.diff()[: size // 2] >= 0)


def test_fixed_phase_signed_projection() -> None:
    """One POCS step uses the signed real projection, including negative amplitudes."""
    fft = FastFourierOp((0,))
    kspace = torch.tensor([0, 0, 1 + 4j, 3, 2 - 1j, -4j, 2, -3j], dtype=torch.complex128)
    mask = torch.arange(8) >= 2
    taper = kspace.real.new_tensor([0, 0, 0.25, 0.75, 1, 0.75, 0.25, 0])
    (low_resolution,) = fft.H(kspace * taper)
    phase = torch.exp(1j * low_resolution.angle())
    (image,) = fft.H(kspace)
    demodulated = (phase.conj() * image).real
    assert (demodulated < 0).any()
    (expected,) = fft(phase * demodulated)
    completed = pocs(kspace, mask, (0,), 0, 1)
    torch.testing.assert_close(completed, torch.where(mask, kspace, expected))


@pytest.mark.parametrize(
    ('mask', 'dim', 'axis', 'error'),
    [
        ([False] * 8, (0,), 0, ValueError),
        ([False, False, False, False, True, True, True, True], (0,), 0, ValueError),
        ([False, True, True, True, True, True, True, False], (0,), 0, ValueError),
        ([True, False, True, True, True, True, True, True], (0,), 0, ValueError),
        ([[False, False, True, True, True, True, True, True], [True] * 8], (1,), 1, ValueError),
        ([True] * 8, (0,), -2, IndexError),
        ([[True] * 8], (0,), 1, ValueError),
    ],
    ids=['empty', 'no-conjugate-pair', 'interior', 'hole', 'varying', 'out-of-range', 'missing-pf-axis'],
)
def test_pocs_unsupported_inputs(
    mask: list[bool] | list[list[bool]], dim: tuple[int, ...], axis: int, error: type[Exception]
) -> None:
    """Reject unsupported acquisition support and invalid transform axes."""
    acquired = torch.tensor(mask)
    kspace = torch.ones(acquired.shape, dtype=torch.complex64)
    with pytest.raises(error):
        pocs(kspace, acquired, dim, axis)


@pytest.mark.cuda
def test_pocs_cuda() -> None:
    """CUDA completion matches CPU and preserves acquired zeros and outer samples exactly."""
    kspace = torch.tensor([0, 0, 1 + 4j, 0, 2 - 1j, -4j, 2, -3j], dtype=torch.complex64)
    mask = torch.arange(8) >= 2
    expected = pocs(kspace, mask, (0,), 0)
    data = kspace.cuda()
    acquired = mask.cuda()
    actual = pocs(data, acquired, (0,), 0)
    torch.testing.assert_close(actual.cpu(), expected)
    assert torch.equal(actual[acquired], data[acquired])
    assert actual.device == data.device
    assert actual.dtype == data.dtype
