"""Tests for homodyne phase correction and ramp scaling."""

import pytest
import torch
from mr2.algorithms.partial_fourier import homodyne
from mr2.operators import FastFourierOp


@pytest.mark.parametrize(('size', 'side'), [(15, 'low'), (15, 'high'), (16, 'low'), (16, 'high')])
def test_homodyne_ramp(size: int, side: str) -> None:
    """A constant-phase object recovers exact scale, signed detail and acquired Nyquist."""
    phase = torch.tensor(0.7j, dtype=torch.complex128).exp()
    coordinate = torch.arange(size, dtype=phase.real.dtype) - size // 2
    amplitude = 1 + 2.5 * torch.cos(2 * torch.pi * 5 * coordinate / size)
    if size % 2 == 0 and side == 'low':
        amplitude += 0.2 * torch.cos(torch.pi * coordinate)
    image = amplitude * phase
    fft = FastFourierOp((0,))
    (kspace,) = fft(image)
    mask = torch.arange(size) >= 4 if side == 'high' else torch.arange(size) < size - 4
    measured = torch.where(mask, kspace, torch.nan)
    corrected = homodyne(measured, mask, (0,), -1)
    restored = homodyne(measured, mask, (0,), 0, restore_phase=True)
    torch.testing.assert_close(corrected, amplitude)
    torch.testing.assert_close(restored, image)
    assert corrected.min() < 0
    assert corrected.dtype == kspace.real.dtype
    assert restored.dtype == kspace.dtype


@pytest.mark.parametrize(('size', 'axis', 'side'), [(31, 0, 'low'), (32, -1, 'high')])
def test_homodyne_smooth_phase(size: int, axis: int, side: str) -> None:
    """Smooth complex-phase images improve over zero fill, including partial echo."""
    coordinate = torch.linspace(-1, 1, size)
    y, x = torch.meshgrid(coordinate, coordinate, indexing='ij')
    amplitude = ((x.square() + 1.4 * y.square()) < 0.55).float()
    image = amplitude * torch.exp(0.3j * (x + y.square()))
    fft = FastFourierOp((0, 1))
    (kspace,) = fft(image)
    support = torch.arange(size) >= size // 4 if side == 'high' else torch.arange(size) < size - size // 4
    mask = support[:, None] if axis == 0 else support[None, :]
    measured = torch.where(mask, kspace, 0)
    restored = homodyne(measured, mask, (0, -1), axis, restore_phase=True)
    (zero_fill,) = fft.H(measured)
    assert torch.linalg.vector_norm(restored - image) < 0.7 * torch.linalg.vector_norm(zero_fill - image)
    assert restored.dtype == image.dtype


def test_homodyne_full_and_batched() -> None:
    """Full sampling has unit scale, while partial sampling preserves batch/device conventions."""
    kspace = torch.ones(2, 3, 16, dtype=torch.complex64)
    fft = FastFourierOp((-1,))
    (image,) = fft.H(kspace)
    full = torch.tensor(True)
    torch.testing.assert_close(homodyne(kspace, full, (-1,), -1), image.abs())
    torch.testing.assert_close(homodyne(kspace, full, (-1,), -1, restore_phase=True), image)
    mask = torch.arange(16) >= 4
    batched = homodyne(kspace, mask, (-1,), -1)
    single = homodyne(kspace[0, 0], mask, (0,), 0)
    torch.testing.assert_close(batched, single.expand_as(batched))
    assert batched.device == kspace.device
    assert batched.dtype == kspace.real.dtype


def test_homodyne_linear_transition() -> None:
    """Check explicit ramp and phase correction against a small centered FFT calculation."""
    kspace = torch.tensor([0, 0, 2j, 1, 5, 1j, 2, -1j], dtype=torch.complex128)
    mask = torch.arange(8) >= 2
    fft = FastFourierOp((0,))
    taper = kspace.real.new_tensor([0, 0, 0.25, 0.75, 1, 0.75, 0.25, 0])
    (low_resolution,) = fft.H(kspace * taper)
    phase = torch.exp(1j * low_resolution.angle())
    weight = kspace.real.new_tensor([1, 0, 0, 0.5, 1, 1.5, 2, 2])
    (weighted,) = fft.H(kspace * weight)
    expected = (phase.conj() * weighted).real
    torch.testing.assert_close(homodyne(kspace, mask, (0,), 0), expected)


@pytest.mark.cuda
def test_homodyne_cuda() -> None:
    """CUDA phase correction matches CPU for signed real and phase-restored complex output."""
    kspace = torch.tensor([0, 0, 2j, 1, 5, 1j, 2, -1j], dtype=torch.complex64)
    mask = torch.arange(8) >= 2
    data = kspace.cuda()
    acquired = mask.cuda()
    for restore_phase in (False, True):
        expected = homodyne(kspace, mask, (0,), 0, restore_phase=restore_phase)
        actual = homodyne(data, acquired, (0,), 0, restore_phase=restore_phase)
        torch.testing.assert_close(actual.cpu(), expected)
        assert actual.device == data.device
        assert actual.dtype == expected.dtype
