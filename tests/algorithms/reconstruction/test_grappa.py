"""Tests for Cartesian GRAPPA calibration and interpolation."""

from itertools import product

import pytest
import torch
from mr2.algorithms.reconstruction import grappa


def _point_object(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Fourier data of three points observed by three distinct coil profiles."""
    coordinates = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) for n in shape), indexing='ij')
    frequencies = torch.tensor([[0.13, 0.21, -0.17], [-0.19, 0.07, 0.11], [0.05, -0.16, 0.23]])
    phase = torch.stack([coordinate[..., None] * frequencies[axis] for axis, coordinate in enumerate(coordinates)]).sum(
        0
    )
    spectra = torch.exp(2j * torch.pi * phase)
    amplitudes = torch.tensor([[1, 0.4j, 0.2], [0.3j, 1, -0.5], [0.2, -0.6j, 1]], dtype=spectra.dtype)
    return torch.einsum('...p,cp->c...', spectra, amplitudes).to(dtype)


@pytest.mark.parametrize(
    ('shape', 'kernel', 'acceleration'),
    [((24, 20), (3, 3), 2), ((24, 20), (5, 3), 3), ((9, 16, 12), (3, 3, 3), 2)],
)
@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_grappa_point_object(
    shape: tuple[int, ...], kernel: tuple[int, ...], acceleration: int, dtype: torch.dtype
) -> None:
    """Recover known multi-coil Fourier data, including one-sided boundary patterns."""
    full = _point_object(shape, dtype)
    acquired = torch.zeros(shape, dtype=torch.bool)
    acquired[..., ::acceleration, :] = True
    acquired[..., 6:10, :] = True  # Embedded ACS changes the local patterns.
    data = torch.where(acquired, full, complex('nan'))
    output = grappa(data, full, acquired, kernel, regularization=0)
    torch.testing.assert_close(output, full, atol=2e-5, rtol=2e-5)
    assert torch.equal(output[..., acquired], full[..., acquired])
    assert torch.isnan(data[..., ~acquired]).all()


@pytest.mark.parametrize('regularization', [0.0, 0.03])
def test_grappa_explicit_calibration(regularization: float) -> None:
    """Match explicit complex solves with center exclusion and reduced edge kernels."""
    generator = torch.Generator().manual_seed(13)
    acs = torch.randn(2, 5, 9, generator=generator, dtype=torch.complex128)
    data = torch.randn(2, 3, 5, generator=generator, dtype=torch.complex128)
    acquired = torch.tensor([[False, True, False, True, False]]).expand(3, 5)
    data[:, 0, 1] = 0  # A real acquired zero must remain acquired and act as a source.
    before = data.clone()
    result = grappa(data, acs, acquired, (3, 3), regularization=regularization)
    for y, x in product(range(3), range(5)):
        if acquired[y, x]:
            continue
        offsets = [
            (dy, dx)
            for dy, dx in product(range(-1, 2), repeat=2)
            if 0 <= y + dy < 3 and 0 <= x + dx < 5 and acquired[y + dy, x + dx]
        ]
        sources = torch.stack(
            [
                torch.stack([acs[:, ay + dy, ax + dx] for dy, dx in offsets], dim=-1).flatten()
                for ay, ax in product(range(1, 4), range(1, 8))
            ]
        )
        targets = torch.stack([acs[:, ay, ax] for ay, ax in product(range(1, 4), range(1, 8))])
        if regularization:
            ridge = regularization * sources.abs().square().sum() / sources.shape[-1]
            weights = torch.linalg.solve(
                sources.mH @ sources + ridge * torch.eye(sources.shape[-1]), sources.mH @ targets
            )
        else:
            weights = torch.linalg.lstsq(sources, targets, driver='gelsd').solution
        values = torch.stack([data[:, y + dy, x + dx] for dy, dx in offsets], dim=-1).flatten()
        torch.testing.assert_close(result[:, y, x], values @ weights)
    assert torch.equal(result[..., acquired], data[..., acquired])
    assert torch.equal(data, before)


def test_grappa_batches_and_scale() -> None:
    """Keep batches independent, broadcast ACS, and scale ridge with source energy."""
    generator = torch.Generator().manual_seed(42)
    data = torch.randn(2, 3, 2, 8, 9, generator=generator, dtype=torch.complex128)
    acs = torch.randn(1, 3, 2, 12, 13, generator=generator, dtype=data.dtype)
    acquired = torch.ones((8, 9), dtype=torch.bool)
    acquired[1::2] = False
    output = grappa(data, acs, acquired, (3, 3))
    for i, j in product(range(2), range(3)):
        torch.testing.assert_close(output[i, j], grappa(data[i, j], acs[0, j], acquired, (3, 3)))
    scale = 3.1 - 2.7j
    torch.testing.assert_close(grappa(scale * data, scale * acs, acquired, (3, 3)), scale * output)
    torch.testing.assert_close(grappa(data, scale * acs, acquired, (3, 3)), output)


def test_grappa_fully_acquired() -> None:
    """A full grid needs no interpolation and is copied exactly, including zeros."""
    data = torch.zeros((2, 4, 5), dtype=torch.complex64)
    output = grappa(data, data, torch.ones((4, 5), dtype=torch.bool), (3, 3))
    assert torch.equal(output, data)
    assert output.data_ptr() != data.data_ptr()


@pytest.mark.parametrize(
    ('case', 'message'),
    [
        ('kernel_even', 'positive odd'),
        ('kernel_dim', 'two or three'),
        ('regularization', 'finite and nonnegative'),
        ('nan_regularization', 'finite and nonnegative'),
        ('real', 'complex64 or complex128'),
        ('dtype', 'same complex64 or complex128'),
        ('mask_dtype', 'boolean mask'),
        ('mask_batch', 'exactly the spatial shape'),
        ('coils', 'same number of coils'),
        ('batch', 'batch dimensions'),
        ('small', 'at least kernel_size'),
        ('insufficient', 'Insufficient ACS'),
        ('empty_neighborhood', 'acquired neighbors'),
        ('zero', 'source energy'),
        ('nonfinite', 'must be finite'),
    ],
)
def test_grappa_invalid(case: str, message: str) -> None:
    """Reject incompatible inputs and unsupported or uncalibratable neighborhoods."""
    data = torch.ones((2, 8, 9), dtype=torch.complex64)
    acs = data.clone()
    acquired = torch.ones((8, 9), dtype=torch.bool)
    acquired[1::2] = False
    kernel: tuple[int, ...] = (3, 3)
    regularization = 0.01
    if case == 'kernel_even':
        kernel = (2, 3)
    elif case == 'kernel_dim':
        kernel = (3,)
    elif case == 'regularization':
        regularization = -1
    elif case == 'nan_regularization':
        regularization = float('nan')
    elif case == 'real':
        data, acs = data.real, acs.real
    elif case == 'dtype':
        acs = acs.to(torch.complex128)
    elif case == 'mask_dtype':
        acquired = acquired.float()
    elif case == 'mask_batch':
        acquired = acquired[None]
    elif case == 'coils':
        acs = acs[:1]
    elif case == 'batch':
        acs = acs[None].expand(3, -1, -1, -1)
    elif case == 'small':
        acs = acs[:, :2]
    elif case == 'insufficient':
        acs = acs[:, :3, :3]
    elif case == 'empty_neighborhood':
        acquired[:] = False
    elif case == 'zero':
        acs = acs * 0
    elif case == 'nonfinite':
        acs[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match=message):
        grappa(data, acs, acquired, kernel, regularization=regularization)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
@pytest.mark.parametrize('regularization', [0.0, 0.01])
def test_grappa_cuda(dtype: torch.dtype, regularization: float) -> None:
    """Keep interpolation and calibration on CUDA with the original complex dtype."""
    data = _point_object((18, 16), dtype).cuda()
    acquired = torch.ones((18, 16), dtype=torch.bool, device='cuda')
    acquired[1::2] = False
    result = grappa(data, data, acquired, (3, 3), regularization=regularization)
    expected = grappa(data.cpu(), data.cpu(), acquired.cpu(), (3, 3), regularization=regularization)
    assert result.dtype == dtype
    assert result.device == data.device
    torch.testing.assert_close(result.cpu(), expected, atol=3e-4, rtol=3e-4)
    assert torch.equal(result[..., acquired], data[..., acquired])
