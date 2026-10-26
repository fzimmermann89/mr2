"""Tests for trajectory-driven Cartesian GRAPPA reconstruction."""

import pytest
import torch
from mr2.algorithms.reconstruction import GrappaReconstruction
from mr2.data import AcqInfo, CsmData, KData, KHeader, KTrajectory, SpatialDimension
from mr2.operators import FastFourierOp


def _point_object(shape: tuple[int, int, int], dtype: torch.dtype) -> torch.Tensor:
    """Fourier data of three points observed by three distinct coil profiles."""
    coordinates = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) for n in shape), indexing='ij')
    frequencies = torch.tensor([[0.13, 0.21, -0.17], [-0.19, 0.07, 0.11], [0.05, -0.16, 0.23]])
    phase = torch.stack([coordinate[..., None] * frequencies[axis] for axis, coordinate in enumerate(coordinates)]).sum(
        0
    )
    spectra = torch.exp(2j * torch.pi * phase)
    amplitudes = torch.tensor([[1, 0.4j, 0.2], [0.3j, 1, -0.5], [0.2, -0.6j, 1]], dtype=spectra.dtype)
    return torch.einsum('...p,cp->c...', spectra, amplitudes).to(dtype)


def _kdata(full: torch.Tensor, acquired: torch.Tensor) -> KData:
    """Store samples compactly with an explicit Cartesian acquisition trajectory."""
    shape = full.shape[-3:]
    coordinates = torch.meshgrid(*(torch.arange(n, device=full.device) - n // 2 for n in shape), indexing='ij')
    kz, ky, kx = (coordinate[acquired].reshape(1, 1, 1, 1, -1) for coordinate in coordinates)
    trajectory = KTrajectory(kz, ky, kx)
    header = KHeader(
        recon_matrix=SpatialDimension(*shape),
        encoding_matrix=SpatialDimension(*shape),
        recon_fov=SpatialDimension(1.0, 1.0, 1.0),
        encoding_fov=SpatialDimension(1.0, 1.0, 1.0),
        acq_info=AcqInfo(),
    )
    return KData(data=full[..., acquired].unsqueeze(-2).unsqueeze(-2), header=header, traj=trajectory)


@pytest.mark.parametrize(
    ('shape', 'kernel', 'acceleration'),
    [((1, 24, 20), (1, 3, 3), 2), ((1, 24, 20), (1, 5, 3), 3), ((9, 16, 12), (3, 3, 3), 2)],
)
@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_grappa_reconstruction(
    shape: tuple[int, int, int], kernel: tuple[int, int, int], acceleration: int, dtype: torch.dtype
) -> None:
    """Recover known coil data using cropped ACS and trajectory-derived sampling."""
    full = _point_object(shape, dtype)[None]
    acquired = torch.zeros(shape, dtype=torch.bool)
    acquired[..., ::acceleration, :] = True
    acquired[..., 6:10, :] = True
    data = _kdata(full, acquired)
    acs = full[..., 2:-2, 2:-2]
    reconstruction = GrappaReconstruction(data, acs, kernel, regularization=0, csm=None)
    completed = reconstruction.interpolate(data)
    torch.testing.assert_close(completed, full, atol=2e-5, rtol=2e-5)
    assert torch.equal(completed[..., acquired], full[..., acquired])
    (expected,) = FastFourierOp().H(full)
    torch.testing.assert_close(reconstruction(data).data, expected, atol=2e-5, rtol=2e-5)
    # Reusing calibrated kernels must not require ACS again.
    doubled = data.clone()
    doubled.data = data.data * 2
    torch.testing.assert_close(reconstruction(doubled).data, expected * 2, atol=4e-5, rtol=2e-5)


def test_grappa_reconstruction_zero_and_coil_combination() -> None:
    """Preserve acquired zeros and use the normal IData coil combination interface."""
    full = _point_object((1, 16, 18), torch.complex64)[None]
    acquired = torch.ones((1, 16, 18), dtype=torch.bool)
    acquired[..., 1::2, :] = False
    full[..., 0, 0, 0] = 0
    data = _kdata(full, acquired)
    reconstruction = GrappaReconstruction(data, full, (1, 3, 3))
    completed = reconstruction.interpolate(data)
    assert torch.equal(completed[..., acquired], full[..., acquired])
    assert reconstruction(data).data.shape == (1, 1, 1, 16, 18)
    csm = CsmData.from_idata_walsh(GrappaReconstruction(data, full, (1, 3, 3), csm=None)(data))
    explicit = GrappaReconstruction(data, full, (1, 3, 3), csm=csm)
    torch.testing.assert_close(explicit(data).data, reconstruction(data).data)


def test_grappa_batched_volume() -> None:
    """Use noncontiguous shared cropped ACS for separate 3D data batches."""
    full = _point_object((9, 16, 12), torch.complex128)
    full = torch.stack((full, (2 - 1j) * full)).transpose(-1, -2)
    acs = full[:1, :, 1:-1, 2:-2, 2:-2]
    acquired = torch.ones(full.shape[-3:], dtype=torch.bool)
    acquired[..., 1::2] = False
    data = _kdata(full, acquired)
    reconstruction = GrappaReconstruction(data, acs, (3, 3, 3), regularization=0, csm=None)
    torch.testing.assert_close(reconstruction.interpolate(data), full, atol=2e-5, rtol=2e-5)
    assert reconstruction.kernels.shape[1] == 1


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_grappa_reconstruction_cuda(dtype: torch.dtype) -> None:
    """Move calibration, sampling and interpolation together with the reconstruction."""
    full = _point_object((1, 18, 16), dtype)[None]
    acquired = torch.ones(full.shape[-3:], dtype=torch.bool)
    acquired[..., 1::2, :] = False
    data = _kdata(full, acquired)
    reconstruction = GrappaReconstruction(data, full, (1, 3, 3), csm=None)
    expected = reconstruction(data).data
    actual = reconstruction.cuda()(data.cuda()).data
    assert actual.device.type == 'cuda'
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.cpu(), expected, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(reconstruction.cpu()(data).data, expected)


def test_grappa_full_sampling_and_recalibration() -> None:
    """A full grid is unchanged, and updating a trajectory recalibrates its kernels."""
    full = _point_object((1, 24, 20), torch.complex128)[None]
    mask = torch.ones(full.shape[-3:], dtype=torch.bool)
    data = _kdata(full, mask)
    reconstruction = GrappaReconstruction(data, full, (1, 5, 3), regularization=0, csm=None)
    assert torch.equal(reconstruction.interpolate(data), full)
    for acceleration in (2, 3):
        mask[:] = False
        mask[..., ::acceleration, :] = True
        data = _kdata(full, mask)
        reconstruction.recalculate_fourierop(data)
        torch.testing.assert_close(reconstruction.interpolate(data), full, atol=2e-5, rtol=2e-5)


def test_grappa_reject_noncartesian() -> None:
    """Reject non-Cartesian acquisition coordinates before creating interpolation kernels."""
    full = _point_object((1, 12, 12), torch.complex64)[None]
    data = _kdata(full, torch.ones(full.shape[-3:], dtype=torch.bool))
    data.traj = KTrajectory(data.traj.kz, data.traj.ky + 0.25, data.traj.kx)
    with pytest.raises(ValueError, match='Cartesian data'):
        GrappaReconstruction(data, full, (1, 3, 3), csm=None)


def test_grappa_reject_repeated_samples() -> None:
    """Avoid silently selecting one of several measurements at a repeated grid point."""
    full = _point_object((1, 12, 12), torch.complex64)[None]
    data = _kdata(full, torch.ones(full.shape[-3:], dtype=torch.bool))
    data.data = torch.cat((data.data, data.data), dim=-1)
    kz, ky, kx = (torch.cat((k, k), dim=-1) for k in (data.traj.kz, data.traj.ky, data.traj.kx))
    data.traj = KTrajectory(kz, ky, kx)
    with pytest.raises(ValueError, match='distinct samples'):
        GrappaReconstruction(data, full, (1, 3, 3), csm=None)


def test_grappa_reject_varying_batch_masks() -> None:
    """Reject different sampling supports across batches explicitly."""
    full = _point_object((1, 12, 12), torch.complex64)[None]
    mask = torch.ones(full.shape[-3:], dtype=torch.bool)
    mask[..., 1::2, :] = False
    first, second = _kdata(full, mask), _kdata(full, ~mask)
    first.data = torch.cat((first.data, second.data), dim=0)
    kz, ky, kx = (
        torch.cat(torch.broadcast_tensors(a, b), dim=0)
        for a, b in zip(
            (first.traj.kz, first.traj.ky, first.traj.kx), (second.traj.kz, second.traj.ky, second.traj.kx), strict=True
        )
    )
    first.traj = KTrajectory(kz, ky, kx)
    with pytest.raises(ValueError, match='shared sampling pattern'):
        GrappaReconstruction(first, full, (1, 3, 3), csm=None)


@pytest.mark.parametrize('failure', ['repeated_samples', 'insufficient_acs'])
def test_grappa_failed_recalibration(failure: str) -> None:
    """A failed sampling update or kernel fit leaves the original reconstruction usable."""
    full = _point_object((1, 16, 18), torch.complex128)[None]
    mask = torch.ones(full.shape[-3:], dtype=torch.bool)
    mask[..., 1::2, :] = False
    original = _kdata(full, mask)
    # Twenty calibration rows suffice for R2 patterns, but not an isolated hole.
    acs = full[..., 3:9, 3:10]
    reconstruction = GrappaReconstruction(original, acs, (1, 3, 3), regularization=0, csm=None)
    expected = reconstruction(original).data
    expected_direct = reconstruction.direct_reconstruction(original).data
    if failure == 'repeated_samples':
        invalid = original.clone()
        invalid.data = torch.cat((invalid.data, invalid.data), dim=-1)
        kz, ky, kx = (torch.cat((k, k), dim=-1) for k in (invalid.traj.kz, invalid.traj.ky, invalid.traj.kx))
        invalid.traj = KTrajectory(kz, ky, kx)
        message = 'distinct samples'
    else:
        mask[:] = True
        mask[..., 8, 9] = False
        invalid = _kdata(full, mask)
        message = 'Insufficient ACS'
    with pytest.raises(ValueError, match=message):
        reconstruction.recalculate_fourierop(invalid)
    torch.testing.assert_close(reconstruction(original).data, expected)
    torch.testing.assert_close(reconstruction.direct_reconstruction(original).data, expected_direct)
