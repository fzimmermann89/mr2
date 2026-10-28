"""Trajectory-derived partial Fourier completion for KData."""

from typing import Literal

import pytest
import torch
from mr2.algorithms.partial_fourier import homodyne, pocs
from mr2.data import AcqInfo, KData, KHeader, KTrajectory, SpatialDimension
from mr2.operators import CartesianSamplingOp


@pytest.mark.parametrize('method', ['pocs', 'homodyne'])
@pytest.mark.parametrize(('size', 'axis', 'side'), [(15, -2, 'high'), (16, -1, 'low')])
def test_kdata_partial_fourier(
    random_kheader: KHeader, method: Literal['pocs', 'homodyne'], size: int, axis: int, side: str
) -> None:
    """Infer acquired zeros from reversed trajectories and return consistent independent metadata."""
    header = random_kheader
    header.encoding_matrix = SpatialDimension(1, size, size)
    header.recon_matrix = SpatialDimension(1, size, size)
    header.acq_info = AcqInfo()
    frequency = torch.arange(size) - size // 2
    keep = torch.arange(size // 4, size) if side == 'high' else torch.arange(size - size // 4)
    keep = keep.flip(0)
    ky = frequency[keep] if axis == -2 else frequency
    kx = frequency[keep] if axis == -1 else frequency
    trajectory = KTrajectory(torch.zeros(1, 1, 1, 1, 1), ky.reshape(1, 1, 1, -1, 1), kx.reshape(1, 1, 1, 1, -1))
    data = torch.arange(2 * ky.numel() * kx.numel()).reshape(1, 2, 1, ky.numel(), kx.numel()).to(torch.complex64)
    data[..., 0, 0] = 0
    header.acq_info.idx.k1 = (ky + size // 2).reshape(1, 1, 1, -1, 1)
    header.acq_info.acquisition_time_stamp = header.acq_info.idx.k1.float()
    original = data.clone()
    kdata = KData(header, data, trajectory)
    sampling = CartesianSamplingOp(header.encoding_matrix, trajectory)
    (grid,) = sampling.H(data)
    (mask,) = sampling.H(torch.ones_like(data[:, :1].real))
    expected = (
        pocs(grid, mask.bool(), (-3, -2, -1), axis, n_iterations=3)
        if method == 'pocs'
        else homodyne(grid, mask.bool(), (-3, -2, -1), axis)
    )

    result = kdata.partial_fourier(axis, method, n_iterations=3)

    torch.testing.assert_close(result.data, expected)
    assert result.data.shape == (1, 2, 1, size, size)
    assert result.data.dtype == data.dtype
    (resampled,) = CartesianSamplingOp(header.encoding_matrix, result.traj)(result.data)
    assert torch.equal(resampled, result.data)
    assert torch.equal(result.header.acq_info.idx.k1.flatten(), torch.arange(size))
    timestamp = result.header.acq_info.acquisition_time_stamp.flatten()
    torch.testing.assert_close(
        timestamp,
        torch.arange(size).float().clamp(keep.min(), keep.max()) if axis == -2 else torch.arange(size).float(),
    )
    assert result.header is not header
    assert torch.equal(kdata.data, original)
    assert torch.equal(header.acq_info.idx.k1.flatten(), ky + size // 2)
    result.header.acq_info.acquisition_time_stamp.zero_()
    assert header.acq_info.acquisition_time_stamp.sum() > 0
    if method == 'pocs':
        (measured,) = sampling(result.data)
        assert torch.equal(measured, data)


def test_kdata_partial_fourier_invalid(random_kheader: KHeader) -> None:
    """Reject unknown methods and non-Cartesian trajectories without mutating the input."""
    random_kheader.encoding_matrix = SpatialDimension(1, 1, 8)
    trajectory = KTrajectory(
        torch.zeros(1, 1, 1, 1, 1), torch.zeros(1, 1, 1, 1, 1), torch.arange(8).reshape(1, 1, 1, 1, -1) + 0.25
    )
    kdata = KData(random_kheader, torch.ones(1, 1, 1, 1, 8, dtype=torch.complex64), trajectory)
    with pytest.raises(ValueError, match='integer Cartesian'):
        kdata.partial_fourier(-1)
    with pytest.raises(ValueError, match='method'):
        kdata.partial_fourier(method='unknown')  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='last three'):
        kdata.partial_fourier(0)


@pytest.mark.parametrize('method', ['pocs', 'homodyne'])
def test_kdata_partial_fourier_partition_and_full(random_kheader: KHeader, method: Literal['pocs', 'homodyne']) -> None:
    """Complete the partition axis with broadcast data; full sampling is unchanged on a second call."""
    header = random_kheader
    header.encoding_matrix = SpatialDimension(8, 1, 1)
    header.recon_matrix = SpatialDimension(8, 1, 1)
    header.acq_info = AcqInfo()
    kz = torch.arange(3, -3, -1).reshape(1, 1, -1, 1, 1)
    trajectory = KTrajectory(kz, torch.zeros(1, 1, 1, 1, 1), torch.zeros(1, 1, 1, 1, 1))
    # Constant data is reduced to singleton axes by KData and must broadcast to the trajectory.
    kdata = KData(header, torch.ones(1, 1, 6, 1, 1, dtype=torch.complex64), trajectory)
    completed = kdata.partial_fourier(-3, method)
    again = completed.partial_fourier(-3, method)
    torch.testing.assert_close(again.data, completed.data)
    assert torch.broadcast_shapes(completed.data.shape, completed.traj.shape)[-3:] == (8, 1, 1)
    assert torch.equal(completed.traj.kz.flatten(), torch.arange(-4, 4).float())
    if method == 'pocs':
        (measured,) = CartesianSamplingOp(header.encoding_matrix, trajectory)(completed.data)
        torch.testing.assert_close(measured, torch.ones_like(measured))
