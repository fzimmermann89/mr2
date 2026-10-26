"""Trajectory-based apodization of KData."""

from typing import Literal

import pytest
import torch
from mr2.data import KData, KHeader, KTrajectory, SpatialDimension


@pytest.fixture
def kdata() -> KData:
    """Small Cartesian acquisition with two coils."""
    shape = SpatialDimension(1, 5, 6)
    fov = SpatialDimension(0.01, 0.2, 0.3)
    header = KHeader(shape, shape.clone(), fov, fov.clone())
    kz = torch.zeros(1, 1, 1, 1, 1)
    ky = torch.arange(-2, 3).reshape(1, 1, 1, 5, 1).float()
    kx = torch.arange(-3, 3).reshape(1, 1, 1, 1, 6).float()
    data = torch.randn((1, 2, 1, 5, 6), dtype=torch.complex64, generator=torch.Generator().manual_seed(1))
    return KData(header, data, KTrajectory(kz, ky, kx))


@pytest.mark.parametrize('size', [5, 6])
@pytest.mark.parametrize('window', ['hann', 'gaussian'])
def test_apodize_center_and_width(kdata: KData, size: int, window: Literal['hann', 'gaussian']) -> None:
    """Widths refer to encoding steps, with DC gain one for odd and even grids."""
    coordinate = (torch.arange(size) - size // 2).reshape(1, 1, 1, 1, size)
    kdata.header.encoding_matrix.x = size
    kdata.traj = KTrajectory(torch.zeros_like(coordinate), torch.zeros_like(coordinate), coordinate.float())
    kdata.data = torch.ones(1, 2, 1, 1, size, dtype=torch.complex128)
    result = kdata.apodize(window=window)
    q = coordinate.double() / (size / 2)
    expected = (
        (1 + torch.cos(torch.pi * q)) / 2
        if window == 'hann'
        else torch.exp(-torch.log(torch.tensor(2.0, dtype=torch.float64)) * q.square())
    )
    torch.testing.assert_close(result.data, expected.expand_as(result.data).to(result.data.dtype))
    assert result.data[..., size // 2].eq(1).all()


@pytest.mark.parametrize('radial', [False, True])
@pytest.mark.parametrize('window', ['hann', 'gaussian'])
def test_apodize_coordinates(kdata: KData, radial: bool, window: Literal['hann', 'gaussian']) -> None:
    """Permuting and packing coordinates into a non-Cartesian readout preserves weights."""
    result = kdata.apodize(SpatialDimension(2.0, 4.0, 6.0), window=window, radial=radial)
    permutation = torch.randperm(30, generator=torch.Generator().manual_seed(0))
    kz, ky, kx = (
        k.expand(1, 1, 1, 5, 6).flatten(-3)[..., permutation].unsqueeze(-2).unsqueeze(-2)
        for k in (kdata.traj.kz, kdata.traj.ky, kdata.traj.kx)
    )
    packed = KData(
        kdata.header.clone(),
        kdata.data.flatten(-3)[..., permutation].unsqueeze(-2).unsqueeze(-2),
        KTrajectory(kz, ky, kx),
    )
    actual = packed.apodize(SpatialDimension(2.0, 4.0, 6.0), window=window, radial=radial)
    torch.testing.assert_close(actual.data.flatten(-3), result.data.flatten(-3)[..., permutation])


@pytest.mark.parametrize('radial', [False, True])
def test_apodize_nonuniform_batch_support(kdata: KData, radial: bool) -> None:
    """Fractional, batched coordinates determine Hann weights and compact support."""
    kx = torch.tensor([0, 0.5, 1, 2, 3, 4, 1.0, 1, 1, 1, 1, 1]).reshape(2, 1, 1, 1, 6)
    ky = torch.ones(1, 1, 1, 1, 1)
    kdata.traj = KTrajectory(torch.zeros_like(ky), ky, kx)
    kdata.data = torch.ones(2, 2, 1, 1, 6, dtype=torch.complex64)
    result = kdata.apodize(4, radial=radial)
    q = torch.sqrt((kx / 2).square() + 0.25) if radial else kx / 2
    expected = torch.where(q < 1, (1 + (torch.pi * q).cos()) / 2, 0)
    if not radial:
        expected *= 0.5
    torch.testing.assert_close(result.data, expected.expand_as(result.data).to(result.data.dtype))
    assert result.data[0, ..., 3:].eq(0).all()
    gaussian = kdata.apodize(2, window='gaussian', radial=radial)
    torch.testing.assert_close(gaussian.data[:, 0, 0, 0, 0].real, torch.tensor([0.5, 0.25]))


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_apodize_copy(kdata: KData, dtype: torch.dtype, device: str) -> None:
    """Data dtype/device, acquired zeros, readout metadata and trajectory are preserved."""
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    kdata = kdata.to(device=device)
    kdata.data = kdata.data.to(dtype=dtype)
    kdata.data[..., 0] = 0
    kdata.header.acq_info.sample_time_us = torch.tensor(2.5, device=device)
    kdata.header.acq_info.flags = torch.tensor(17, device=device)
    original = kdata.clone()
    result = kdata.apodize()
    assert result.data.dtype == dtype
    assert result.data.device == kdata.data.device
    assert result.data.shape == kdata.data.shape
    assert result.data[..., 0].eq(0).all()
    torch.testing.assert_close(kdata.data, original.data)
    assert result.header is not kdata.header
    assert result.traj is not kdata.traj
    for name in ('kz', 'ky', 'kx'):
        torch.testing.assert_close(getattr(result.traj, name), getattr(original.traj, name))
    assert result.header.encoding_matrix == original.header.encoding_matrix
    assert result.header.encoding_fov == original.header.encoding_fov
    torch.testing.assert_close(result.header.acq_info.sample_time_us, original.header.acq_info.sample_time_us)
    torch.testing.assert_close(result.header.acq_info.flags, original.header.acq_info.flags)


@pytest.mark.parametrize('width', [0, -1, float('nan'), float('inf'), SpatialDimension(1.0, 0.0, 1.0)])
def test_apodize_invalid_width(kdata: KData, width: float | SpatialDimension[float]) -> None:
    """Invalid widths fail before constructing weights."""
    with pytest.raises(ValueError, match='width'):
        kdata.apodize(width)


def test_apodize_invalid_window(kdata: KData) -> None:
    """Unknown window names are rejected."""
    with pytest.raises(ValueError, match='window'):
        kdata.apodize(window='unknown')  # type: ignore[arg-type]


@pytest.mark.parametrize('scale', [0.0, 1e-8])
def test_apodize_preserves_small_data(kdata: KData, scale: float) -> None:
    """Zero and small samples keep their shape and weighted values."""
    expected = kdata.apodize().data * scale
    kdata.data = kdata.data * scale
    result = kdata.apodize()
    assert result.data.shape == kdata.data.shape
    torch.testing.assert_close(result.data, expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize('radial', [False, True])
def test_apodize_trajectory_gradient_at_dc(kdata: KData, radial: bool) -> None:
    """Hann coordinate gradients are zero at DC and finite throughout support."""
    kx = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float64, requires_grad=True)
    ky = torch.zeros_like(kx, requires_grad=True)
    kz = torch.zeros_like(kx, requires_grad=True)
    kdata.traj = KTrajectory(
        kz.reshape(1, 1, 1, 1, 4),
        ky.reshape(1, 1, 1, 1, 4),
        kx.reshape(1, 1, 1, 1, 4),
        repeat_detection_tolerance=None,
    )
    kdata.data = torch.ones(1, 1, 1, 1, 4, dtype=torch.complex128)
    kdata.apodize(4, radial=radial).data.real.sum().backward()
    expected = -torch.pi / 4 * torch.sin(torch.pi * kx.detach() / 2)
    expected[-1] = 0
    assert kx.grad is not None
    torch.testing.assert_close(kx.grad, expected)
    assert ky.grad is not None
    assert kz.grad is not None
    torch.testing.assert_close(ky.grad, torch.zeros_like(ky))
    torch.testing.assert_close(kz.grad, torch.zeros_like(kz))
