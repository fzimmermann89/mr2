"""Trajectory-based apodization of KData."""

import pytest
import torch
from mr2.data import KData, KTrajectory, SpatialDimension


def test_apodize_copy_and_coordinates(consistently_shaped_kdata: KData) -> None:
    """Default encoding widths, copies and weights are independent of sample order."""
    kdata = consistently_shaped_kdata
    kdata.header.encoding_matrix = SpatialDimension(1, 8, 12)
    ky = torch.arange(-2, 3).reshape(1, 1, 1, 5, 1).float()
    kx = torch.arange(-3, 3).reshape(1, 1, 1, 1, 6).float()
    kdata.traj = KTrajectory(torch.zeros(1, 1, 1, 1, 1), ky, kx)
    kdata.data = torch.randn((1, 2, 1, 5, 6), dtype=torch.complex128, generator=torch.Generator().manual_seed(1))
    kdata.data[..., 0] = 0
    original = kdata.clone()
    result = kdata.apodize()
    expected = kdata.data * (1 + (torch.pi * ky / 4).cos()) / 2
    expected *= (1 + (torch.pi * kx / 6).cos()) / 2
    torch.testing.assert_close(result.data, expected)
    torch.testing.assert_close(kdata.data, original.data)
    torch.testing.assert_close(result.traj.as_tensor(), original.traj.as_tensor())
    assert result.header is not kdata.header
    assert result.traj is not kdata.traj
    permutation = torch.randperm(30, generator=torch.Generator().manual_seed(0))
    kz, ky, kx = (
        k.expand(1, 1, 1, 5, 6).reshape(1, 1, 1, 1, 30)[..., permutation]
        for k in (kdata.traj.kz, kdata.traj.ky, kdata.traj.kx)
    )
    packed = KData(kdata.header.clone(), kdata.data.reshape(1, 2, 1, 1, 30)[..., permutation], KTrajectory(kz, ky, kx))
    torch.testing.assert_close(packed.apodize().data.flatten(-3), result.data.flatten(-3)[..., permutation])


@pytest.mark.parametrize('radial', [False, True])
def test_apodize_nonuniform_support(consistently_shaped_kdata: KData, radial: bool) -> None:
    """Fractional batched coordinates determine Hann support and Gaussian FWHM."""
    kdata = consistently_shaped_kdata
    kx = torch.tensor([0, 0.5, 1, 2, 3, 4, 1.0, 1, 1, 1, 1, 1]).reshape(2, 1, 1, 1, 6)
    ky = torch.ones(1, 1, 1, 1, 1)
    kdata.traj = KTrajectory(torch.zeros_like(ky), ky, kx)
    kdata.data = torch.ones(2, 2, 1, 1, 6, dtype=torch.complex64)
    q = ((kx / 2).square() + 0.25).sqrt() if radial else kx / 2
    expected = torch.where(q < 1, (1 + (torch.pi * q).cos()) / 2, 0) * (1 if radial else 0.5)
    torch.testing.assert_close(
        kdata.apodize(4, radial=radial).data.expand_as(kdata.data), expected.expand_as(kdata.data).to(kdata.data.dtype)
    )
    gaussian = kdata.apodize(SpatialDimension(2.0, 2.0, 4.0), window='gaussian', radial=radial)
    torch.testing.assert_close(gaussian.data.expand_as(kdata.data)[:, 0, 0, 0, 0].real, torch.tensor([0.5, 2**-1.25]))


def test_apodize_constructor_and_invalid_width(consistently_shaped_kdata: KData) -> None:
    """Use normal constructor repeat reduction and reject nonpositive widths."""
    kdata = consistently_shaped_kdata
    kdata.data = torch.zeros_like(kdata.data)
    assert kdata.apodize().data.shape == (1,) * kdata.data.ndim
    with pytest.raises(ValueError, match='width'):
        kdata.apodize(0)
