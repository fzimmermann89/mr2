"""Tests for RING gradient delay estimation."""

import pytest
import torch
from einops import einsum
from mr2.algorithms import Ring
from mr2.data import KData, KHeader, KTrajectory, SpatialDimension
from mr2.data.traj_calculators import KTrajectoryRadial2D
from mr2.phantoms import EllipsePhantom
from mr2.phantoms.phantom_elements import EllipseParameters
from mr2.utils import RandomGenerator

DELAY = torch.tensor([[0.3, 0.05], [0.05, -0.2]])


def simulate_kdata(n_spokes: int, k0_center: float = 128.0, reverse_odd: bool = False) -> KData:
    """Simulate noisy multi-coil radial data acquired with gradient delays, but with the nominal trajectory."""
    k1_idx = torch.arange(n_spokes)[None, None, None, :, None]
    nominal = KTrajectoryRadial2D()(n_k0=256, k0_center=torch.full((1, 1, 1, 1, 1), k0_center), k1_idx=k1_idx)
    ky, kx = nominal.ky / 2, nominal.kx / 2  # readout oversampling
    if reverse_odd:
        ky, kx = torch.where(k1_idx % 2 == 1, ky.flip(-1), ky), torch.where(k1_idx % 2 == 1, kx.flip(-1), kx)
    angle = k1_idx * KTrajectoryRadial2D().angle + torch.pi * (reverse_odd * k1_idx % 2)
    direction = torch.stack((angle.sin(), angle.cos()))
    acquired_ky, acquired_kx = torch.stack((ky, kx)) + einsum(DELAY, direction, 'shift yx, yx ... -> shift ...')
    data = torch.cat(
        [
            EllipsePhantom(
                [
                    EllipseParameters(center_x=x, center_y=y, radius_x=0.15, radius_y=0.1, intensity=1.0),
                    EllipseParameters(center_x=0.0, center_y=0.0, radius_x=0.3, radius_y=0.35, intensity=0.3),
                ]
            ).kspace(acquired_ky, acquired_kx)
            for x, y in ((0.2, 0.2), (-0.2, 0.2), (0.2, -0.2), (-0.2, -0.2))
        ],
        dim=1,
    )
    data = data + 0.01 * data.abs().max() * RandomGenerator(0).complex64_tensor(data.shape)
    fov, matrix = SpatialDimension(0.01, 1.0, 1.0), SpatialDimension(1, 128, 128)
    header = KHeader(recon_fov=fov, encoding_fov=fov, encoding_matrix=matrix, recon_matrix=matrix)
    return KData(header, data, KTrajectory(nominal.kz, ky, kx))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_ring(device: str) -> None:
    """RING recovers the simulated gradient delays and no delay remains after correction."""
    kdata = simulate_kdata(n_spokes=32).to(device)
    ring = Ring(kdata[..., ::2, :])
    torch.testing.assert_close(ring.delay.cpu(), DELAY, atol=0.02, rtol=0)
    torch.testing.assert_close(Ring(ring(kdata)).delay.cpu(), torch.zeros(2, 2), atol=0.02, rtol=0)
    assert ring(kdata.traj) == ring(kdata).traj


def test_ring_few_reversed_asymmetric_spokes() -> None:
    """RING works with few spokes, multiple intersections, reversed readouts, and an off-center k-space center."""
    kdata = simulate_kdata(n_spokes=6, k0_center=100.3, reverse_odd=True)
    torch.testing.assert_close(Ring(kdata, n_intersections=3).delay, DELAY, atol=0.03, rtol=0)


def test_ring_invalid() -> None:
    """Too small search regions and unsupported trajectories are reported."""
    kdata = simulate_kdata(n_spokes=8)
    with pytest.warns(UserWarning, match='search_radius'):
        Ring(kdata, search_radius=0.1)
    with pytest.raises(ValueError, match='not covered'):
        Ring(kdata, search_radius=100.0)
    with pytest.raises(ValueError, match='three different directions'):
        Ring(kdata[..., :2, :])
    curved = KTrajectory(kdata.traj.kz, kdata.traj.ky + 1e-3 * kdata.traj.kx**2, kdata.traj.kx)
    with pytest.raises(ValueError, match='straight'):
        Ring(KData(kdata.header, kdata.data, curved))
    tilted = KTrajectory(kdata.traj.kx, kdata.traj.ky, kdata.traj.kx)
    with pytest.raises(ValueError, match='same kx-ky plane'):
        Ring(KData(kdata.header, kdata.data, tilted))
