"""Tests for algorithms to calculate the DCF with voronoi."""

import math

import numpy as np
import pytest
import torch
from einops import repeat
from mr2.algorithms.dcf import dcf_1d, dcf_2d3d_voronoi
from mr2.data import KTrajectory
from scipy.spatial import ConvexHull, HalfspaceIntersection


def example_traj_rad_2d(n_kr, n_ka, phi0=0.0, broadcast=True):
    """Create 2D radial trajectory with uniform angular gap."""
    krad = repeat(
        torch.linspace(-n_kr // 2, n_kr // 2 - 1, n_kr) / n_kr,
        'k0 -> other coils k2 k1 k0',
        other=1,
        coils=1,
        k2=1,
        k1=1,
    )
    kang = repeat(
        torch.linspace(0, n_ka - 1, n_ka) * (torch.pi / n_ka) + phi0,
        'k1 -> other coils k2 k1 k0',
        other=1,
        coils=1,
        k2=1,
        k0=1,
    )
    kz = torch.zeros(1, 1, 1, 1, 1)
    ky = torch.sin(kang) * krad
    kx = torch.cos(kang) * krad
    trajectory = KTrajectory(kz, ky, kx, repeat_detection_tolerance=1e-8 if broadcast else None)
    return trajectory


@pytest.mark.parametrize(
    ('n_kr', 'n_ka', 'phi0', 'broadcast'),
    [
        (20, 20, 0, True),
        (20, 1, 0, True),
        (20, 20, torch.pi / 4, True),
        (20, 1, torch.pi / 4, True),
        (20, 1, 0, False),
    ],
)
def test_dcf_rad_traj_voronoi(n_kr, n_ka, phi0, broadcast):
    """Compare voronoi-based dcf calculation for 2D radial trajectory to
    analytical solution."""
    # 2D radial trajectory
    traj = example_traj_rad_2d(n_kr, n_ka, phi0, broadcast)
    trajectory = traj.as_tensor()

    if n_ka > 1:  # only for for multiple spokes, analytical dcf is defined
        dcf = dcf_2d3d_voronoi(trajectory[1:3, 0, 0, ...])
        krad_idx = torch.linspace(-n_kr // 2, n_kr // 2 - 1, n_kr)
        dcf_analytical = torch.pi / n_ka * torch.abs(krad_idx) * (1 / n_kr) ** 2
        dcf_analytical[krad_idx == 0] = 2 * torch.pi / n_ka * 1 / 8 * (1 / n_kr) ** 2
        dcf_analytical = torch.repeat_interleave(repeat(dcf_analytical, 'k0->k2 k1 k0', k1=1, k2=1), n_ka, dim=-2)
        # Do not test outer points because they have to be approximated and cannot be calculated
        # accurately using voronoi
        torch.testing.assert_close(dcf_analytical[:, :, 1:-1], dcf[:, :, 1:-1])
    else:
        dcf = dcf_1d(trajectory[0, ...])
        dcf_ptp = dcf.max() - dcf.min()
        assert dcf_ptp / dcf.max() < 0.1, 'DCF for a single spoke should be constant-ish'
        assert dcf.sum() > 1e-3, 'DCF sum should not be zero'
        assert dcf.shape == traj.shape, 'DCF shape should match broadcasted trajectory shape'


@pytest.mark.parametrize(('n_k2', 'n_k1', 'n_k0'), [(4, 6, 8), (1, 2, 2)])
def test_dcf_3d_cart_traj_broadcast_voronoi(n_k2, n_k1, n_k0):
    """Compare voronoi-based dcf calculation for broadcasted 3D regular
    Cartesian trajectory to analytical solution which is 1 for each k-space
    point."""
    # 3D trajectory with points on Cartesian grid with step size of 1
    kx = repeat(
        torch.linspace(-n_k0 // 2, n_k0 // 2 - 1, n_k0), 'k0 -> other coils k2 k1 k0', other=1, coils=1, k2=1, k1=1
    )
    ky = repeat(
        torch.linspace(-n_k1 // 2, n_k1 // 2 - 1, n_k1), 'k1 -> other  coils k2 k1 k0', other=1, coils=1, k2=1, k0=1
    )
    kz = repeat(
        torch.linspace(-n_k2 // 2, n_k2 // 2 - 1, n_k2), 'k2 -> other  coils k2 k1 k0', other=1, coils=1, k1=1, k0=1
    )
    trajectory = KTrajectory(kx, ky, kz)

    # Analytical dcf
    dcf_analytical = torch.ones((1, 1, n_k2, n_k1, n_k0))
    # calculate dcf
    dcf = dcf_2d3d_voronoi(trajectory.as_tensor())
    # Do not test outer points because they have to be approximated and cannot be calculated
    # accurately using voronoi.
    torch.testing.assert_close(dcf[..., 1:-1, 1:-1, 1:-1], dcf_analytical[..., 1:-1, 1:-1, 1:-1])


@pytest.mark.parametrize(('n_k2', 'n_k1', 'n_k0'), [(4, 6, 8), (1, 2, 2)])
def test_dcf_3d_cart_full_traj_voronoi(n_k2, n_k1, n_k0):
    """Compare voronoi-based dcf calculation for full 3D regular Cartesian
    trajectory to analytical solution which is 1 for each k-space point."""
    # 3D trajectory with points on Cartesian grid with step size of 1
    ky, kz, kx = torch.meshgrid(
        torch.linspace(-n_k1 // 2, n_k1 // 2 - 1, n_k1),
        torch.linspace(-n_k2 // 2, n_k2 // 2 - 1, n_k2),
        torch.linspace(-n_k0 // 2, n_k0 // 2 - 1, n_k0),
        indexing='xy',
    )
    trajectory = KTrajectory(
        repeat(kz, '... -> other coils ...', other=1, coils=1),
        repeat(ky, '... -> other coils ...', other=1, coils=1),
        repeat(kx, '... -> other coils ...', other=1, coils=1),
        repeat_detection_tolerance=None,
    )
    # Analytical dcf
    dcf_analytical = torch.ones((1, 1, n_k2, n_k1, n_k0))
    dcf = dcf_2d3d_voronoi(trajectory.as_tensor())
    # Do not test outer points because they have to be approximated and cannot be calculated
    # accurately using voronoi
    torch.testing.assert_close(dcf[:, 1:-1, 1:-1, 1:-1], dcf_analytical[:, 1:-1, 1:-1, 1:-1])


@pytest.mark.parametrize(
    ('n_k2', 'n_k1', 'n_k0', 'k2_steps', 'k1_steps', 'k0_steps'),
    [(4, 6, 8, (1.0, 0.5, 0.25), (1.0, 0.5), (1.0,))],
)
def test_dcf_3d_cart_nonuniform_traj_voronoi(n_k2, n_k1, n_k0, k2_steps, k1_steps, k0_steps):
    """Compare voronoi-based dcf calculation for 3D nonuniform Cartesian
    trajectory to analytical solution which is 1 for each k-space point."""

    def k_range(n: int, *steps: float):
        """Create a tensor with n values, steps apart."""
        r = torch.tensor(steps).repeat(math.ceil(n / len(steps))).ravel()[:n]
        r = torch.cumsum(r, 0)
        r -= r.mean()
        return r

    k0_range = k_range(n_k0, *k0_steps)
    k1_range = k_range(n_k1, *k1_steps)
    k2_range = k_range(n_k2, *k2_steps)

    ky_full, kz_full, kx_full = torch.meshgrid(k1_range, k2_range, k0_range, indexing='xy')
    trajectory_full = KTrajectory(
        repeat(kz_full, '... -> other  coils ...', other=1, coils=1),
        repeat(ky_full, '... -> other  coils ...', other=1, coils=1),
        repeat(kx_full, '... -> other  coils ...', other=1, coils=1),
        repeat_detection_tolerance=None,
    )

    kx_broadcast = repeat(k0_range, 'k0 -> k2 k1 k0', k2=1, k1=1)
    ky_broadcast = repeat(k1_range, 'k1 -> k2 k1 k0', k2=1, k0=1)
    kz_broadcast = repeat(k2_range, 'k2 -> k2 k1 k0', k1=1, k0=1)
    trajectory_broadcast = KTrajectory(
        repeat(kz_broadcast, '... -> other  coils ...', other=1, coils=1),
        repeat(ky_broadcast, '... -> other  coils ...', other=1, coils=1),
        repeat(kx_broadcast, '... -> other  coils ...', other=1, coils=1),
        repeat_detection_tolerance=None,
    )

    # Sanity check inputs
    torch.testing.assert_close(trajectory_full.as_tensor(), trajectory_broadcast.as_tensor())
    assert trajectory_full.shape == (1, 1, len(k2_range), len(k1_range), len(k0_range))
    torch.testing.assert_close(kx_full[0, 0, :], k0_range)  # kx changes along k0
    torch.testing.assert_close(ky_full[0, :, 0], k1_range)  # ky changes along k1
    torch.testing.assert_close(kz_full[:, 0, 0], k2_range)  # kz changes along k2

    dcf_full = dcf_2d3d_voronoi(trajectory_full.as_tensor()[1:3, 0, ...])
    dcf_broadcast = dcf_2d3d_voronoi(trajectory_broadcast.as_tensor()[1:3, 0, ...])

    # Do not test outer points because they have to be approximated and cannot be calculated
    # accurately using voronoi
    torch.testing.assert_close(dcf_full[1:-1, 1:-1, 1:-1], dcf_broadcast[1:-1, 1:-1, 1:-1])


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('n_points', [1, 2, 40])
def test_dcf_1d_float64(device: str, n_points: int) -> None:
    """All 1D size branches preserve float64 and give the expected cell widths."""
    traj = torch.arange(n_points, dtype=torch.float64, device=device) * 0.25
    result = dcf_1d(traj)
    expected = torch.full_like(traj, 1 if n_points == 1 else 0.25)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('ndim', [2, 3])
def test_dcf_voronoi_float64(device: str, ndim: int) -> None:
    """Voronoi preserves float64 without changing the calculated cell volumes."""
    generator = torch.Generator().manual_seed(42)
    traj = torch.rand(ndim, 40, generator=generator, dtype=torch.float64).to(device)
    result = dcf_2d3d_voronoi(traj)
    assert result.dtype == traj.dtype
    assert result.device == traj.device
    assert result.shape == traj.shape[1:]
    assert (result.isfinite() & (result > 0)).all()
    torch.testing.assert_close(result.float(), dcf_2d3d_voronoi(traj.float()), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize('offset', [1e-6, 1e-12])
def test_dcf_3d_voronoi_near_duplicate(offset: float) -> None:
    """A second point very close to a grid point shares the cell of that grid point.

    At 1e-12 Qhull no longer separates the two points.
    """
    grid = torch.stack(torch.meshgrid(*[torch.arange(6.0, dtype=torch.float64)] * 3, indexing='ij')).reshape(3, -1)
    index = 3 * 36 + 3 * 6 + 2
    extra = grid[:, index : index + 1] + torch.tensor([[offset], [0.0], [0.0]], dtype=torch.float64)
    dcf = dcf_2d3d_voronoi(torch.cat((grid, extra), dim=1)[:, None, None, :]).ravel()
    torch.testing.assert_close(dcf[[index, -1]], torch.full((2,), 0.5, dtype=torch.float64), rtol=0, atol=1e-5)
    torch.testing.assert_close(dcf[index - 1], torch.tensor(1.0, dtype=torch.float64), rtol=0, atol=1e-5)


@pytest.mark.parametrize('ndim', [2, 3])
def test_dcf_voronoi_repeated_points(ndim: int) -> None:
    """Repeated and nearly coincident samples preserve the total weight of their cell."""
    grid = torch.stack(torch.meshgrid(*[torch.arange(6.0, dtype=torch.float64)] * ndim, indexing='ij')).reshape(
        ndim, -1
    )
    index = int((grid == 3).all(dim=0).nonzero().item())
    extra = grid[:, index : index + 1].repeat(1, 3)
    if ndim == 3:
        extra[0, -1] += 1e-12
    reference = dcf_2d3d_voronoi(grid)
    dcf = dcf_2d3d_voronoi(torch.cat((grid, extra), dim=1)[:, None, None, :]).ravel()
    repeated = torch.tensor([index, grid.shape[1], grid.shape[1] + 1, grid.shape[1] + 2])
    torch.testing.assert_close(dcf[repeated], reference[index].expand(4) / 4)
    keep = torch.arange(grid.shape[1]) != index
    torch.testing.assert_close(dcf[: grid.shape[1]][keep], reference[keep])


def test_dcf_3d_voronoi_stack_of_stars() -> None:
    """Cospherical stack-of-stars cells agree with their defining bisector halfspaces."""
    angle = np.pi * np.arange(32) / 32
    radius = np.arange(-8, 8) + 0.5
    xy = (np.stack((np.cos(angle), np.sin(angle)), axis=1)[:, None] * radius[:, None]).reshape(-1, 2)
    points = np.concatenate([np.column_stack((xy, np.full(len(xy), z))) for z in range(-3, 3)])
    points = np.round(points, 9)
    points = points[np.random.default_rng(0).permutation(len(points))]
    dcf = dcf_2d3d_voronoi(torch.from_numpy(points.T)).numpy()
    inner = np.flatnonzero((np.linalg.norm(points[:, :2], axis=1) < 0.6) & (points[:, 2] == 1))
    for index in inner:
        offset = np.delete(points - points[index], index, axis=0)
        distance = np.linalg.norm(offset, axis=1)
        # Relative to the sample, a Voronoi cell is the intersection of the neighbor bisector halfspaces.
        halfspaces = np.column_stack((offset / distance[:, None], -distance / 2))
        vertices = HalfspaceIntersection(halfspaces, np.zeros(3)).intersections
        np.testing.assert_allclose(dcf[index], ConvexHull(vertices).volume, rtol=1e-6)


def test_dcf_3d_voronoi_near_point_chain() -> None:
    """A chain of close points preserves the total volume, including cells shared by Qhull."""
    grid = torch.stack(torch.meshgrid(*[torch.arange(6.0, dtype=torch.float64)] * 3, indexing='ij')).reshape(3, -1)
    index = 3 * 36 + 3 * 6 + 2
    extra = grid[:, index : index + 1].repeat(1, 20)
    extra[0] += torch.arange(1, 21, dtype=torch.float64) * 3e-7
    dcf = dcf_2d3d_voronoi(torch.cat((grid, extra), dim=1))
    total = dcf[index] + dcf[-20:].sum()
    torch.testing.assert_close(total, torch.tensor(1.0, dtype=total.dtype), rtol=1e-4, atol=0)
