"""Tests for DcfData class."""

import pytest
import torch
from einops import repeat
from mr2.data import DcfData, KTrajectory, SpatialDimension
from mr2.utils import RandomGenerator


def example_traj_rpe(n_ka: int, n_kr: int, n_k0: int, broadcast: bool = True):
    """Create RPE trajectory with uniform angular gap."""
    krad = repeat(
        torch.linspace(-n_kr // 2, n_kr // 2 - 1, n_kr) / n_kr,
        'k1 -> other coils k2 k1 k0',
        other=1,
        coils=1,
        k2=1,
        k0=1,
    )
    kang = repeat(
        torch.linspace(0, n_ka - 1, n_ka) * (torch.pi / n_ka),
        'k2 -> other coils k2 k1 k0',
        other=1,
        coils=1,
        k1=1,
        k0=1,
    )
    kz = torch.sin(kang) * krad
    ky = torch.cos(kang) * krad
    kx = repeat(
        torch.linspace(-n_k0 // 2, n_k0 // 2 - 1, n_k0) / n_k0,
        'k0 -> other coils k2 k1 k0',
        other=1,
        coils=1,
        k2=1,
        k1=1,
    )
    trajectory = KTrajectory(kz, ky, kx, repeat_detection_tolerance=1e-8 if broadcast else None)
    return trajectory


def example_traj_spiral_2d(n_kr: int, n_ki: int, n_ka: int, broadcast: bool = True) -> KTrajectory:
    """Create 2D spiral trajectory with n_kr points along each spiral arm, n_ki
    turns per spiral arm and n_ka spiral arms."""
    ang = repeat(
        torch.linspace(0, 2 * torch.pi * n_ki, n_kr), 'k0 -> other coils k2 k1 k0', other=1, coils=1, k2=1, k1=1
    )
    start_ang = repeat(
        torch.linspace(0, 2 * torch.pi * (1 - 1 / n_ka), n_ka),
        'k1 -> other  coils  k2 k1 k0',
        other=1,
        coils=1,
        k2=1,
        k0=1,
    )
    kz = torch.zeros(1, 1, 1, 1, 1)
    kx = ang * torch.cos(ang + start_ang)
    ky = ang * torch.sin(ang + start_ang)
    trajectory = KTrajectory(kz, ky, kx, repeat_detection_tolerance=1e-8 if broadcast else None)
    return trajectory


@pytest.mark.parametrize(
    ('n_ka', 'n_kr', 'n_k0', 'expected_shape'),
    [
        (6, 10, 20, (1, 1, 1, 10, 1)),  # no k2 or k0, dcf are the same and reduced
        (1, 10, 20, (1, 1, 1, 1, 1)),  # without angular dimension all dcf are the same
        (6, 10, 1, (1, 1, 1, 10, 1)),
    ],
)
def test_dcf_rpe_traj_voronoi(n_ka: int, n_kr: int, n_k0: int, expected_shape: tuple[int, ...]) -> None:
    """Voronoi-based dcf calculation for RPE trajectory."""
    trajectory = example_traj_rpe(n_ka, n_kr, n_k0)
    dcf = DcfData.from_traj_voronoi(trajectory)
    assert dcf.data.shape == expected_shape


@pytest.mark.parametrize(('n_kr', 'n_ki', 'n_ka'), [(10, 2, 1)])
def test_dcf_spiral_traj_voronoi(n_kr: int, n_ki: int, n_ka: int) -> None:
    """Voronoi-based dcf calculation for spiral trajectory."""
    # nkr points along each spiral arm, nki turns per spiral arm, nka spiral arms
    trajectory = example_traj_spiral_2d(n_kr, n_ki, n_ka)
    dcf = DcfData.from_traj_voronoi(trajectory)
    assert dcf.data.shape == trajectory.shape


def test_dcf_spiral_traj_voronoi_singlespiral() -> None:
    """For three z-stacked spirals in the x,y plane, the center spiral should
    be the same as a single 2D spiral.
    """
    n_kr = 30  # points along each spiral arm
    n_ki = 5  # turns per spiral arm
    trajectory_single = example_traj_spiral_2d(n_kr, n_ki, 1)

    # A new trajectroy with three spirals stacked in z direction.
    three_spirals: torch.Tensor = trajectory_single.as_tensor().repeat_interleave(repeats=3, dim=-2)
    three_spirals[0, :, :, :, 0] = -1  # z of first spiral
    three_spirals[0, :, :, :, 1] = 0  # z of second spiral
    three_spirals[0, :, :, :, 2] = 1  # z of third spiral
    trajectory_three = KTrajectory.from_tensor(three_spirals)

    dcf_single = DcfData.from_traj_voronoi(trajectory_single)
    dcf_three = DcfData.from_traj_voronoi(trajectory_three)

    ignore_last = int(n_kr / n_ki)  # ignore the outer points of the spirals
    torch.testing.assert_close(dcf_three.data[..., :-ignore_last], dcf_single.data[..., :-ignore_last])


@pytest.mark.cuda
@pytest.mark.parametrize(('n_ka', 'n_kr', 'n_k0'), [(10, 6, 20)])
def test_dcf_rpe_traj_voronoi_cuda(n_ka: int, n_kr: int, n_k0: int) -> None:
    """Voronoi-based dcf calculation for RPE trajectory in CUDA memory."""
    trajectory = example_traj_rpe(n_ka, n_kr, n_k0)
    dcf = DcfData.from_traj_voronoi(trajectory.cuda())
    assert dcf.data.is_cuda


def test_dcf_broadcast() -> None:
    """Test broadcasting within voronoi dcf calculation."""
    rng = RandomGenerator(0)
    # kx and ky force voronoi calculation and need to be broadcasted
    kx = rng.float32_tensor((1, 1, 1, 4, 4))
    ky = rng.float32_tensor((1, 1, 4, 1, 4))
    kz = torch.zeros(1, 1, 1, 1, 1)
    trajectory = KTrajectory(kz, ky, kx)
    dcf = DcfData.from_traj_voronoi(trajectory)
    assert dcf.data.shape == trajectory.shape


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_dcf_pipe_mixed_batch(device: str) -> None:
    """Mixed radial/Cartesian factors work with dense, broadcast and batched geometry."""
    trajectory = example_traj_rpe(12, 32, 8)
    # Scale radial coordinates to the encoded matrix, and use a full Cartesian readout.
    trajectory = KTrajectory(trajectory.kz * 32, trajectory.ky * 32, trajectory.kx * 8).to(device)
    sizes = SpatialDimension(32, 32, 8)
    reference = DcfData.from_traj_pipe(trajectory, sizes)
    radial = DcfData.from_traj_pipe(
        KTrajectory(trajectory.kz, trajectory.ky, torch.zeros_like(trajectory.kx[..., :1])), sizes
    )
    torch.testing.assert_close(reference.data, radial.data)
    # Change the second batch geometry, retaining broadcast singleton coil/k0 axes.
    batch = KTrajectory(
        trajectory.kz.expand(2, -1, -1, -1, -1), torch.cat((trajectory.ky, trajectory.ky * 0.7)), trajectory.kx
    )
    batched = DcfData.from_traj_pipe(batch, sizes)
    torch.testing.assert_close(batched.data[0:1], reference.data)
    second = KTrajectory(batch.kz, batch.ky[1:2], batch.kx)
    torch.testing.assert_close(batched.data[1:2], DcfData.from_traj_pipe(second, sizes).data)
    dense = KTrajectory.from_tensor(batch.as_tensor(), repeat_detection_tolerance=None)
    torch.testing.assert_close(DcfData.from_traj_pipe(dense, sizes).data, batched.data)
    data = torch.ones(2, 4, 12, 32, 8, dtype=torch.complex64, device=device)
    (weighted,) = batched.as_operator()(data)
    torch.testing.assert_close(weighted, batched.data.expand_as(data).to(data.dtype))


def test_dcf_pipe_singleton() -> None:
    """Constant batch geometries give unit weights without a gridding plan."""
    k = torch.zeros(2, 1, 1, 1, 1)
    result = DcfData.from_traj_pipe(KTrajectory(k, k, k), SpatialDimension(1, 1, 1))
    torch.testing.assert_close(result.data.expand_as(k), torch.ones_like(k))


def test_dcf_pipe_coupled_3d() -> None:
    """Three coupled coordinate directions form one group, including across leading batch axes."""
    from mr2.algorithms.dcf import dcf_pipe

    kz = torch.linspace(-4, 3, 8).reshape(1, 1, 1, 8, 1, 1)
    kx = torch.linspace(-4, 3, 8).reshape(1, 1, 1, 1, 1, 8)
    ky = torch.stack((kz + kx, kz - kx), dim=0).squeeze(1)
    traj = KTrajectory(kz, ky, kx)
    result = DcfData.from_traj_pipe(traj, SpatialDimension(8, 16, 8), n_iterations=5)
    coordinates = traj.as_tensor().expand(3, *traj.shape)
    for i in range(2):
        expected = dcf_pipe(coordinates[:, i, 0, 0], (8, 16, 8), n_iterations=5)
        torch.testing.assert_close(result.data[i, 0, 0].expand_as(expected), expected)


def test_dcf_pipe_coil_axis() -> None:
    """Coils must broadcast over one geometry rather than create independent DCFs."""
    k = torch.arange(8.0).reshape(1, 2, 1, 1, 4)
    with pytest.raises(ValueError, match='coil'):
        DcfData.from_traj_pipe(KTrajectory(k, k, k), SpatialDimension(8, 8, 8))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_dcf_pipe_batch_constant_directions(device: str) -> None:
    """Unrelated batch entries must not change active directions or normalization."""
    kx = torch.arange(-4, 4, dtype=torch.float32, device=device).reshape(1, 1, 1, 1, 8)
    ky = torch.cat((torch.zeros_like(kx), kx))
    kz = torch.zeros_like(kx)
    sizes = SpatialDimension(1, 8, 8)
    batched = DcfData.from_traj_pipe(KTrajectory(kz, ky, kx), sizes).data
    for index in range(2):
        single = DcfData.from_traj_pipe(KTrajectory(kz, ky[index : index + 1], kx), sizes).data
        torch.testing.assert_close(batched[index : index + 1], single)


@pytest.mark.parametrize('varying', [False, True])
@pytest.mark.parametrize('invalid', ['nan_coordinate', 'iterations', 'oversampling', 'encoding'])
def test_dcf_pipe_constant_input_validation(varying: bool, invalid: str) -> None:
    """Constant directions and singleton geometries must not bypass input validation."""
    zero = torch.zeros(1, 1, 1, 1, 1)
    kz = torch.full_like(zero, float('nan')) if invalid == 'nan_coordinate' else zero
    kx = torch.arange(8.0).reshape(1, 1, 1, 1, 8) if varying else zero
    traj = KTrajectory(kz, zero, kx, repeat_detection_tolerance=None)
    with pytest.raises(ValueError):
        DcfData.from_traj_pipe(
            traj,
            SpatialDimension(0 if invalid == 'encoding' else 1, 1, 8),
            n_iterations=0 if invalid == 'iterations' else 10,
            oversampling=0.5 if invalid == 'oversampling' else 2.0,
        )
