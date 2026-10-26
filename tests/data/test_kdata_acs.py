"""Tests for Cartesian and non-Cartesian calibration preparation."""

from unittest.mock import patch

import pytest
import torch
from mr2.data import AcqInfo, CsmData, DcfData, KData, KHeader, KTrajectory, SpatialDimension
from mr2.data.traj_calculators import KTrajectoryCartesian
from mr2.operators import FastFourierOp, FourierOp


def make_kdata(header: KHeader, shape: SpatialDimension[int], traj: KTrajectory, data: torch.Tensor) -> KData:
    """Build an acquisition with consistent encoding geometry."""
    header = header.clone()
    header.acq_info = AcqInfo()
    header.trajectory = None
    header.encoding_matrix = shape
    header.recon_matrix = shape
    header.encoding_fov = SpatialDimension(*[float(n) for n in shape.zyx])
    header.recon_fov = header.encoding_fov
    return KData(header, data, traj)


def radial_trajectory(n: int, spokes: int) -> KTrajectory:
    """Create radial spokes in encoding-grid units."""
    angle = torch.arange(spokes) * torch.pi / spokes
    radius = torch.arange(n) - n // 2
    return KTrajectory(
        torch.zeros(1, 1, 1, 1, 1),
        (angle.sin()[:, None] * radius).reshape(1, 1, 1, spokes, n),
        (angle.cos()[:, None] * radius).reshape(1, 1, 1, spokes, n),
    )


@pytest.mark.parametrize('shape', [SpatialDimension(1, 9, 12), SpatialDimension(7, 8, 9)])
def test_acs_cartesian_sizes(random_kheader: KHeader, shape: SpatialDimension[int]) -> None:
    """Scalar widths produce squares/cubes and explicit widths preserve zero frequency."""
    data = torch.arange(2 * 3 * shape.z * shape.y * shape.x).reshape(2, 3, *shape.zyx).to(torch.complex64)
    kdata = make_kdata(random_kheader, shape, KTrajectoryCartesian.fullysampled(shape), data)
    for width in (4, 5):
        actual = kdata.acs(width)
        target = tuple(width if n > 1 else 1 for n in shape.zyx)
        slices = tuple(slice(n // 2 - m // 2, n // 2 - m // 2 + m) for n, m in zip(shape.zyx, target, strict=True))
        torch.testing.assert_close(actual, data[..., slices[0], slices[1], slices[2]])
    width = min(n for n in shape.zyx if n > 1)
    assert kdata.acs().shape == (2, 3, *(width if n > 1 else 1 for n in shape.zyx))
    rectangle = SpatialDimension(1, 4, 6) if shape.z == 1 else SpatialDimension(3, 4, 5)
    assert kdata.acs(rectangle).shape == (2, 3, *rectangle.zyx)


def test_acs_cartesian_missing_outer_samples(random_kheader: KHeader) -> None:
    """Complete central ACS is usable in an otherwise undersampled acquisition."""
    shape = SpatialDimension(1, 16, 16)
    traj = KTrajectoryCartesian.fullysampled(shape)
    keep = torch.tensor([0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 14])
    data = torch.zeros(2, 2, 1, len(keep), 16, dtype=torch.complex64)
    kdata = make_kdata(random_kheader, shape, KTrajectory(traj.kz, traj.ky[..., keep, :], traj.kx), data)
    assert kdata.acs().shape[-3:] == (1, 8, 8)
    assert torch.count_nonzero(kdata.acs()) == 0
    with pytest.raises(ValueError, match='covered centered ACS'):
        kdata.acs(9)


def test_acs_cartesian_batches_and_reversal(random_kheader: KHeader) -> None:
    """Choose a common region across batches and sort reversed readouts."""
    shape = SpatialDimension(1, 12, 12)
    traj = KTrajectoryCartesian.fullysampled(shape)
    ky = traj.ky.expand(2, 1, 1, 12, 1).clone()
    ky[1, ..., 1, :] = 2  # duplicate inside size 6, outside size 4
    data = torch.randn(2, 2, 1, 12, 12, dtype=torch.complex64)
    kx = traj.kx.flip(-1)
    kdata = make_kdata(random_kheader, shape, KTrajectory(traj.kz, ky, kx), data.flip(-1))
    torch.testing.assert_close(kdata.acs(), data[..., 4:8, 4:8])
    with pytest.raises(ValueError, match='covered centered ACS'):
        kdata.acs(6)


@pytest.mark.parametrize('size', [0, -1, True, 20, SpatialDimension(2, 4, 4)])
def test_acs_invalid_size(random_kheader: KHeader, size: int | SpatialDimension[int]) -> None:
    """Reject invalid widths and widths exceeding inactive dimensions."""
    shape = SpatialDimension(1, 8, 8)
    kdata = make_kdata(random_kheader, shape, KTrajectoryCartesian.fullysampled(shape), torch.zeros(1, 1, 1, 8, 8))
    with pytest.raises(ValueError, match='ACS size'):
        kdata.acs(size)


def test_acs_noncartesian_reuses_density(random_kheader: KHeader) -> None:
    """Infer radial ACS and reuse a single density calculation for the adjoint."""
    shape = SpatialDimension(1, 24, 24)
    traj = radial_trajectory(24, 32)
    kdata = make_kdata(random_kheader, shape, traj, torch.randn(1, 2, 1, 32, 24, dtype=torch.complex64))
    compute = DcfData.from_traj_voronoi
    with patch.object(DcfData, 'from_traj_voronoi', wraps=compute) as density:
        actual = kdata.acs()
    density.assert_called_once()
    assert actual.shape[-2] == actual.shape[-1]
    assert 4 <= actual.shape[-1] < 24
    assert torch.isfinite(actual).all()
    dcf = compute(traj)
    fourier = FourierOp.from_kdata(kdata, recon_shape=shape)
    grid = FastFourierOp(dim=(-3, -2, -1))(*fourier.H(*dcf.as_operator()(kdata.data)))[0]
    width = actual.shape[-1]
    start = 12 - width // 2
    torch.testing.assert_close(actual, grid[..., start : start + width, start : start + width])
    torch.testing.assert_close(actual, kdata.acs(width))
    rectangle = kdata.acs(SpatialDimension(1, 4, 6))
    torch.testing.assert_close(rectangle, grid[..., 10:14, 9:15])
    torch.testing.assert_close(kdata.acs(24), grid)  # explicit sizes are trusted overrides


def test_acs_repeated_spokes_do_not_improve_coverage(random_kheader: KHeader) -> None:
    """Repeating samples changes weights but not geometric coverage or inferred size."""
    shape = SpatialDimension(1, 24, 24)
    traj = radial_trajectory(24, 12)
    data = torch.randn(1, 2, 1, 12, 24, dtype=torch.complex64)
    single = make_kdata(random_kheader, shape, traj, data)
    repeated = make_kdata(
        random_kheader,
        shape,
        KTrajectory(traj.kz, traj.ky.repeat_interleave(3, -2), traj.kx.repeat_interleave(3, -2)),
        data.repeat_interleave(3, -2),
    )
    torch.testing.assert_close(single.acs(), repeated.acs(), atol=1e-5, rtol=1e-5)


def test_acs_occupancy_limits_size(random_kheader: KHeader) -> None:
    """More distinct radial spokes support a larger central square."""
    shape = SpatialDimension(1, 24, 24)
    widths = []
    for spokes in (8, 32):
        traj = radial_trajectory(24, spokes)
        kdata = make_kdata(random_kheader, shape, traj, torch.ones(1, 1, 1, spokes, 24, dtype=torch.complex64))
        widths.append(kdata.acs().shape[-1])
    assert widths[0] < widths[1]


def test_acs_mixed_cartesian_noncartesian(random_kheader: KHeader) -> None:
    """A Cartesian slice axis and radial plane share the existing Fourier/DCF path."""
    shape = SpatialDimension(3, 16, 16)
    radial = radial_trajectory(16, 24)
    traj = KTrajectory(torch.arange(-1, 2).reshape(1, 1, 3, 1, 1), radial.ky, radial.kx)
    kdata = make_kdata(random_kheader, shape, traj, torch.randn(1, 2, 3, 24, 16, dtype=torch.complex64))
    actual = kdata.acs()
    assert actual.shape == (1, 2, 3, 3, 3)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize('coupled', [False, True])
def test_acs_noncartesian_3d(random_kheader: KHeader, coupled: bool) -> None:
    """Rounded occupancy selects a cube from an oversampled noninteger 3D lattice."""
    shape = SpatialDimension(5, 5, 5)
    axis = torch.arange(-3, 4) * 0.6
    traj = KTrajectory(axis.reshape(1, 1, 7, 1, 1), axis.reshape(1, 1, 1, 7, 1), axis.reshape(1, 1, 1, 1, 7))
    if coupled:
        generator = torch.Generator().manual_seed(0)
        xyz = traj.as_tensor().expand(3, 1, 1, 7, 7, 7).clone()
        xyz += 0.02 * torch.randn(xyz.shape, generator=generator)
        traj = KTrajectory(*xyz)
    kdata = make_kdata(random_kheader, shape, traj, torch.randn(1, 2, 7, 7, 7, dtype=torch.complex64))
    actual = kdata.acs()
    assert actual.shape == (1, 2, 5, 5, 5)
    assert torch.isfinite(actual).all()


def test_radial_acs_espirit_preserves_relative_coil_phase(random_kheader: KHeader) -> None:
    """The complete radial preparation and ESPIRiT path recovers complex coil ratios."""
    shape = SpatialDimension(1, 24, 24)
    traj = radial_trajectory(24, 32)
    reference = torch.tensor([1, 1j, 0.5 - 0.3j], dtype=torch.complex64)
    reference /= reference.norm()
    y, x = torch.meshgrid(torch.linspace(-1, 1, 24), torch.linspace(-1, 1, 24), indexing='ij')
    image = torch.exp(-6 * (x.square() + y.square()))
    coil_images = reference[None, :, None, None, None] * image
    fourier = FourierOp(recon_matrix=shape, encoding_matrix=shape, traj=traj)
    kdata = make_kdata(random_kheader, shape, traj, fourier(coil_images)[0])
    maps = CsmData.from_kdata_espirit(kdata).data.expand(1, 3, 1, 24, 24)[0, :, 0]
    roi = x.square() + y.square() < 0.25
    torch.testing.assert_close(maps[:, roi].norm(dim=0), torch.ones_like(x[roi]))
    overlap = (maps[:, roi].conj() * reference[:, None]).sum(dim=0).abs()
    torch.testing.assert_close(overlap, torch.ones_like(overlap))


def test_acs_noncartesian_common_batch_size(random_kheader: KHeader) -> None:
    """Non-Cartesian batches use the smallest shared coverage without averaging coils."""
    shape = SpatialDimension(1, 24, 24)
    sparse = radial_trajectory(24, 8)
    dense = radial_trajectory(24, 32)
    traj = KTrajectory(
        dense.kz,
        torch.cat([dense.ky, sparse.ky.repeat_interleave(4, -2)]),
        torch.cat([dense.kx, sparse.kx.repeat_interleave(4, -2)]),
    )
    data = torch.randn(2, 2, 1, 32, 24, dtype=torch.complex64)
    acquisition = make_kdata(random_kheader, shape, traj, data)
    combined = acquisition.acs()
    sparse_only = make_kdata(random_kheader, shape, sparse, data[1:2, ..., ::4, :]).acs()
    assert combined.shape == (2, 2, *sparse_only.shape[-3:])
    width = combined.shape[-1]
    for i in range(2):
        single = acquisition[i : i + 1].acs(width)
        torch.testing.assert_close(combined[i : i + 1], single)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('encoded', [(1, 25, 32), (1, 24, 33)])
def test_noncartesian_coverage_window(encoded: tuple[int, int, int], device: str) -> None:
    """A parity-safe window matches the full occupancy check, including repeated batches."""
    import importlib

    module = importlib.import_module('mr2.algorithms.prepare_acs')
    traj = radial_trajectory(32, 32)
    coords = traj.as_tensor().expand(3, 1, 1, 1, 32, 32).reshape(3, 1, -1).to(device)
    full = module._noncartesian_coverage(coords, encoded)
    window = (1, 7, 8)
    cropped = module._noncartesian_coverage(coords.expand(-1, 3, -1), window)
    torch.testing.assert_close(cropped, module._center_crop(full, window).expand(3, *window))


def test_acs_noncartesian_explicit_size_skips_estimate(random_kheader: KHeader) -> None:
    """Explicit sizes bypass occupancy and a shared trajectory broadcasts over data batches."""
    import importlib

    module = importlib.import_module('mr2.algorithms.prepare_acs')
    shape = SpatialDimension(1, 24, 32)
    traj = radial_trajectory(24, 32)
    kdata = make_kdata(random_kheader, shape, traj, torch.randn(4, 2, 1, 32, 24, dtype=torch.complex64))
    with patch.object(module, '_noncartesian_coverage', wraps=module._noncartesian_coverage) as coverage:
        actual = kdata.acs(SpatialDimension(1, 4, 6))
    assert actual.shape == (4, 2, 1, 4, 6)
    coverage.assert_not_called()


@pytest.mark.parametrize('mixed', [False, True])
def test_espirit_crop_matches_fourier_origin(random_kheader: KHeader, mixed: bool) -> None:
    """Odd-to-even cropping follows NUFFT origins and retains FFT-axis cropping."""
    shape = SpatialDimension(3 if mixed else 1, 33, 33)
    radial = radial_trajectory(33, 32)
    kz = torch.arange(-1, 2).reshape(1, 1, 3, 1, 1) if mixed else radial.kz
    trajectory = KTrajectory(kz, radial.ky, radial.kx)
    generator = torch.Generator().manual_seed(4)
    data = torch.randn(1, 2, shape.z, 32, 33, dtype=torch.complex64, generator=generator)
    kdata = make_kdata(random_kheader, shape, trajectory, data)
    full_maps = CsmData.from_kdata_espirit(kdata, kernel_width=2).data
    full_image = FourierOp.from_kdata(kdata).H(data)[0]
    kdata.header.recon_matrix = SpatialDimension(2 if mixed else 1, 32, 32)
    kdata.header.recon_fov = SpatialDimension(2.0 if mixed else 1.0, 32.0, 32.0)
    actual = CsmData.from_kdata_espirit(kdata, kernel_width=2).data
    # NUFFT y/x retain index zero; the mixed FFT z axis drops its first voxel.
    expected = full_maps[..., 1:3, :32, :32] if mixed else full_maps[..., :32, :32]
    torch.testing.assert_close(actual, expected)
    fourier = FourierOp.from_kdata(kdata)
    cropped_image = fourier.H(data)[0]
    torch.testing.assert_close(fourier.crop_to_recon(full_image), cropped_image, atol=2e-5, rtol=2e-5)
    expected_image = full_image[..., 1:3, :32, :32] if mixed else full_image[..., :32, :32]
    torch.testing.assert_close(cropped_image, expected_image, atol=2e-5, rtol=2e-5)
