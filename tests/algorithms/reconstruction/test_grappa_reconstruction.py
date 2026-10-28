"""Tests for trajectory-driven Cartesian GRAPPA reconstruction."""

from itertools import product

import pytest
import torch
from mr2.algorithms.reconstruction import GrappaReconstruction
from mr2.data import AcqInfo, CsmData, IData, KData, KHeader, KNoise, KTrajectory, SpatialDimension
from mr2.operators import FastFourierOp, FourierOp
from mr2.phantoms.coils import birdcage_2d
from tests.phantoms import EllipsePhantomTestData
from torch.nn.functional import conv3d


def _phantom_kspace(phantom: EllipsePhantomTestData, shape: tuple[int, int, int], dtype: torch.dtype) -> torch.Tensor:
    """Sample the existing ellipse phantom with birdcage coil sensitivities."""
    dimensions = SpatialDimension(*shape)
    image = phantom.phantom.image_space(dimensions).to(dtype)
    image = image * torch.linspace(0.5, 1.0, shape[0]).reshape(1, 1, -1, 1, 1)
    (kspace,) = FastFourierOp()(image * birdcage_2d(8, dimensions).to(dtype))
    return kspace[0]


def _kdata(header: KHeader, full: torch.Tensor, acquired: torch.Tensor) -> KData:
    """Pack acquired Cartesian samples using a copy of the existing header fixture."""
    shape = full.shape[-3:]
    coordinates = torch.meshgrid(*(torch.arange(n, device=full.device) - n // 2 for n in shape), indexing='ij')
    kz, ky, kx = (coordinate[acquired].reshape(1, 1, 1, 1, -1) for coordinate in coordinates)
    trajectory = KTrajectory(kz, ky, kx)
    header = header.clone()
    header.recon_matrix = SpatialDimension(*shape)
    header.encoding_matrix = SpatialDimension(*shape)
    header.recon_fov = SpatialDimension(1.0, 1.0, 1.0)
    header.encoding_fov = SpatialDimension(1.0, 1.0, 1.0)
    header.acq_info = AcqInfo()
    header.trajectory = None
    return KData(data=full[..., acquired].unsqueeze(-2).unsqueeze(-2), header=header, traj=trajectory)


@pytest.fixture
def grappa_data(
    ellipse_phantom: EllipsePhantomTestData, random_kheader: KHeader
) -> tuple[torch.Tensor, torch.Tensor, KData]:
    """Ellipse acquisition with alternate phase-encoding lines missing."""
    full = _phantom_kspace(ellipse_phantom, (1, 16, 18), torch.complex128)[None]
    acquired = torch.ones(full.shape[-3:], dtype=torch.bool)
    acquired[..., 1::2, :] = False
    return full, acquired, _kdata(random_kheader, full, acquired)


@pytest.mark.parametrize(
    ('shape', 'kernel', 'acceleration'),
    [((1, 24, 20), (1, 3, 3), 2), ((1, 24, 20), (1, 5, 3), 3), ((9, 16, 12), (3, 3, 3), 2)],
)
@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_grappa_reconstruction(
    shape: tuple[int, int, int],
    kernel: tuple[int, int, int],
    acceleration: int,
    dtype: torch.dtype,
    ellipse_phantom: EllipsePhantomTestData,
    random_kheader: KHeader,
) -> None:
    """Reconstruct the ellipse phantom using cropped ACS and trajectory-derived sampling."""
    full = _phantom_kspace(ellipse_phantom, shape, dtype)[None]
    acquired = torch.zeros(shape, dtype=torch.bool)
    acquired[..., ::acceleration, :] = True
    acquired[..., 6:10, :] = True
    data = _kdata(random_kheader, full, acquired)
    acs = full[..., 2:-2, 2:-2]
    reconstruction = GrappaReconstruction(data, acs, kernel, regularization=0, csm=None)
    completed = reconstruction.interpolate(data)
    assert (completed - full).norm() < 0.03 * full.norm()
    assert torch.equal(completed[..., acquired], full[..., acquired])
    (expected,) = FastFourierOp().H(completed)
    torch.testing.assert_close(reconstruction(data).data, expected, atol=2e-5, rtol=2e-5)


def test_grappa_reconstruction_zero_and_coil_combination(
    grappa_data: tuple[torch.Tensor, torch.Tensor, KData], random_kheader: KHeader
) -> None:
    """Preserve acquired zeros and use the normal IData coil combination interface."""
    full, acquired, _ = grappa_data
    full[..., 0, 0, 0] = 0
    data = _kdata(random_kheader, full, acquired)
    reconstruction = GrappaReconstruction(data, full, (1, 3, 3))
    completed = reconstruction.interpolate(data)
    assert torch.equal(completed[..., acquired], full[..., acquired])
    assert reconstruction(data).data.shape == (1, 1, 1, 16, 18)
    csm = CsmData.from_idata_walsh(GrappaReconstruction(data, full, (1, 3, 3), csm=None)(data))
    explicit = GrappaReconstruction(data, full, (1, 3, 3), csm=csm)
    torch.testing.assert_close(explicit(data).data, reconstruction(data).data)


def test_grappa_batched_volume(ellipse_phantom: EllipsePhantomTestData, random_kheader: KHeader) -> None:
    """Use noncontiguous shared cropped ACS for separate 3D data batches."""
    full = _phantom_kspace(ellipse_phantom, (9, 16, 12), torch.complex128)
    full = torch.stack((full, (2 - 1j) * full)).transpose(-1, -2)
    acs = full[:1, :, 1:-1, 2:-2, 2:-2]
    acquired = torch.ones(full.shape[-3:], dtype=torch.bool)
    acquired[..., 1::2] = False
    data = _kdata(random_kheader, full, acquired)
    reconstruction = GrappaReconstruction(data, acs, (3, 3, 3), regularization=0, csm=None)
    assert (reconstruction.interpolate(data) - full).norm() < 0.03 * full.norm()


@pytest.mark.cuda
@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_grappa_reconstruction_cuda(grappa_data: tuple[torch.Tensor, torch.Tensor, KData], dtype: torch.dtype) -> None:
    """Move calibration, sampling and interpolation together with the reconstruction."""
    full, _, data = grappa_data
    full = full.to(dtype)
    data.data = data.data.to(dtype)
    reconstruction = GrappaReconstruction(data, full, (1, 3, 3), csm=None)
    expected = reconstruction(data).data
    actual = reconstruction.cuda()(data.cuda()).data
    assert actual.device.type == 'cuda'
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.cpu(), expected, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(reconstruction.cpu()(data).data, expected)


def test_grappa_full_sampling_and_recalibration(
    ellipse_phantom: EllipsePhantomTestData, random_kheader: KHeader
) -> None:
    """A full grid is unchanged, and updating a trajectory recalibrates its kernels."""
    full = _phantom_kspace(ellipse_phantom, (1, 24, 20), torch.complex128)[None]
    mask = torch.ones(full.shape[-3:], dtype=torch.bool)
    data = _kdata(random_kheader, full, mask)
    reconstruction = GrappaReconstruction(data, full, (1, 5, 3), regularization=0, csm=None)
    assert torch.equal(reconstruction.interpolate(data), full)
    for acceleration in (2, 3):
        mask[:] = False
        mask[..., ::acceleration, :] = True
        data = _kdata(random_kheader, full, mask)
        reconstruction.recalculate_fourierop(data)
        assert (reconstruction.interpolate(data) - full).norm() < 0.03 * full.norm()


def test_grappa_reject_noncartesian(grappa_data: tuple[torch.Tensor, torch.Tensor, KData]) -> None:
    """Reject non-Cartesian acquisition coordinates before creating interpolation kernels."""
    full, _, data = grappa_data
    data.traj = KTrajectory(data.traj.kz, data.traj.ky + 0.25, data.traj.kx)
    with pytest.raises(ValueError, match='Cartesian data'):
        GrappaReconstruction(data, full, (1, 3, 3), csm=None)


def test_grappa_reject_varying_batch_masks(
    grappa_data: tuple[torch.Tensor, torch.Tensor, KData], random_kheader: KHeader
) -> None:
    """Reject different sampling supports across batches explicitly."""
    full, acquired, first = grappa_data
    second = _kdata(random_kheader, full, ~acquired)
    first.data = torch.cat((first.data, second.data), dim=0)
    first.traj = KTrajectory.from_tensor(torch.cat((first.traj.as_tensor(), second.traj.as_tensor()), dim=1))
    with pytest.raises(ValueError, match='shared sampling pattern'):
        GrappaReconstruction(first, full, (1, 3, 3), csm=None)


@pytest.mark.parametrize('failure', ['repeated_samples', 'no_sources'])
def test_grappa_failed_recalibration(
    grappa_data: tuple[torch.Tensor, torch.Tensor, KData], failure: str, random_kheader: KHeader
) -> None:
    """A failed sampling update or kernel fit leaves the original reconstruction usable."""
    full, mask, original = grappa_data
    acs = full[..., 3:9, 3:10]
    reconstruction = GrappaReconstruction(original, acs, (1, 3, 3), regularization=0, csm=None)
    expected = reconstruction(original).data
    expected_direct = reconstruction.direct_reconstruction(original).data
    if failure == 'repeated_samples':
        invalid = original.clone()
        invalid.data = torch.cat((invalid.data, invalid.data), dim=-1)
        trajectory = invalid.traj.as_tensor()
        invalid.traj = KTrajectory.from_tensor(torch.cat((trajectory, trajectory), dim=-1))
        message = 'distinct samples'
    else:
        mask[:] = True
        mask[..., 6:11, 6:11] = False
        invalid = _kdata(random_kheader, full, mask)
        message = 'No acquired neighbor within kernel_size'
    with pytest.raises(ValueError, match=message):
        reconstruction.recalculate_fourierop(invalid)
    torch.testing.assert_close(reconstruction(original).data, expected)
    torch.testing.assert_close(reconstruction.direct_reconstruction(original).data, expected_direct)


def test_grappa_recalculate_csm(grappa_data: tuple[torch.Tensor, torch.Tensor, KData]) -> None:
    """Estimate sensitivities from completed coil images even when CSMs already exist."""
    full, _, data = grappa_data
    reconstruction = GrappaReconstruction(data, full, (1, 3, 3), regularization=0)
    (expected,) = FastFourierOp().H(reconstruction.interpolate(data))

    def calculate(image: IData) -> CsmData:
        torch.testing.assert_close(image.data, expected, atol=2e-5, rtol=2e-5)
        return CsmData.from_idata_walsh(image)

    reference = reconstruction(data).data
    reconstruction.recalculate_csm(data, calculate, noise=False)
    torch.testing.assert_close(reconstruction(data).data, reference)

    noise = KNoise(torch.randn(1, full.shape[-4], 1, 1, 100, dtype=full.dtype))
    with pytest.raises(ValueError, match='Prewhiten acquisition and ACS together'):
        reconstruction.recalculate_csm(data, noise=noise)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_grappa_gradient_calibration(grappa_data: tuple[torch.Tensor, torch.Tensor, KData], device: str) -> None:
    """Differentiate calibration on CPU and CUDA."""
    full, _, data = grappa_data
    data = data.to(device)
    acs = full.to(device).requires_grad_()
    reconstruction = GrappaReconstruction(data, acs, (1, 3, 3), csm=None)
    reference = reconstruction(data).data
    (gradient,) = torch.autograd.grad(reference.abs().square().sum(), acs)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


@pytest.mark.parametrize(('depth', 'recon_y', 'recon_x'), [(1, 12, 14), (4, 16, 18)])
def test_grappa_image_geometry(
    grappa_data: tuple[torch.Tensor, torch.Tensor, KData],
    depth: int,
    recon_y: int,
    recon_x: int,
) -> None:
    """Match FourierOp for image cropping and a constant trajectory axis."""
    full, _, data = grappa_data
    data.header.encoding_matrix.z = depth
    data.header.recon_matrix = SpatialDimension(depth, recon_y, recon_x)
    reconstruction = GrappaReconstruction(data, full, (1, 3, 3), regularization=0, csm=None)
    (expected,) = FastFourierOp(
        dim=(-2, -1), recon_matrix=data.header.recon_matrix, encoding_matrix=data.header.encoding_matrix
    ).H(reconstruction.interpolate(data))
    torch.testing.assert_close(reconstruction(data).data, expected)
    reference = FourierOp.from_kdata(data)
    (direct,) = reference.H(data.data)
    torch.testing.assert_close(reconstruction.direct_reconstruction(data).data, direct)
    torch.testing.assert_close(reconstruction.fourier_op(expected)[0], reference(expected)[0])


@pytest.mark.parametrize('regularization', [0.0, 0.03])
@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_grappa_calibration_explicit(regularization: float, dtype: torch.dtype) -> None:
    """Fit the acquired neighbors of a missing center, matching an explicit solve."""
    acs = torch.randn(2, 1, 5, 9, generator=torch.Generator().manual_seed(13), dtype=dtype)
    pattern = torch.zeros((1, 3, 3), dtype=torch.bool)
    pattern[..., ::2] = True
    offsets = [(dy, dx) for dy, dx in product(range(-1, 2), repeat=2) if dx != 0]
    sources = torch.stack(
        [
            torch.stack([acs[:, 0, y + dy, x + dx] for dy, dx in offsets], dim=-1).flatten()
            for y, x in product(range(1, 4), range(1, 8))
        ]
    )
    targets = torch.stack([acs[:, 0, y, x] for y, x in product(range(1, 4), range(1, 8))])
    if regularization:
        ridge = regularization * sources.abs().square().sum() / sources.shape[-1]
        weights = torch.linalg.solve(sources.mH @ sources + ridge * torch.eye(sources.shape[-1]), sources.mH @ targets)
    else:
        weights = torch.linalg.lstsq(sources, targets, driver='gelsd').solution
    kernel = GrappaReconstruction.calibrate(acs, pattern, regularization=regularization)
    assert kernel.shape == (2, 2, 1, 3, 3)
    actual = kernel.flatten(start_dim=-3)[..., pattern.flatten()].flatten(start_dim=-2).mT
    torch.testing.assert_close(actual, weights, atol=2e-5, rtol=2e-5)
    assert torch.count_nonzero(kernel[..., 0, 1, 1]) == 0


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
@pytest.mark.parametrize(('shape', 'kernel_size'), [((1, 48, 50), (1, 3, 3)), ((7, 8, 9), (3, 3, 3))])
def test_grappa_interpolation(
    dtype: torch.dtype, shape: tuple[int, int, int], kernel_size: tuple[int, int, int]
) -> None:
    """Match complex cross-correlation, including zero boundaries."""
    generator = torch.Generator().manual_seed(42)
    data = torch.randn(2, 3, *shape, generator=generator, dtype=dtype)
    pattern = torch.zeros(kernel_size, dtype=torch.bool)
    pattern[..., 0] = True
    kernel = GrappaReconstruction.calibrate(data[0], pattern)
    expected = conv3d(data, kernel, padding=tuple(k // 2 for k in kernel_size))
    actual = GrappaReconstruction._interpolate_kernel(data, kernel)
    torch.testing.assert_close(actual, expected)
    locations = torch.arange(1, data[0, 0].numel(), 2)
    selected = GrappaReconstruction._interpolate_kernel(data, kernel, locations)
    torch.testing.assert_close(selected.flatten(-3)[..., locations], expected.flatten(-3)[..., locations])
    assert torch.equal(selected.flatten(-3)[..., ::2], data.flatten(-3)[..., ::2])


def test_grappa_shared_calibration_and_scale() -> None:
    """Broadcast fitted kernels over batches without repeating calibration."""
    generator = torch.Generator().manual_seed(42)
    acs = torch.randn(1, 3, 2, 1, 12, 13, generator=generator, dtype=torch.complex128)
    data = torch.randn(2, 3, 2, 1, 8, 9, generator=generator, dtype=acs.dtype)
    pattern = torch.zeros((1, 3, 3), dtype=torch.bool)
    pattern[..., ::2] = True
    kernel = GrappaReconstruction.calibrate(acs, pattern)
    assert kernel.shape[:2] == (1, 3)
    output = GrappaReconstruction._interpolate_kernel(data, kernel)
    for i, j in product(range(2), range(3)):
        torch.testing.assert_close(output[i, j], GrappaReconstruction._interpolate_kernel(data[i, j], kernel[0, j]))
    scale = 3.1 - 2.7j
    torch.testing.assert_close(GrappaReconstruction.calibrate(scale * acs, pattern), kernel)
    torch.testing.assert_close(GrappaReconstruction._interpolate_kernel(scale * data, kernel), scale * output)


def test_grappa_invalid_regularization() -> None:
    """Reject invalid ridge strengths before solving."""
    pattern = torch.tensor([[[True, False, True]]])
    with pytest.raises(ValueError, match='finite and nonnegative'):
        GrappaReconstruction.calibrate(torch.ones(2, 1, 4, 5, dtype=torch.complex64), pattern, regularization=-1)


def test_grappa_reject_center_source() -> None:
    """A kernel cannot regress its own center."""
    with pytest.raises(ValueError, match='exclude the center'):
        GrappaReconstruction.calibrate(
            torch.ones(2, 1, 4, 5, dtype=torch.complex64), torch.ones(1, 3, 3, dtype=torch.bool)
        )


@pytest.mark.parametrize('regularization', [0.0, 0.03])
def test_grappa_underdetermined_calibration(regularization: float) -> None:
    """Allow more source features than rows and match an augmented least-squares solve."""
    acs = torch.randn(4, 1, 1, 5, dtype=torch.complex128, generator=torch.Generator().manual_seed(27))
    pattern = torch.tensor([[[True, False, True]]])
    sources = torch.stack((acs[:, 0, 0, :-2].mT, acs[:, 0, 0, 2:].mT), dim=-1).flatten(-2)
    targets = acs[:, 0, 0, 1:-1].mT
    ridge = regularization * sources.abs().square().sum() / sources.shape[-1]
    augmented = torch.cat((sources, ridge.sqrt() * torch.eye(sources.shape[-1], dtype=acs.dtype)))
    right = torch.cat((targets, targets.new_zeros(sources.shape[-1], targets.shape[-1])))
    expected = torch.linalg.lstsq(augmented, right, driver='gelsd').solution
    kernel = GrappaReconstruction.calibrate(acs, pattern, regularization=regularization)
    actual = kernel.flatten(-3)[..., pattern.flatten()].flatten(-2).mT
    torch.testing.assert_close(actual, expected)


def test_grappa_zero_calibration_batch() -> None:
    """An empty-signal batch yields zero kernels without affecting neighboring batches."""
    acs = torch.randn(2, 2, 1, 4, 5, dtype=torch.complex128, generator=torch.Generator().manual_seed(28))
    acs[0] = 0
    pattern = torch.tensor([[[True, False, True]]])
    kernels = GrappaReconstruction.calibrate(acs, pattern)
    assert torch.isfinite(kernels).all()
    assert torch.count_nonzero(kernels[0]) == 0
    torch.testing.assert_close(kernels[1], GrappaReconstruction.calibrate(acs[1], pattern))
