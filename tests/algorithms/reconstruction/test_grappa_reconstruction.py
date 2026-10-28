"""Tests for trajectory-driven Cartesian GRAPPA reconstruction."""

from itertools import product

import pytest
import torch
from mr2.algorithms.reconstruction import GrappaReconstruction
from mr2.data import CsmData, IData, KData, KNoise, KTrajectory, SpatialDimension
from mr2.operators import FastFourierOp, FourierOp
from mr2.utils import RandomGenerator, unsqueeze_at
from tests import relative_image_difference
from torch.nn.functional import conv3d


def _acquired(kdata: KData, mask: torch.Tensor) -> KData:
    """Keep the samples selected by a k-space mask as a single readout."""
    trajectory = KTrajectory.from_tensor(unsqueeze_at(kdata.traj.as_tensor()[..., mask], dim=-2, n=2))
    return KData(header=kdata.header, data=unsqueeze_at(kdata.data[..., mask], dim=-2, n=2), traj=trajectory)


@pytest.fixture
def grappa_data(fully_sampled_kdata: KData) -> tuple[KData, torch.Tensor, KData]:
    """Ellipse acquisition with alternate phase-encoding lines missing."""
    full = fully_sampled_kdata
    acquired = torch.ones(full.data.shape[-3:], dtype=torch.bool)
    acquired[..., 1::2, :] = False
    return full, acquired, _acquired(full, acquired)


@pytest.mark.parametrize(
    ('fully_sampled_kdata', 'kernel', 'acceleration'),
    [
        ((shape, dtype), kernel, acceleration)
        for shape, kernel, acceleration in [
            ((1, 24, 20), (1, 3, 3), 2),
            ((1, 24, 20), (1, 5, 3), 3),
            ((9, 16, 12), (3, 3, 3), 2),
        ]
        for dtype in (torch.complex64, torch.complex128)
    ],
    indirect=['fully_sampled_kdata'],
)
def test_grappa_reconstruction(fully_sampled_kdata: KData, kernel: tuple[int, int, int], acceleration: int) -> None:
    """Reconstruct the ellipse phantom using cropped ACS and trajectory-derived sampling."""
    full = fully_sampled_kdata
    acquired = torch.zeros(full.data.shape[-3:], dtype=torch.bool)
    acquired[..., ::acceleration, :] = True
    acquired[..., 6:10, :] = True
    data = _acquired(full, acquired)
    acs = full.data[..., 2:-2, 2:-2]
    reconstruction = GrappaReconstruction(data, acs, kernel, regularization=0, csm=None)
    completed = reconstruction.interpolate(data)
    assert relative_image_difference(completed, full.data) < 0.03
    assert torch.equal(completed[..., acquired], full.data[..., acquired])
    (expected,) = FastFourierOp().H(completed)
    torch.testing.assert_close(reconstruction(data).data, expected, atol=2e-5, rtol=2e-5)


def test_grappa_reconstruction_zero_and_coil_combination(grappa_data: tuple[KData, torch.Tensor, KData]) -> None:
    """Preserve acquired zeros and use the normal IData coil combination interface."""
    full, acquired, _ = grappa_data
    full.data[..., 0, 0, 0] = 0
    data = _acquired(full, acquired)
    reconstruction = GrappaReconstruction(data, full.data, (1, 3, 3))
    completed = reconstruction.interpolate(data)
    assert torch.equal(completed[..., acquired], full.data[..., acquired])
    assert reconstruction(data).data.shape == (1, 1, 1, 16, 18)
    csm = CsmData.from_idata_walsh(GrappaReconstruction(data, full.data, (1, 3, 3), csm=None)(data))
    explicit = GrappaReconstruction(data, full.data, (1, 3, 3), csm=csm)
    torch.testing.assert_close(explicit(data).data, reconstruction(data).data)


@pytest.mark.parametrize('fully_sampled_kdata', [((9, 16, 12), torch.complex128)], indirect=True)
def test_grappa_batched_volume(fully_sampled_kdata: KData) -> None:
    """Use noncontiguous shared cropped ACS for separate 3D data batches."""
    full = fully_sampled_kdata
    full.data = torch.cat((full.data, (2 - 1j) * full.data))
    acs = full.data[:1, :, 1:-1, 2:-2, 2:-2]
    acquired = torch.ones(full.data.shape[-3:], dtype=torch.bool)
    acquired[..., 1::2] = False
    data = _acquired(full, acquired)
    reconstruction = GrappaReconstruction(data, acs, (3, 3, 3), regularization=0, csm=None)
    assert relative_image_difference(reconstruction.interpolate(data), full.data) < 0.03


@pytest.mark.cuda
@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_grappa_reconstruction_cuda(grappa_data: tuple[KData, torch.Tensor, KData], dtype: torch.dtype) -> None:
    """Move calibration, sampling and interpolation together with the reconstruction."""
    full, _, data = grappa_data
    data = data.to(dtype)
    reconstruction = GrappaReconstruction(data, full.data.to(dtype), (1, 3, 3), csm=None)
    expected = reconstruction(data).data
    actual = reconstruction.cuda()(data.cuda()).data
    assert actual.device.type == 'cuda'
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.cpu(), expected, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(reconstruction.cpu()(data).data, expected)


@pytest.mark.parametrize('fully_sampled_kdata', [((1, 24, 20), torch.complex128)], indirect=True)
def test_grappa_full_sampling_and_recalibration(fully_sampled_kdata: KData) -> None:
    """A full grid is unchanged, and updating a trajectory recalibrates its kernels."""
    full = fully_sampled_kdata
    reconstruction = GrappaReconstruction(full, full.data, (1, 5, 3), regularization=0, csm=None)
    assert torch.equal(reconstruction.interpolate(full), full.data)
    for acceleration in (2, 3):
        mask = torch.zeros(full.data.shape[-3:], dtype=torch.bool)
        mask[..., ::acceleration, :] = True
        data = _acquired(full, mask)
        reconstruction.recalculate_fourierop(data)
        assert relative_image_difference(reconstruction.interpolate(data), full.data) < 0.03


def test_grappa_reject_noncartesian(grappa_data: tuple[KData, torch.Tensor, KData]) -> None:
    """Reject non-Cartesian acquisition coordinates before creating interpolation kernels."""
    full, _, data = grappa_data
    data.traj = KTrajectory(data.traj.kz, data.traj.ky + 0.25, data.traj.kx)
    with pytest.raises(ValueError, match='Cartesian data'):
        GrappaReconstruction(data, full.data, (1, 3, 3), csm=None)


def test_grappa_reject_varying_batch_masks(grappa_data: tuple[KData, torch.Tensor, KData]) -> None:
    """Reject different sampling supports across batches explicitly."""
    full, acquired, data = grappa_data
    data = data.concatenate(_acquired(full, ~acquired), dim=0)
    with pytest.raises(ValueError, match='shared sampling pattern'):
        GrappaReconstruction(data, full.data, (1, 3, 3), csm=None)


@pytest.mark.parametrize('failure', ['repeated_samples', 'no_sources'])
def test_grappa_failed_recalibration(grappa_data: tuple[KData, torch.Tensor, KData], failure: str) -> None:
    """A failed sampling update or kernel fit leaves the original reconstruction usable."""
    full, mask, original = grappa_data
    acs = full.data[..., 3:9, 3:10]
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
        invalid = _acquired(full, mask)
        message = 'No acquired neighbor within kernel_size'
    with pytest.raises(ValueError, match=message):
        reconstruction.recalculate_fourierop(invalid)
    torch.testing.assert_close(reconstruction(original).data, expected)
    torch.testing.assert_close(reconstruction.direct_reconstruction(original).data, expected_direct)


def test_grappa_recalculate_csm(grappa_data: tuple[KData, torch.Tensor, KData]) -> None:
    """Estimate sensitivities from completed coil images even when CSMs already exist."""
    full, _, data = grappa_data
    reconstruction = GrappaReconstruction(data, full.data, (1, 3, 3), regularization=0)
    (expected,) = FastFourierOp().H(reconstruction.interpolate(data))

    def calculate(image: IData) -> CsmData:
        torch.testing.assert_close(image.data, expected, atol=2e-5, rtol=2e-5)
        return CsmData.from_idata_walsh(image)

    reference = reconstruction(data).data
    reconstruction.recalculate_csm(data, calculate, noise=False)
    torch.testing.assert_close(reconstruction(data).data, reference)

    noise = KNoise(RandomGenerator(0).randn_tensor((1, full.data.shape[-4], 1, 1, 100), dtype=full.data.dtype))
    with pytest.raises(ValueError, match='Prewhiten acquisition and ACS together'):
        reconstruction.recalculate_csm(data, noise=noise)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_grappa_gradient_calibration(grappa_data: tuple[KData, torch.Tensor, KData], device: str) -> None:
    """Differentiate calibration on CPU and CUDA."""
    full, _, data = grappa_data
    data = data.to(device)
    acs = full.data.to(device).requires_grad_()
    reconstruction = GrappaReconstruction(data, acs, (1, 3, 3), csm=None)
    reference = reconstruction(data).data
    (gradient,) = torch.autograd.grad(reference.abs().square().sum(), acs)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


@pytest.mark.parametrize(('depth', 'recon_y', 'recon_x'), [(1, 12, 14), (4, 16, 18)])
def test_grappa_image_geometry(
    grappa_data: tuple[KData, torch.Tensor, KData],
    depth: int,
    recon_y: int,
    recon_x: int,
) -> None:
    """Match FourierOp for image cropping and a constant trajectory axis."""
    full, _, data = grappa_data
    data.header.encoding_matrix.z = depth
    data.header.recon_matrix = SpatialDimension(depth, recon_y, recon_x)
    reconstruction = GrappaReconstruction(data, full.data, (1, 3, 3), regularization=0, csm=None)
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
    acs = RandomGenerator(13).randn_tensor((2, 1, 5, 9), dtype=dtype)
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
    rng = RandomGenerator(42)
    data = rng.randn_tensor((2, 3, *shape), dtype=dtype)
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
    rng = RandomGenerator(42)
    acs = rng.randn_tensor((1, 3, 2, 1, 12, 13), dtype=torch.complex128)
    data = rng.randn_tensor((2, 3, 2, 1, 8, 9), dtype=acs.dtype)
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
    acs = RandomGenerator(27).randn_tensor((4, 1, 1, 5), dtype=torch.complex128)
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
    acs = RandomGenerator(28).randn_tensor((2, 2, 1, 4, 5), dtype=torch.complex128)
    acs[0] = 0
    pattern = torch.tensor([[[True, False, True]]])
    kernels = GrappaReconstruction.calibrate(acs, pattern)
    assert torch.isfinite(kernels).all()
    assert torch.count_nonzero(kernels[0]) == 0
    torch.testing.assert_close(kernels[1], GrappaReconstruction.calibrate(acs[1], pattern))
