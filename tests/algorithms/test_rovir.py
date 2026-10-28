"""Tests for ROVir coil compression."""

import pytest
import scipy.linalg
import torch
from mr2.algorithms import rovir
from mr2.utils import RandomGenerator


def test_rovir_returns_operator_with_expected_shape() -> None:
    """ROVir should return a coil compression operator."""
    rng = RandomGenerator(0)
    img = rng.complex64_tensor((4, 3, 5, 6))
    roi = torch.zeros((3, 5, 6), dtype=torch.bool)
    roi[1, 1:4, 2:5] = True

    operator = rovir(img, roi, n_compressed_coils=2)
    test_data = rng.complex64_tensor((7, 4, 3, 5, 6))
    (compressed,) = operator(test_data)

    assert compressed.shape == (7, 2, 3, 5, 6)
    gram = operator.matrix @ operator.matrix.conj().mT
    torch.testing.assert_close(gram, torch.eye(2, dtype=img.dtype), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_rovir_energy_ratio_matches_generalized_eigenvector(device: str, dtype: torch.dtype) -> None:
    """ROVir matches the optimal ROI/background energy ratio for real and complex coil images."""
    rng = RandomGenerator(seed=0)
    phases = torch.tensor([0.0, 0.7, 1.8, -1.2], dtype=torch.float64)
    weights = (torch.exp(1j * phases) if dtype.is_complex else phases.cos()).to(dtype)
    pattern = torch.tensor([1, 2, -1, -2, 2, 1, -2, -1], dtype=dtype)
    roi_data = 10 * weights[:, None] * pattern
    background_data = rng.randn_tensor((4, 8), dtype) + 3 * weights.conj()[:, None] * pattern
    img = torch.cat((roi_data, background_data), dim=1).reshape(4, 1, 2, 8)
    roi = torch.zeros((1, 2, 8), dtype=torch.bool)
    roi[:, 0] = True

    # Use column coil vectors and an independent generalized eigensolver for w^H x.
    reference_roi = roi_data.to(torch.complex128 if dtype.is_complex else torch.float64)
    reference_background = background_data.to(reference_roi.dtype)
    roi_covariance = reference_roi @ reference_roi.mH / reference_roi.shape[1]
    background_covariance = reference_background @ reference_background.mH / reference_background.shape[1]
    _, eigenvectors = scipy.linalg.eigh(roi_covariance.numpy(), background_covariance.numpy())
    reference_weights = torch.from_numpy(eigenvectors[:, -1].copy())
    reference_weights /= torch.linalg.vector_norm(reference_weights)
    reference = (reference_weights.conj() @ img.to(reference_roi.dtype).flatten(1)).reshape(1, 2, 8)
    reference_roi_energy = reference[roi].abs().square().mean()
    reference_background_energy = reference[~roi].abs().square().mean()

    operator = rovir(img.to(device), roi.to(device), n_compressed_coils=1)
    (compressed,) = operator(img.to(device))
    assert compressed.device.type == device
    compressed = compressed.squeeze(0).cpu()
    roi_energy = compressed[roi].abs().square().mean()
    background_energy = compressed[~roi].abs().square().mean()

    torch.testing.assert_close(roi_energy, reference_roi_energy.to(roi_energy.dtype), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        background_energy, reference_background_energy.to(background_energy.dtype), atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        roi_energy / background_energy,
        (reference_roi_energy / reference_background_energy).to(roi_energy.dtype),
        atol=1e-4,
        rtol=1e-4,
    )


def test_rovir_raises_for_invalid_roi() -> None:
    """ROVir requires a non-empty ROI and background."""
    rng = RandomGenerator(0)
    img = rng.complex64_tensor((4, 3, 5, 6))

    with pytest.raises(ValueError, match='roi does not contain any voxels'):
        rovir(img, torch.zeros((3, 5, 6), dtype=torch.bool), n_compressed_coils=2)

    with pytest.raises(ValueError, match='roi covers the full image'):
        rovir(img, torch.ones((3, 5, 6), dtype=torch.bool), n_compressed_coils=2)


@pytest.mark.cuda
def test_rovir_cuda_image_with_cpu_roi() -> None:
    """ROVir accepts a CPU ROI with CUDA coil images and preserves orthonormality."""
    img = RandomGenerator(1).complex64_tensor((4, 2, 3, 4)).cuda()
    roi = torch.zeros((2, 3, 4), dtype=torch.bool)
    roi[:, :2, :2] = True

    operator = rovir(img, roi, n_compressed_coils=2)
    gram = operator.matrix @ operator.matrix.conj().mT

    assert operator.matrix.is_cuda
    torch.testing.assert_close(gram, torch.eye(2, dtype=img.dtype, device='cuda'), atol=1e-5, rtol=1e-5)
