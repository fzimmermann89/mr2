"""Tests for ROVir coil compression."""

import pytest
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


def test_rovir_raises_for_invalid_roi() -> None:
    """ROVir requires a non-empty ROI and background."""
    rng = RandomGenerator(0)
    img = rng.complex64_tensor((4, 3, 5, 6))

    with pytest.raises(ValueError, match='roi does not contain any voxels'):
        rovir(img, torch.zeros((3, 5, 6), dtype=torch.bool), n_compressed_coils=2)

    with pytest.raises(ValueError, match='roi covers the full image'):
        rovir(img, torch.ones((3, 5, 6), dtype=torch.bool), n_compressed_coils=2)
