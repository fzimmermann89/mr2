"""Tests the iterative Walsh algorithm."""

import pytest
import torch
from mr2.algorithms.csm import walsh
from mr2.data import SpatialDimension
from mr2.utils import RandomGenerator
from tests import relative_image_difference
from tests.algorithms.csm.conftest import multi_coil_image


def test_walsh(ellipse_phantom, random_kheader):
    """Test the Walsh method."""
    idata, csm_ref = multi_coil_image(n_coils=4, ph_ellipse=ellipse_phantom, random_kheader=random_kheader)

    # Estimate coil sensitivity maps.
    # walsh should be applied for each other dimension separately
    smoothing_width = SpatialDimension(z=1, y=5, x=5)
    csm = walsh(idata.data[0, ...], smoothing_width)

    # Phase is only relative in csm calculation, therefore only the abs values are compared.
    assert relative_image_difference(torch.abs(csm), torch.abs(csm_ref[0, ...])) <= 0.01


def test_walsh_extrapolates_to_low_signal_regions() -> None:
    """Test that Walsh extrapolation fills low-signal regions with normalized CSMs."""
    coil_img = torch.ones(2, 1, 16, 16, dtype=torch.complex64)
    coil_img[:, :, :, 8:] = 0

    csm = walsh(coil_img, smoothing_width=5, extrapolate=True)

    torch.testing.assert_close(csm.abs().square().sum(dim=0).sqrt(), torch.ones(1, 16, 16))
    assert torch.isfinite(csm).all()


def test_walsh_pools_echoes_without_phase_cancellation() -> None:
    """Opposite echo phases preserve shared maps and independent slice batches."""
    rng = RandomGenerator(0)
    coils = rng.randn_tensor((2, 4, 1, 8, 8), dtype=torch.complex64)
    echoes = torch.stack((coils, -coils, 0.5j * coils), dim=1)
    expected = walsh(coils, smoothing_width=3).unsqueeze(1)
    actual = walsh(echoes, smoothing_width=3, pool_dim=1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(walsh(echoes, 3, pool_dim=-5), actual)
    assert walsh(echoes, 3).shape == echoes.shape


@pytest.mark.parametrize('pool_dim', [2, 5, -4, -1])
def test_walsh_rejects_nonleading_pool_dimension(pool_dim: int) -> None:
    """Pooling must not conflate coils or spatial axes with echoes."""
    with pytest.raises(ValueError, match='leading dimension'):
        walsh(torch.ones(2, 3, 4, 1, 8, 8, dtype=torch.complex64), 3, pool_dim=pool_dim)


@pytest.mark.parametrize('pool_dim', [0, 1, -6, -5])
def test_walsh_pools_extrapolation_confidence(pool_dim: int) -> None:
    """Pooled extrapolation uses signal energy from all echoes in each batch."""
    rng = RandomGenerator(0)
    coils = rng.randn_tensor((2, 4, 1, 8, 8), dtype=torch.complex64)
    weights = rng.rand_tensor((2, 3, 1, 1, 8, 8), dtype=torch.float32)
    weights[:, 0, ..., 4:] = 0
    weights[..., 6:] = 0
    echoes = coils.unsqueeze(1) * weights
    echoes[:, 1] *= -1
    expected = walsh(coils * weights.square().sum(dim=1).sqrt(), 3, align_phase=False, extrapolate=True).unsqueeze(
        pool_dim
    )

    actual = walsh(echoes.movedim(1, pool_dim), 3, align_phase=False, extrapolate=True, pool_dim=pool_dim)

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual.norm(dim=-4), torch.ones_like(actual[:, :, 0].real))
