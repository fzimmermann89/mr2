"""Tests the iterative Walsh algorithm."""

import torch
from mr2.algorithms.csm import walsh
from mr2.data import SpatialDimension
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
