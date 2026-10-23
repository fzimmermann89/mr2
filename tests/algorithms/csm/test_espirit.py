"""Tests the espirit algorithm."""

import torch
from mr2.algorithms.csm import espirit
from mr2.data import SpatialDimension
from mr2.operators import FastFourierOp
from tests.algorithms.csm.conftest import multi_coil_image
from tests.helper import relative_image_difference


def test_espirit(ellipse_phantom, random_kheader):
    """Test the espirit algorithm."""
    idata, csm_ref = multi_coil_image(n_coils=4, ph_ellipse=ellipse_phantom, random_kheader=random_kheader)

    # Estimate coil sensitivity maps from Cartesian k-space.
    (kspace,) = FastFourierOp(dim=(-3, -2, -1))(idata.data[0, ...])
    csm = espirit(kspace, img_shape=SpatialDimension(*idata.shape[-3:]))
    # Phase is only relative in csm calculation, therefore only the abs values are compared.

    assert relative_image_difference(torch.abs(csm), torch.abs(csm_ref[0, ...])) <= 0.04


def test_espirit_zero_calibration() -> None:
    """An empty retained subspace produces finite zero maps."""
    csm = espirit(torch.zeros(2, 1, 8, 8, dtype=torch.complex64), SpatialDimension(1, 12, 12), kernel_width=3)

    assert csm.shape == (2, 1, 12, 12)
    assert torch.isfinite(csm).all()
    assert torch.count_nonzero(csm) == 0
