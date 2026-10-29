"""Tests the espirit algorithm."""

import pytest
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


@pytest.mark.parametrize('coils', [[1, 1j], [1, -1], [0, 1], [1, -0.505, -0.505]])
def test_espirit_complex_rank_one(coils: list[complex]) -> None:
    """Cancellation and a missing first coil must not erase a rank-one map."""
    reference = torch.tensor(coils, dtype=torch.complex128)
    reference = reference / reference.norm()
    image = reference[:, None, None, None].expand(-1, 1, 7, 8)
    kspace = FastFourierOp(dim=(-3, -2, -1))(image)[0]
    actual = espirit(kspace, SpatialDimension(1, 7, 8), kernel_width=1)
    torch.testing.assert_close(actual.norm(dim=0), torch.ones_like(actual[0].real))
    overlap = (actual.conj() * reference[:, None, None, None]).sum(dim=0).abs()
    torch.testing.assert_close(overlap, torch.ones_like(overlap))


@pytest.mark.parametrize('ny', [15, 16])
def test_espirit_spatial_complex_phase(ny: int) -> None:
    """Odd/even grids retain spatially varying relative coil phases and origin."""
    phase = torch.arange(ny, dtype=torch.float64) * (2 * torch.pi / ny)
    reference = torch.stack([torch.ones(ny, dtype=torch.complex128), torch.exp(1j * phase)])
    reference = reference[:, None, :, None].expand(2, 1, ny, 16) / 2**0.5
    kspace = FastFourierOp(dim=(-3, -2, -1))(reference)[0]
    actual = espirit(kspace, SpatialDimension(1, ny, 16), kernel_width=3, n_iterations=30)
    overlap = (actual.conj() * reference).sum(dim=0).abs()
    torch.testing.assert_close(overlap, torch.ones_like(overlap), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
@pytest.mark.parametrize('support', [(1, 3, 3), (2, 3, 3)])
def test_espirit_correlations(support, dtype, device) -> None:
    """Kernel correlations reproduce direct image-domain outer products."""
    from math import prod

    from mr2.algorithms.csm.espirit import _espirit_correlations

    kernels = torch.randn(7, 3, *support, dtype=dtype, device=device)
    shape = tuple(2 * n - 1 for n in support)
    images = torch.fft.ifftn(kernels, s=shape, dim=(-3, -2, -1))
    expected = torch.einsum('nczyx,ndzyx->cdzyx', images, images.conj())
    actual = torch.fft.ifftn(_espirit_correlations(kernels), dim=(-3, -2, -1)) / prod(shape)
    torch.testing.assert_close(actual, expected)


def test_espirit_multiple_slabs() -> None:
    """A volume larger than the voxel batch preserves all slices and relative coil phase."""
    reference = torch.tensor([1, 2j], dtype=torch.complex64) / 5**0.5
    calibration = reference[:, None, None, None] * torch.ones(1, 2, 3, 4)
    actual = espirit(calibration, SpatialDimension(64, 65, 66), kernel_width=1)
    assert actual.shape == (2, 64, 65, 66)
    torch.testing.assert_close(actual, reference[:, None, None, None].expand_as(actual))


@pytest.mark.parametrize('shape', [(1, 4, 4), (1, 8, 9)])
def test_espirit_gram_subspace(shape) -> None:
    """Gram eigenvectors retain the same complex subspace as SVD for wide and tall calibration."""
    from importlib import import_module
    from unittest.mock import patch

    from mr2.algorithms.csm.espirit import _espirit_correlations

    calibration = torch.randn(3, *shape, dtype=torch.complex128)
    matrix = calibration.unfold(1, 1, 1).unfold(2, 3, 1).unfold(3, 3, 1)
    matrix = matrix.permute(1, 2, 3, 0, 4, 5, 6).reshape(-1, 27)
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    expected = vh[singular_values > 0.02 * singular_values.max()]
    with patch.object(
        import_module('mr2.algorithms.csm.espirit'), '_espirit_correlations', wraps=_espirit_correlations
    ) as correlations:
        espirit(calibration, SpatialDimension(*shape), kernel_width=3)
    actual = correlations.call_args.args[0].reshape(-1, 27)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.H @ actual, expected.H @ expected)
