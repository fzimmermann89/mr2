"""Tests for ConvDecoderReconstruction."""

import pytest
import torch
from mr2.algorithms.optimizers.OptimizerStatus import OptimizerStatus
from mr2.algorithms.reconstruction import ConvDecoderReconstruction
from mr2.data import KData, SpatialDimension
from mr2.operators import FourierOp, SensitivityOp


@pytest.mark.parametrize(
    ('latent_shape', 'dtype'),
    [((2, 2), torch.complex64), ((2, 2, 2), torch.complex128), ((2, 2, 2, 2), torch.complex64)],
)
def test_reconstruction(
    convdecoder_acquisition: tuple[KData, FourierOp, SensitivityOp], latent_shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    """Test ND fitting, complex layout, fresh initialization and fixed acquisition gradients."""
    kdata, fourier, csm = convdecoder_acquisition
    kdata = kdata.to(dtype=dtype)
    maps = csm.csm_tensor.to(dtype=dtype).detach().requires_grad_()
    csm = SensitivityOp(maps * 1.0)
    recon = ConvDecoderReconstruction(
        fourier_op=fourier, csm=csm, latent_shape=latent_shape, n_features=4, n_levels=1, max_iterations=0
    )
    torch.manual_seed(0)
    initial = recon(kdata).data
    recon.max_iterations = 20
    torch.manual_seed(0)
    result = recon(kdata)
    operator = fourier @ csm
    assert (operator(result.data)[0] - kdata.data).abs().square().mean() < (
        operator(initial)[0] - kdata.data
    ).abs().square().mean()
    assert result.data.shape == (3, 1, 3, 5, 7)
    assert result.data.dtype == dtype
    assert result.header.resolution == SpatialDimension(0.01, 0.01, 0.01)
    assert maps.grad is None


def test_automatic(cartesian_kdata: KData) -> None:
    """Test automatic setup from kdata only."""
    reconstruction = ConvDecoderReconstruction(kdata=cartesian_kdata, n_features=2, n_levels=1, max_iterations=1)
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert reconstruction.csm_op is not None


def test_callback(convdecoder_acquisition: tuple[KData, FourierOp, SensitivityOp]) -> None:
    """Test stopping with the current image after an update."""
    kdata, fourier, csm = convdecoder_acquisition
    statuses: list[OptimizerStatus] = []

    def callback(status: OptimizerStatus) -> bool:
        statuses.append(status)
        return False

    recon = ConvDecoderReconstruction(
        fourier_op=fourier, csm=csm, n_features=4, n_levels=1, latent_shape=(2, 2), callback=callback
    )
    result = recon(kdata)
    assert len(statuses) == 1
    assert statuses[0]['iteration_number'] == 0
    torch.testing.assert_close(result.data, statuses[0]['solution'][0])


def test_independent_slices(convdecoder_acquisition: tuple[KData, FourierOp, SensitivityOp]) -> None:
    """Changing one batch entry's intensity leaves the others unchanged."""
    kdata, fourier, csm = convdecoder_acquisition
    recon = ConvDecoderReconstruction(
        fourier_op=fourier, csm=csm, n_features=4, n_levels=1, latent_shape=(2, 2), max_iterations=3
    )
    torch.manual_seed(0)
    original = recon(kdata).data
    changed = kdata.data.clone()
    changed[1] *= 100
    torch.manual_seed(0)
    result = recon(KData(header=kdata.header, data=changed, traj=kdata.traj)).data
    torch.testing.assert_close(original[0], result[0], rtol=0, atol=0)


@pytest.mark.cuda
def test_cuda(convdecoder_acquisition: tuple[KData, FourierOp, SensitivityOp]) -> None:
    """Test reconstruction on CUDA."""
    kdata, fourier, csm = convdecoder_acquisition
    recon = ConvDecoderReconstruction(
        fourier_op=fourier, csm=csm, n_features=4, n_levels=1, latent_shape=(2, 2), max_iterations=1
    ).cuda()
    assert recon(kdata.cuda()).is_cuda
