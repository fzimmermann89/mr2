"""Small synthetic acquisition tests for the untrained ConvDecoder prior."""

from collections.abc import Iterator

import pytest
import torch
from mr2.algorithms.optimizers.OptimizerStatus import OptimizerStatus
from mr2.algorithms.reconstruction import ConvDecoderReconstruction
from mr2.data import KData, KHeader, KTrajectory, SpatialDimension
from mr2.operators import FourierOp, SensitivityOp


@pytest.fixture(autouse=True)
def single_thread() -> Iterator[None]:
    """Avoid thread-pool overhead for tiny convolutions."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def acquisition() -> tuple[KData, FourierOp, SensitivityOp]:
    """Two-coil Cartesian acquisition with distinct complex sensitivities."""
    size = SpatialDimension(1, 8, 10)
    header = KHeader(
        recon_matrix=size,
        encoding_matrix=size,
        recon_fov=SpatialDimension(0.01, 0.08, 0.10),
        encoding_fov=SpatialDimension(0.01, 0.08, 0.10),
        te=[0.012],
    )
    # Only every second phase-encoding line is measured.
    traj = KTrajectory(
        torch.zeros(1, 1, 1, 1, 1),
        torch.arange(-4, 4, 2).reshape(1, 1, 1, 4, 1),
        torch.arange(-5, 5).reshape(1, 1, 1, 1, 10),
    )
    fourier = FourierOp(size, size, traj)
    y, x = torch.meshgrid(torch.linspace(-1, 1, 8), torch.linspace(-1, 1, 10), indexing='ij')
    image = (torch.exp(-3 * (x.square() + y.square())) * torch.exp(1j * x)).reshape(1, 1, 1, 8, 10)
    csm = SensitivityOp(torch.stack((torch.ones_like(x), (1 + 0.2 * y) * torch.exp(1j * x)))[None, :, None])
    (data,) = (fourier @ csm)(image)
    return KData(header=header, data=data, traj=traj), fourier, csm


def make_reconstruction(fourier: FourierOp, csm: SensitivityOp | None, **kwargs) -> ConvDecoderReconstruction:
    """Use a small decoder to keep fitting tests fast."""
    return ConvDecoderReconstruction(
        fourier_op=fourier, csm=csm, n_features=8, n_levels=2, latent_shape=(2, 2), **kwargs
    )


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_fit_acquired_complex_data(acquisition: tuple[KData, FourierOp, SensitivityOp], dtype: torch.dtype) -> None:
    """Gradient updates reduce the residual through both sensitivity and Fourier operators."""
    kdata, fourier, csm = acquisition
    kdata = kdata.to(dtype=dtype)
    csm = SensitivityOp(csm.csm_tensor.to(dtype=dtype))
    initial = make_reconstruction(fourier, csm, max_iterations=0)(kdata)
    result = make_reconstruction(fourier, csm, max_iterations=50)(kdata)
    operator = fourier @ csm
    initial_error = (operator(initial.data)[0] - kdata.data).abs().square().mean()
    final_error = (operator(result.data)[0] - kdata.data).abs().square().mean()
    assert final_error < initial_error * 0.2
    assert result.data.dtype == dtype
    assert result.data.shape == (1, 1, 1, 8, 10)
    assert not result.data.requires_grad
    assert result.data.imag.abs().max() > 0
    assert result.header.te == kdata.header.te
    assert result.header.resolution == SpatialDimension(0.01, 0.01, 0.01)


def test_reset_and_local_rng(acquisition: tuple[KData, FourierOp, SensitivityOp]) -> None:
    """Calls reset weights, input and Adam state without consuming global random numbers."""
    kdata, fourier, csm = acquisition
    recon = make_reconstruction(fourier, csm, max_iterations=3)
    state = torch.random.get_rng_state().clone()
    first = recon(kdata)
    torch.testing.assert_close(state, torch.random.get_rng_state(), rtol=0, atol=0)
    other = KData(header=kdata.header, data=kdata.data * 2j, traj=kdata.traj)
    recon(other)
    with torch.no_grad():
        repeated = recon(kdata)
    torch.testing.assert_close(first.data, repeated.data, rtol=0, atol=0)
    different = make_reconstruction(fourier, csm, max_iterations=3, seed=1)(kdata)
    assert not torch.allclose(first.data, different.data)
    assert not list(recon.parameters())  # No fitted generator is retained.


def test_callback_stops_after_update(acquisition: tuple[KData, FourierOp, SensitivityOp]) -> None:
    """Callbacks receive current complex images and can stop with False."""
    kdata, fourier, csm = acquisition
    statuses: list[OptimizerStatus] = []

    def callback(status: OptimizerStatus) -> bool:
        statuses.append(status)
        return status['iteration_number'] < 1

    result = make_reconstruction(fourier, csm, max_iterations=20, callback=callback)(kdata)
    assert [status['iteration_number'] for status in statuses] == [0, 1]
    assert not statuses[-1]['solution'][0].requires_grad
    torch.testing.assert_close(result.data, statuses[-1]['solution'][0], rtol=0, atol=0)
    expected = make_reconstruction(fourier, csm, max_iterations=2)(kdata)
    torch.testing.assert_close(result.data, expected.data, rtol=0, atol=0)
    assert not torch.equal(statuses[0]['solution'][0], statuses[1]['solution'][0])


def test_independent_slices(acquisition: tuple[KData, FourierOp, SensitivityOp]) -> None:
    """Leading batch dimensions are restored and slices start with independent weights."""
    kdata, fourier, csm = acquisition
    batched = KData(
        header=kdata.header, data=kdata.data[None] * torch.tensor([1, 2]).reshape(2, 1, 1, 1, 1, 1), traj=kdata.traj
    )
    result = make_reconstruction(fourier, csm, max_iterations=0)(batched)
    assert result.data.shape == (2, 1, 1, 1, 8, 10)
    assert not torch.equal(result.data[0], result.data[1])


def test_single_coil_without_maps(acquisition: tuple[KData, FourierOp, SensitivityOp]) -> None:
    """A single coil can use the Fourier operator directly."""
    kdata, fourier, _ = acquisition
    single = KData(header=kdata.header, data=kdata.data[:, :1], traj=kdata.traj)
    result = make_reconstruction(fourier, None, max_iterations=1)(single)
    assert result.data.shape == (1, 1, 1, 8, 10)
    with pytest.raises(ValueError, match='Sensitivity maps'):
        make_reconstruction(fourier, None, max_iterations=1)(kdata)


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'max_iterations': -1}, 'max_iterations'),
        ({'learning_rate': 0}, 'learning_rate'),
        ({'learning_rate': float('nan')}, 'learning_rate'),
        ({'n_features': 0}, 'n_features'),
        ({'n_levels': 0}, 'n_levels'),
        ({'latent_shape': (1, 1)}, 'latent_shape'),
    ],
)
def test_invalid_settings(kwargs: dict, message: str) -> None:
    """Reject invalid fitting budgets and architecture settings before setup."""
    with pytest.raises(ValueError, match=message):
        ConvDecoderReconstruction(**kwargs)


def test_automatic_fourier_setup(acquisition: tuple[KData, FourierOp, SensitivityOp]) -> None:
    """KData configures the Fourier operator and preserves the usual reconstruction interface."""
    kdata, fourier, csm = acquisition
    automatic = ConvDecoderReconstruction(
        kdata, csm=csm, n_features=8, n_levels=2, latent_shape=(2, 2), max_iterations=1
    )
    expected = make_reconstruction(fourier, csm, max_iterations=1)(kdata)
    torch.testing.assert_close(automatic(kdata).data, expected.data)
    assert automatic.dcf_op is not None


@pytest.mark.cuda
def test_cuda(acquisition: tuple[KData, FourierOp, SensitivityOp]) -> None:
    """Fit on the input device after moving the reconstruction operators to CUDA."""
    kdata, fourier, csm = acquisition
    recon = make_reconstruction(fourier, csm, max_iterations=2).cuda()
    state = torch.cuda.get_rng_state().clone()
    result = recon(kdata.cuda())
    assert result.is_cuda
    assert result.data.dtype == torch.complex64
    torch.testing.assert_close(state, torch.cuda.get_rng_state(), rtol=0, atol=0)
