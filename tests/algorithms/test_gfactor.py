"""Tests for measurement-domain g-factor estimation."""

import weakref

import pytest
import torch
from mr2.algorithms import gfactor
from mr2.data import KTrajectory
from mr2.operators import IdentityOp, NonUniformFastFourierOp


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_covariance_and_online_variance(dtype):
    """Noise has the requested complex covariance; online variance matches a stack."""
    covariance = torch.tensor([[2, 0.3 + 0.7j], [0.3 - 0.7j, 1]], dtype=dtype)
    data = torch.zeros(2, 1, 1, 1024, dtype=dtype)
    replicas = []

    def reconstruct(measurements):
        replicas.append(measurements.clone())
        return measurements + 10 + 7j

    result = gfactor(
        data,
        reconstruct,
        data,
        reconstruct,
        acceleration=1,
        noise_covariance=covariance,
        n_replicas=128,
        generator=torch.Generator().manual_seed(42),
    )
    accelerated, reference = torch.stack(replicas).split(128)
    expected = (accelerated.var(dim=0) / reference.var(dim=0)).sqrt()
    torch.testing.assert_close(result, expected)
    noise = accelerated.movedim(1, 0).reshape(2, -1)
    empirical_covariance = noise @ noise.mH / noise.shape[-1]
    torch.testing.assert_close(empirical_covariance, covariance, atol=0.02, rtol=0.02)
    # Circular noise: zero pseudo-covariance and half the total variance per real part.
    torch.testing.assert_close(noise @ noise.T / noise.shape[-1], torch.zeros_like(covariance), atol=0.02, rtol=0)
    torch.testing.assert_close(noise.real.square().mean(-1), covariance.diag().real / 2, atol=0.02, rtol=0.02)


@pytest.mark.parametrize(('scale', 'acceleration'), [(1.0, 1.0), (3.0, 2.0)])
def test_identity_scale_and_reproducibility(scale, acceleration):
    """Explicit signal scale and acceleration determine the variance ratio."""
    data = torch.full((1, 1, 1, 8), 4 + 3j)

    def estimate():
        return gfactor(
            data,
            lambda x: scale * x,
            data,
            IdentityOp(),
            acceleration=acceleration,
            noise_covariance=torch.eye(1),
            n_replicas=4096,
            generator=torch.Generator().manual_seed(9),
        )

    actual = estimate()
    torch.testing.assert_close(actual, estimate(), rtol=0, atol=0)
    torch.testing.assert_close(actual, torch.full_like(actual, scale / acceleration**0.5), rtol=0.04, atol=0)


def test_batched_noise():
    """Covariance broadcasts over batches without mixing them."""
    data = torch.zeros(2, 1, 1, 1, 256, dtype=torch.complex64)
    covariance = torch.tensor([[[1.0]], [[9.0]]])
    power = torch.zeros(2)
    calls = 0

    def reconstruct(x):
        nonlocal calls
        if calls < 64:
            power.add_(x.abs().square().mean(dim=(-4, -3, -2, -1)) / 64)
        calls += 1
        return x

    gfactor(
        data,
        reconstruct,
        data,
        reconstruct,
        acceleration=1,
        noise_covariance=covariance,
        n_replicas=64,
        generator=torch.Generator().manual_seed(1),
    )
    torch.testing.assert_close(power, torch.tensor([1.0, 9.0]), rtol=0.025, atol=0)


def test_noncartesian_encoding():
    """Unbiased nonuniform multi-coil inverses match explicit covariance propagation."""
    covariance = torch.tensor([[1, 0.2 + 0.3j], [0.2 - 0.3j, 1.5]], dtype=torch.complex128)
    coils = torch.tensor([[1, 0.8j], [0.6j, 1.2]], dtype=torch.complex128).reshape(2, 1, 1, 2)

    def acquisition(frequencies):
        kx = torch.tensor(frequencies, dtype=torch.float64).reshape(1, 1, 1, 1, -1)
        trajectory = KTrajectory(torch.zeros_like(kx), torch.zeros_like(kx), kx)
        fourier = NonUniformFastFourierOp(['x'], [2], [2], trajectory)
        basis = torch.eye(2, dtype=torch.complex128).reshape(2, 1, 1, 1, 2)
        matrix = fourier(basis * coils)[0].reshape(2, -1).T
        inverse = torch.linalg.pinv(matrix)
        torch.testing.assert_close(inverse @ matrix, torch.eye(2, dtype=matrix.dtype))
        data = (matrix @ torch.tensor([2 + 1j, -1j], dtype=matrix.dtype)).reshape(2, 1, 1, -1)

        def reconstruct(x):
            return (inverse @ x.flatten()).reshape(1, 1, 1, 2)

        propagated = inverse @ torch.kron(covariance, torch.eye(len(frequencies))) @ inverse.mH
        return data, reconstruct, propagated.diag().real

    data, reconstruct, variance = acquisition([-0.8, -0.1, 0.35, 0.7])
    full_data, full_reconstruct, full_variance = acquisition([-0.95, -0.7, -0.4, -0.15, 0.1, 0.3, 0.55, 0.85])
    result = gfactor(
        data,
        reconstruct,
        full_data,
        full_reconstruct,
        acceleration=2,
        noise_covariance=covariance,
        n_replicas=4096,
        generator=torch.Generator().manual_seed(12),
    )
    torch.testing.assert_close(result.flatten(), (variance / (2 * full_variance)).sqrt(), rtol=0.04, atol=0)


def test_nonlinear_signal_dependence_and_zero_reference():
    """Nonlinear ratios depend on the baseline; zero reference variance is undefined."""

    def estimate(amplitude, reference):
        data = torch.full((1, 1, 1, 8), complex(amplitude))
        return gfactor(
            data,
            lambda x: x.abs().square(),
            data,
            reference,
            acceleration=1,
            noise_covariance=torch.eye(1),
            n_replicas=2048,
            generator=torch.Generator().manual_seed(3),
        )

    low = estimate(0, IdentityOp())
    high = estimate(10, IdentityOp())
    assert (high > 10 * low).all()
    assert estimate(0, lambda x: torch.zeros_like(x)).isnan().all()


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'acceleration': 0}, 'acceleration'),
        ({'acceleration': float('nan')}, 'acceleration'),
        ({'n_replicas': 1}, 'n_replicas'),
        ({'n_replicas': 2.5}, 'n_replicas'),
        ({'data': torch.zeros(1, 1, 1, 1)}, 'complex'),
        ({'reference_data': torch.zeros(2, 1, 1, 1, dtype=torch.complex64)}, 'batch and coil'),
        ({'reference_data': torch.zeros(1, 1, 1, 1, dtype=torch.complex128)}, 'dtype and device'),
        ({'noise_covariance': torch.eye(2)}, 'coils'),
        ({'noise_covariance': torch.tensor([[1j]])}, 'Hermitian'),
    ],
)
def test_invalid_arguments(kwargs, message):
    """Reject incompatible inputs and undefined normalization."""
    data = torch.zeros(1, 1, 1, 1, dtype=torch.complex64)
    with pytest.raises(ValueError, match=message):
        gfactor(
            kwargs.get('data', data),
            IdentityOp(),
            kwargs.get('reference_data', data),
            IdentityOp(),
            acceleration=kwargs.get('acceleration', 1),
            noise_covariance=kwargs.get('noise_covariance', torch.eye(1)),
            n_replicas=kwargs.get('n_replicas', 2),
        )


@pytest.mark.parametrize('as_tuple', [False, True])
def test_reconstruction_graph_released(as_tuple):
    """Release saved activations before constructing the next replica graph."""
    data = torch.zeros(1, 1, 1, 4, dtype=torch.complex64)
    scale = torch.tensor(2.0, requires_grad=True)
    saved: list[weakref.ReferenceType[torch.Tensor]] = []

    def reconstruct(x):
        assert all(reference() is None for reference in saved)
        intermediate = x * scale
        saved.append(weakref.ref(intermediate))
        image = intermediate.square()
        # Reconstructions may need to differentiate through their own operations.
        torch.autograd.grad(image.real.sum(), scale, retain_graph=True)
        return (image,) if as_tuple else image

    gfactor(
        data,
        reconstruct,
        data,
        reconstruct,
        acceleration=1,
        noise_covariance=torch.eye(1),
        n_replicas=2,
    )
    assert all(reference() is None for reference in saved)


@pytest.mark.parametrize('scale', [1.0, 1e-10])
def test_reject_nonhermitian_covariance(scale):
    """Covariance validation must not depend on scanner noise amplitude."""
    data = torch.zeros(2, 1, 1, 4, dtype=torch.complex64)
    covariance = scale * torch.tensor([[1.0, 5.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match='Hermitian'):
        gfactor(
            data,
            IdentityOp(),
            data,
            IdentityOp(),
            acceleration=1,
            noise_covariance=covariance,
            n_replicas=2,
        )


def test_reject_mismatched_image_shapes():
    """Accelerated and reference outputs must describe the same image grid."""
    data = torch.zeros(1, 1, 1, 4, dtype=torch.complex64)
    with pytest.raises(ValueError, match='same shape'):
        gfactor(
            data,
            IdentityOp(),
            data,
            lambda x: x[..., :2],
            acceleration=1,
            noise_covariance=torch.eye(1),
            n_replicas=2,
        )
