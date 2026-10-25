"""Rician likelihood values and derivatives, including the physical boundary."""

import pytest
import torch
from mr2.operators.functionals import RicianNegLogLikelihood

# 80-digit mpmath evaluation of the defining I0/I1 expressions, rounded to double.
REFERENCES = [
    (0.0, 2.0, 1.0, 2.0, 0.0, 2.0, -2.0, -1.0),
    (1e-8, 2.0, 1.0, 2.0, -1e-8, 2.0, -2.0, -1.0),
    (
        0.7,
        1.3,
        0.8,
        0.8038534090169546,
        -0.07669366310723812,
        1.4010111044807179,
        0.29046391043766673,
        0.47852418187537293,
    ),
    (
        1.0,
        32.0,
        1.0,
        483.14783710897024,
        -30.495965297183584,
        31.015751084463012,
        -960.0080694056328,
        0.49179529857619214,
    ),
    (
        1.0,
        32.00001,
        1.0,
        483.14814726653105,
        -30.49597529848671,
        31.015761079500077,
        -960.0086894031266,
        0.49179530127159166,
    ),
    (
        1000.0,
        1000.0,
        1.0,
        7.826693687186747,
        0.000500000125000125,
        0.000500000125000125,
        0.99999974999975,
        0.99999949999975,
    ),
    (
        1.0,
        15.999,
        1.0,
        114.78212729637322,
        -14.49064352659498,
        15.030774265479407,
        -223.98671394681006,
        0.48267939463389953,
    ),
    (
        1.0,
        16.0,
        1.0,
        114.79715856962822,
        -14.49164408685056,
        15.03177224457184,
        -224.0167118262989,
        0.48268060050244344,
    ),
    (
        1.0,
        16.001,
        1.0,
        114.81219084086246,
        -14.492644647030778,
        15.032770223921581,
        -224.04671170593843,
        0.48268180620217777,
    ),
    (
        1.0,
        31.999,
        1.0,
        483.1168218576376,
        -30.494965166866915,
        31.01475158077231,
        -959.9460706662662,
        0.49179502902728983,
    ),
    (10000.0, 10000.0, 1.0, 10.129278903930855, 5.0000000125e-5, 5.0000000125e-5, 0.9999999975, 0.999999995),
    (1e6, 1e6, 1.0, 14.734449091168822, 5.00000000000125e-7, 5.00000000000125e-7, 0.99999999999975, 0.9999999999995),
]


@pytest.mark.parametrize(
    'device',
    ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))],
)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('reference', REFERENCES)
def test_reference(reference: tuple[float, ...], dtype: torch.dtype, device: str) -> None:
    """Compare value, all first derivatives and amplitude curvature to references."""
    a, y, s = (torch.tensor(v, dtype=dtype, device=device, requires_grad=True) for v in reference[:3])
    loss = RicianNegLogLikelihood(y, s)(a)[0]
    gradients = torch.autograd.grad(loss, (a, y, s), create_graph=True)
    curvature = torch.autograd.grad(gradients[0], a)[0]
    actual = torch.stack((loss, *gradients, curvature))
    expected = actual.new_tensor(reference[3:])
    tolerance = 3e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(actual[:4], expected[:4], rtol=tolerance, atol=tolerance * 0.01)
    torch.testing.assert_close(actual[4], expected[4], rtol=tolerance, atol=8e-5 if dtype == torch.float32 else 2e-14)
    if reference[0] <= 1e-8 or reference[0] >= 1000:
        torch.testing.assert_close(actual[1], expected[1], rtol=3e-7 if dtype == torch.float32 else 2e-15, atol=0)
    assert loss.dtype == dtype


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_zero_observation(dtype: torch.dtype) -> None:
    """The reduced loss has a finite Gaussian limit and zero observation gradient."""
    a = torch.tensor([0.0, 0.2, 3.0], dtype=dtype, requires_grad=True)
    y = torch.zeros_like(a, requires_grad=True)
    s = torch.tensor(0.7, dtype=dtype, requires_grad=True)
    loss = RicianNegLogLikelihood(y, s)(a)[0]
    expected = (a.square() / (2 * s.square()) + 2 * s.log()).sum()
    torch.testing.assert_close(loss, expected)
    gradients = torch.autograd.grad(loss, (a, y, s), create_graph=True)
    torch.testing.assert_close(gradients[0], a / s.square())
    torch.testing.assert_close(gradients[1], torch.zeros_like(y))
    torch.testing.assert_close(gradients[2], 2 * a.numel() / s - a.square().sum() / s**3)
    # Observation curvature is generally nonzero, despite its zero first derivative.
    curvature = torch.autograd.grad(gradients[1].sum(), y)[0]
    torch.testing.assert_close(curvature, 1 / s.square() - a.square() / (2 * s**4))


@pytest.mark.parametrize('dim', [None, 0, (0, 2), (-1,)])
@pytest.mark.parametrize('mean', [False, True])
@pytest.mark.parametrize('keepdim', [False, True])
def test_broadcast_reduction(dim: int | tuple[int, ...] | None, mean: bool, keepdim: bool) -> None:
    """Weights multiply likelihoods; averaging counts zero weights as elements."""
    a = torch.linspace(0, 3, 6, dtype=torch.float64).reshape(2, 1, 3).requires_grad_()
    y = torch.tensor([0.2, 0.7], dtype=a.dtype).reshape(1, 2, 1).requires_grad_()
    s = torch.tensor([0.6, 1.0, 1.3], dtype=a.dtype, requires_grad=True)
    w = torch.tensor([0.0, 2.0], dtype=a.dtype).reshape(1, 2, 1).requires_grad_()
    actual = RicianNegLogLikelihood(y, s, w, dim, mean, keepdim)(a)[0]
    z = a * y / s.square()
    values = w * ((a.square() + y.square()) / (2 * s.square()) - torch.special.i0(z).log() + 2 * s.log())
    expected = (values.mean if mean else values.sum)(dim=dim, keepdim=keepdim)
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual.sum(), (a, y, s, w)), torch.autograd.grad(expected.sum(), (a, y, s, w)), strict=True
    ):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_higher_derivatives_and_func() -> None:
    """Check joint amplitude/data/noise HVPs, forward AD and batching."""
    a = torch.tensor([0.1, 2.0, 20.0], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([2.0, 1.0, 21.0], dtype=a.dtype, requires_grad=True)
    s = torch.tensor(0.8, dtype=a.dtype, requires_grad=True)

    def loss(a: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return RicianNegLogLikelihood(y, s)(a)[0]

    assert torch.autograd.gradcheck(loss, (a, y, s))
    assert torch.autograd.gradgradcheck(loss, (a, y, s))
    direction = (torch.ones_like(a), torch.ones_like(y), torch.ones_like(s))
    _, hvp, *_ = torch.func.jvp(torch.func.grad(loss, argnums=(0, 1, 2)), (a, y, s), direction)
    gradients = torch.autograd.grad(loss(a, y, s), (a, y, s), create_graph=True)
    expected = torch.autograd.grad(torch.stack([g.sum() for g in gradients]).sum(), (a, y, s))
    for actual, reference in zip(hvp, expected, strict=True):
        torch.testing.assert_close(actual, reference)
    functional = RicianNegLogLikelihood(y.detach(), s.detach())
    batched = torch.stack((a, a * 2))
    torch.testing.assert_close(
        torch.vmap(lambda x: functional(x)[0])(batched), torch.stack([functional(x)[0] for x in batched])
    )
    torch.testing.assert_close(torch.func.jacfwd(lambda x: functional(x)[0])(a), gradients[0])


@pytest.mark.parametrize(('dtype', 'amplitude'), [(torch.float32, 1e20), (torch.float64, 1e160)])
def test_overflowing_bessel_argument(dtype: torch.dtype, amplitude: float) -> None:
    """Equal amplitudes can have finite loss/gradients even when their product overflows."""
    a = torch.tensor(amplitude, dtype=dtype, requires_grad=True)
    loss = RicianNegLogLikelihood(a.detach(), 1.0)(a)[0]
    gradient = torch.autograd.grad(loss, a)[0]
    torch.testing.assert_close(loss, a.log() + 0.5 * torch.log(a.new_tensor(2 * torch.pi)))
    torch.testing.assert_close(gradient, 0.5 / a, rtol=1e-6, atol=0)


@pytest.mark.parametrize(
    ('keyword', 'value'),
    [
        ('target', -1.0),
        ('target', 1j),
        ('weight', -1.0),
        ('weight', 1j),
        ('noise_std', 0.0),
        ('noise_std', -1.0),
        ('noise_std', 1j),
    ],
)
def test_invalid_parameters(keyword: str, value: complex) -> None:
    """Reject invalid magnitude, weight and noise parameters."""
    parameters: dict[str, torch.Tensor | float] = {'target': 1.0, 'noise_std': 1.0, 'weight': 1.0}
    parameters[keyword] = torch.as_tensor(value)
    with pytest.raises(ValueError):
        RicianNegLogLikelihood(parameters['target'], parameters['noise_std'], parameters['weight'])


def test_unsupported_precision() -> None:
    """Low precision amplitudes are explicitly unsupported."""
    with pytest.raises(TypeError, match='float32 and float64'):
        RicianNegLogLikelihood(1.0, 1.0)(torch.ones(1, dtype=torch.float16))


def test_reference_hvp() -> None:
    """Joint amplitude/observation/noise curvature agrees with an 80-digit reference."""
    parameters = torch.tensor([0.7, 1.3, 0.8], dtype=torch.float64, requires_grad=True)
    a, y, s = parameters.unbind()
    loss = RicianNegLogLikelihood(y, s)(a)[0]
    gradient = torch.autograd.grad(loss, parameters, create_graph=True)[0]
    hvp = torch.autograd.grad(gradient.sum(), parameters)[0]
    expected = parameters.new_tensor([1.083195455365814, -2.716898422178374, 1.4486872228308374])
    torch.testing.assert_close(hvp, expected, rtol=2e-12, atol=2e-14)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_large_near_equal(dtype: torch.dtype) -> None:
    """Nearby large amplitudes keep the residual and small Bessel deficit separate."""
    a = torch.tensor(10000.0, dtype=dtype, requires_grad=True)
    y = torch.nextafter(a.detach(), a.new_tensor(float('inf')))
    loss = RicianNegLogLikelihood(y, 1.0)(a)[0]
    gradient = torch.autograd.grad(loss, a, create_graph=True)[0]
    inverse_z = (1 / a) * (1 / y)
    # At z~1e8, remaining q terms change this reference by < 2e-20.
    expected = (a - y) + (0.5 + inverse_z / 8) / a
    torch.testing.assert_close(gradient, expected, rtol=2e-7 if dtype == torch.float32 else 2e-15, atol=0)
    assert torch.isfinite(torch.autograd.grad(gradient, a)[0])


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('scale', [1e-6, 1e6])
def test_noise_scale_equivariance(dtype: torch.dtype, scale: float) -> None:
    """Rescaling all physical units shifts the loss and rescales its derivatives."""
    original = torch.tensor([0.7, 1.3, 0.8], dtype=dtype, requires_grad=True)
    scaled = (original.detach() * scale).requires_grad_()

    def evaluate(parameters: torch.Tensor) -> torch.Tensor:
        a, y, s = parameters.unbind()
        return RicianNegLogLikelihood(y, s)(a)[0]

    value = evaluate(original)
    scaled_value = evaluate(scaled)
    torch.testing.assert_close(scaled_value, value + 2 * original.new_tensor(scale).log())
    gradient = torch.autograd.grad(value, original)[0]
    scaled_gradient = torch.autograd.grad(scaled_value, scaled)[0]
    torch.testing.assert_close(
        scaled_gradient * scale, gradient, rtol=3e-5 if dtype == torch.float32 else 2e-12, atol=0
    )
