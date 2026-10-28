"""Rician likelihood values and derivatives, including the physical boundary."""

import pytest
import torch
from mr2.operators.functionals import RicianNegLogLikelihood

from tests import autodiff_test

# 80-digit mpmath evaluation of the defining I0/I1 expressions, rounded to double.
# fmt: off
# a, y, s, followed by loss, amplitude/data/noise gradients and amplitude curvature.
REFERENCES = [
    (0.0, 2.0, 1.0, 2.0, 0.0, 2.0, -2.0, -1.0),
    (1e-08, 2.0, 1.0, 2.0, -1e-08, 2.0, -2.0, -1.0),
    (0.7, 1.3, 0.8,
     0.8038534090169546, -0.07669366310723812, 1.4010111044807179, 0.29046391043766673, 0.47852418187537293),
    (1.0, 32.0, 1.0,
     483.14783710897024, -30.495965297183584, 31.015751084463012, -960.0080694056328, 0.49179529857619214),
    (1.0, 32.00001, 1.0,
     483.14814726653105, -30.49597529848671, 31.015761079500077, -960.0086894031266, 0.49179530127159166),
    (1000.0, 1000.0, 1.0,
     7.826693687186747, 0.000500000125000125, 0.000500000125000125, 0.99999974999975, 0.99999949999975),
    (1.0, 15.999, 1.0,
     114.78212729637322, -14.49064352659498, 15.030774265479407, -223.98671394681006, 0.48267939463389953),
    (1.0, 16.0, 1.0,
     114.79715856962822, -14.49164408685056, 15.03177224457184, -224.0167118262989, 0.48268060050244344),
    (1.0, 16.001, 1.0,
     114.81219084086246, -14.492644647030778, 15.032770223921581, -224.04671170593843, 0.48268180620217777),
    (1.0, 31.999, 1.0,
     483.1168218576376, -30.494965166866915, 31.01475158077231, -959.9460706662662, 0.49179502902728983),
    (10000.0, 10000.0, 1.0, 10.129278903930855, 5.0000000125e-05, 5.0000000125e-05, 0.9999999975, 0.999999995),
    (1000000.0, 1000000.0, 1.0,
     14.734449091168822, 5.00000000000125e-07, 5.00000000000125e-07, 0.99999999999975, 0.9999999999995),
]
# fmt: on


@pytest.mark.parametrize(
    'device',
    [
        'cpu',
        pytest.param('cuda', marks=pytest.mark.cuda),
    ],
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
        torch.testing.assert_close(actual[1], expected[1], rtol=5e-7 if dtype == torch.float32 else 2e-15, atol=0)
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


@pytest.mark.parametrize('noise_scale', [1.0, 0.25])
@pytest.mark.parametrize('dim', [None, 0, (0, 2), (-1,)])
@pytest.mark.parametrize('mean', [False, True])
@pytest.mark.parametrize('keepdim', [False, True])
def test_broadcast_reduction(dim: int | tuple[int, ...] | None, mean: bool, keepdim: bool, noise_scale: float) -> None:
    """Weights multiply likelihoods; averaging counts zero weights as elements."""
    a = torch.linspace(0, 3, 6, dtype=torch.float64).reshape(2, 1, 3).requires_grad_()
    y = torch.tensor([0.2, 0.7], dtype=a.dtype).reshape(1, 2, 1).requires_grad_()
    s = (torch.tensor([0.6, 1.0, 1.3], dtype=a.dtype) * noise_scale).requires_grad_()
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
    parameters = torch.tensor([[0.1, 2.0, 20.0], [2.0, 1.0, 21.0], [0.8, 0.8, 0.8]], dtype=torch.float64)
    parameters.requires_grad_()

    def loss(parameters: torch.Tensor) -> torch.Tensor:
        a, y, s = parameters.unbind()
        return RicianNegLogLikelihood(y, s)(a)[0]

    assert torch.autograd.gradcheck(loss, (parameters,))
    assert torch.autograd.gradgradcheck(loss, (parameters,))
    direction = torch.ones_like(parameters)
    _, hvp, *_ = torch.func.jvp(torch.func.grad(loss), (parameters,), (direction,))
    torch.testing.assert_close(hvp, torch.autograd.functional.hvp(loss, parameters, direction)[1])
    torch.testing.assert_close(
        torch.func.hessian(loss)(parameters), torch.autograd.functional.hessian(loss, parameters)
    )

    a, y, s = parameters.detach().unbind()
    functional = RicianNegLogLikelihood(y, s)
    batched = torch.stack((a, a * 2))
    torch.testing.assert_close(
        torch.vmap(lambda x: functional(x)[0])(batched), torch.stack([functional(x)[0] for x in batched])
    )
    autodiff_test(functional, a)


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


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ('parameters', 'expected'),
    [
        (
            (1.0, 2.0, 0.5),
            [
                [3.4604713630996753, -4.010706292567916, 16.085650340543335],
                [-4.010706292567916, 3.8651178407749187, -15.957174829728332],
                [16.085650340543335, -15.957174829728332, 43.80232705194259],
            ],
        ),
        (
            (1.0, 2.5, 0.5),
            [
                [3.470161239704864, -4.006334807937439, 24.063348079374382],
                [-4.006334807937439, 3.915225798352778, -23.974660768250246],
                [24.063348079374382, -23.974660768250246, 103.8586216061148],
            ],
        ),
        (
            (2.0, 4.0, 0.5),
            [
                [3.872948824644048, -4.000521249825924, 32.00833999721478],
                [-4.000521249825924, 3.968237206161012, -31.99583000139261],
                [32.00833999721478, -31.99583000139261, 187.96555764481306],
            ],
        ),
    ],
)
def test_full_hessian(parameters: tuple[float, ...], expected: list[list[float]], dtype: torch.dtype) -> None:
    """Check noise and mixed curvature against 80-digit references at z=8,10,32."""
    p = torch.tensor(parameters, dtype=dtype, requires_grad=True)
    hessian = torch.autograd.functional.hessian(lambda p: RicianNegLogLikelihood(p[1], p[2])(p[0])[0], p)
    tolerance = 5e-4 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(hessian, p.new_tensor(expected), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    ('name', 'invalid'),
    [(name, value) for name in ('target', 'weight', 'noise_std') for value in (float('nan'), float('inf'), -1.0)]
    + [('noise_std', 0.0)],
)
def test_updated_parameters(name: str, invalid: float) -> None:
    """Reject invalid constructor values and later in-place optimizer updates."""
    parameters = {key: torch.tensor([1.0, 2.0], requires_grad=True) for key in ('target', 'weight', 'noise_std')}
    functional = RicianNegLogLikelihood(parameters['target'], parameters['noise_std'], parameters['weight'])
    with torch.no_grad():
        parameters[name][1] = invalid
    with pytest.raises(ValueError, match=name):
        functional(torch.ones(2))
    with pytest.raises(ValueError, match=name):
        RicianNegLogLikelihood(parameters['target'], parameters['noise_std'], parameters['weight'])


@pytest.mark.parametrize('kind', ['constant', 'parameter'])
@pytest.mark.parametrize(
    'device',
    [
        'cpu',
        pytest.param('cuda', marks=pytest.mark.cuda),
    ],
)
def test_tensor_moves(kind: str, device: str) -> None:
    """Constants move, and learnable Parameters keep receiving optimizer updates."""
    values = [torch.tensor([1.0, 2.0]) for _ in range(3)]
    if kind == 'parameter':
        values = [torch.nn.Parameter(value) for value in values]
    target, noise, weight = values
    functional = RicianNegLogLikelihood(target, noise, weight).to(device=device, dtype=torch.float64)
    moved = [getattr(functional, name) for name in ('target', 'noise_std', 'weight')]
    assert all(value.device.type == device and value.dtype == torch.float64 for value in moved)
    assert all(isinstance(value, torch.nn.Parameter) == (kind == 'parameter') for value in moved)
    loss = functional(torch.ones(2, device=device, dtype=torch.float64))[0]
    if kind == 'parameter':
        optimizer = torch.optim.SGD(functional.parameters(), lr=0.01)
        loss.backward()
        assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in moved)
        before = functional.noise_std.detach().clone()
        optimizer.step()
        assert not torch.equal(functional.noise_std, before)
        optimizer.zero_grad()
        functional(torch.ones(2, device=device, dtype=torch.float64))[0].backward()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_signed_extension(dtype: torch.dtype) -> None:
    """Negative optimizer overshoots retain the finite, even likelihood extension."""
    a = torch.tensor([0.0, 1e-8, 5.0, 50.0, 1000.0], dtype=dtype, requires_grad=True)
    functional = RicianNegLogLikelihood(2.0, 1.0)
    positive, negative = functional(a)[0], functional(-a)[0]
    torch.testing.assert_close(positive, negative)
    gp = torch.autograd.grad(positive, a, create_graph=True)[0]
    gn = torch.autograd.grad(negative, a, create_graph=True)[0]
    torch.testing.assert_close(gp, gn)
    torch.testing.assert_close(torch.autograd.grad(gp.sum(), a)[0], torch.autograd.grad(gn.sum(), a)[0])


def test_leaf_noise_updates() -> None:
    """Ordinary leaf noise retains identity and follows repeated optimizer steps."""
    noise = torch.tensor(1.0, requires_grad=True)
    functional = RicianNegLogLikelihood(2.0, noise)
    optimizer = torch.optim.SGD([noise], lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        functional(torch.tensor(1.0))[0].backward()
        optimizer.step()
        assert functional.noise_std is noise
