import pytest
import torch
from mr2.operators import ProximableFunctional, ProximableFunctionalSeparableSum
from mr2.operators.Functional import ScaledProximableFunctional
from mr2.operators.functionals import L1Norm, L2NormSquared
from mr2.utils import RandomGenerator


class Dummy(ProximableFunctional):
    """Only for testing purposes."""

    def __init__(self, prox_scale: float = 1.0):
        super().__init__()
        self.dummy = torch.tensor(1.0)
        self.prox_scale = prox_scale

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x.max() + self.dummy,)

    def prox(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        """This is not a real proximal operator, just a dummy function."""
        return (self.prox_scale * x * sigma,)

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        """This is not a real proximal operator, just a dummy function."""

        return (-self.prox_scale * x * sigma,)


def test_separablesum_forward():
    a = Dummy()
    b = Dummy()
    rng = RandomGenerator(123)
    x1 = rng.float32_tensor(3)
    x2 = rng.float32_tensor(3)
    summed = ProximableFunctionalSeparableSum(a, b)
    torch.testing.assert_close(summed(x1, x2)[0], a(x1)[0] + b(x2)[0])


@pytest.mark.cuda
def test_separablesum_forward_cuda():
    a = Dummy()
    b = Dummy()
    rng = RandomGenerator(123)
    x1 = rng.float32_tensor(3)
    x2 = rng.float32_tensor(3)
    summed = ProximableFunctionalSeparableSum(a, b)
    torch.testing.assert_close(summed(x1, x2)[0], a(x1)[0] + b(x2)[0])


def test_separablesum_shorthand():
    a = Dummy()
    b = Dummy()

    rng = RandomGenerator(123)
    x1 = rng.float64_tensor(3)
    x2 = rng.float64_tensor(3)
    x3 = rng.float64_tensor(3)
    x4 = rng.float64_tensor(3)

    summed = a | b
    torch.testing.assert_close(summed(x1, x2)[0], a(x1)[0] + b(x2)[0])

    summed2left = b | (a | b)
    torch.testing.assert_close(summed2left(x1, x2, x3)[0], b(x1)[0] + a(x2)[0] + b(x3)[0])

    summed2right = (a | b) | b
    torch.testing.assert_close(summed2right(x1, x2, x3)[0], a(x1)[0] + b(x2)[0] + b(x3)[0])

    summed3 = (a | b) | (b | a)
    torch.testing.assert_close(summed3(x1, x2, x3, x4)[0], a(x1)[0] + b(x2)[0] + b(x3)[0] + a(x4)[0])


def test_ProximableFunctionalSeparableSum_prox():
    rng = RandomGenerator(123)
    xs = rng.float32_tensor(4)
    summed = Dummy(1.0) | (Dummy(2.0) | Dummy(3.0)) | Dummy(4.0)
    expected_prox = tuple(torch.tensor([1.0, 2.0, 3.0, 4.0]) * xs)
    actual_prox = summed.prox(*xs, sigma=1.0)
    torch.testing.assert_close(actual_prox, expected_prox)


def test_ProximableFunctionalSeparableSum_prox_convex_conj():
    rng = RandomGenerator(123)
    xs = rng.float32_tensor(4)
    summed = Dummy(1.0) | (Dummy(2.0) | Dummy(3.0)) | Dummy(4.0)
    expected_proxcc = tuple(torch.tensor([1.0, 2.0, 3.0, 4.0]) * xs)
    actual_prox = summed.prox_convex_conj(*xs, sigma=-1.0)
    torch.testing.assert_close(actual_prox, expected_proxcc)


def test_ProximableFunctionalSeparableSum_iter_order():
    a = Dummy()
    b = Dummy()
    c = Dummy()
    assert tuple(a | b) == (a, b)
    assert tuple(b | c) == (b, c)
    assert tuple((b | c) | (b | c) | b) == (b, c, b, c, b)
    assert tuple((a | c) | (b | c) | b) == (a, c, b, c, b)


def test_summedfunctionls_len():
    a = Dummy()
    b = Dummy()
    assert len(a | b) == 2
    assert len(b | b | b) == 3


@pytest.mark.parametrize('shorthand', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_separablesum_to(shorthand: bool, dtype: torch.dtype, device: str):
    """Device and dtype conversions reach every member, including scaled functionals."""
    a = L2NormSquared(target=torch.ones(3), weight=torch.full((3,), 2.0))
    b = L1Norm(target=torch.full((3,), 2.0), weight=torch.full((3,), 3.0))
    c = L2NormSquared(target=torch.full((3,), 3.0))
    scaled_c = torch.tensor(2.0) * c
    assert isinstance(scaled_c, ScaledProximableFunctional)
    summed = a | (b | scaled_c) if shorthand else ProximableFunctionalSeparableSum(a, b, scaled_c)
    summed.to(dtype).to(device)

    for functional in (a, b, c):
        for tensor in (functional.target, functional.weight):
            assert tensor.dtype == dtype
            assert tensor.device.type == device
    assert scaled_c.scale.dtype == dtype
    assert scaled_c.scale.device.type == device
    assert summed.functionals == (a, b, scaled_c)
    assert summed.functionals[0] is a
    assert summed.functionals[1:] == (b, scaled_c)

    x = torch.arange(3, dtype=dtype, device=device)
    torch.testing.assert_close(summed(x, x, x), (a(x)[0] + b(x)[0] + scaled_c(x)[0],))
    torch.testing.assert_close(summed.prox(x, x, x), (a.prox(x)[0], b.prox(x)[0], scaled_c.prox(x)[0]))
    torch.testing.assert_close(
        summed.prox_convex_conj(x, x, x),
        (a.prox_convex_conj(x)[0], b.prox_convex_conj(x)[0], scaled_c.prox_convex_conj(x)[0]),
    )


def test_separablesum_registered_members():
    """Parameters and state include the members' parameters and buffers."""
    a = L2NormSquared(target=torch.nn.Parameter(torch.ones(3)), weight=torch.nn.Parameter(torch.ones(3)))
    b = L1Norm(target=torch.zeros(3), weight=torch.full((3,), 2.0))
    summed = a | b

    assert {id(parameter) for parameter in summed.parameters()} == {id(a.target), id(a.weight)}
    assert {id(tensor) for tensor in summed.state_dict(keep_vars=True).values()} == {
        id(a.target),
        id(a.weight),
        id(b.target),
        id(b.weight),
    }
