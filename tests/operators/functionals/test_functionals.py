from copy import deepcopy
from typing import Literal

import pytest
import torch
from mr2.operators.Functional import ElementaryFunctional, ElementaryProximableFunctional
from mr2.operators.functionals import MSE, L1Norm, L1NormViewAsReal, L2NormSquared, L21Norm, ZeroFunctional
from mr2.utils import RandomGenerator
from mr2.utils.reshape import normalize_indices
from typing_extensions import TypedDict

from tests.operators.functionals.conftest import (
    FUNCTIONALS,
    PROXIMABLE_FUNCTIONALS,
    FunctionalTestCase,
    functional_test_cases,
    make_functional,
)


@pytest.mark.parametrize('shape', [[1, 2, 3], ()])
@pytest.mark.parametrize('dim', [None, -2])
@pytest.mark.parametrize('keepdim', [True, False])
@pytest.mark.parametrize('functional', FUNCTIONALS)
def test_functional_shape(functional: type[ElementaryFunctional], shape: torch.Size, dim: int | None, keepdim: bool):
    """Test the shape returned by the forward of the functional."""
    f = make_functional(functional, dim=dim, keepdim=keepdim)
    x = torch.ones(shape)

    if (functional is L21Norm and not shape) or (dim is not None and (dim >= len(shape) or dim < -len(shape))):
        # dim is out of bounds
        # easier to check here than to remove the cases from the parametrization
        with pytest.raises(IndexError):
            f(x)
        return

    (fx,) = f(x)

    reduced = set(range(len(shape))) if dim is None else set(normalize_indices(len(shape), dim))
    if functional is L21Norm:
        reduced.add(len(shape) - 1)
    expected_shape = (
        tuple(1 if i in reduced else size for i, size in enumerate(shape))
        if keepdim
        else tuple(size for i, size in enumerate(shape) if i not in reduced)
    )
    assert fx.shape == expected_shape


@pytest.mark.parametrize('shape', [[2, 3, 4], ()])
@pytest.mark.parametrize('dim', [None, (0, -2)])
@pytest.mark.parametrize('keepdim', [True, False])
@pytest.mark.parametrize('functional', FUNCTIONALS)
def test_functional_divide_by_n(
    functional: type[ElementaryFunctional], shape: torch.Size, dim: tuple[int, ...] | None, keepdim: bool
):
    """Test if divide_by_n scales by number of elements indexed by dim."""

    f_mean = make_functional(functional, divide_by_n=True, dim=dim, keepdim=keepdim)
    f_sum = make_functional(functional, divide_by_n=False, dim=dim, keepdim=keepdim)
    x = RandomGenerator(13).float32_tensor(shape)

    if (functional is L21Norm and not shape) or (
        dim is not None and any(d >= len(shape) or d < -len(shape) for d in dim)
    ):
        # dim is out of bounds
        # easier to check here than to remove the cases from the parametrization
        with pytest.raises(IndexError):
            f_mean(x)
        with pytest.raises(IndexError):
            f_sum(x)
        return

    outer = tuple(range(len(shape))) if dim is None else normalize_indices(len(shape), dim)
    if functional is L21Norm:
        outer = tuple(d for d in outer if d != len(shape) - 1)
    n = 1
    for d in outer:
        n *= shape[d]
    torch.testing.assert_close(f_mean(x)[0], f_sum(x)[0] / n)


@functional_test_cases
def test_functional_moreau(case: FunctionalTestCase):
    """Test if Moreua identity holds."""
    functional = case.functional
    x = case.rand_x()
    (prox,) = functional.prox(x, sigma=case.sigma)
    (prox_convex_conj,) = functional.prox_convex_conj(x / case.sigma, 1.0 / case.sigma)
    x_new = prox + case.sigma * prox_convex_conj
    torch.testing.assert_close(x.to(case.result_dtype), x_new, rtol=1e-3, atol=1e-3)


@functional_test_cases
def test_functional_prox_optimality(case: FunctionalTestCase):
    """Use autograd to check if prox criterion is minimized."""
    functional = case.functional
    x = case.rand_x()

    (prox,) = functional.prox(x, sigma=case.sigma)

    def prox_criterion(p: torch.Tensor) -> torch.Tensor:
        diff = x - p
        l2 = torch.sum((diff * diff.conj()).real, dim=functional.dim, keepdim=functional.keepdim)
        return (case.sigma * functional(p)[0] + 1 / 2 * l2).sum()

    for perturbation in (0, 1e-3, 0.1):
        p = (prox + perturbation * case.rng.rand_like(prox)).requires_grad_()
        optim = torch.optim.SGD([p], lr=1e-2)
        for _ in range(200):
            optim.zero_grad()
            prox_criterion(p).backward()
            optim.step()
            optim.param_groups[0]['lr'] *= 0.97

        assert prox_criterion(p) + 1e-5 >= prox_criterion(prox)


@functional_test_cases
def test_functional_shift(case: FunctionalTestCase):
    """Test if target shift is correct."""
    # translation property
    # \mathrm{prox}_{\sigma F( \cdot - s)}(x) = s + \mathrm{prox}_{\sigma F}(x - s)
    shift = case.rand_x()
    x = case.rand_x()
    functional = case.functional
    prox_target = functional.prox(x - shift, sigma=case.sigma)[0] + shift
    value_target = functional(x - shift)[0]
    shifted_functional = deepcopy(functional)
    shifted_functional.target = functional.target + shift
    prox_shifted = shifted_functional.prox(x, sigma=case.sigma)[0]
    value_shifted = shifted_functional(x)[0]
    torch.testing.assert_close(prox_shifted, prox_target)
    torch.testing.assert_close(value_shifted, value_target)


@functional_test_cases
def test_functional_scaling(case: FunctionalTestCase):
    """Test if scaling with scalar weight is correct"""
    # scaling property
    # \mathrm{prox}_{\sigma F(\alpha \,\cdot)}(x)=\frac{1}{\alpha}\mathrm{prox}_{\sigma \alpha^2 F(\cdot)}(\alpha x)
    alpha = case.rng.float32_tensor(1, 1, 2)
    functional = case.functional
    x = case.rand_x()
    fx_target = functional(alpha * x)
    prox_target = 1 / alpha * functional.prox(alpha * x, sigma=alpha**2 * case.sigma)[0]
    scaled_functional = deepcopy(functional)
    scaled_functional.weight = functional.weight * alpha
    scaled_functional.target = functional.target / alpha
    fx_scaled = scaled_functional(x)
    prox_scaled = scaled_functional.prox(x, sigma=case.sigma)[0]
    torch.testing.assert_close(prox_scaled, prox_target)
    torch.testing.assert_close(fx_scaled, fx_target)


@functional_test_cases
def test_functional_prox_non_expansive(case: FunctionalTestCase):
    """Tests |function(x) - function(y)|_2 <= |x-y|_2"""
    f = case.functional
    x = case.rand_x()
    y = case.rand_x()

    def l2(x):
        return torch.sum((x * x.conj()).real, dim=case.functional.dim, keepdim=case.functional.keepdim)

    eps = torch.finfo(x.real.dtype).eps
    # Squared norms are subject to rounding errors relative to the magnitude of the inputs.
    tolerance = eps * (l2(x) + l2(y))
    assert (l2(f.prox(x)[0] - f.prox(y)[0]) <= l2(x - y) + tolerance).all()


@functional_test_cases
def test_functional_prox_zero_weight(case: FunctionalTestCase):
    """Test if scaling with scalar zero weight is correct"""
    functional = case.functional
    functional.weight *= 0
    x = case.rand_x()
    x2 = case.rand_x()
    torch.testing.assert_close(functional(x), functional(x2))  # independent of x if weight is zero
    (prox,) = functional.prox(x, sigma=case.sigma)
    torch.testing.assert_close(prox, x.to(case.result_dtype))  # prox is identity if weight is zero


@pytest.mark.parametrize('function', ['prox', 'prox_convex_conj'])
@pytest.mark.parametrize('functional', PROXIMABLE_FUNCTIONALS)
def test_functional_prox_zero_sigma(
    functional: type[ElementaryProximableFunctional], function: Literal['prox', 'prox_convex_conj']
):
    """Test prox with sigma=0"""
    p = getattr(make_functional(functional), function)
    x = torch.tensor([-1.0, 0.0, 1.0])
    (prox,) = p(x, sigma=0.0)
    assert not torch.any(torch.isnan(prox))
    assert not torch.any(torch.isinf(prox))
    assert torch.allclose(prox, x)


@pytest.mark.parametrize('function', ['prox', 'prox_convex_conj'])
@pytest.mark.parametrize('functional', PROXIMABLE_FUNCTIONALS)
def test_functional_negative_sigma(
    functional: type[ElementaryProximableFunctional], function: Literal['prox', 'prox_convex_conj']
):
    """Test if error is raised for negative or complex sigma"""

    def case(sigma):
        x = torch.zeros(2)
        f = getattr(make_functional(functional, keepdim=True), function)
        with pytest.raises(ValueError):
            f(x, sigma)

    # all these cases should throw an exception:
    case(sigma=-1)  # negative
    case(sigma=1j)  # complex
    case(sigma=torch.tensor((1, -1)))  # one element negative
    case(sigma=torch.tensor(1.0, dtype=torch.complex64))  # complex dtype


class NumericCase(TypedDict):
    """The expected values are taken from the ODL implementation of the functionals.
    You can use the following code to generate the expected values:

    import odl
    import torch
    torch.manual_seed(42)
    torch.set_printoptions(precision=6)
    functional=odl.solvers.L1Norm # set functional here
    weight,sigma=2,0.5
    space=odl.rn((1,2,3))
    x=torch.arange(-3,3,1).reshape((1,2,3)).to(torch.float32)
    target=torch.randn(1,2,3).round(decimals=2)
    f=(functional(space)*weight).translated(target)
    {'x':x, 'weight':weight, 'target':target, 'sigma':sigma,
    'fx_expected':torch.tensor(f(x)), 'prox_expected:torch.tensor(f.proximal(sigma)(x)),
    'prox_convex_conj_expected':torch.tensor(f.convex_conj.proximal(sigma)(x)))
    }
    """

    functional: type[ElementaryProximableFunctional]
    x: torch.Tensor
    weight: complex | torch.Tensor
    target: torch.Tensor
    sigma: float
    fx_expected: torch.Tensor
    prox_expected: torch.Tensor
    prox_convex_conj_expected: torch.Tensor


# This is more readable than using pytest.mark.parametrize directly
NUMERICCASES: dict[str, NumericCase] = {  # Name: Case
    'L1Norm real odl': {
        # Generated with ODL
        'functional': L1Norm,
        'x': torch.tensor([[[-3.0, -2.0, -1.0], [0.0, 1.0, 2.0]]]),
        'weight': 2.0,
        'target': torch.tensor([[[0.340, 0.130, 0.230], [0.230, -1.120, -0.190]]]),
        'sigma': 0.5,
        'fx_expected': torch.tensor(22.480),
        'prox_expected': torch.tensor([[[-2.0, -1.0, 0.0], [0.230, 0.0, 1.0]]]),
        'prox_convex_conj_expected': torch.tensor([[[-2.0, -2.0, -1.115], [-0.115, 1.560, 2.0]]]),
    },
    'ZeroFunctional with complex weight': {
        'functional': ZeroFunctional,
        'x': torch.tensor([1.0, 2.0, 3.0]),
        'weight': 1j,
        'target': torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        'sigma': 0.5,
        'fx_expected': torch.tensor(0.0),
        'prox_expected': torch.tensor([1.0 + 0j, 2.0 + 0j, 3.0 + 0j]),
        'prox_convex_conj_expected': torch.tensor([0j, 0j, 0j]),
    },
    'L1NormViewAsReal real': {
        # The same as the L1Norm case
        'functional': L1NormViewAsReal,
        'x': torch.tensor([[[-3.0, -2.0, -1.0], [0.0, 1.0, 2.0]]]),
        'weight': 2.0,
        'target': torch.tensor([[[0.340, 0.130, 0.230], [0.230, -1.120, -0.190]]]),
        'sigma': 0.5,
        'fx_expected': torch.tensor(22.480),
        'prox_expected': torch.tensor([[[-2.0, -1.0, 0.0], [0.230, 0.0, 1.0]]]),
        'prox_convex_conj_expected': torch.tensor([[[-2.0, -2.0, -1.115], [-0.115, 1.560, 2.0]]]),
    },
    'L1NormViewAsReal complex': {
        # The same as the real case, weight, target and x are complex
        'functional': L1NormViewAsReal,
        'x': torch.tensor([-3.0 + 0j, -2.0 + 1.0j, -1.0 + 2.0j]),
        'weight': 2.0 + 2.0j,
        'target': torch.tensor([0.340 + 0.230j, 0.130 - 1.120j, 0.230 - 0.190j]),
        'sigma': 0.5,
        'fx_expected': torch.tensor(22.480),
        'prox_expected': torch.tensor([-2.0 + 0.230j, -1.0 + 0.0j, 0.0 + 1.0j]),
        'prox_convex_conj_expected': torch.tensor([-2.0 - 0.115j, -2.0 + 1.560j, -1.115 + 2.0j]),
    },
    'L2NormSquared real odl': {
        # Generated with ODL
        'functional': L2NormSquared,
        'x': torch.tensor([[[-3.0, -2.0, -1.0], [0.0, 1.0, 2.0]]]),
        'weight': 2.0,
        'target': torch.tensor([[[0.340, 0.130, 0.230], [0.230, -1.120, -0.190]]]),
        'sigma': 0.5,
        'fx_expected': torch.tensor(106.195198),
        'prox_expected': torch.tensor([[[-0.328, -0.296, -0.016], [0.184, -0.696, 0.248]]]),
        'prox_convex_conj_expected': torch.tensor(
            [[[-2.983529, -1.943529, -1.049412], [-0.108235, 1.468235, 1.971765]]]
        ),
    },
    'L21Norm complex groups': {
        # Two translated complex groups with norms 5 and 10, weighted by 2.
        'functional': L21Norm,
        'x': torch.tensor([[1 + 3j, 4 + 2j], [7 + 1j, 2 + 8j]]),
        'weight': 2.0,
        'target': torch.tensor([[1 + 0j, 0 + 2j], [1 + 1j, 2 + 0j]]),
        'sigma': 0.5,
        'fx_expected': torch.tensor(30.0),
        'prox_expected': torch.tensor([[1 + 2.4j, 3.2 + 2j], [6.4 + 1j, 2 + 7.2j]]),
        # Shift by sigma*target, then project each row onto the radius-two ball.
        'prox_convex_conj_expected': torch.tensor([[0.5 + 3j, 4 + 1j], [6.5 + 0.5j, 1 + 8j]])
        * (2 / torch.tensor([[26.25], [107.5]]).sqrt()),
    },
    'MSE': {
        # Generated with ODL
        'functional': MSE,
        'x': torch.tensor([[[-3.0, -2.0, -1.0], [0.0, 1.0, 2.0]]]),
        'weight': 2.0,
        'target': torch.tensor([[[0.340, 0.130, 0.230], [0.230, -1.120, -0.190]]]),
        'sigma': 0.5,
        'fx_expected': torch.tensor(17.6992),
        'prox_expected': torch.tensor([[[-1.6640, -1.1480, -0.5080], [0.0920, 0.1520, 1.1240]]]),
        'prox_convex_conj_expected': torch.tensor(
            [[[-2.305455, -1.501818, -0.810909], [-0.083636, 1.134545, 1.523636]]]
        ),
    },
}


@pytest.mark.parametrize('case_name', NUMERICCASES.keys())
def test_functional_values(case_name: str):
    """Test if functional values match expected values."""
    _test_functional_values(**NUMERICCASES[case_name])


def _test_functional_values(
    functional: type[ElementaryProximableFunctional],
    x: torch.Tensor,
    weight: complex | torch.Tensor,
    target: torch.Tensor,
    sigma: float,
    fx_expected: torch.Tensor,
    prox_expected: torch.Tensor,
    prox_convex_conj_expected: torch.Tensor,
):
    f = make_functional(functional, weight=weight, target=target, dim=None, keepdim=False)
    (fx,) = f(x)
    (prox,) = f.prox(x, sigma=sigma)
    (prox_convex_conj,) = f.prox_convex_conj(x, sigma=sigma)
    torch.testing.assert_close(fx, fx_expected)
    torch.testing.assert_close(prox, prox_expected)
    torch.testing.assert_close(prox_convex_conj, prox_convex_conj_expected)


@pytest.mark.parametrize('functional', PROXIMABLE_FUNCTIONALS)
def test_functional_has_testcase(functional):
    """Check if there is at least one test case for each functional."""
    cases = [k for k in NUMERICCASES if NUMERICCASES[k]['functional'] == functional]
    assert len(cases), f'No test case found for {functional.__name__}!'


@pytest.mark.cuda
@pytest.mark.parametrize('functional', FUNCTIONALS)
@pytest.mark.parametrize('parameters', ['scalar', 'none', 'tensor'])
def test_functional_cuda_forward(
    functional: type[ElementaryFunctional], parameters: Literal['scalar', 'none', 'tensor']
):
    """Test if the forward pass works on the GPU."""
    match parameters:
        case 'scalar':
            f = make_functional(functional, weight=1.0, target=0.0)
        case 'none':
            f = make_functional(functional)
        case 'tensor':
            f = make_functional(functional, weight=torch.tensor(1.0), target=torch.tensor(0.0))
    x = (torch.ones(1) if functional is L21Norm else torch.tensor(1.0)).cuda()
    f.cuda()
    (fx,) = f(x)
    assert fx.is_cuda


@pytest.mark.cuda
@pytest.mark.parametrize('parameters', ['scalar', 'none', 'tensor'])
@pytest.mark.parametrize('functional', PROXIMABLE_FUNCTIONALS)
def test_functional_cuda_prox(
    functional: type[ElementaryProximableFunctional], parameters: Literal['scalar', 'none', 'tensor']
):
    """Test if prox and prox_convex_conj work on the GPU."""
    x = (torch.ones(1) if functional is L21Norm else torch.tensor(1.0)).cuda()
    match parameters:
        case 'scalar':
            f = make_functional(functional, weight=1.0, target=0.0).cuda()
            (prox,) = f.prox(x, sigma=1)
            (prox_convex_conj,) = f.prox_convex_conj(x, sigma=1)
        case 'none':
            f = make_functional(functional).cuda()
            (prox,) = f.prox(x)
            (prox_convex_conj,) = f.prox_convex_conj(x)
        case 'tensor':
            f = make_functional(functional, weight=torch.tensor(1.0), target=torch.tensor(0.0)).cuda()
            sigma = torch.tensor(1.0).cuda()
            (prox,) = f.prox(x, sigma)
            (prox_convex_conj,) = f.prox_convex_conj(x, sigma)

    assert prox.is_cuda
    assert prox_convex_conj.is_cuda


@pytest.mark.parametrize('functional', FUNCTIONALS)
def test_functional_scalar_arguments_forward(functional: type[ElementaryFunctional]):
    """Test if the forward pass works with scalar weight and target."""
    x = torch.ones(1) if functional is L21Norm else torch.tensor(1.0)
    f_scalar = make_functional(functional, weight=1.0, target=0.0)
    f_tensor = make_functional(functional, weight=torch.tensor(1.0), target=torch.tensor(0.0))
    (fx_scalar,) = f_scalar(x)
    (fx_tensor,) = f_tensor(x)
    assert torch.allclose(fx_scalar, fx_tensor)


@pytest.mark.parametrize('function', ['prox', 'prox_convex_conj'])
@pytest.mark.parametrize('functional', PROXIMABLE_FUNCTIONALS)
def test_functional_scalar_arguments_prox(
    functional: type[ElementaryProximableFunctional], function: Literal['prox', 'prox_convex_conj']
):
    """Test if prox and prox_convex_conj work with scalar weight and target."""
    x = torch.ones(1) if functional is L21Norm else torch.tensor(1.0)
    p_scalar = getattr(make_functional(functional, weight=1.0, target=0.0), function)
    p_tensor = getattr(make_functional(functional, weight=torch.tensor(1.0), target=torch.tensor(0.0)), function)
    (px_scalar,) = p_scalar(x, sigma=1.0)
    (px_tensor,) = p_tensor(x, sigma=torch.tensor(1.0))
    assert torch.allclose(px_scalar, px_tensor)


@pytest.mark.parametrize('functional', FUNCTIONALS)
def test_functional_default_arguments_forward(functional: type[ElementaryFunctional]):
    """Test if the forward pass works with default weight and target."""
    x = torch.ones(1) if functional is L21Norm else torch.tensor(1.0)
    f_default = make_functional(functional)
    f_tensor = make_functional(functional, weight=torch.tensor(1.0), target=torch.tensor(0.0))
    (fx_default,) = f_default(x)
    (fx_tensor,) = f_tensor(x)
    assert torch.allclose(fx_default, fx_tensor)


@pytest.mark.parametrize('function', ['prox', 'prox_convex_conj'])
@pytest.mark.parametrize('functional', PROXIMABLE_FUNCTIONALS)
def test_functional_default_arguments_prox(
    functional: type[ElementaryProximableFunctional], function: Literal['prox', 'prox_convex_conj']
):
    """Test if prox and prox_convex_conj work with default weight and target."""
    x = torch.ones(1) if functional is L21Norm else torch.tensor(1.0)
    p_default = getattr(make_functional(functional), function)
    p_tensor = getattr(make_functional(functional, weight=torch.tensor(1.0), target=torch.tensor(0.0)), function)
    (px_default,) = p_default(x)
    (px_tensor,) = p_tensor(x, sigma=torch.tensor(1.0))
    assert torch.allclose(px_default, px_tensor)
