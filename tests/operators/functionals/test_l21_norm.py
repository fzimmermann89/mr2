"""Independent analytic and regression checks for the group norm."""

from collections.abc import Sequence

import pytest
import torch
from mr2.operators.functionals import L21Norm


@pytest.mark.parametrize(
    ('divide_by_n', 'expected_value', 'expected_prox'), [(False, 5.0, [2.4, 3.2]), (True, 2.5, [2.7, 3.6])]
)
def test_l21_mean(divide_by_n: bool, expected_value: float, expected_prox: list[float]):
    """Mean counts two outer groups, not four group components."""
    x = torch.tensor([[3.0, 0.0], [4.0, 0.0]])
    f = L21Norm(group_dim=0, divide_by_n=divide_by_n)
    torch.testing.assert_close(f(x)[0], torch.tensor(expected_value))
    torch.testing.assert_close(f.prox(x)[0], torch.tensor([expected_prox, [0.0, 0.0]]).T)
    radius = 0.5 if divide_by_n else 1.0
    torch.testing.assert_close(f.prox_convex_conj(x)[0], x * (radius / 5))


@pytest.mark.parametrize('keepdim', [False, True])
@pytest.mark.parametrize('divide_by_n', [False, True])
def test_l21_batches(keepdim: bool, divide_by_n: bool):
    """Two grouping axes, explicit outer axes, batch weights and step sizes."""
    x = torch.zeros(2, 2, 3, 2, dtype=torch.complex128)
    x[:, 0, :, 0] = torch.tensor(3j, dtype=x.dtype)
    x[:, 1, :, 1] = 4
    target = torch.full_like(x, 1 + 2j)
    weight = torch.tensor([1.0, 2.0], dtype=torch.float64).reshape(2, 1, 1, 1)
    sigma = torch.tensor([0.5, 1.0], dtype=torch.float64).reshape(2, 1, 1, 1)
    f = L21Norm(group_dim=(1, -1), dim=2, weight=weight, target=target, keepdim=keepdim, divide_by_n=divide_by_n)
    count = 3 if divide_by_n else 1
    expected_value = torch.tensor([15.0, 30.0], dtype=torch.float64) / count
    if keepdim:
        expected_value = expected_value.reshape(2, 1, 1, 1)
    torch.testing.assert_close(f(x + target)[0], expected_value)
    expected = target + x * (1 - sigma * weight / (5 * count))
    prox = f.prox(x + target, sigma)[0]
    torch.testing.assert_close(prox, expected)
    dual = f.prox_convex_conj((x + target) / sigma, 1 / sigma)[0]
    torch.testing.assert_close(prox + sigma * dual, x + target)


@pytest.mark.parametrize('dim', [None, ()])
@pytest.mark.parametrize('keepdim', [False, True])
def test_l21_single_group(dim: tuple[()] | None, keepdim: bool):
    """Grouping all axes yields one norm and no additional normalization."""
    f = L21Norm(group_dim=(0, 1), dim=dim, keepdim=keepdim, divide_by_n=True)
    value = f(torch.tensor([[3.0, 4.0]]))[0]
    assert value.shape == ((1, 1) if keepdim else ())
    torch.testing.assert_close(value.squeeze(), torch.tensor(5.0))


def test_l21_no_outer_reduction():
    """An explicit empty outer axis list keeps separate group norms."""
    f = L21Norm(group_dim=-1, dim=(), divide_by_n=True)
    x = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
    torch.testing.assert_close(f(x)[0], torch.tensor([5.0, 5.0]))
    torch.testing.assert_close(f.prox(x)[0], x * 0.8)


@pytest.mark.parametrize(
    ('group', 'outer', 'error'),
    [
        ((0, -2), None, IndexError),
        ((0, 0), None, IndexError),
        (0, (-1, 1), IndexError),
        (0, -2, ValueError),
        (2, None, IndexError),
        (-3, None, IndexError),
        (0, 2, IndexError),
    ],
)
def test_l21_axes(group: int | Sequence[int], outer: int | Sequence[int] | None, error: type[Exception]):
    """Reject out-of-range, duplicate, aliased and overlapping axes."""
    with pytest.raises(error):
        L21Norm(group_dim=group, dim=outer)(torch.ones(2, 3))


def test_l21_empty_group_axes():
    """At least one grouping axis is required."""
    with pytest.raises(ValueError, match='nonempty'):
        L21Norm(group_dim=())


@pytest.mark.parametrize('method', ['forward', 'prox', 'prox_convex_conj'])
def test_l21_weight_validation(method: str):
    """Radial formulas require singleton group axes even for dense equal weights."""
    with pytest.raises(ValueError, match=r'weight.*singleton'):
        getattr(L21Norm(group_dim=0, weight=torch.ones(2, 1)), method)(torch.ones(2, 3))


@pytest.mark.parametrize('method', ['prox', 'prox_convex_conj'])
@pytest.mark.parametrize('sigma', [torch.ones(2, 1), torch.ones(1, 2, 3)])
def test_l21_sigma_validation(method: str, sigma: torch.Tensor):
    """Reject within-group varying step sizes and shape expansion."""
    with pytest.raises(ValueError):
        getattr(L21Norm(group_dim=0), method)(torch.ones(2, 3), sigma)


@pytest.mark.parametrize('method', ['prox', 'prox_convex_conj'])
def test_l21_zero_sigma(method: str):
    """Zero steps return the input even outside the conjugate domain."""
    x = torch.tensor([[3 + 4j, 0], [0, 0]])
    f = L21Norm(group_dim=0, weight=torch.tensor([[0.0, 1.0]]), target=2j)
    sigma = torch.tensor([[0.0, 1.0]])
    torch.testing.assert_close(getattr(f, method)(x, sigma)[0][:, 0], x[:, 0])


@pytest.mark.parametrize(('weight', 'sigma'), [(1.0, 0.0), (0.0, 1.0)])
def test_l21_zero_threshold_large_target(weight: float, sigma: float):
    """Zero thresholds preserve the input without subtracting a large target."""
    x = torch.tensor([[3.0, 5.0], [4.0, 12.0]], requires_grad=True)
    f = L21Norm(group_dim=0, target=1e8, weight=torch.tensor([[weight, 1.0]]))
    result = f.prox(x, torch.tensor([[sigma, 1.0]]))[0]
    torch.testing.assert_close(result[:, 0], x[:, 0], rtol=0, atol=0)
    gradient = torch.autograd.grad(result[:, 0].sum(), x)[0]
    torch.testing.assert_close(gradient, torch.tensor([[1.0, 0.0], [1.0, 0.0]]))


def test_l21_zero_groups():
    """Zero groups have exact zero norms and finite proximal derivatives."""
    x = torch.tensor([[0.0, 3.0], [0.0, 4.0]], requires_grad=True)
    f = L21Norm(group_dim=0, weight=0)
    torch.testing.assert_close(f(x)[0], torch.tensor(0.0))
    torch.testing.assert_close(f.prox_convex_conj(x)[0], torch.zeros_like(x))
    torch.testing.assert_close(torch.autograd.grad(f.prox(x)[0].sum(), x)[0], torch.ones_like(x))
    f = L21Norm(group_dim=0)
    torch.testing.assert_close(f.prox(x)[0][:, 0], torch.zeros(2))
    torch.testing.assert_close(torch.autograd.grad(f.prox(x)[0][:, 0].sum(), x)[0], torch.zeros_like(x))


@pytest.mark.parametrize('method', ['forward', 'prox', 'prox_convex_conj'])
def test_l21_empty_axes_sizes(method: str):
    """Empty component axes represent zero norms; an empty outer mean is undefined."""
    f = L21Norm(group_dim=0)
    x = torch.empty(0, 3)
    result = getattr(f, method)(x)[0]
    torch.testing.assert_close(result, torch.tensor(0.0) if method == 'forward' else x)
    with pytest.raises(ValueError, match='zero outer groups'):
        getattr(L21Norm(group_dim=0, divide_by_n=True), method)(torch.empty(2, 0))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_l21_dtype(dtype: torch.dtype):
    """Forward returns the real dtype; proximal mappings preserve the input dtype."""
    x = torch.tensor([[3.0, 4.0], [5.0, 12.0]], dtype=dtype)
    if dtype.is_complex:
        x = x * 1j
    f = L21Norm(group_dim=1, divide_by_n=True, target=0.1, weight=-2.0)
    assert f.prox(x, 0.7)[0].dtype == dtype
    assert f.prox_convex_conj(x)[0].dtype == dtype
    assert f(x)[0].dtype == x.real.dtype


@pytest.mark.parametrize('method', ['forward', 'prox', 'prox_convex_conj'])
def test_l21_autodiff(method: str):
    """Check derivatives of real and complex inputs, targets and group weights."""
    for dtype in (torch.float64, torch.complex128):
        x = torch.tensor([[3.0, 4.0], [5.0, 12.0]], dtype=dtype, requires_grad=True)
        target = torch.tensor(0.1, dtype=dtype, requires_grad=True)
        weight = torch.tensor([[1.1], [1.3]], dtype=torch.float64, requires_grad=True)

        def apply(x: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return getattr(L21Norm(group_dim=1, target=target, weight=weight), method)(x)[0]

        assert torch.autograd.gradcheck(apply, (x, target, weight))
        assert torch.autograd.gradgradcheck(apply, (x, target, weight))


@pytest.mark.cuda
def test_l21_cuda():
    """Module conversion moves target and weight; all outputs remain on device."""
    x = torch.tensor([[3j, 4.0]], device='cuda', dtype=torch.complex128)
    f = L21Norm(group_dim=1, weight=-2.0, target=1j).cuda()
    for method in ('forward', 'prox', 'prox_convex_conj'):
        result = getattr(f, method)(x)[0]
        assert result.device == x.device
        torch.testing.assert_close(result.cpu(), getattr(f.cpu(), method)(x.cpu())[0])
        f.cuda()


@pytest.mark.parametrize('method', ['prox', 'prox_convex_conj'])
@pytest.mark.parametrize('parameter', ['target', 'weight'])
@pytest.mark.parametrize('value', [torch.tensor([[1.0, 2.0]]), torch.ones(1, 1, 2)])
def test_l21_prox_rejects_expansion(method, parameter, value):
    """Proximal maps cannot duplicate input variables along expanded or added axes."""
    x = torch.tensor([[3.0], [4.0]])
    f = L21Norm(group_dim=-2, **{parameter: value})
    assert torch.isfinite(f(x)[0])
    with pytest.raises(ValueError, match='cannot expand'):
        getattr(f, method)(x)


@pytest.mark.parametrize('method', ['prox', 'prox_convex_conj'])
def test_l21_step_precision(method):
    """Python steps use input precision; tensor steps and weights retain their precision."""
    x = torch.tensor([3.0, 4.0], dtype=torch.float64)
    f = L21Norm(group_dim=0)
    if method == 'prox':
        torch.testing.assert_close(f.prox(x, 1 / 3)[0], x * (1 - 1 / 15), atol=1e-15, rtol=1e-15)
    weight = torch.tensor(1.0000000001, dtype=torch.float64)
    sigma = torch.tensor(0.333333333333, dtype=torch.float64)
    f = L21Norm(group_dim=0, weight=weight)
    expected = x * (1 - sigma * weight / 5) if method == 'prox' else x * (weight / 5)
    torch.testing.assert_close(getattr(f, method)(x, sigma)[0], expected, atol=1e-15, rtol=1e-15)
