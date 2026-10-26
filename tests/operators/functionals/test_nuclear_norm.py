"""Mathematical checks for the nuclear norm and its proximal maps."""

import pytest
import torch
from mr2.operators.functionals import NuclearNorm


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('shape', [(3, 5), (5, 3), (2, 3, 5), (2, 1, 5, 3)])
@pytest.mark.parametrize('divide_by_n', [False, True])
def test_diagonal(dtype: torch.dtype, shape: tuple[int, ...], divide_by_n: bool):
    """Check sums, soft thresholding and clipping against diagonal formulas."""
    x = torch.zeros(shape, dtype=dtype)
    singular_values = torch.tensor([5.0, 2.0, 0.0], dtype=torch.float64)
    phase = 1j if dtype.is_complex else -1
    x[..., :3, :3] = torch.diag(phase * singular_values)
    target = torch.full_like(x, 0.2 + 0.3j if dtype.is_complex else 0.2)
    weight = torch.arange(1, x[..., :1, :1].numel() + 1, dtype=torch.float64).reshape(*shape[:-2], 1, 1)
    sigma = weight * 0.4
    f = NuclearNorm(target=target, weight=weight, divide_by_n=divide_by_n)
    radius = weight / (shape[-2] * shape[-1]) if divide_by_n else weight
    expected_value = radius * 7
    torch.testing.assert_close(f(x + target)[0], expected_value[..., 0, 0])
    shrunk = torch.relu(singular_values - (sigma * radius)[..., 0])
    clipped = torch.minimum(singular_values, radius[..., 0])
    primal = torch.zeros_like(x)
    dual = torch.zeros_like(x)
    primal[..., :3, :3] = torch.diag_embed(phase * shrunk)
    dual[..., :3, :3] = torch.diag_embed(phase * clipped)
    torch.testing.assert_close(f.prox(x + target, sigma)[0], primal + target)
    torch.testing.assert_close(f.prox_convex_conj(x + sigma * target, sigma)[0], dual)


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('divide_by_n', [False, True])
def test_moreau(dtype: torch.dtype, divide_by_n: bool):
    """Check Moreau for dense rectangular batches, translation and complex weights."""
    generator = torch.Generator().manual_seed(31)
    x = torch.randn(2, 3, 5, dtype=dtype, generator=generator)
    target = torch.randn(3, 5, dtype=dtype, generator=generator)
    weight = torch.tensor([-2.0, 3j], dtype=torch.complex128).reshape(2, 1, 1)
    step = torch.tensor([0.3, 1.7], dtype=torch.float64).reshape(2, 1, 1)
    f = NuclearNorm(target, weight, divide_by_n=divide_by_n)
    primal = f.prox(x, step)[0]
    dual = f.prox_convex_conj(x / step, 1 / step)[0]
    torch.testing.assert_close(primal + step * dual, x)
    expected = torch.linalg.matrix_norm(x - target, ord='nuc') * weight.abs().flatten()
    if divide_by_n:
        expected /= 15
    torch.testing.assert_close(f(x)[0], expected)
    # Existing functional algebra must retain the same scaling.
    torch.testing.assert_close((2 * f).prox(x, step)[0], f.prox(x, 2 * step)[0])


@pytest.mark.parametrize('method', ['forward', 'prox', 'prox_convex_conj'])
@pytest.mark.parametrize('shape', [(2,), (2, 3, 1), (2, 1, 5), (4, 2, 1, 1)])
def test_weight_shape(method: str, shape: tuple[int, ...]):
    """Reject matrix variation, ambiguous batch vectors and output expansion."""
    f = NuclearNorm(weight=torch.ones(shape))
    with pytest.raises(RuntimeError, match='expand'):
        getattr(f, method)(torch.ones(2, 3, 5))


@pytest.mark.parametrize('method', ['prox', 'prox_convex_conj'])
@pytest.mark.parametrize('step', [torch.ones(2), torch.ones(2, 3, 1), torch.ones(4, 2, 1, 1)])
def test_step_shape(method: str, step: torch.Tensor):
    """Reject unsupported steps consistently in both proximal maps."""
    with pytest.raises(RuntimeError, match='expand'):
        getattr(NuclearNorm(), method)(torch.ones(2, 3, 5), step)


@pytest.mark.parametrize('method', ['forward', 'prox', 'prox_convex_conj'])
def test_input_and_target_shape(method: str):
    """Require matrices and disallow target-induced output expansion."""
    with pytest.raises(ValueError, match='at least two'):
        getattr(NuclearNorm(), method)(torch.ones(3))
    with pytest.raises(RuntimeError, match='expand'):
        getattr(NuclearNorm(target=torch.ones(2, 3, 5)), method)(torch.ones(1, 3, 5))


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('method', ['forward', 'prox', 'prox_convex_conj'])
def test_gradients(dtype: torch.dtype, method: str):
    """Check gradients away from repeated/zero singular values and thresholds."""
    x = torch.zeros(3, 5, dtype=dtype)
    x[:, :3] = torch.diag(torch.tensor([4.0, 2.0, 0.5], dtype=dtype))
    x.requires_grad_()
    f = NuclearNorm(weight=1.0)
    assert torch.autograd.gradcheck(getattr(f, method), (x,))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda():
    """Parameters follow the functional onto CUDA."""
    x = torch.randn(2, 3, 5, dtype=torch.complex64)
    step = torch.tensor([0.3, 0.7]).reshape(2, 1, 1)
    f = NuclearNorm(target=x / 2, weight=torch.tensor([1.0, 2.0]).reshape(2, 1, 1))
    expected = (f(x)[0], f.prox(x, step)[0], f.prox_convex_conj(x, step)[0])
    torch.testing.assert_close(NuclearNorm()(x.cuda())[0].cpu(), NuclearNorm()(x)[0])
    actual = (f(x.cuda())[0], f.prox(x.cuda(), step.cuda())[0], f.prox_convex_conj(x.cuda(), step.cuda())[0])
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result.cpu(), reference)
    f = f.cuda()
    actual = (f(x.cuda())[0], f.prox(x.cuda(), step.cuda())[0], f.prox_convex_conj(x.cuda(), step.cuda())[0])
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result.cpu(), reference)


def test_batch_broadcasting():
    """Batch-axis broadcasting agrees with independent matrix calls, including zero steps."""
    x = torch.randn(2, 3, 4, 5, generator=torch.Generator().manual_seed(13))
    weight = torch.tensor([0.0, -2.0, 1.0]).reshape(3, 1, 1)
    step = torch.tensor([0.0, 0.4]).reshape(2, 1, 1, 1)
    target = torch.tensor(0.5)
    f = NuclearNorm(target, weight)
    values, primal, dual = f(x)[0], f.prox(x, step)[0], f.prox_convex_conj(x, step)[0]
    for i in range(2):
        for j in range(3):
            matrix_f = NuclearNorm(target, weight[j, 0, 0])
            torch.testing.assert_close(values[i, j], matrix_f(x[i, j])[0])
            torch.testing.assert_close(primal[i, j], matrix_f.prox(x[i, j], step[i, 0, 0, 0])[0])
            torch.testing.assert_close(dual[i, j], matrix_f.prox_convex_conj(x[i, j], step[i, 0, 0, 0])[0])


@pytest.mark.parametrize('method', ['prox', 'prox_convex_conj'])
def test_step_precision(method: str):
    """Python steps use input precision; tensor steps retain their precision."""
    x = torch.diag(torch.tensor([3.0, 0.5], dtype=torch.float64))
    target = torch.diag(torch.tensor([1.0, -0.25], dtype=torch.float64))
    f = NuclearNorm(target=target)
    expected = x - torch.eye(2, dtype=x.dtype) / 3 if method == 'prox' else torch.diag(x.new_tensor([1.0, 7 / 12]))
    torch.testing.assert_close(getattr(f, method)(x, 1 / 3)[0], expected, atol=1e-15, rtol=1e-15)
    weight = torch.tensor(1.0000000001, dtype=torch.float64)
    sigma = torch.tensor(0.333333333333, dtype=torch.float64)
    f = NuclearNorm(target=target, weight=weight)
    expected = (
        x - sigma * weight * torch.eye(2, dtype=x.dtype)
        if method == 'prox'
        else torch.diag(torch.minimum(x.diag() - sigma * target.diag(), weight))
    )
    torch.testing.assert_close(getattr(f, method)(x, sigma)[0], expected, atol=1e-15, rtol=1e-15)


def test_zero_threshold_identity():
    """Zero weights and steps preserve dense matrices exactly in a mixed batch."""
    x = 100 * torch.randn(3, 4, 5, dtype=torch.complex64, generator=torch.Generator().manual_seed(13))
    weight = torch.tensor([0.0, 1.0, 2.0]).reshape(3, 1, 1)
    step = torch.tensor([1.0, 0.0, 0.5]).reshape(3, 1, 1)
    f = NuclearNorm(target=0.2j, weight=weight)
    torch.testing.assert_close(f.prox(x, step)[0][:2], x[:2], atol=0, rtol=0)
    torch.testing.assert_close(f.prox_convex_conj(x, step)[0][1], x[1], atol=0, rtol=0)
