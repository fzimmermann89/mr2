"""Mathematical and regression tests for dictionary denoising recipes."""

from typing import Literal

import pytest
import torch
from mr2.algorithms import analysis_denoising, synthesis_denoising
from mr2.data import IData
from mr2.operators import ConvAnalysisDictionaryOp, ConvSynthesisDictionaryOp, FiniteDifferenceOp, IdentityOp


@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_identity(denoise, dtype: torch.dtype) -> None:
    """Identity dictionaries have the magnitude soft-threshold solution, including weighted complex data."""
    image = torch.tensor([[-3, 0, 2, 5], [4, -2, 0.1, -1]], dtype=dtype)
    if image.is_complex():
        image = image * (0.6 + 0.8j)
    weight = torch.tensor([[0.0], [0.7]], dtype=image.real.dtype)
    expected = torch.sgn(image) * (image.abs() - weight).clamp_min(0)
    result = denoise(image, IdentityOp(), weight, max_iterations=150)
    torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-6)
    assert result.dtype == dtype


@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
@pytest.mark.parametrize('spatial_dim', [1, 2, 3])
@pytest.mark.parametrize(
    ('image_dtype', 'kernel_dtype'),
    [
        (torch.float64, torch.float32),
        (torch.float32, torch.float64),
        (torch.complex128, torch.float64),
        (torch.float64, torch.complex64),
    ],
)
def test_convolution_promotion(denoise, spatial_dim: int, image_dtype: torch.dtype, kernel_dtype: torch.dtype) -> None:
    """Codes and images use image/kernel promotion with arbitrary leading batch axes."""
    image = torch.full((2, 1, *([3] * spatial_dim)), 2.0, dtype=image_dtype)
    kernel = torch.ones((1, *([1] * spatial_dim)), dtype=kernel_dtype)
    operator = (ConvAnalysisDictionaryOp if denoise is analysis_denoising else ConvSynthesisDictionaryOp)(kernel)
    result = denoise(image, operator, 0.5, max_iterations=80)
    assert result.shape == image.shape
    assert result.dtype == torch.promote_types(image_dtype, kernel_dtype)
    torch.testing.assert_close(result, torch.full_like(result, 1.5), atol=1e-6, rtol=1e-6)
    assert operator.kernel is kernel
    assert operator.kernel.dtype == kernel_dtype


def test_analysis_difference_reference() -> None:
    """A two-pixel difference penalty has an independent fused-pair solution."""
    image = torch.tensor([[0.0, 4.0]], dtype=torch.float64)
    result = analysis_denoising(image, FiniteDifferenceOp(dim=(-1,), pad_mode='zeros'), 0.5, max_iterations=200)
    torch.testing.assert_close(result, torch.tensor([[0.5, 3.0]], dtype=image.dtype))


@pytest.mark.parametrize('low_pass', [False, True])
def test_synthesis_baseline(low_pass: bool) -> None:
    """Baseline agrees with a dense normal equation and is explicitly restored after residual shrinkage."""
    image = torch.tensor([[0.0, 3.0, -2.0, 1.0]], dtype=torch.float64)
    difference = FiniteDifferenceOp(dim=(-1,), pad_mode='circular') * 1.3
    basis = torch.eye(4, dtype=image.dtype)
    matrix = difference(basis)[0].reshape(4, 4).T
    baseline = torch.linalg.solve(torch.eye(4, dtype=image.dtype) + matrix.T @ matrix, image.flatten()).reshape_as(
        image
    )
    weight = 0.4
    residual = image - baseline if low_pass else image
    expected = torch.sgn(residual) * (residual.abs() - weight).clamp_min(0)
    if low_pass:
        expected += baseline
    result = synthesis_denoising(
        image,
        IdentityOp(),
        weight,
        low_pass_operator=difference if low_pass else None,
        max_iterations=50,
        tolerance_low_pass=1e-12,
    )
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
@pytest.mark.parametrize('weight', [0.0, 0.5])
def test_zero_dictionary(denoise, weight: float) -> None:
    """Zero dictionaries are finite and have the correct image-space limits."""
    image = torch.ones(2, 3, 4, dtype=torch.float64)
    operator = (ConvAnalysisDictionaryOp if denoise is analysis_denoising else ConvSynthesisDictionaryOp)(
        torch.zeros(2, 1, 1, dtype=image.dtype)
    )
    result = denoise(image, operator, weight, max_iterations=8)
    expected = image if denoise is analysis_denoising else torch.zeros_like(image)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
@pytest.mark.parametrize('weight', [-1.0, float('inf'), float('nan'), 1j, torch.ones(2, 1, 3, 4)])
def test_invalid_weights(denoise, weight) -> None:
    """Weights cannot be negative/nonfinite/complex or expand the coefficient tensor."""
    with pytest.raises(ValueError, match='regularization_weight'):
        denoise(torch.ones(1, 3, 4), IdentityOp(), weight)


@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
def test_incompatible_weights(denoise) -> None:
    """Broadcast failures produce an actionable error."""
    with pytest.raises(ValueError, match='regularization_weight'):
        denoise(torch.ones(1, 3, 4), IdentityOp(), torch.ones(5))


@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
def test_weight_conversion_overflow(denoise) -> None:
    """Finite double-precision weights must remain finite after conversion to image precision."""
    with pytest.raises(ValueError, match='finite in the coefficient dtype'):
        denoise(torch.ones(1, 3, 4), IdentityOp(), torch.tensor(1e40, dtype=torch.float64))


@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
def test_even_kernel(denoise) -> None:
    """Reject the operators' unsupported even spatial kernels before running a solver."""
    operator = (ConvAnalysisDictionaryOp if denoise is analysis_denoising else ConvSynthesisDictionaryOp)(
        torch.ones(2, 2, 3)
    )
    with pytest.raises(ValueError, match='odd spatial kernel'):
        denoise(torch.ones(1, 3, 4), operator, 0.2)


@pytest.mark.parametrize(
    ('denoise', 'initial_name'),
    [
        (analysis_denoising, 'initial_image'),
        (synthesis_denoising, 'initial_codes'),
    ],
)
def test_initial_shape(denoise, initial_name: str) -> None:
    """Initial values must have the exact domain shape, even when broadcasting is possible."""
    with pytest.raises(ValueError, match='initial value'):
        denoise(torch.ones(1, 3, 4), IdentityOp(), 0.2, **{initial_name: torch.ones(2, 1, 3, 4)})


@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
@pytest.mark.parametrize('controls', [{'max_iterations': 0}, {'tolerance': -1}, {'tolerance': float('nan')}])
def test_invalid_controls(denoise, controls) -> None:
    """Invalid iteration counts and tolerances fail before solver execution."""
    with pytest.raises(ValueError):
        denoise(torch.ones(2, 3), IdentityOp(), 0.2, **controls)


@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
def test_idata(denoise, random_iheader) -> None:
    """The wrappers preserve image shape and header identity."""
    image = IData(torch.full((2, 1, 1, 3, 4), 2.0), random_iheader)
    result = denoise(image, IdentityOp(), 0.5)
    assert isinstance(result, IData)
    assert result.header is image.header
    assert result.data.shape == image.data.shape
    torch.testing.assert_close(result.data, torch.full_like(image.data, 1.5))


def test_synthesis_backtracking() -> None:
    """A large dictionary gain needs backtracking; the solution still matches soft thresholding."""
    image = torch.tensor([[2.0, -4.0]], dtype=torch.float64)
    result = synthesis_denoising(image, 10 * IdentityOp(), 2.0, max_iterations=200)
    torch.testing.assert_close(result, torch.tensor([[1.8, -3.8]], dtype=image.dtype), atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
def test_tolerance(denoise) -> None:
    """Stopping on a stationary zero solution remains finite."""
    image = torch.zeros(1, 3)
    torch.testing.assert_close(denoise(image, IdentityOp(), 0.2, tolerance=1e-5), image)


@pytest.mark.cuda
@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
def test_cuda(denoise) -> None:
    """The image device controls execution, without modifying a CPU dictionary."""
    image = torch.full((2, 3, 4), 2.0, dtype=torch.complex128, device='cuda')
    operator = (ConvAnalysisDictionaryOp if denoise is analysis_denoising else ConvSynthesisDictionaryOp)(
        torch.ones(1, 1, 1)
    )
    result = denoise(image, operator, torch.tensor(0.5))
    assert result.device == image.device
    assert result.dtype == image.dtype
    assert operator.kernel.device.type == 'cpu'
    torch.testing.assert_close(result, torch.full_like(result, 1.5))


@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
@pytest.mark.parametrize('pad_mode', ['constant', 'reflect', 'replicate', 'circular'])
def test_filter_weights(denoise, pad_mode: Literal['constant', 'reflect', 'replicate', 'circular']) -> None:
    """Redundant delta banks distinguish analysis and synthesis objectives and filter-wise weights."""
    image = torch.tensor([[2.0, -3.0, 4.0]], dtype=torch.complex128)
    kernel = torch.tensor([[0, 1j, 0], [0, -2j, 0]], dtype=image.dtype)
    operator = (ConvAnalysisDictionaryOp if denoise is analysis_denoising else ConvSynthesisDictionaryOp)(
        kernel, pad_mode=pad_mode
    )
    weight = torch.tensor([0.4, 0.6], dtype=image.real.dtype).reshape(2, 1, 1)
    threshold = 1.6 if denoise is analysis_denoising else 0.3
    expected = torch.sgn(image) * (image.abs() - threshold).clamp_min(0)
    result = denoise(image, operator, weight, max_iterations=300)
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
def test_zero_weight_nonlocal(denoise) -> None:
    """A nonlocal invertible convolution reconstructs the data at zero regularization."""
    image = torch.tensor([[2.0, -3.0, 4.0, 1.0]], dtype=torch.float64)
    kernel = torch.tensor([[0.2, 1.0, 0.3]], dtype=image.dtype)
    operator = (ConvAnalysisDictionaryOp if denoise is analysis_denoising else ConvSynthesisDictionaryOp)(kernel)
    result = denoise(image, operator, 0.0, max_iterations=300)
    torch.testing.assert_close(result, image, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_operator_norm_double(dtype: torch.dtype) -> None:
    """PDHG's norm estimation must retain double precision in its stopping comparison."""
    result = IdentityOp().operator_norm(torch.ones(2, 3, dtype=dtype), dim=None)
    assert result.dtype == torch.float64
    torch.testing.assert_close(result, torch.ones_like(result))


def test_low_pass_zero_image() -> None:
    """CG must stop safely before dividing by a zero residual."""
    image = torch.zeros(1, 4, dtype=torch.float64)
    result = synthesis_denoising(image, IdentityOp(), 0.2, low_pass_operator=FiniteDifferenceOp(dim=(-1,)))
    torch.testing.assert_close(result, image)


@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
def test_solver_controls(denoise, monkeypatch: pytest.MonkeyPatch) -> None:
    """The recipes forward exact initial values and iteration budgets to existing solvers."""
    from importlib import import_module

    module = import_module(f'mr2.algorithms.{denoise.__name__}')
    image = torch.tensor([[3.0, -2.0]], dtype=torch.float64)
    initial = torch.ones_like(image)
    calls = []
    solver_name = 'pdhg' if denoise is analysis_denoising else 'pgd'
    solver = getattr(module, solver_name)

    def record(**kwargs):
        calls.append(kwargs)
        return solver(**kwargs)

    monkeypatch.setattr(module, solver_name, record)
    initial_name = 'initial_image' if denoise is analysis_denoising else 'initial_codes'
    result = denoise(image, IdentityOp(), 0.2, max_iterations=1, tolerance=0.01, **{initial_name: initial})
    assert len(calls) == 1
    assert calls[0]['max_iterations'] == 1
    if denoise is analysis_denoising:
        assert calls[0]['tolerance'] == 0.01
        assert calls[0]['initial_values'][0] is initial
    else:
        assert calls[0]['initial_value'] is initial
        assert calls[0]['callback'] is not None
    assert torch.isfinite(result).all()


def test_synthesis_rearranged_codes() -> None:
    """Coefficient shapes can differ from image shapes for reusable non-convolutional operators."""
    from mr2.operators import RearrangeOp

    operator = RearrangeOp('a b -> b a')
    # Its adjoint and forward are consistent, so this also exercises non-convolutional code shapes.
    image = torch.full((2, 3), 2.0, dtype=torch.float64)
    result = synthesis_denoising(image, operator, 0.5)
    torch.testing.assert_close(result, torch.full_like(image, 1.5))
