"""Mathematical and regression tests for dictionary denoising recipes."""

import pytest
import torch
from mr2.algorithms.denoising import analysis_denoising, synthesis_denoising
from mr2.data import IData
from mr2.operators import ConvolutionOp, FiniteDifferenceOp, IdentityOp, RearrangeOp


@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_identity(denoise, dtype: torch.dtype) -> None:
    """Identity dictionaries have the magnitude soft-threshold solution, including weighted complex data."""
    image = torch.tensor([[-3, 0, 2, 5], [4, -2, 0.1, -1]], dtype=dtype)
    if image.is_complex():
        image = image * (0.6 + 0.8j)
    weight = torch.tensor([[0.0], [0.7]], dtype=image.real.dtype)
    expected = torch.sgn(image) * (image.abs() - weight).clamp_min(0)
    result = denoise(image, IdentityOp(), weight, max_iterations=150, tolerance=1e-7)
    torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-6)
    assert result.dtype == dtype


@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_convolution_supplied_operator(denoise, dtype: torch.dtype) -> None:
    """Supplied dictionaries retain tensor identity, dtype and device."""
    image = torch.full((2, 1, 3, 3), 2.0, dtype=dtype)
    kernel = torch.ones((1, 1, 1, 1), dtype=dtype)
    convolution = ConvolutionOp(kernel)
    insert_channel = RearrangeOp('... -> 1 ...')
    operator = convolution @ insert_channel if denoise is analysis_denoising else insert_channel.H @ convolution
    result = denoise(image, operator, 0.5, max_iterations=80)
    torch.testing.assert_close(result, torch.full_like(image, 1.5), atol=1e-6, rtol=1e-6)
    assert convolution.kernel is kernel


@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_trainable_kernel_gradient(denoise, dtype: torch.dtype) -> None:
    """Delta-kernel solutions and parameter gradients match independent analytic shrinkage."""
    image = torch.tensor([[2.0, -4.0]], dtype=dtype)
    if image.is_complex():
        image = image * (0.6 + 0.8j)
    kernel = torch.nn.Parameter(torch.full((1, 1, 1), 1.3, dtype=dtype))
    if kernel.is_complex():
        kernel = torch.nn.Parameter(kernel.detach() + 0.2j)
    convolution = ConvolutionOp(kernel)
    insert_channel = RearrangeOp('... -> 1 ...')
    operator = convolution @ insert_channel if denoise is analysis_denoising else insert_channel.H @ convolution
    result = denoise(image, operator, 0.2, max_iterations=100 if denoise is analysis_denoising else 20)
    gain = kernel.squeeze().abs()
    threshold = 0.2 * gain if denoise is analysis_denoising else 0.2 / gain
    expected = torch.sgn(image) * (image.abs() - threshold).clamp_min(0)
    actual_gradient = torch.autograd.grad(result.abs().square().sum(), kernel)[0]
    expected_gradient = torch.autograd.grad(expected.abs().square().sum(), kernel)[0]
    torch.testing.assert_close(result, expected, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-5, rtol=2e-5)
    assert convolution.kernel is kernel


@pytest.mark.parametrize('warm_start', [False, True])
def test_analysis_real_image_complex_dictionary(warm_start: bool) -> None:
    """Complex dictionaries promote real images and allow compatible complex warm starts."""
    image = torch.tensor([[2.0, -4.0]], dtype=torch.float64)
    kernel = torch.tensor([[[1 + 1j]]], dtype=torch.complex128)
    operator = ConvolutionOp(kernel) @ RearrangeOp('... -> 1 ...')
    initial = image.to(kernel.dtype) if warm_start else None
    result = analysis_denoising(image, operator, 0.2, initial_image=initial, max_iterations=150)
    expected = torch.sgn(image) * (image.abs() - 0.2 * kernel.abs().squeeze()).clamp_min(0)
    torch.testing.assert_close(result, expected.to(kernel.dtype), atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
def test_python_weight_precision(denoise) -> None:
    """Python weights retain the precision of float64 coefficients."""
    weight = 0.1
    image = torch.tensor([[2.0, -4.0]], dtype=torch.float64)
    result = denoise(image, IdentityOp(), weight, max_iterations=150)
    expected = denoise(image, IdentityOp(), torch.tensor(weight, dtype=image.dtype), max_iterations=150)
    torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)


def test_synthesis_rng_state() -> None:
    """Synthesis denoising is repeatable and leaves the global random generator unchanged."""
    image = torch.tensor([[2.0, -4.0]], dtype=torch.float64)
    state = torch.random.get_rng_state()
    first = synthesis_denoising(image, IdentityOp(), 0.1, max_iterations=20)
    torch.testing.assert_close(torch.random.get_rng_state(), state)
    second = synthesis_denoising(image, IdentityOp(), 0.1, max_iterations=20)
    torch.testing.assert_close(first, second, atol=0, rtol=0)


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
def test_zero_dictionary(denoise) -> None:
    """Zero dictionaries are finite and have the correct image-space limits."""
    image = torch.ones(2, 3, 4, dtype=torch.float64)
    kernel = torch.zeros(2, 1, 1, dtype=image.dtype)
    convolution = ConvolutionOp(kernel[:, None] if denoise is analysis_denoising else kernel[None])
    insert_channel = RearrangeOp('... -> 1 ...')
    operator = convolution @ insert_channel if denoise is analysis_denoising else insert_channel.H @ convolution
    result = denoise(image, operator, 0.5, max_iterations=8)
    expected = image if denoise is analysis_denoising else torch.zeros_like(image)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
@pytest.mark.parametrize('weight', [-1.0, float('inf'), float('nan'), 1j, torch.ones(2, 1, 3, 4), torch.ones(5)])
def test_invalid_weights(denoise, weight) -> None:
    """Weights cannot be negative/nonfinite/complex or expand the coefficient tensor."""
    with pytest.raises((ValueError, RuntimeError)):
        denoise(torch.ones(1, 3, 4), IdentityOp(), weight)


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


@pytest.mark.parametrize('gain', [0.1, 10.0])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_synthesis_scaled_dictionary(gain: float, dtype: torch.dtype) -> None:
    """Norm-based steps converge for both low and high dictionary gains."""
    image = torch.tensor([[2.0, -4.0]], dtype=dtype)
    result = synthesis_denoising(image, gain * IdentityOp(), 0.01, max_iterations=10)
    expected = torch.sgn(image) * (image.abs() - 0.01 / gain).clamp_min(0)
    torch.testing.assert_close(result, expected, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
def test_tolerance(denoise) -> None:
    """Stopping on a stationary zero solution remains finite."""
    image = torch.zeros(1, 3)
    torch.testing.assert_close(denoise(image, IdentityOp(), 0.2, tolerance=1e-5), image)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
def test_cuda(denoise) -> None:
    """Caller-supplied dictionaries on the image device support CUDA."""
    image = torch.full((2, 3, 4), 2.0, dtype=torch.complex128, device='cuda')
    convolution = ConvolutionOp(torch.ones(1, 1, 1, 1, dtype=image.dtype, device=image.device))
    insert_channel = RearrangeOp('... -> 1 ...')
    operator = convolution @ insert_channel if denoise is analysis_denoising else insert_channel.H @ convolution
    result = denoise(image, operator, torch.tensor(0.5))
    assert result.device == image.device
    assert result.dtype == image.dtype
    assert convolution.kernel.device == image.device
    torch.testing.assert_close(result, torch.full_like(result, 1.5))


@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
def test_filter_weights(denoise) -> None:
    """Redundant delta banks distinguish analysis and synthesis objectives and filter-wise weights."""
    image = torch.tensor([[2.0, -3.0, 4.0]], dtype=torch.complex128)
    kernel = torch.tensor([[0, 1j, 0], [0, -2j, 0]], dtype=image.dtype)
    convolution = ConvolutionOp(kernel[:, None] if denoise is analysis_denoising else kernel[None])
    insert_channel = RearrangeOp('... -> 1 ...')
    operator = convolution @ insert_channel if denoise is analysis_denoising else insert_channel.H @ convolution
    weight = torch.tensor([0.4, 0.6], dtype=image.real.dtype).reshape(2, 1, 1)
    threshold = 1.6 if denoise is analysis_denoising else 0.3
    expected = torch.sgn(image) * (image.abs() - threshold).clamp_min(0)
    result = denoise(image, operator, weight, max_iterations=300)
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('denoise', [analysis_denoising, synthesis_denoising])
def test_zero_weight_nonlocal(denoise) -> None:
    """A nonlocal invertible convolution reconstructs the data at zero regularization."""
    image = torch.tensor([[2.0, -3.0, 4.0, 1.0]], dtype=torch.float64)
    kernel = torch.tensor([[[0.2, 1.0, 0.3]]], dtype=image.dtype)
    convolution = ConvolutionOp(kernel)
    insert_channel = RearrangeOp('... -> 1 ...')
    operator = convolution @ insert_channel if denoise is analysis_denoising else insert_channel.H @ convolution
    result = denoise(image, operator, 0.0, max_iterations=300)
    torch.testing.assert_close(result, image, atol=1e-6, rtol=1e-6)


def test_synthesis_rearranged_codes() -> None:
    """Coefficient shapes can differ from image shapes for reusable non-convolutional operators."""
    operator = RearrangeOp('a b -> b a')
    image = torch.full((2, 3), 2.0, dtype=torch.float64)
    result = synthesis_denoising(image, operator, 0.5)
    torch.testing.assert_close(result, torch.full_like(image, 1.5))


def test_synthesis_adjoint_convolution() -> None:
    """An adjoint filter bank can be supplied as the synthesis map."""
    kernel = torch.tensor([[[1j, 0, 0]]], dtype=torch.complex128)
    analysis = ConvolutionOp(kernel) @ RearrangeOp('... -> 1 ...')
    image = torch.tensor([[2j, -3, 4, 1]], dtype=kernel.dtype)
    # A periodic shift with unit-magnitude phase is unitary, so image-space shrinkage is exact.
    expected = torch.sgn(image) * (image.abs() - 0.5).clamp_min(0)
    result = synthesis_denoising(image, analysis.H, 0.5, max_iterations=30, tolerance=1e-10)
    torch.testing.assert_close(result, expected)


def test_synthesis_backtracking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backtracking corrects an underestimated dictionary norm."""
    operator = 3 * IdentityOp()
    monkeypatch.setattr(operator, 'operator_norm', lambda *_args, **_kwargs: torch.tensor(1.0))
    image = torch.tensor([[2.0, -4.0]], dtype=torch.float64)
    result = synthesis_denoising(image, operator, 0.3, max_iterations=50)
    torch.testing.assert_close(result, torch.tensor([[1.9, -3.9]], dtype=image.dtype), atol=1e-7, rtol=1e-7)
