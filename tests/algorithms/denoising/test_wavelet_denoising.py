"""Tests for wavelet thresholding and reconstruction."""

import math

import pytest
import torch
from mr2.algorithms.denoising import wavelet_denoising
from mr2.data import IData, KHeader
from mr2.operators import WaveletOp
from mr2.operators.WaveletOp import WaveletType


def test_haar_analytic() -> None:
    """One-level Haar shrinkage includes the approximation band."""
    image = torch.tensor([3.0, 1.0], dtype=torch.float64)
    result = wavelet_denoising(image, (-1,), 1.0, wavelet_name='haar', level=1)
    torch.testing.assert_close(result, torch.tensor([3 - math.sqrt(2), 1.0], dtype=image.dtype))
    torch.testing.assert_close(wavelet_denoising(image, (-1,), 4.0, wavelet_name='haar'), torch.zeros_like(image))


def test_weighted_haar() -> None:
    """Packed thresholds may vary by coefficient and batch without expanding the result."""
    image = torch.tensor([[3.0, 1.0], [3.0, 1.0]], dtype=torch.float64)
    threshold = torch.tensor([[0.0, math.sqrt(2)], [math.sqrt(2), 0.0]], dtype=torch.float64)
    result = wavelet_denoising(image, (-1,), threshold, wavelet_name='haar', level=1)
    torch.testing.assert_close(result, torch.tensor([[2.0, 2.0], [2.0, 0.0]], dtype=image.dtype))
    per_batch = torch.tensor([[0.0], [4.0]])
    result = wavelet_denoising(image, (-1,), per_batch, wavelet_name='haar', level=1)
    torch.testing.assert_close(result, torch.stack([image[0], torch.zeros_like(image[1])]))


def test_complex_shrinkage() -> None:
    """Real/imaginary shrinkage differs from magnitude shrinkage."""
    image = torch.full((2,), complex(3, 4), dtype=torch.complex128) / math.sqrt(2)
    result = wavelet_denoising(image, (-1,), 1.0, wavelet_name='haar', level=1)
    torch.testing.assert_close(result, torch.full_like(image, complex(2, 3)) / math.sqrt(2))
    assert not torch.allclose(result, image * (1 - 1 / 5))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize('wavelet_name', ['haar', 'db4', 'sym2', 'coif1'])
@pytest.mark.parametrize('dim', [(-1,), (-2, -1), (-3, -2, -1), (3, 1)])
def test_zero_threshold(dtype: torch.dtype, wavelet_name: WaveletType, dim: tuple) -> None:
    """Zero threshold reconstructs batched and reordered axes with the original dtype/shape."""
    image = torch.arange(2 * 16**3, dtype=torch.float64).reshape(2, 16, 16, 16).to(dtype) / 100
    if image.is_complex():
        image = image + 0.5j * image
    result = wavelet_denoising(image, dim, 0.0, wavelet_name=wavelet_name, level=1)
    torch.testing.assert_close(result, image, atol=2e-5, rtol=2e-5)
    assert result.dtype == image.dtype
    assert result.device == image.device


def test_idata(random_kheader: KHeader) -> None:
    """Return IData with the same header and the tensor result."""
    image = IData.from_tensor_and_kheader(torch.arange(4.0).reshape(1, 1, 1, 2, 2), random_kheader)
    result = wavelet_denoising(image, (-2, -1), 0.5, wavelet_name='haar', level=1)
    assert isinstance(result, IData)
    assert result.header is image.header
    torch.testing.assert_close(result.data, wavelet_denoising(image.data, (-2, -1), 0.5, wavelet_name='haar'))


@pytest.mark.parametrize('weight', [-1.0, float('nan'), float('inf'), complex(1, 0), torch.tensor([0.0, -1.0])])
def test_invalid_weight(weight: float | complex | torch.Tensor) -> None:
    """Reject negative, nonfinite and complex thresholds."""
    with pytest.raises(ValueError, match='finite, real and nonnegative'):
        wavelet_denoising(torch.ones(2), (-1,), weight, wavelet_name='haar')  # type: ignore[call-overload]


@pytest.mark.parametrize('weight', [torch.ones(3), torch.ones(1, 2), torch.ones(2, 2)])
def test_invalid_weight_shape(weight: torch.Tensor) -> None:
    """Reject incompatible broadcasting and broadcasting that would expand the coefficient shape."""
    with pytest.raises((ValueError, RuntimeError), match=r'broadcast|mismatch'):
        wavelet_denoising(torch.ones(2), (-1,), weight, wavelet_name='haar')


@pytest.mark.parametrize('dim', [(), (0, -1), (0, 1, 2, 3)])
def test_invalid_axes(dim: tuple) -> None:
    """Reject empty, repeated and excessive transform axes."""
    with pytest.raises(ValueError, match=r'wavelet transforms|Repeated'):
        wavelet_denoising(torch.ones((2,) * (4 if len(dim) > 3 else 1)), dim, 1.0, wavelet_name='haar')


def test_axis_out_of_range() -> None:
    """Reject axes outside the image rank."""
    with pytest.raises(IndexError, match='Invalid index'):
        wavelet_denoising(torch.ones(2), (1,), 1.0, wavelet_name='haar')


@pytest.mark.parametrize('level', [-1, 1.5, True])
def test_invalid_level(level: int) -> None:
    """Reject invalid decomposition levels."""
    with pytest.raises(ValueError, match='level must be'):
        wavelet_denoising(torch.ones(2), (-1,), 1.0, wavelet_name='haar', level=level)


@pytest.mark.parametrize('wavelet_name', ['bior2.2', 'rbio2.2'])
def test_biorthogonal_rejected(wavelet_name: WaveletType) -> None:
    """Reconstruction filters are not a mathematical adjoint for biorthogonal families."""
    with pytest.raises(ValueError, match='only supports orthogonal'):
        wavelet_denoising(torch.ones(32), (-1,), 1.0, wavelet_name=wavelet_name)


def test_invalid_image() -> None:
    """Reject integer images, empty transform axes and unsupported odd lengths."""
    with pytest.raises(ValueError, match='dtype'):
        wavelet_denoising(torch.ones(2, dtype=torch.int64), (-1,), 1.0)
    with pytest.raises(ValueError, match='positive lengths'):
        wavelet_denoising(torch.ones(0), (-1,), 1.0)
    with pytest.raises(NotImplementedError, match='even number'):
        wavelet_denoising(torch.ones(3), (-1,), 0.0, wavelet_name='haar')


def test_level_zero() -> None:
    """Level zero shrinks image values directly."""
    image = torch.tensor([-2.0, 0.5, 3.0, 0.0])
    torch.testing.assert_close(
        wavelet_denoising(image, (-1,), 1.0, wavelet_name='haar', level=0),
        torch.tensor([-1.0, 0.0, 2.0, 0.0]),
    )


def test_boundary_redundancy() -> None:
    """Even orthogonal filters can produce redundant coefficients due to zero padding."""
    image = torch.arange(16, dtype=torch.float64)
    operator = WaveletOp((16,), (-1,), wavelet_name='db4', level=1)
    (coefficients,) = operator(image)
    assert coefficients.numel() > image.numel()
    torch.testing.assert_close(wavelet_denoising(image, (-1,), 0.0, level=1), image)


@pytest.mark.cuda
def test_cuda() -> None:
    """CPU threshold tensors follow the CUDA image device without promoting dtype."""
    image = torch.tensor([3.0, 1.0], device='cuda', dtype=torch.complex64)
    result = wavelet_denoising(image, (-1,), torch.tensor(1.0, dtype=torch.float64), wavelet_name='haar')
    assert result.dtype == image.dtype
    assert result.device == image.device
    torch.testing.assert_close(result.cpu(), torch.tensor([3 - math.sqrt(2), 1.0], dtype=image.dtype))


def test_excessive_level_warning() -> None:
    """Keep the existing warning about levels dominated by boundary effects."""
    with pytest.warns(UserWarning, match='boundary effects'):
        wavelet_denoising(torch.arange(8.0), (-1,), 0.0, level=1)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_threshold_dtype(dtype: torch.dtype) -> None:
    """A float64 threshold does not promote image coefficients."""
    image = torch.tensor([3.0, 1.0], dtype=dtype)
    result = wavelet_denoising(image, (-1,), torch.tensor(1.0, dtype=torch.float64), wavelet_name='haar')
    assert result.dtype == dtype
    torch.testing.assert_close(result, torch.tensor([3 - math.sqrt(2), 1.0], dtype=dtype))


def test_large_scalar_threshold() -> None:
    """A finite float64 threshold must not overflow through the default float32 dtype."""
    image = torch.tensor([3.0, 1.0], dtype=torch.float64)
    result = wavelet_denoising(image, (-1,), 1e40, wavelet_name='haar')
    torch.testing.assert_close(result, torch.zeros_like(image))
    with pytest.raises(ValueError, match='finite'):
        wavelet_denoising(image.float(), (-1,), torch.tensor(1e40, dtype=torch.float64), wavelet_name='haar')


@pytest.mark.parametrize('dim', [(-1,), (-2, -1), (-3, -2, -1)])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_unbatched(dim: tuple[int] | tuple[int, int] | tuple[int, int, int], dtype: torch.dtype) -> None:
    """ptwt's implicit batch axis must not appear in packed coefficients or reconstructed images."""
    image = torch.arange(8 ** len(dim), dtype=torch.float64).reshape((8,) * len(dim)).to(dtype)
    operator = WaveletOp(image.shape, dim, wavelet_name='haar', level=1)
    (coefficients,) = operator(image)
    assert coefficients.shape == (image.numel(),)
    torch.testing.assert_close(operator.H(coefficients)[0], image)
    actual = wavelet_denoising(image, dim, 0.2, wavelet_name='haar', level=1)
    expected = wavelet_denoising(image.unsqueeze(0), dim, 0.2, wavelet_name='haar', level=1).squeeze(0)
    torch.testing.assert_close(actual, expected)


def test_default_wavelet_reduces_noise() -> None:
    """The default db4 recipe reduces seeded noise in a smooth image."""
    image = torch.zeros(1, 64, 64)
    image[:, 16:48, 16:48] = 1
    noise = 0.1 * torch.randn(image.shape, generator=torch.Generator().manual_seed(0))
    result = wavelet_denoising(image + noise, (-2, -1), 0.1)
    assert (result - image).square().mean() < noise.square().mean()
