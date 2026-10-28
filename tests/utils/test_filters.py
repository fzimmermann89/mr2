"""Tests for filters."""

import math

import pytest
import torch
from einops import repeat
from mr2.utils.filters import (
    dct_filters,
    dct_matrix,
    fft_filters,
    filter_separable,
    gabor_filters_2d,
    gaussian_filter,
    gaussian_window,
    hann_window,
    radial_hann_window,
    uniform_filter,
)
from scipy.fft import dct, dctn


@pytest.fixture
def data():
    """Create a simple 3D tensor with a single voxel set to 1.0."""
    data = torch.zeros(1, 1, 5, 5, 5)
    data[..., 2, 2, 2] = 1.0
    return data


@pytest.mark.parametrize(
    ('pad_mode', 'center_value', 'edge_value'),
    [('constant', 44, 7), ('reflect', 44, 6), ('replicate', 44, 5), ('circular', 44, 24), ('none', 48, 8)],
)
def test_filter_separable(pad_mode, center_value, edge_value):
    """Test filter_separable and different padding modes."""

    data = repeat(torch.arange(1, 21), 'x -> y x', y=1).to(dtype=torch.float32)
    kernels = (torch.as_tensor([1.0, 2.0, 1.0]),)
    result = filter_separable(
        data, kernels, dim=(1,), pad_mode=pad_mode, pad_value=3.0 if pad_mode == 'constant' else 0.0
    )
    if pad_mode == 'none':
        assert result.shape == (data.shape[0], data.shape[1] - len(kernels[0]) + 1)
    else:
        assert data.shape == result.shape
    assert result[0, 10] == center_value
    assert result[0, 0] == edge_value


@pytest.mark.parametrize('filter_dtype', [torch.float32, torch.float64, torch.int32, torch.complex64, torch.complex128])
@pytest.mark.parametrize('data_dtype', [torch.float32, torch.float64, torch.int32, torch.complex64, torch.complex128])
def test_filter_separable_dtype(filter_dtype, data_dtype):
    """Test filter_separable and different padding modes."""

    data = repeat(torch.arange(1, 21), 'x -> y x', y=1).to(dtype=data_dtype)
    kernels = (torch.tensor([1, 2, 1], dtype=filter_dtype),)
    result = filter_separable(data, kernels, dim=(1,))
    expected_dtype = torch.result_type(data, kernels[0])
    assert result.dtype == expected_dtype
    assert result[0, 10] == 44


def test_gaussian_filter_int_axis(data):
    """Test Gaussian filter."""
    result = gaussian_filter(data, 0.5, -1)
    expected = torch.tensor(
        [
            [0, 0, 2, 2, 0],
            [0, 0, 2, 2, 1],
            [0, 0, 2, 2, 2],
            [0, 0, 2, 2, 3],
            [0, 0, 2, 2, 4],
        ],
    )
    assert torch.equal(result.nonzero(), expected)
    assert result.shape == data.shape
    torch.testing.assert_close(result.sum(), torch.tensor(1.0), atol=1e-3, rtol=0)


def test_gaussian_filter_two_axis(data):
    """Test Gaussian filter along two axes."""
    result = gaussian_filter(data, 0.5, (-1, 3))
    assert result.shape == data.shape
    torch.testing.assert_close(result.sum(), torch.tensor(1.0), atol=1e-3, rtol=0)


def test_gaussian_filter_two_sigmas(data):
    """Test Gaussian filter with different sigma values."""
    result = gaussian_filter(data, (0.2, 0.5), (-1, 3))
    assert result.shape == data.shape
    torch.testing.assert_close(result.sum(), torch.tensor(1.0), atol=1e-3, rtol=0)


def test_gaussian_filter_noaxis(data):
    """Test Gaussian filter is applied along all axes if not specified."""
    result = gaussian_filter(data, sigmas=torch.tensor(0.2))
    assert result.shape == data.shape
    torch.testing.assert_close(result.sum(), torch.tensor(1.0), atol=1e-3, rtol=0)


def test_gaussian_invalid_sigmas(data):
    """Test Gaussian filter with invalid sigma values."""
    with pytest.raises(ValueError, match='positive'):
        gaussian_filter(data, dim=(-1, 2), sigmas=torch.tensor([0.2, 0.0]))
    with pytest.raises(ValueError, match='positive'):
        gaussian_filter(data, sigmas=torch.tensor(-1.0))
    with pytest.raises(ValueError, match='positive'):
        gaussian_filter(data, sigmas=torch.nan)
    with pytest.raises(ValueError, match='length'):
        gaussian_filter(data, sigmas=(1.0, 2.0))


def test_uniform_filter_int_axis(data):
    """Test uniform filter."""
    result = uniform_filter(data, 3, -1)
    assert result.shape == data.shape
    expected = torch.zeros_like(data)
    expected[..., 2, 2, 1:4] = 1 / 3
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(result.sum(), torch.tensor(1.0), atol=1e-3, rtol=0)


def test_uniform_filter_two_axis(data):
    """Test uniform filter along two axes."""
    result = uniform_filter(data, 3, (-1, -2))
    assert result.shape == data.shape
    expected = torch.zeros_like(data)
    expected[..., 2, 1:4, 1:4] = 1 / 9
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(result.sum(), torch.tensor(1.0), atol=1e-3, rtol=0)


def test_uniform_filter_two_widths(data):
    """Test uniform filter with two widths."""
    result = uniform_filter(data, (3, 5), (-1, -2))
    assert result.shape == data.shape
    expected = torch.zeros_like(data)
    expected[..., 2, :, 1:4] = 1 / 15
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(result.sum(), torch.tensor(1.0), atol=1e-3, rtol=0)


def test_uniform_filter_noaxis(data):
    """Test uniform filter is applied along all axes if not specified."""
    result = uniform_filter(data, width=torch.tensor(3))
    assert result.shape == data.shape
    torch.testing.assert_close(result.sum(), torch.tensor(1.0), atol=1e-3, rtol=0)
    expected = torch.zeros_like(data)
    expected[..., 1:4, 1:4, 1:4] = 1 / 27
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)


def test_uniform_invalid_width(data):
    """Test uniform filter with invalid width."""
    with pytest.raises(ValueError, match='positive'):
        uniform_filter(data, dim=(-1, 2), width=torch.tensor([3, 0]))
    with pytest.raises(ValueError, match='positive'):
        uniform_filter(data, width=torch.tensor(-1.0))
    with pytest.raises(ValueError, match='positive'):
        uniform_filter(data, width=torch.nan)  # type: ignore[arg-type]
    with pytest.warns(UserWarning, match='odd'):
        uniform_filter(data, width=2)
    with pytest.raises(ValueError, match='length'):
        uniform_filter(data, width=(3.0, 3.0))  # type: ignore[arg-type]


@pytest.mark.parametrize('kernel_size', [(9, 9), (7, 9)])
def test_gabor_filters_2d(kernel_size: tuple[int, int]) -> None:
    """Gabor banks have the requested shape, zero mean, and unit filter norms."""
    filters = gabor_filters_2d(4, kernel_size)
    assert filters.shape == (4, *kernel_size)
    torch.testing.assert_close(filters.mean(dim=(-2, -1)), torch.zeros(4), atol=1e-7, rtol=0)
    torch.testing.assert_close(torch.linalg.vector_norm(filters, dim=(-2, -1)), torch.ones(4))


@pytest.mark.parametrize('sigma', [0.0, -1.0, float('nan'), float('inf')])
def test_gabor_invalid_sigma(sigma: float) -> None:
    """Invalid Gaussian widths fail before generating nonfinite filters."""
    with pytest.raises(ValueError):
        gabor_filters_2d(4, (7, 7), sigma=sigma)


def test_gabor_orientation() -> None:
    """A ninety-degree carrier rotation rotates the corresponding spatial filter."""
    filters = gabor_filters_2d(4, (9, 9))
    torch.testing.assert_close(filters[2], filters[0].rot90(1, (-2, -1)))


def test_dct_matrix() -> None:
    """The matrix agrees with SciPy's orthonormal DCT-II."""
    matrix = dct_matrix(5, dtype=torch.float64)
    expected = dct(torch.eye(5, dtype=torch.float64).numpy(), axis=0, norm='ortho')
    torch.testing.assert_close(matrix, torch.from_numpy(expected))


@pytest.mark.parametrize('kernel_size', [(5,), (3, 5), (2, 3, 4)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_dct_filters(kernel_size: tuple[int, ...], dtype: torch.dtype) -> None:
    """Flattened filters give the same patch transform as SciPy's N-dimensional DCT."""
    size = math.prod(kernel_size)
    filters = dct_filters(kernel_size, dtype=dtype)
    patches = torch.eye(size, dtype=dtype).reshape(size, *kernel_size)
    transformed = dctn(patches.numpy(), axes=tuple(range(1, patches.ndim)), norm='ortho')
    expected = torch.from_numpy(transformed).reshape(size, size).T
    assert filters.shape == (size, *kernel_size)
    assert filters.dtype == dtype
    torch.testing.assert_close(filters.flatten(1), expected)


@pytest.mark.parametrize('kernel_size', [(5,), (3, 5), (3, 3, 3)])
@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_fft_filters(kernel_size: tuple[int, ...], dtype: torch.dtype) -> None:
    """Hermitian inner products with Fourier modes reproduce the orthonormal FFT."""
    size = math.prod(kernel_size)
    filters = fft_filters(kernel_size, dtype=dtype)
    patch = torch.arange(size, dtype=filters.real.dtype).reshape(kernel_size).to(dtype)
    patch = patch + 1j * patch.flip(-1)
    coefficients = filters.flatten(1).conj() @ patch.flatten()
    expected = torch.fft.fftn(patch, norm='ortho').flatten()
    assert filters.shape == (size, *kernel_size)
    assert filters.dtype == dtype
    torch.testing.assert_close(coefficients, expected)


def test_apodization_windows() -> None:
    """Check continuous Hann support, Gaussian FWHM and finite radial DC gradients."""
    coordinate = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float64, requires_grad=True)
    expected = torch.tensor([1.0, (2 + 2**0.5) / 4, 0.5, 0.0], dtype=torch.float64)
    torch.testing.assert_close(hann_window(coordinate, 4), expected)
    torch.testing.assert_close(gaussian_window(coordinate[[0, 3]], 4), torch.tensor([1.0, 0.5], dtype=torch.float64))
    radial = radial_hann_window((coordinate,), (4.0,))
    torch.testing.assert_close(radial, expected)
    radial.sum().backward()
    gradient = -torch.pi / 4 * torch.sin(torch.pi * coordinate.detach() / 2)
    gradient[-1] = 0
    assert coordinate.grad is not None
    torch.testing.assert_close(coordinate.grad, gradient)
