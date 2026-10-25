"""Tests for total variation denoising."""

import pytest
import torch
from mr2.algorithms.denoising import total_variation_denoising
from mr2.data import IData, IHeader, SpatialDimension
from mr2.utils import RandomGenerator
from tests.helper import relative_image_difference


@pytest.fixture
def idata_single_coil(ellipse_phantom, random_kheader) -> IData:
    """Create single-coil image."""
    image_dimensions = SpatialDimension(z=1, y=ellipse_phantom.n_y, x=ellipse_phantom.n_x)
    img = ellipse_phantom.phantom.image_space(image_dimensions)
    return IData.from_tensor_and_kheader(data=img, header=random_kheader)


@pytest.mark.parametrize('tensor_input', [True, False], ids=['tensor', 'idata'])
def test_denoising(idata_single_coil: IData, tensor_input: bool) -> None:
    rng = RandomGenerator(seed=0)
    noisy = IData(idata_single_coil.data + rng.rand_like(idata_single_coil.data), idata_single_coil.header)
    if tensor_input:
        denoised = total_variation_denoising(noisy.data, regularization_dim=(-2, -1), regularization_weight=[1.0, 1.0])
    else:
        denoised = total_variation_denoising(noisy, regularization_dim=(-2, -1), regularization_weight=[1.0, 1.0]).data
    assert relative_image_difference(denoised, idata_single_coil.data) < relative_image_difference(
        noisy.data, idata_single_coil.data
    )


def test_denoising_same_weight_for_all_dims(idata_single_coil: IData) -> None:
    """Test denoising with same weight for all dims."""
    rng = RandomGenerator(seed=0)
    noisy = IData(idata_single_coil.data + rng.rand_like(idata_single_coil.data), idata_single_coil.header)
    denoised_same_weight = total_variation_denoising(noisy, regularization_dim=(-2, -1), regularization_weight=1.0)
    denoised = total_variation_denoising(noisy, regularization_dim=(-2, -1), regularization_weight=(1.0, 1.0))
    assert relative_image_difference(denoised.data, denoised_same_weight.data) < 1e-2


def test_denoising_weight_dim_mismatch() -> None:
    """Error for different length of dim and weights."""
    with pytest.raises(ValueError, match='Regularization dimensions and weights must have the same length'):
        _ = total_variation_denoising(torch.zeros(1, 1), regularization_dim=(-2,), regularization_weight=[1.0, 1.0])


def test_denoising_repeated_dims() -> None:
    """Error for repeated dims."""
    with pytest.raises(ValueError, match='Repeated values are not allowed in regularization_dim'):
        _ = total_variation_denoising(torch.zeros(1, 1), regularization_dim=(-2, 0), regularization_weight=[1.0, 1.0])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('tensor_input', [True, False], ids=['tensor', 'idata'])
@pytest.mark.parametrize('image_device', ['cpu', 'cuda'])
def test_denoising_device(random_iheader: IHeader, tensor_input: bool, image_device: str) -> None:
    """Initialization follows the image device for Tensor and IData inputs."""
    image = torch.zeros(1, 1, 1, 8, 8, dtype=torch.complex64, device=image_device)
    image[..., 3:5, 3:5] = torch.tensor(1 + 0.5j, device=image.device)
    initial_image = torch.zeros_like(image)
    # PDHG estimates the operator norm from a random starting vector.
    torch.manual_seed(0)
    expected = total_variation_denoising(image, (-2, -1), 0.2, initial_image=initial_image, max_iterations=20)
    torch.manual_seed(0)
    result = total_variation_denoising(
        image if tensor_input else IData(image, random_iheader),
        (-2, -1),
        0.2,
        initial_image=initial_image.to('cuda' if image_device == 'cpu' else 'cpu'),
        max_iterations=20,
    )
    if isinstance(result, IData):
        assert result.header is random_iheader
        result = result.data
    assert result.device == image.device
    torch.testing.assert_close(result, expected)


def test_denoising_two_pixel_solution() -> None:
    """Preserve the existing squared-fidelity scaling and zero-padded forward differences."""
    image = torch.tensor([[0.0, 2.0]])
    weight = 0.4
    result = total_variation_denoising(image, (-1,), weight, max_iterations=100)
    # min a² + (b-2)² + weight * (|b-a| + |b|), with 0 < a < b.
    expected = torch.tensor([[weight / 2, 2 - weight]], dtype=image.dtype)
    torch.testing.assert_close(result, expected)
