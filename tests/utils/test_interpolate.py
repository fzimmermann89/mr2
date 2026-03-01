"""Tests interpolation."""

from typing import Literal

import pytest
import torch
from mr2.utils.interpolate import apply_lowres, interpolate


@pytest.fixture
def data() -> torch.Tensor:
    """Create a simple 5D tensor with a linear ramp of step size 1."""
    data = torch.arange(0, 20).repeat(10, 10, 1).unsqueeze(0).unsqueeze(0)
    return data


@pytest.mark.parametrize('size', [10, 20, 30])
@pytest.mark.parametrize('data_dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_interpolate_linear(data: torch.Tensor, size: int, data_dtype: torch.dtype) -> None:
    """Linear ramp should remain a linear ramp after inear interpolation."""
    result = interpolate(data.to(dtype=data_dtype), size=(size,), dim=(4,), mode='linear')
    assert result.dtype == data_dtype
    assert torch.diff(result[..., 1:-1], dim=-1).isclose(torch.tensor(data.shape[-1] / size, dtype=data_dtype)).all()


@pytest.mark.parametrize('data_dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_interpolate_nearest(data: torch.Tensor, data_dtype: torch.dtype) -> None:
    """Tensor is unchanged after nearest upsampling and then downsampling."""
    result = interpolate(data.to(dtype=data_dtype), size=(data.shape[-1] * 2,), dim=(4,), mode='nearest')
    result = interpolate(result, size=(data.shape[-1],), dim=(4,), mode='nearest')
    torch.testing.assert_close(result, data.to(dtype=data_dtype))


def test_interpolate_size_dim_mismatch() -> None:
    """Test mismatch between size and dim."""
    with pytest.raises(ValueError, match='matching length'):
        interpolate(torch.zeros((2, 2, 2)), dim=(-1, -2), size=(2,))


def test_interpolate_unique_dim() -> None:
    """Test non-unique interpolate dimensions."""
    with pytest.raises(IndexError, match='unique'):
        interpolate(torch.zeros((2, 2, 2)), dim=(-1, -2, 1), size=(2, 2, 2))


@pytest.mark.parametrize('data_dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize('size', [10, 20, 30])
def test_interpolate_area(data: torch.Tensor, data_dtype: torch.dtype, size: int) -> None:
    """Area interpolation should return expected shape and dtype for up to 3 dims."""
    data = data.to(dtype=data_dtype).repeat(2, 3, 4, 5, 3)
    result = interpolate(data, size=(size, size, size), dim=(-3, -2, -1), mode='area')
    assert result.shape == (2, 3, size, size, size)
    assert result.dtype == data_dtype


@pytest.mark.parametrize('data_dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize('size', [10, 20, 30])
def test_interpolate_bicubic(data: torch.Tensor, data_dtype: torch.dtype, size: int) -> None:
    """Bicubic interpolation should return expected shape and dtype in 2D."""
    data = data[..., 0, :, :].to(dtype=data_dtype).repeat(2, 3, 4, 3)
    result = interpolate(data, size=(size, size), dim=(-2, -1), mode='bicubic')
    assert result.shape == (2, 3, size, size)
    assert result.dtype == data_dtype


def test_interpolate_bicubic_requires_2d() -> None:
    """Bicubic mode should only work with exactly two interpolation dimensions."""
    with pytest.raises(ValueError, match='exactly 2 interpolation dimensions'):
        interpolate(torch.zeros((2, 3, 4, 5, 6)), size=(2, 3, 4), dim=(-3, -2, -1), mode='bicubic')


def test_interpolate_area_requires_1_to_3d() -> None:
    """Area mode should only work with one to three interpolation dimensions."""
    with pytest.raises(ValueError, match='requires 1-3 interpolation dimensions'):
        interpolate(
            torch.zeros((2, 3, 4, 5, 6, 7)),
            size=(2, 2, 2, 2),
            dim=(-4, -3, -2, -1),
            mode='area',
        )


def test_interpolate_align_corners_mode_validation() -> None:
    """align_corners should only be accepted for linear and bicubic modes."""
    with pytest.raises(ValueError, match='align_corners is only supported'):
        interpolate(torch.zeros((2, 3, 4, 5)), size=(2, 3), dim=(-2, -1), mode='area', align_corners=True)


@pytest.mark.parametrize('mode', ['nearest', 'linear', 'area', 'bicubic'])
def test_interpolate_vmap_2d(data: torch.Tensor, mode: Literal['nearest', 'linear', 'area', 'bicubic']) -> None:
    """Interpolate should support vmap for 2D modes."""
    data = data.to(torch.float32).repeat(5, 1, 1, 1, 1)
    vmapped = torch.vmap(lambda tensor: interpolate(tensor, size=(6, 7), dim=(-2, -1), mode=mode))
    result = vmapped(data)
    assert result.shape == (5, 1, 10, 6, 7)


def test_interpolate_linear_four_dims() -> None:
    """Linear interpolation should work for four interpolation dimensions."""
    data = torch.zeros((2, 3, 4, 5, 6, 7), dtype=torch.float32)
    result = interpolate(data, size=(3, 4, 5, 6), dim=(-4, -3, -2, -1), mode='linear')
    assert result.shape == (2, 3, 3, 4, 5, 6)


def test_apply_lowres(data: torch.Tensor) -> None:
    """Applying identity function should not change linear ramp data except for edges."""
    data = data.to(torch.complex64)

    def identity_function(x: torch.Tensor):
        return x

    apply = apply_lowres(identity_function, size=(8, 8, 8), dim=(-3, -2, -1))
    torch.testing.assert_close(data[..., 1:-1], apply(data)[..., 1:-1])
