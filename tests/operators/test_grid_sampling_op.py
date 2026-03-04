"""Tests for grid sampling operator."""

import contextlib
from typing import Any, Literal

import pytest
import torch
from mr2.data import SpatialDimension
from mr2.operators import GridSamplingOp
from mr2.utils import RandomGenerator
from torch.autograd.gradcheck import gradcheck

from tests import dotproduct_adjointness_test


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64'])
def test_grid_sampling_op_dtype(dtype: str) -> None:
    """Test for different data types."""
    _test_grid_sampling_op_adjoint(dtype=dtype)


@pytest.mark.parametrize('dim_str', ['2D', '3D'])
@pytest.mark.parametrize('batched', ['batched', 'non_batched'])
@pytest.mark.parametrize('channel', ['multi_channel', 'single_channel'])
@pytest.mark.parametrize('dtype', ['float32', 'complex64'])
def test_grid_sampling_op_dim_batch_channel(dim_str: str, batched: str, channel: str, dtype: str) -> None:
    """Test for different dimensions."""
    _test_grid_sampling_op_adjoint(dim=int(dim_str[0]), batched=batched, channel=channel, dtype=dtype)


@pytest.mark.parametrize('interpolation_mode', ['bilinear', 'nearest', 'bicubic'])
def test_grid_sampling_op_interpolation_mode(interpolation_mode: str) -> None:
    """Test for different interpolation_modes."""
    # bicubic only supports 2D
    _test_grid_sampling_op_adjoint(dim=2, interpolation_mode=interpolation_mode)


@pytest.mark.parametrize('padding_mode', ['zeros', 'border', 'reflection'])
def test_grid_sampling_op_padding_mode(padding_mode: str) -> None:
    """Test for different padding_modes."""
    _test_grid_sampling_op_adjoint(padding_mode=padding_mode)


@pytest.mark.parametrize('align_corners', ['no_align', 'align'])
def test_grid_sampling_op_align_mode(align_corners: str) -> None:
    """Test for different align modes ."""
    _test_grid_sampling_op_adjoint(align_corners=align_corners)


def _test_grid_sampling_op_adjoint(
    dtype='float32',
    dim: int = 2,
    interpolation_mode='bilinear',
    padding_mode='zeros',
    align_corners='no_align',
    batched='non_batched',
    channel='single_channel',
):
    """Used in the tests above."""
    rng = getattr(RandomGenerator(0), f'{dtype}_tensor')
    batch = (2, 3) if batched == 'batched' else (1,)
    channel = (5, 6) if channel == 'multi_channel' else (1,)
    align_corners_bool = align_corners == 'align'
    zyx_v = (7, 8, 9)[-dim:]
    zyx_u = (11, 12, 13)[-dim:]
    grid = RandomGenerator(42).float64_tensor((*batch, *zyx_v, 3), -1, 1)
    input_shape = SpatialDimension(z=(99 if dim == 2 else zyx_u[-3]), y=zyx_u[-2], x=zyx_u[-1])
    operator = GridSamplingOp(
        grid_z=grid[..., 0] if dim == 3 else None,
        grid_y=grid[..., 1],
        grid_x=grid[..., 2],
        input_shape=input_shape,
        interpolation_mode=interpolation_mode,
        padding_mode=padding_mode,
        align_corners=align_corners_bool,
    )
    operator = operator.to(dtype=getattr(torch, dtype).to_real())
    u = rng((*batch, *channel, *zyx_u))
    v = rng((*batch, *channel, *zyx_v))
    dotproduct_adjointness_test(operator, u, v)


@pytest.mark.parametrize('interpolation_mode', ['bilinear', 'nearest', 'bicubic'])
def test_grid_sampling_op_interpolation_mode_backward_is_adjoint(
    interpolation_mode: Literal['bilinear', 'nearest', 'bicubic'],
) -> None:
    """Test for different interpolation_modes."""
    # bicubic only supports 2D
    dim = 2 if interpolation_mode == 'bicubic' else 3
    _test_grid_sampling_op_x_backward(dim=dim, interpolation_mode=interpolation_mode)


@pytest.mark.parametrize('padding_mode', ['zeros', 'border', 'reflection'])
def test_grid_sampling_op_padding_mode_backward_is_adjoint(
    padding_mode: Literal['zeros', 'border', 'reflection'],
) -> None:
    """Test for different padding_modes."""
    _test_grid_sampling_op_x_backward(padding_mode=padding_mode)


@pytest.mark.parametrize('align_corners', ['no_align', 'align'])
def test_grid_sampling_op_align_mode_backward_is_adjoint(align_corners: Literal['no_align', 'align']) -> None:
    """Test for different align modes ."""
    _test_grid_sampling_op_x_backward(align_corners=align_corners == 'align')


def _test_grid_sampling_op_x_backward(
    dim: int = 3,
    interpolation_mode: Literal['bilinear', 'nearest', 'bicubic'] = 'bilinear',
    padding_mode: Literal['zeros', 'border', 'reflection'] = 'zeros',
    align_corners: bool = False,
) -> None:
    """Used in the tests above."""
    rng = RandomGenerator(0).float32_tensor
    batch = (2, 3)
    channel = (5, 7)
    zyx_v = (7, 10, 20)[-dim:]
    zyx_u = (9, 22, 30)[-dim:]
    grid = rng((*batch, *zyx_v, 3), -1, 1.0)
    input_shape = SpatialDimension(z=99 if dim == 2 else zyx_u[-3], y=zyx_u[-2], x=zyx_u[-1])
    operator = GridSamplingOp(
        grid_z=grid[..., 0] if dim == 3 else None,
        grid_y=grid[..., 1],
        grid_x=grid[..., 2],
        input_shape=input_shape,
        interpolation_mode=interpolation_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    u = rng((*batch, *channel, *zyx_u)).requires_grad_(True)
    v = rng((*batch, *channel, *zyx_v)).requires_grad_(True)
    (forward_u,) = operator(u)
    forward_u.backward(v.detach())
    (adjoint_v,) = operator.adjoint(v)
    adjoint_v.backward(u.detach())
    torch.testing.assert_close(u.grad, adjoint_v)
    torch.testing.assert_close(v.grad, forward_u)


def test_grid_sampling_op_gradcheck_x_forward() -> None:
    """Gradient check for forward wrt x."""
    rng = RandomGenerator(0).float64_tensor
    grid = rng((2, 1, 2, 2), -0.8, 0.8)
    u = rng((1, 1, 3, 5)).requires_grad_(True)
    gradcheck(
        lambda grid, u: GridSamplingOp(
            grid_z=None, grid_y=grid[..., 0], grid_x=grid[..., 1], input_shape=SpatialDimension(1, 3, 5)
        )(u),
        (grid, u),
        fast_mode=True,
    )


def test_grid_sampling_op_gradcheck_grid_forward() -> None:
    """Gradient check for forward wrt grid."""
    rng = RandomGenerator(0).float64_tensor
    grid = rng((2, 1, 2, 2), -0.8, 0.8).requires_grad_(True)
    u = rng((1, 1, 3, 5))
    gradcheck(
        lambda grid, u: GridSamplingOp(
            grid_z=None, grid_y=grid[..., 0], grid_x=grid[..., 1], input_shape=SpatialDimension(1, 3, 5)
        )(u),
        (grid, u),
        fast_mode=True,
    )


def test_grid_sampling_op_gradcheck_x_adjoint() -> None:
    """Gradient check for adjoint wrt x."""
    rng = RandomGenerator(0).float64_tensor
    grid = rng((2, 1, 2, 2), -0.8, 0.8)
    v = rng((2, 1, 1, 2)).requires_grad_(True)
    gradcheck(
        lambda grid, v: GridSamplingOp(
            grid_z=None, grid_y=grid[..., 0], grid_x=grid[..., 1], input_shape=SpatialDimension(1, 2, 3)
        ).adjoint(v),
        (grid, v),
        fast_mode=True,
    )


def test_grid_sampling_op_gradcheck_grid_adjoint() -> None:
    """Gradient check for adjoint wrt grid."""
    rng = RandomGenerator(0).float64_tensor
    grid = rng((2, 1, 2, 2), -0.8, 0.8).requires_grad_(True)
    v = rng((2, 1, 1, 2))
    gradcheck(
        lambda grid, v: GridSamplingOp(
            grid_z=None, grid_y=grid[..., 0], grid_x=grid[..., 1], input_shape=SpatialDimension(1, 2, 3)
        ).adjoint(v),
        (grid, v),
        fast_mode=True,
    )


def test_grid_sampling_op_errormsg_gridshape_3d() -> None:
    """Test if error message on mismatch of grid shape is raised."""
    with pytest.raises(ValueError, match='should have the same shape'):
        _ = GridSamplingOp(
            grid_z=torch.ones(1, 2, 1, 1),
            grid_y=torch.ones(1, 1, 1, 1),
            grid_x=torch.ones(1, 1, 1, 1),
        )


def test_grid_sampling_op_errormsg_gridshape_2d() -> None:
    """Test if error message on mismatch of grid shape is raised."""
    with pytest.raises(ValueError, match='should have the same shape'):
        _ = GridSamplingOp(grid_z=None, grid_y=torch.ones(1, 3, 1), grid_x=torch.ones(1, 1, 1))


def test_grid_sampling_op_errormsg_gridndims_3d() -> None:
    """Test if error message on missing batch dim is raised."""
    with pytest.raises(ValueError, match='batch z y x'):
        _ = GridSamplingOp(
            grid_z=torch.ones(1, 1, 1),
            grid_y=torch.ones(1, 1, 1),
            grid_x=torch.ones(1, 1, 1),
        )


def test_grid_sampling_op_errormsg_gridndims_2d() -> None:
    """Test if error message on missing batch dim is raised."""
    with pytest.raises(ValueError, match='batch y x'):
        _ = GridSamplingOp(grid_z=None, grid_y=torch.ones(1, 1), grid_x=torch.ones(1, 1))


def test_grid_sampling_op_errormsg_cubic3d() -> None:
    """Test if error for 3D cubic is raised."""
    grid = torch.ones(1, 1, 1, 1, 3)  # 3d
    with pytest.raises(NotImplementedError, match='cubic'):
        _ = GridSamplingOp(
            grid_z=grid[..., 0],
            grid_y=grid[..., 1],
            grid_x=grid[..., 2],
            input_shape=SpatialDimension(1, 1, 1),
            interpolation_mode='bicubic',
        )


def test_grid_sampling_op_errormsg_complexgrid() -> None:
    """Test if error for complex grid is raised."""
    grid = torch.ones(1, 1, 1, 1, 3) + 0j
    with pytest.raises(ValueError, match='real'):
        _ = GridSamplingOp(
            grid_z=grid[..., 0], grid_y=grid[..., 1], grid_x=grid[..., 2], input_shape=SpatialDimension(1, 1, 1)
        )


@pytest.mark.parametrize(
    ('value', 'error_message'),
    [(1.0001, 'values outside range'), (-1.0001, 'values outside range'), (1.0, None), (-1.0, None)],
)
def test_grid_sampling_op_warning_gridrange(value: float, error_message: str | None) -> None:
    """Test if warning for grid values outside [-1,1] is raised"""
    grid = torch.zeros(1, 1, 1, 1, 3)
    grid[..., 1] = value
    conditional_warn: contextlib.AbstractContextManager[None] | pytest.WarningsRecorder = (
        pytest.warns(UserWarning, match=error_message) if error_message else contextlib.nullcontext()
    )
    with conditional_warn:
        _ = GridSamplingOp(
            grid_z=grid[..., 0], grid_y=grid[..., 1], grid_x=grid[..., 2], input_shape=SpatialDimension(1, 1, 1)
        )


def test_grid_sampling_op_errormsg_inputdim_3d() -> None:
    """Test if error for wrong input dimensions is raised."""
    grid = torch.ones(1, 1, 1, 1, 3)
    input_shape = SpatialDimension(2, 3, 4)
    operator = GridSamplingOp(grid_z=grid[..., 0], grid_y=grid[..., 1], grid_x=grid[..., 2], input_shape=input_shape)
    u = torch.zeros(1, 2, 3, 4)
    with pytest.raises(ValueError, match='5 dimensions: batch channel z y x'):
        _ = operator(u)


def test_grid_sampling_op_warningmsg_inputshape_3d() -> None:
    """Test if warning for wrong input_shape is raised in forward"""
    grid = torch.ones(1, 1, 1, 1, 3)
    input_shape = SpatialDimension(2, 3, 4)
    operator = GridSamplingOp(grid_z=grid[..., 0], grid_y=grid[..., 1], grid_x=grid[..., 2], input_shape=input_shape)
    u = torch.zeros(1, 1, 3, 3, 4)
    with pytest.warns(UserWarning, match='Mismatch'):
        _ = operator(u)


def test_grid_sampling_op_errormsg_inputdim_2d() -> None:
    """Test if error for wrong input dimensions is raised."""
    grid = torch.ones(1, 1, 1, 2)
    input_shape = SpatialDimension(2, 3, 4)
    operator = GridSamplingOp(grid_z=None, grid_y=grid[..., 0], grid_x=grid[..., 1], input_shape=input_shape)
    u = torch.zeros(1, 3, 4)
    with pytest.raises(ValueError, match='4 dimensions: batch channel y x'):
        _ = operator(u)


def test_grid_sampling_op_warningmsg_inputshape_2d() -> None:
    """Test if warning for wrong input_shape is raised in forward"""
    grid = torch.ones(1, 1, 1, 2)
    input_shape = SpatialDimension(2, 3, 4)
    operator = GridSamplingOp(grid_z=None, grid_y=grid[..., 0], grid_x=grid[..., 1], input_shape=input_shape)
    u = torch.zeros(1, 2, 3, 5)
    with pytest.warns(UserWarning, match='Mismatch'):
        _ = operator(u)


def test_grid_sampling_op_errormsg_inputdim_z_2d() -> None:
    """Test if no error for wrong input dimensions is raised if only z is wrong for 2d."""
    grid = torch.ones(1, 1, 1, 2)
    input_shape = SpatialDimension(2, 3, 4)
    operator = GridSamplingOp(grid_z=None, grid_y=grid[..., 0], grid_x=grid[..., 1], input_shape=input_shape)
    u = torch.zeros(1, 17, 3, 4)
    _ = operator(u)  # works, as z is ignored.


@pytest.mark.parametrize(
    ('grid_batch', 'u_batch', 'channel', 'expected_output'),
    [
        ((1,), (1,), (1,), (1, 1)),
        ((7, 1, 2), (1, 8, 2), (2, 3), (7, 8, 2, 2, 3)),
        ((3,), (4,), (1,), 'not broadcastable'),
        ((7, 1, 2), (1, 1, 2), (4,), (7, 1, 2, 4)),
        ((7, 1, 2), (2,), (4,), 'not broadcastable'),
    ],
)
def test_grid_sampling_op_batchdims(
    grid_batch: tuple[int, ...],
    u_batch: tuple[int, ...],
    channel: tuple[int, ...],
    expected_output: tuple[int, ...] | str,
) -> None:
    """Test if error for wrong input dimensions is raised."""
    grid = torch.ones(*grid_batch, 7, 8, 9, 3)  # 3d
    input_shape = SpatialDimension(2, 3, 4)
    u = torch.zeros(*u_batch, *channel, *input_shape.zyx)
    operator = GridSamplingOp(grid_z=grid[..., 0], grid_y=grid[..., 1], grid_x=grid[..., 2], input_shape=input_shape)
    if isinstance(expected_output, str):
        with pytest.raises(ValueError, match=expected_output):
            _ = operator(u)
    else:
        (result,) = operator(u)
        assert result.shape == (*expected_output, 7, 8, 9)


# MRtwo uses (z,y,x)-convention for grid sampling
# PyTorch uses (x,y,z)-convention for grid sampling
@pytest.mark.parametrize(('dim', 'grid_sample_dim'), [(-1, -3), (-2, -2), (-3, -1)])
def test_grid_sampling_op_orientation(dim: int, grid_sample_dim: int) -> None:
    """Test orientation of transformation."""
    phantom = torch.zeros(1, 1, 20, 30, 40)
    phantom[..., 5:15, 10:20, 10:30] = 1

    # shift phantom along dim
    shift = 5
    phantom_shifted = torch.roll(phantom, shifts=shift, dims=dim)

    # create grid
    unity_matrix = torch.cat((torch.eye(3), torch.zeros(3, 1)), dim=1).unsqueeze(0)
    grid = torch.nn.functional.affine_grid(unity_matrix, list(phantom.shape), align_corners=False)
    grid[..., grid_sample_dim] -= shift / phantom.shape[dim] * 2
    grid[grid > 1] = 1
    grid[grid < -1] = -1
    operator = GridSamplingOp(
        grid_z=grid[..., 2], grid_y=grid[..., 1], grid_x=grid[..., 0], interpolation_mode='nearest'
    )

    torch.testing.assert_close(phantom_shifted, operator(phantom)[0])


def test_grid_sampling_op_from_displacement_3d() -> None:
    """Test transformation created from displacement."""
    phantom = torch.zeros(3, 4, 20, 30, 40)
    phantom[..., 6:10, 10:20, 10:30] = 1

    # shift phantom along dim
    shift = (2, 3, 4)
    phantom_shifted = torch.roll(phantom, shifts=shift, dims=(-3, -2, -1))

    # Create displacement with border to avoid shifts outside of the image
    displacement = torch.zeros(3, 20, 30, 40, 3)
    displacement[:, 5:-5, 5:-5, 5:-5, 0] = -shift[0]
    displacement[:, 5:-5, 5:-5, 5:-5, 1] = -shift[1]
    displacement[:, 5:-5, 5:-5, 5:-5, 2] = -shift[2]

    operator = GridSamplingOp.from_displacement(
        displacement_z=displacement[..., 0],
        displacement_y=displacement[..., 1],
        displacement_x=displacement[..., 2],
        interpolation_mode='nearest',
    )

    torch.testing.assert_close(phantom_shifted, operator(phantom)[0])


def test_grid_sampling_op_from_displacement_2d() -> None:
    """Test transformation created from displacement."""
    phantom = torch.zeros(3, 4, 20, 30, 40)
    phantom[..., 6:10, 10:20, 10:30] = 1

    # shift phantom along dim
    shift = (3, 4)
    phantom_shifted = torch.roll(phantom, shifts=shift, dims=(-2, -1))

    # Create displacement with border to avoid shifts outside of the image
    displacement = torch.zeros(3, 30, 40, 2)
    displacement[:, 5:-5, 5:-5, 0] = -shift[0]
    displacement[:, 5:-5, 5:-5, 1] = -shift[1]

    operator = GridSamplingOp.from_displacement(
        displacement_z=None,
        displacement_y=displacement[..., 0],
        displacement_x=displacement[..., 1],
        interpolation_mode='nearest',
    )

    torch.testing.assert_close(phantom_shifted, operator(phantom)[0])


@pytest.mark.parametrize('dim', [3, 2])
def test_grid_sampling_op_to_displacement(dim: int) -> None:
    """Test conversion from grid back to displacement."""
    if dim == 3:
        displacement = torch.zeros(2, 5, 6, 7, 3)
        displacement[:, 1:-1, 1:-1, 1:-1, 0] = 1.0
        displacement[:, 1:-1, 1:-1, 1:-1, 1] = -1.0
        displacement[:, 1:-1, 1:-1, 1:-1, 2] = 0.5
        operator = GridSamplingOp.from_displacement(
            displacement_z=displacement[..., 0],
            displacement_y=displacement[..., 1],
            displacement_x=displacement[..., 2],
        )
    else:
        displacement = torch.zeros(2, 6, 7, 2)
        displacement[:, 1:-1, 1:-1, 0] = -1.0
        displacement[:, 1:-1, 1:-1, 1] = 0.5
        operator = GridSamplingOp.from_displacement(
            displacement_z=None,
            displacement_y=displacement[..., 0],
            displacement_x=displacement[..., 1],
        )

    displacement_recovered = operator.to_displacement().movedim(1, -1)
    torch.testing.assert_close(displacement_recovered, displacement)


def test_grid_sampling_op_from_affine_3d_identity() -> None:
    """Test identity transformation created from affine matrix in 3D."""
    image = torch.zeros(2, 3, 4, 8, 9, 10)
    image[..., 2:6, 2:7, 3:8] = 1

    affine = torch.zeros(2, 3, 3, 4)
    affine[..., 0, 0] = 1
    affine[..., 1, 1] = 1
    affine[..., 2, 2] = 1

    operator = GridSamplingOp.from_affine(
        affine,
        input_shape=SpatialDimension(8, 9, 10),
        interpolation_mode='nearest',
        padding_mode='border',
    )
    torch.testing.assert_close(image, operator(image)[0])


def test_grid_sampling_op_from_affine_2d_identity() -> None:
    """Test identity transformation created from affine matrix in 2D."""
    image = torch.zeros(2, 3, 4, 5, 10, 11)
    image[..., 2:7, 3:8] = 1

    affine = torch.zeros(2, 3, 2, 3)
    affine[..., 0, 0] = 1
    affine[..., 1, 1] = 1

    operator = GridSamplingOp.from_affine(
        affine,
        input_shape=SpatialDimension(1, 10, 11),
        interpolation_mode='nearest',
        padding_mode='border',
    )
    torch.testing.assert_close(image, operator(image)[0])


def test_grid_sampling_op_from_affine_unbatched_2d_identity() -> None:
    """Test unbatched 2D affine identity."""
    image = torch.zeros(1, 3, 10, 11)
    image[..., 2:7, 3:8] = 1

    affine = torch.zeros(2, 3)
    affine[0, 0] = 1
    affine[1, 1] = 1

    operator = GridSamplingOp.from_affine(
        affine,
        input_shape=SpatialDimension(1, 10, 11),
        interpolation_mode='nearest',
        padding_mode='border',
    )
    torch.testing.assert_close(image, operator(image)[0])


def test_grid_sampling_op_from_affine_unbatched_3d_identity() -> None:
    """Test unbatched 3D affine identity."""
    image = torch.zeros(1, 3, 8, 9, 10)
    image[..., 2:6, 2:7, 3:8] = 1

    affine = torch.zeros(3, 4)
    affine[0, 0] = 1
    affine[1, 1] = 1
    affine[2, 2] = 1

    operator = GridSamplingOp.from_affine(
        affine,
        input_shape=SpatialDimension(8, 9, 10),
        interpolation_mode='nearest',
        padding_mode='border',
    )
    torch.testing.assert_close(image, operator(image)[0])


def test_grid_sampling_op_from_bspline_3d_zero_displacement() -> None:
    """Test from_bspline with zero control points in 3D."""
    shape = SpatialDimension(8, 9, 10)
    spacing = SpatialDimension(4, 4, 4)
    control_points_shape = (shape - 1) // spacing + 4
    image = torch.zeros(2, 3, *shape.zyx)
    image[..., 2:6, 2:7, 3:8] = 1
    control_points_z = torch.zeros(2, *control_points_shape.zyx)
    control_points_y = torch.zeros(2, *control_points_shape.zyx)
    control_points_x = torch.zeros(2, *control_points_shape.zyx)

    operator = GridSamplingOp.from_bspline(
        control_points_z,
        control_points_y,
        control_points_x,
        input_shape=SpatialDimension(8, 9, 10),
        control_point_spacing=spacing,
        interpolation_mode='nearest',
        padding_mode='border',
    )
    torch.testing.assert_close(image, operator(image)[0])


def test_grid_sampling_op_from_bspline_2d_zero_displacement() -> None:
    """Test from_bspline with zero control points in 2D."""
    shape = SpatialDimension(1, 10, 11)
    image = torch.zeros(4, 2, 3, *shape.zyx)
    image[..., 2:7, 3:8] = 1
    spacing = SpatialDimension(1, 4, 4)
    control_points_shape = (shape - 1) // spacing + 4
    control_points_y = torch.zeros(4, 2, *control_points_shape.zyx[1:])
    control_points_x = torch.zeros(4, 2, *control_points_shape.zyx[1:])

    operator = GridSamplingOp.from_bspline(
        None,
        control_points_y,
        control_points_x,
        input_shape=shape,
        control_point_spacing=spacing,
        interpolation_mode='nearest',
        padding_mode='border',
    )
    torch.testing.assert_close(image, operator(image)[0])


def test_grid_sampling_op_from_bspline_return_displacement() -> None:
    """Test from_bspline can return dense displacement."""
    control_points_z = torch.zeros(2, 5, 5, 5)
    control_points_y = torch.zeros(2, 5, 5, 5)
    control_points_x = torch.zeros(2, 5, 5, 5)
    operator, displacement = GridSamplingOp.from_bspline(
        control_points_z,
        control_points_y,
        control_points_x,
        input_shape=SpatialDimension(8, 9, 10),
        control_point_spacing=SpatialDimension(4.0, 4.0, 4.0),
        interpolation_mode='nearest',
        padding_mode='border',
        return_displacement=True,
    )
    assert isinstance(operator, GridSamplingOp)
    torch.testing.assert_close(displacement, torch.zeros(2, 3, 8, 9, 10))


def test_grid_sampling_op_from_stationary_identity_3d() -> None:
    """Test from_stationary_velocity with zero velocity creates identity mapping."""
    velocity = torch.zeros(3, 2, 4, 5, 6)
    operator = GridSamplingOp.from_stationary_velocity(
        velocity[0],
        velocity[1],
        velocity[2],
    )
    image = RandomGenerator(0).float32_tensor((2, 3, 4, 5, 6), -1.0, 1.0)
    (result,) = operator(image)
    torch.testing.assert_close(result, image, atol=2e-5, rtol=2e-5)


def test_grid_sampling_op_from_stationary_identity_2d() -> None:
    """Test from_stationary_velocity with zero velocity creates identity mapping."""
    velocity = torch.zeros(2, 2, 5, 6)
    operator = GridSamplingOp.from_stationary_velocity(
        None,
        velocity[0],
        velocity[1],
    )
    image = RandomGenerator(0).float32_tensor((2, 3, 1, 5, 6), -1.0, 1.0)
    (result,) = operator(image)
    torch.testing.assert_close(result, image, atol=2e-5, rtol=2e-5)


def test_grid_sampling_op_matmul_composition() -> None:
    """Test composition of two GridSamplingOp with @."""
    image = torch.zeros(2, 3, 8, 9, 10)
    image[..., 2:6, 2:7, 3:8] = 1

    affine = torch.zeros(2, 3, 4)
    affine[..., 0, 0] = 1
    affine[..., 1, 1] = 1
    affine[..., 2, 2] = 1
    affine_operator = GridSamplingOp.from_affine(
        affine,
        input_shape=SpatialDimension(8, 9, 10),
        interpolation_mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )
    spline_operator = GridSamplingOp.from_bspline(
        torch.zeros(2, 5, 5, 5),
        torch.zeros(2, 5, 5, 5),
        torch.zeros(2, 5, 5, 5),
        input_shape=SpatialDimension(8, 9, 10),
        control_point_spacing=SpatialDimension(4.0, 4.0, 4.0),
        interpolation_mode='bilinear',
        padding_mode='border',
    )

    joint_operator = spline_operator @ affine_operator
    (moved_affine,) = affine_operator(image)
    (moved_sequential,) = spline_operator(moved_affine)
    (moved_joint,) = joint_operator(image)
    torch.testing.assert_close(moved_joint, moved_sequential, atol=1e-5, rtol=1e-5)


def test_grid_sampling_op_matmul_composition_3d_2d() -> None:
    """Test composition of 3D and 2D GridSamplingOp."""
    image = torch.zeros(2, 3, 8, 9, 10)
    image[..., 2:6, 2:7, 3:8] = 1

    affine_3d = torch.zeros(2, 3, 4)
    affine_3d[..., 0, 0] = 1
    affine_3d[..., 1, 1] = 1
    affine_3d[..., 2, 2] = 1
    operator_3d = GridSamplingOp.from_affine(
        affine_3d,
        input_shape=SpatialDimension(8, 9, 10),
        interpolation_mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )

    affine_2d = torch.zeros(2, 2, 3)
    affine_2d[..., 0, 0] = 1
    affine_2d[..., 1, 1] = 1
    operator_2d = GridSamplingOp.from_affine(
        affine_2d,
        input_shape=SpatialDimension(1, 9, 10),
        interpolation_mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )

    joint_operator = operator_3d @ operator_2d
    (moved_2d,) = operator_2d(image)
    (moved_sequential,) = operator_3d(moved_2d)
    (moved_joint,) = joint_operator(image)
    torch.testing.assert_close(moved_joint, moved_sequential, atol=1e-5, rtol=1e-5)


def test_grid_sampling_op_matmul_composition_2d_3d() -> None:
    """Test composition of 2D and 3D GridSamplingOp."""
    image = torch.zeros(2, 3, 8, 9, 10)
    image[..., 2:6, 2:7, 3:8] = 1

    affine_2d = torch.zeros(2, 2, 3)
    affine_2d[..., 0, 0] = 1
    affine_2d[..., 1, 1] = 1
    operator_2d = GridSamplingOp.from_affine(
        affine_2d,
        input_shape=SpatialDimension(1, 9, 10),
        interpolation_mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )

    affine_3d = torch.zeros(2, 3, 4)
    affine_3d[..., 0, 0] = 1
    affine_3d[..., 1, 1] = 1
    affine_3d[..., 2, 2] = 1
    operator_3d = GridSamplingOp.from_affine(
        affine_3d,
        input_shape=SpatialDimension(8, 9, 10),
        interpolation_mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )

    joint_operator = operator_2d @ operator_3d
    (moved_3d,) = operator_3d(image)
    (moved_sequential,) = operator_2d(moved_3d)
    (moved_joint,) = joint_operator(image)
    torch.testing.assert_close(moved_joint, moved_sequential, atol=1e-5, rtol=1e-5)


@pytest.mark.cuda
@pytest.mark.parametrize('dim', [3, 2])
def test_grid_sampling_op_from_displacement_cuda(dim: int) -> None:
    """Test operator grid on cuda if the input displacement on cuda."""
    batch, coil = (2, 3), 3
    if dim == 3:
        zyx = (2, 4, 8)
        displacement_cuda: Any = torch.zeros(dim, *batch, *zyx, device='cuda').unbind(0)
        displacement_cpu: Any = torch.zeros(dim, *batch, *zyx, device='cpu').unbind(0)
    elif dim == 2:
        zyx = (1, 4, 8)
        displacement_cuda = (None, *torch.zeros(dim, *batch, *zyx, device='cuda').unbind(0))
        displacement_cpu = (None, *torch.zeros(dim, *batch, *zyx, device='cpu').unbind(0))

    image = torch.ones(*batch, coil, *zyx)

    operator_cuda = GridSamplingOp.from_displacement(*displacement_cuda)
    (result,) = operator_cuda(image.cuda())
    assert result.is_cuda

    operator_cpu = operator_cuda.cpu()
    (result,) = operator_cpu(image)
    assert result.is_cpu

    operator_cpu = GridSamplingOp.from_displacement(*displacement_cpu)
    (result,) = operator_cpu(image)
    assert result.is_cpu

    operator_cuda = operator_cpu.cuda()
    (result,) = operator_cuda(image.cuda())
    assert result.is_cuda


@pytest.mark.cuda
@pytest.mark.parametrize('dim', [3, 2])
def test_grid_sampling_op_from_affine_cuda(dim: int) -> None:
    """Test operator grid on cuda if the input affine is on cuda."""
    batch, coil = (2, 3), 3
    if dim == 3:
        zyx = (2, 4, 8)
        affine_cuda = torch.zeros(*batch, 3, 4, device='cuda')
        affine_cpu = torch.zeros(*batch, 3, 4, device='cpu')
        affine_cuda[..., 0, 0] = 1
        affine_cuda[..., 1, 1] = 1
        affine_cuda[..., 2, 2] = 1
        affine_cpu[..., 0, 0] = 1
        affine_cpu[..., 1, 1] = 1
        affine_cpu[..., 2, 2] = 1
        input_shape = SpatialDimension(*zyx)
    elif dim == 2:
        zyx = (1, 4, 8)
        affine_cuda = torch.zeros(*batch, 2, 3, device='cuda')
        affine_cpu = torch.zeros(*batch, 2, 3, device='cpu')
        affine_cuda[..., 0, 0] = 1
        affine_cuda[..., 1, 1] = 1
        affine_cpu[..., 0, 0] = 1
        affine_cpu[..., 1, 1] = 1
        input_shape = SpatialDimension(*zyx)

    image = torch.ones(*batch, coil, *zyx)

    operator_cuda = GridSamplingOp.from_affine(affine_cuda, input_shape=input_shape)
    (result,) = operator_cuda(image.cuda())
    assert result.is_cuda

    operator_cpu = operator_cuda.cpu()
    (result,) = operator_cpu(image)
    assert result.is_cpu

    operator_cpu = GridSamplingOp.from_affine(affine_cpu, input_shape=input_shape)
    (result,) = operator_cpu(image)
    assert result.is_cpu

    operator_cuda = operator_cpu.cuda()
    (result,) = operator_cuda(image.cuda())
    assert result.is_cuda


@pytest.mark.cuda
@pytest.mark.parametrize('dim', [3, 2])
def test_grid_sampling_op_from_bspline_cuda(dim: int) -> None:
    """Test operator grid on cuda if the input B-spline control points are on cuda."""
    batch, coil = (2, 3), 3
    if dim == 3:
        zyx = (2, 4, 8)
        control_points_z_cuda = torch.zeros(*batch, 5, 6, 7, device='cuda')
        control_points_y_cuda = torch.zeros(*batch, 5, 6, 7, device='cuda')
        control_points_x_cuda = torch.zeros(*batch, 5, 6, 7, device='cuda')
        control_points_z_cpu = torch.zeros(*batch, 5, 6, 7, device='cpu')
        control_points_y_cpu = torch.zeros(*batch, 5, 6, 7, device='cpu')
        control_points_x_cpu = torch.zeros(*batch, 5, 6, 7, device='cpu')
        input_shape = SpatialDimension(*zyx)
        spacing = SpatialDimension(2.0, 2.0, 2.0)
    elif dim == 2:
        zyx = (1, 4, 8)
        control_points_z_cuda = None
        control_points_y_cuda = torch.zeros(*batch, 6, 7, device='cuda')
        control_points_x_cuda = torch.zeros(*batch, 6, 7, device='cuda')
        control_points_z_cpu = None
        control_points_y_cpu = torch.zeros(*batch, 6, 7, device='cpu')
        control_points_x_cpu = torch.zeros(*batch, 6, 7, device='cpu')
        input_shape = SpatialDimension(*zyx)
        spacing = SpatialDimension(1.0, 2.0, 2.0)

    image = torch.ones(*batch, coil, *zyx)

    operator_cuda = GridSamplingOp.from_bspline(
        control_points_z_cuda,
        control_points_y_cuda,
        control_points_x_cuda,
        input_shape=input_shape,
        control_point_spacing=spacing,
    )
    (result,) = operator_cuda(image.cuda())
    assert result.is_cuda

    operator_cpu = operator_cuda.cpu()
    (result,) = operator_cpu(image)
    assert result.is_cpu

    operator_cpu = GridSamplingOp.from_bspline(
        control_points_z_cpu,
        control_points_y_cpu,
        control_points_x_cpu,
        input_shape=input_shape,
        control_point_spacing=spacing,
    )
    (result,) = operator_cpu(image)
    assert result.is_cpu

    operator_cuda = operator_cpu.cuda()
    (result,) = operator_cuda(image.cuda())
    assert result.is_cuda


@pytest.mark.cuda
@pytest.mark.parametrize('dim', [3, 2])
def test_grid_sampling_op_from_stationary_velocity_cuda(dim: int) -> None:
    """Test operator grid on cuda if the input stationary velocity is on cuda."""
    batch, coil = (2, 3), 3
    if dim == 3:
        zyx = (2, 4, 8)
        velocity_cuda: Any = torch.zeros(dim, *batch, *zyx, device='cuda').unbind(0)
        velocity_cpu: Any = torch.zeros(dim, *batch, *zyx, device='cpu').unbind(0)
    elif dim == 2:
        zyx = (1, 4, 8)
        velocity_cuda = (None, *torch.zeros(dim, *batch, *zyx, device='cuda').unbind(0))
        velocity_cpu = (None, *torch.zeros(dim, *batch, *zyx, device='cpu').unbind(0))

    image = torch.ones(*batch, coil, *zyx)

    operator_cuda = GridSamplingOp.from_stationary_velocity(*velocity_cuda)
    (result,) = operator_cuda(image.cuda())
    assert result.is_cuda

    operator_cpu = operator_cuda.cpu()
    (result,) = operator_cpu(image)
    assert result.is_cpu

    operator_cpu = GridSamplingOp.from_stationary_velocity(*velocity_cpu)
    (result,) = operator_cpu(image)
    assert result.is_cpu

    operator_cuda = operator_cpu.cuda()
    (result,) = operator_cuda(image.cuda())
    assert result.is_cuda
