"""Class for Grid Sampling Operator."""

import warnings
from collections.abc import Callable, Sequence
from itertools import product
from typing import Literal, overload

import torch
from typing_extensions import Self, Unpack

from mr2.data.SpatialDimension import SpatialDimension
from mr2.operators.LinearOperator import LinearOperator
from mr2.operators.Operator import Operator, Tin2
from mr2.utils.reshape import unsqueeze_left


class _AdjointGridSampleCtx(torch.autograd.function.FunctionCtx):
    """Context for Adjoint Grid Sample, used for type hinting."""

    shape: Sequence[int]
    interpolation_mode: int
    padding_mode: int
    align_corners: bool
    xshape: Sequence[int]
    backward_2d_or_3d: Callable
    saved_tensors: Sequence[torch.Tensor]
    needs_input_grad: Sequence[bool]


class AdjointGridSample(torch.autograd.Function):
    """Autograd Function for Adjoint Grid Sample.

    Ensures that the Adjoint Operation is differentiable.

    """

    @staticmethod
    def forward(
        y: torch.Tensor,
        grid: torch.Tensor,
        xshape: Sequence[int],
        interpolation_mode: Literal['bilinear', 'nearest', 'bicubic'] = 'bilinear',
        padding_mode: Literal['zeros', 'border', 'reflection'] = 'zeros',
        align_corners: bool = True,
    ) -> tuple[torch.Tensor, tuple[int, int, Callable]]:
        """Adjoint of the linear operator x->gridsample(x,grid).

        Parameters
        ----------
        ctx
            Context
        y
            tensor in the range of gridsample(x,grid). Should not include batch or channel dimension.
        grid
            grid in the shape `(*y.shape, 2/3)`
        xshape
            shape of the domain of gridsample(x,grid), i.e. the shape of `x`
        interpolation_mode
            the kind of interpolation used
        padding_mode
            how to pad the input
        align_corners
             if True, the corner pixels of the input and output tensors are aligned,
             and thus preserve the values at those pixels

        """
        # grid_sampler_and_backward uses integer values instead of strings for the modes
        match interpolation_mode:
            case 'bilinear':
                mode_enum = 0
            case 'nearest':
                mode_enum = 1
            case 'bicubic':
                mode_enum = 2
            case _:
                raise ValueError(f'Interpolation mode {interpolation_mode} not supported')

        match padding_mode:
            case 'zeros':
                padding_mode_enum = 0
            case 'border':
                padding_mode_enum = 1
            case 'reflection':
                padding_mode_enum = 2
            case _:
                raise ValueError(f'Padding mode {padding_mode} not supported')

        match dim := grid.shape[-1]:
            case 3:
                backward_2d_or_3d = torch.ops.aten.grid_sampler_3d_backward
            case 2:
                backward_2d_or_3d = torch.ops.aten.grid_sampler_2d_backward
            case _:
                raise ValueError(f'only 2d and 3d supported, not {dim}')

        if y.shape[0] != grid.shape[0]:
            raise ValueError(f'y and grid must have same batch size, got {y.shape=}, {grid.shape=}')
        if xshape[1] != y.shape[1]:
            raise ValueError(f'xshape and y must have same number of channels, got {xshape[1]} and {y.shape[1]}.')
        if len(xshape) - 2 != dim:
            raise ValueError(f'len(xshape) and dim must either both bei 2 or 3, got {len(xshape)} and {dim}')
        dummy = torch.empty(1, dtype=y.dtype, device=y.device).broadcast_to(xshape)
        x = backward_2d_or_3d(
            y,
            dummy,  # only the shape, device and dtype are relevant
            grid,
            interpolation_mode=mode_enum,
            padding_mode=padding_mode_enum,
            align_corners=align_corners,
            output_mask=[True, False],
        )[0]

        return x, (mode_enum, padding_mode_enum, backward_2d_or_3d)

    @staticmethod
    def setup_context(
        ctx: _AdjointGridSampleCtx,
        inputs: tuple[torch.Tensor, torch.Tensor, Sequence[int], str, str, bool],
        outputs: tuple[torch.Tensor, tuple[int, int, Callable]],
    ) -> None:
        """Save information for backward pass."""
        y, grid, xshape, _, _, align_corners = inputs
        _, (mode_enum, padding_mode_enum, backward_2d_or_3d) = outputs
        ctx.xshape = xshape
        ctx.interpolation_mode = mode_enum
        ctx.padding_mode = padding_mode_enum
        ctx.align_corners = align_corners
        ctx.backward_2d_or_3d = backward_2d_or_3d

        if ctx.needs_input_grad[1]:
            # only if we need to calculate the gradient for grid we need y
            ctx.save_for_backward(grid, y)
        else:
            ctx.save_for_backward(grid)

    @staticmethod
    def backward(
        ctx: _AdjointGridSampleCtx, *grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None, None]:
        """Backward of the Adjoint Gridsample Operator."""
        need_y_grad, need_grid_grad, *_ = ctx.needs_input_grad
        grid = ctx.saved_tensors[0]

        if need_y_grad:
            # torch.grid_sampler has the same signature as the backward
            # (and is used inside F.grid_sample)
            grad_y = torch.grid_sampler(
                grad_output[0],
                grid,
                ctx.interpolation_mode,
                ctx.padding_mode,
                ctx.align_corners,
            )
        else:
            grad_y = None

        if need_grid_grad:
            y = ctx.saved_tensors[1]
            grad_grid = ctx.backward_2d_or_3d(
                y,
                grad_output[0],
                grid,
                interpolation_mode=ctx.interpolation_mode,
                padding_mode=ctx.padding_mode,
                align_corners=ctx.align_corners,
                output_mask=[False, True],
            )[1]
        else:
            grad_grid = None

        return grad_y, grad_grid, None, None, None, None


class GridSamplingOp(LinearOperator):
    """Grid Sampling Operator.

    Given an "input" tensor and a "grid", computes the output by taking the input values at the locations
    determined by grid with interpolation. Thus, the output size will be determined by the grid size.
    For the adjoint to be defined, the grid and the shape of the "input" has to be known.
    """

    grid: torch.Tensor

    def __init__(
        self,
        grid_z: torch.Tensor | None,
        grid_y: torch.Tensor,
        grid_x: torch.Tensor,
        input_shape: SpatialDimension | None = None,
        interpolation_mode: Literal['bilinear', 'nearest', 'bicubic'] = 'bilinear',
        padding_mode: Literal['zeros', 'border', 'reflection'] = 'zeros',
        align_corners: bool = False,
    ):
        r"""Initialize Sampling Operator.

        Parameters
        ----------
        grid_z
            Z-component of sampling grid. Shape `(*batchdim, z,y,x)`. Values should be in ``[-1, 1.]``. Use `None` for a
            2D interpolation along `y` and `x`.
        grid_y
            Y-component of sampling grid. Shape `(*batchdim, z, y, x)` or `(*batchdim, y, x)` if `grid_z` is `None`.
            Values should be in ``[-1, 1.]``.
        grid_x
            X-component of sampling grid. Shape `(*batchdim, z, y, x)` or `(*batchdim, y, x)` if `grid_z` is `None`.
            Values should be in ``[-1, 1.]``.
        input_shape
            Used in the adjoint. The z, y, x shape of the domain of the operator.
            If `grid_z` is `None`, only y and x will be used.
        interpolation_mode
            mode used for interpolation. bilinear is trilinear in 3D, bicubic is only supported in 2D.
        padding_mode
            how the input of the forward is padded.
        align_corners
            if True, the corner pixels of the input and output tensors are aligned,
            and thus preserve the values at those pixels
        """
        super().__init__()

        if grid_y.shape != grid_x.shape:
            raise ValueError('Grid y,x should have the same shape.')
        if grid_z is not None:
            if grid_z.shape != grid_y.shape:
                raise ValueError('Grid z,y,x should have the same shape.')
            if grid_x.ndim < 4:
                raise ValueError(
                    f'For a 3D gridding, grid should have at least 4 dimensions: batch z y x. Got shape {grid_x.shape}.'
                )
            if interpolation_mode == 'bicubic':
                raise NotImplementedError('Bicubic only implemented for 2D')
            grid = torch.stack((grid_x, grid_y, grid_z), dim=-1)  # pytorch expects grid components x,y,z for 3D
        else:
            if grid_x.ndim < 3:
                raise ValueError(
                    f'For a 2D gridding, grid should have at least 3 dimensions: batch y x. Got shape {grid_x.shape}.'
                )
            grid = torch.stack((grid_x, grid_y), dim=-1)  # pytorch expects grid components x,y for 2D

        if not grid.is_floating_point():
            raise ValueError(f'Grid should be a real floating dtype, got {grid.dtype}')
        if grid.max() > 1.0 or grid.min() < -1.0:
            warnings.warn('Grid has values outside range [-1,1].', stacklevel=1)

        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode
        self.grid = grid
        self.input_shape = SpatialDimension.from_array_zyx(grid.shape[-4:-1]) if input_shape is None else input_shape
        self.align_corners = align_corners

    @classmethod
    def from_affine(
        cls,
        affine_matrix: torch.Tensor,
        input_shape: SpatialDimension[int],
        interpolation_mode: Literal['bilinear', 'nearest', 'bicubic'] = 'bilinear',
        padding_mode: Literal['zeros', 'border', 'reflection'] = 'zeros',
        align_corners: bool = True,
    ) -> Self:
        """Create a GridSamplingOp from an affine matrix.

        Parameters
        ----------
        affine_matrix
            Affine matrix of shape ``(*batch, 2, 3)`` for 2D or ``(*batch, 3, 4)`` for 3D.
        input_shape
            Spatial input shape in ``(z, y, x)``.
            For 2D affine matrices, only ``y`` and ``x`` are used.
        interpolation_mode
            mode used for interpolation. bilinear is trilinear in 3D, bicubic is only supported in 2D.
        padding_mode
            how the input of the forward is padded.
        align_corners
            if True, the corner pixels of the input and output tensors are aligned.
            Generated grids are clipped to ``[-1, 1]`` to avoid invalid sampling coordinates.
        """
        match affine_matrix.shape[-2:]:
            case (2, 3):
                dim = 2
            case (3, 4):
                dim = 3
            case _:
                raise ValueError(f'affine_matrix must end with (2,3) or (3,4), got {affine_matrix.shape}.')

        shape = input_shape.zyx[-dim:]
        shape_batch = affine_matrix.shape[:-2]
        affine_flatbatch = affine_matrix.unsqueeze(0) if affine_matrix.ndim == 2 else affine_matrix.flatten(end_dim=-3)
        align_corners_affine = align_corners and min(shape) > 1

        affine_grid = torch.nn.functional.affine_grid(
            affine_flatbatch,
            size=[int(affine_flatbatch.shape[0]), 1, *[int(axis_size) for axis_size in shape]],
            align_corners=align_corners_affine,
        )
        # Keep normalized coordinates in range during optimization line-search steps.
        affine_grid = affine_grid.clamp(-1.0, 1.0)
        affine_grid = affine_grid.reshape(*(shape_batch or (1,)), *affine_grid.shape[1:])
        return cls(
            affine_grid[..., 2] if dim == 3 else None,
            affine_grid[..., 1],
            affine_grid[..., 0],
            input_shape,
            interpolation_mode,
            padding_mode,
            align_corners=align_corners_affine,
        )

    @classmethod
    @overload
    def from_bspline(
        cls,
        control_points_z: torch.Tensor | None,
        control_points_y: torch.Tensor,
        control_points_x: torch.Tensor,
        input_shape: SpatialDimension[int],
        control_point_spacing: SpatialDimension[float],
        interpolation_mode: Literal['bilinear', 'nearest', 'bicubic'] = 'bilinear',
        padding_mode: Literal['zeros', 'border', 'reflection'] = 'zeros',
        *,
        return_displacement: Literal[False] = False,
    ) -> Self: ...

    @classmethod
    @overload
    def from_bspline(
        cls,
        control_points_z: torch.Tensor | None,
        control_points_y: torch.Tensor,
        control_points_x: torch.Tensor,
        input_shape: SpatialDimension[int],
        control_point_spacing: SpatialDimension[float],
        interpolation_mode: Literal['bilinear', 'nearest', 'bicubic'] = 'bilinear',
        padding_mode: Literal['zeros', 'border', 'reflection'] = 'zeros',
        *,
        return_displacement: Literal[True],
    ) -> tuple[Self, torch.Tensor]: ...

    @classmethod
    def from_bspline(
        cls,
        control_points_z: torch.Tensor | None,
        control_points_y: torch.Tensor,
        control_points_x: torch.Tensor,
        input_shape: SpatialDimension[int],
        control_point_spacing: SpatialDimension[float],
        interpolation_mode: Literal['bilinear', 'nearest', 'bicubic'] = 'bilinear',
        padding_mode: Literal['zeros', 'border', 'reflection'] = 'zeros',
        *,
        return_displacement: bool = False,
    ) -> Self | tuple[Self, torch.Tensor]:
        """Create a GridSamplingOp from cubic B-spline control points.

        Parameters
        ----------
        control_points_z
            Z-component of control points. Use ``None`` for 2D.
            Shape ``(*batch, zc, yc, xc)``.
        control_points_y
            Y-component of control points. Shape ``(*batch, zc, yc, xc)`` for 3D
            or ``(*batch, yc, xc)`` for 2D.
        control_points_x
            X-component of control points. Shape ``(*batch, zc, yc, xc)`` for 3D
            or ``(*batch, yc, xc)`` for 2D.
        input_shape
            Spatial input shape in ``(z, y, x)``.
            For 2D, set ``z=1``.
        control_point_spacing
            Control-point spacing in voxel units in ``(z, y, x)``.
            For 2D, set the z-spacing to any positive value.
        interpolation_mode
            mode used for interpolation. bilinear is trilinear in 3D, bicubic is only supported in 2D.
        padding_mode
            how the input of the forward is padded.
        return_displacement
            If ``True``, return dense voxel displacement together with the operator.
        """
        if control_points_z is None:
            dim = 2
            if control_points_y.ndim < 3 or control_points_x.ndim < 3:
                raise ValueError(
                    'For 2D B-spline, control points should have at least 3 dimensions: batch y x. '
                    f'Got shape {control_points_y.shape} and {control_points_x.shape}.'
                )
            control_points = torch.stack(torch.broadcast_tensors(control_points_y, control_points_x), dim=-3)
        else:
            dim = 3
            if control_points_z.ndim < 4 or control_points_y.ndim < 4 or control_points_x.ndim < 4:
                raise ValueError(
                    'For 3D B-spline, control points should have at least 4 dimensions: batch z y x. '
                    f'Got shape {control_points_z.shape}, {control_points_y.shape}, {control_points_x.shape}.'
                )
            control_points = torch.stack(
                torch.broadcast_tensors(control_points_z, control_points_y, control_points_x), dim=-4
            )
        shape = tuple(int(size) for size in input_shape.zyx[-dim:])
        spacing = tuple(float(size) for size in control_point_spacing.zyx[-dim:])

        shape_batch = control_points.shape[: -(dim + 1)]
        # Flatten arbitrary batch dimensions for dense displacement evaluation.
        control_points_flatbatch = (
            unsqueeze_left(control_points, 1) if len(shape_batch) == 0 else control_points.flatten(end_dim=-(dim + 2))
        )

        _, _, *shape_ctrl = control_points_flatbatch.shape
        device = control_points_flatbatch.device
        dtype = control_points_flatbatch.dtype
        coordinate_axes = [torch.arange(size, device=device, dtype=dtype) for size in shape]
        mesh = torch.meshgrid(*coordinate_axes, indexing='ij')

        floor_indices_per_dim: list[torch.Tensor] = []
        basis_weights_per_dim: list[torch.Tensor] = []
        for coordinate, spacing_axis, control_size in zip(mesh, spacing, shape_ctrl, strict=True):
            scaled_coordinate = coordinate / spacing_axis + 1.0
            floor_index = torch.floor(scaled_coordinate).to(torch.int64)
            fractional = (scaled_coordinate - floor_index.to(dtype)).clamp(0.0, 1.0 - 1e-7)
            fractional2 = fractional.square()
            fractional3 = fractional2 * fractional
            basis = torch.stack(
                (
                    (1.0 - fractional).pow(3) / 6.0,
                    (3.0 * fractional3 - 6.0 * fractional2 + 4.0) / 6.0,
                    (-3.0 * fractional3 + 3.0 * fractional2 + 3.0 * fractional + 1.0) / 6.0,
                    fractional3 / 6.0,
                ),
                dim=-1,
            )
            floor_indices_per_dim.append((floor_index - 1).clamp(0, control_size - 4))
            basis_weights_per_dim.append(basis)

        displacement = control_points_flatbatch.new_zeros((control_points_flatbatch.shape[0], dim, *shape))
        for basis_offset in product(range(4), repeat=dim):
            contribution_weight = torch.ones(shape, device=device, dtype=dtype)
            index_components: list[torch.Tensor] = []
            for axis, offset in enumerate(basis_offset):
                contribution_weight = contribution_weight * basis_weights_per_dim[axis][..., offset]
                index_components.append(floor_indices_per_dim[axis] + offset)
            index_tuple: tuple[slice | torch.Tensor, ...] = (slice(None), slice(None), *index_components)
            sampled_control_points = control_points_flatbatch[index_tuple]
            displacement = displacement + sampled_control_points * contribution_weight.unsqueeze(0).unsqueeze(0)

        # Move component axis to front to match from_displacement signature.
        displacement = displacement.reshape(*(shape_batch or (1,)), *displacement.shape[1:])
        component_axis = displacement.ndim - dim - 1
        displacement_components = displacement.movedim(component_axis, 0)
        operator = cls.from_displacement(
            displacement_components[0] if dim == 3 else None,
            displacement_components[1] if dim == 3 else displacement_components[0],
            displacement_components[2] if dim == 3 else displacement_components[1],
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
        )
        if return_displacement:
            return operator, displacement
        return operator

    @overload  # type: ignore[override]
    def __matmul__(self, other: Self) -> Self: ...

    @overload
    def __matmul__(self, other: LinearOperator) -> LinearOperator: ...

    @overload
    def __matmul__(
        self, other: Operator[Unpack[Tin2], tuple[torch.Tensor,]] | Operator[Unpack[Tin2], tuple[torch.Tensor, ...]]
    ) -> Operator[Unpack[Tin2], tuple[torch.Tensor,]]: ...

    def __matmul__(
        self,
        other: 'GridSamplingOp'
        | Operator[Unpack[Tin2], tuple[torch.Tensor,]]
        | LinearOperator
        | Operator[Unpack[Tin2], tuple[torch.Tensor, ...]],
    ) -> (
        Operator[Unpack[Tin2], tuple[torch.Tensor,]] | LinearOperator | Operator[Unpack[Tin2], tuple[torch.Tensor, ...]]
    ):
        """Operator composition.

        For ``GridSamplingOp @ GridSamplingOp``, compose both grids into one sampling op.
        """
        if not isinstance(other, GridSamplingOp):
            return super().__matmul__(other)

        dim = self.grid.shape[-1]
        dim_other = other.grid.shape[-1]
        if self.align_corners != other.align_corners:
            raise ValueError('Cannot compose GridSamplingOp with different align_corners.')
        if self.interpolation_mode != other.interpolation_mode:
            raise ValueError('Cannot compose GridSamplingOp with different interpolation_mode.')
        if self.padding_mode != other.padding_mode:
            raise ValueError('Cannot compose GridSamplingOp with different padding_mode.')

        if dim == dim_other:
            if self.input_shape.zyx[-dim:] != other.grid.shape[-dim - 1 : -1]:
                raise ValueError(
                    f'Cannot compose operators with mismatched shape: expected {self.input_shape.zyx[-dim:]}, '
                    f'got {other.grid.shape[-dim - 1 : -1]}.'
                )
            # Move coordinate channels next to batch dims so we can sample the grid as an image.
            (joint_grid_components,) = self(other.grid.movedim(-1, -dim - 1))
            joint_grid = joint_grid_components.movedim(-dim - 1, -1)
            input_shape = other.input_shape

        elif dim == 3 and dim_other == 2:
            if self.input_shape.zyx[-2:] != other.grid.shape[-3:-1]:
                raise ValueError(
                    f'Cannot compose operators with mismatched shape: expected {self.input_shape.zyx[-2:]}, '
                    f'got {other.grid.shape[-3:-1]}.'
                )
            other_grid_components = other.grid.movedim(-1, -3)
            other_grid_components = other_grid_components.unsqueeze(-3)
            other_grid_components = other_grid_components.expand(
                *other_grid_components.shape[:-3], int(self.input_shape.z), *other_grid_components.shape[-2:]
            )
            (joint_xy_components,) = self(other_grid_components)
            joint_xy_grid = joint_xy_components.movedim(-4, -1)
            joint_grid = torch.stack((joint_xy_grid[..., 0], joint_xy_grid[..., 1], self.grid[..., 2]), dim=-1)
            input_shape = SpatialDimension(self.input_shape.z, other.input_shape.y, other.input_shape.x)

        elif dim == 2 and dim_other == 3:
            if self.input_shape.zyx[-2:] != other.grid.shape[-3:-1]:
                raise ValueError(
                    f'Cannot compose operators with mismatched shape: expected {self.input_shape.zyx[-2:]}, '
                    f'got {other.grid.shape[-3:-1]}.'
                )
            (joint_grid_components,) = self(other.grid.movedim(-1, -4))
            joint_grid = joint_grid_components.movedim(-4, -1)
            input_shape = other.input_shape

        else:
            raise ValueError(f'Unsupported GridSamplingOp composition: {dim}D @ {dim_other}D.')

        return GridSamplingOp(
            joint_grid[..., 2] if joint_grid.shape[-1] == 3 else None,
            joint_grid[..., 1],
            joint_grid[..., 0],
            input_shape=input_shape,
            interpolation_mode=self.interpolation_mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

    @classmethod
    def from_displacement(
        cls,
        displacement_z: torch.Tensor | None,
        displacement_y: torch.Tensor,
        displacement_x: torch.Tensor,
        interpolation_mode: Literal['bilinear', 'nearest', 'bicubic'] = 'bilinear',
        padding_mode: Literal['zeros', 'border', 'reflection'] = 'zeros',
    ) -> Self:
        """Create a GridSamplingOp from a displacement.

        The displacement is expected to describe a pull operation in voxel units. Let's assume we have an input image
        :math:`i(x,y)` and a displacement :math:`d_x(x,y)` and :math:`d_y(x,y)` then the output image :math:`o(x,y)`
        will be calculated as:
        .. math::
            o(x,y) = i(x + d_x(x,y), y + d_y(x,y))

        Parameters
        ----------
        displacement_z
            Z-component of the displacement. Use `None` for a 2D interpolation along `y` and `x`.  Shape is
            `*batchdim, z,y,x`. Values should describe the displacement in voxel.
        displacement_y
            Y-component of sampling grid. Shape is `*batchdim, z,y,x` or `*batchdim, y,x` if `displacement_z` is
            `None`. Values should describe the displacement in voxel. Values should describe the displacement in voxel.
        displacement_x
            X-component of sampling grid. Shape is `*batchdim, z,y,x` or `*batchdim, y,x` if `displacement_z` is
            `None`. Values should describe the displacement in voxel. Values should describe the displacement in voxel.
        interpolation_mode
            mode used for interpolation. bilinear is trilinear in 3D, bicubic is only supported in 2D.
        padding_mode
            how the input of the forward is padded.

        Notes
        -----
        Generated normalized grids are clipped to ``[-1, 1]``.
        """
        if displacement_z is not None:  # 3D
            if displacement_x.ndim < 4 or displacement_y.ndim < 4 or displacement_z.ndim < 4:
                raise ValueError(
                    'For a 3D displacement, displacement should have at least 4 dimensions: batch z y x. ',
                    f'Got shape {displacement_x.shape}.',
                )
            try:
                *_, n_z, n_y, n_x = torch.broadcast_shapes(
                    displacement_z.shape, displacement_y.shape, displacement_x.shape
                )
            except RuntimeError:
                raise ValueError(
                    'Displacement dimensions are not broadcastable. '
                    f'Got shapes {displacement_z.shape}, {displacement_y.shape}, {displacement_x.shape}.'
                ) from None
            grid_z, grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, n_z),
                torch.linspace(-1, 1, n_y),
                torch.linspace(-1, 1, n_x),
                indexing='ij',
            )
            scale_z = 0.0 if n_z == 1 else 2 / (n_z - 1)
            scale_y = 0.0 if n_y == 1 else 2 / (n_y - 1)
            scale_x = 0.0 if n_x == 1 else 2 / (n_x - 1)
            grid_z = (grid_z.to(displacement_z) + displacement_z * scale_z).clamp(-1.0, 1.0)
            grid_y = (grid_y.to(displacement_y) + displacement_y * scale_y).clamp(-1.0, 1.0)
            grid_x = (grid_x.to(displacement_x) + displacement_x * scale_x).clamp(-1.0, 1.0)
            align_corners = n_z > 1 and n_y > 1 and n_x > 1
        else:  # 2D
            if displacement_x.ndim < 3 or displacement_y.ndim < 3:
                raise ValueError(
                    'For a 2D displacement, displacement should have at least 3 dimensions: batch y x. ',
                    f'Got shape {displacement_x.shape} and {displacement_y.shape}.',
                )
            try:
                *_, n_y, n_x = torch.broadcast_shapes(displacement_y.shape, displacement_x.shape)
            except RuntimeError:
                raise ValueError(
                    'Displacement dimensions are not broadcastable. '
                    f'Got shapes {displacement_y.shape}, {displacement_x.shape}.'
                ) from None
            grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, n_y), torch.linspace(-1, 1, n_x), indexing='ij')
            scale_y = 0.0 if n_y == 1 else 2 / (n_y - 1)
            scale_x = 0.0 if n_x == 1 else 2 / (n_x - 1)
            grid_y = (grid_y.to(displacement_y) + displacement_y * scale_y).clamp(-1.0, 1.0)
            grid_x = (grid_x.to(displacement_x) + displacement_x * scale_x).clamp(-1.0, 1.0)
            grid_z = None
            align_corners = n_y > 1 and n_x > 1
        return cls(grid_z, grid_y, grid_x, None, interpolation_mode, padding_mode, align_corners=align_corners)

    @classmethod
    def from_stationary_velocity(
        cls,
        velocity_z: torch.Tensor | None,
        velocity_y: torch.Tensor,
        velocity_x: torch.Tensor,
        squaring_steps: int = 7,
        interpolation_mode: Literal['bilinear', 'nearest', 'bicubic'] = 'bilinear',
        padding_mode: Literal['zeros', 'border', 'reflection'] = 'border',
    ) -> Self:
        """Create a diffeomorphic transform from a stationary velocity field.

        Uses scaling-and-squaring integration as in VoxelMorph.

        Parameters
        ----------
        velocity_z
            Z-component of stationary velocity in voxel units. Use ``None`` for 2D.
        velocity_y
            Y-component of stationary velocity in voxel units.
        velocity_x
            X-component of stationary velocity in voxel units.
        squaring_steps
            Number of squaring steps. Must be non-negative.
        interpolation_mode
            mode used for interpolation. bilinear is trilinear in 3D, bicubic is only supported in 2D.
        padding_mode
            how the input of the forward is padded.
        """
        if squaring_steps < 0:
            raise ValueError(f'squaring_steps must be non-negative, got {squaring_steps}.')

        scaling = float(2**squaring_steps)
        transform = cls.from_displacement(
            None if velocity_z is None else velocity_z / scaling,
            velocity_y / scaling,
            velocity_x / scaling,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
        )
        for _ in range(squaring_steps):
            transform = transform @ transform

        return transform

    def __reshape_wrapper(
        self, x: torch.Tensor, inner: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> tuple[torch.Tensor]:
        """Do all the reshaping pre- and post- sampling."""
        if x.is_complex():
            # apply to real and imaginary part separately
            (real,) = self.__reshape_wrapper(x.real, inner)
            (imag,) = self.__reshape_wrapper(x.imag, inner)
            return (torch.complex(real, imag),)

        # First, we need to do a lot of reshaping ..
        dim = self.grid.shape[-1]
        if x.ndim < dim + 2:
            raise ValueError(
                f'For a {dim}D sampling operation, x should have at least have {dim + 2} dimensions:'
                f' batch channel {"z y x" if dim == 3 else "y x"}.'
            )
        shape_grid_batch = self.grid.shape[: -dim - 1]  # the batch dimensions of grid
        n_batchdim = len(shape_grid_batch)
        shape_x_batch = x.shape[:n_batchdim]  # the batch dimensions of the input
        try:
            shape_batch = torch.broadcast_shapes(shape_x_batch, shape_grid_batch)
        except RuntimeError:
            raise ValueError(
                'Batch dimensions in x and grid are not broadcastable.'
                f' Got batch dimensions x: {shape_x_batch} and grid: {shape_grid_batch},'
                f' (shapes are x: {x.shape}, grid: {self.grid.shape}).'
            ) from None

        shape_channels = x.shape[n_batchdim:-dim]
        #   reshape to 3D: (*batch_dim) z y x 3 or 2D: (*batch_dim) y x 2
        grid_flatbatch = self.grid.broadcast_to(*shape_batch, *self.grid.shape[n_batchdim:]).flatten(
            end_dim=n_batchdim - 1
        )
        x_flatbatch = x.broadcast_to(*shape_batch, *x.shape[n_batchdim:]).flatten(end_dim=n_batchdim - 1)
        #   reshape to 3D: (*batch_dim) (*channel_dim) z y x or 2D: (*batch_dim) (*channel_dim) y x
        x_flatbatch_flatchannel = x_flatbatch.flatten(start_dim=1, end_dim=-dim - 1)

        # .. now we can perform the actual sampling implementation..
        sampled = inner(x_flatbatch_flatchannel, grid_flatbatch)

        # .. and reshape back.
        result = sampled.reshape(*shape_batch, *shape_channels, *sampled.shape[-dim:])
        return (result,)

    def _forward_implementation(
        self, x_flatbatch_flatchannel: torch.Tensor, grid_flatbatch: torch.Tensor
    ) -> torch.Tensor:
        """Apply the actual forward after reshaping."""
        sampled = torch.nn.functional.grid_sample(
            x_flatbatch_flatchannel,
            grid_flatbatch,
            mode=self.interpolation_mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

        return sampled

    def to_displacement(self) -> torch.Tensor:
        """Convert the normalized sampling grid to displacement in voxel units.

        Returns
        -------
            Displacement with shape ``(*batch, dim, *spatial)`` where ``dim`` is 2 or 3.
        """
        dim = self.grid.shape[-1]
        spatial_shape = self.grid.shape[-dim - 1 : -1]

        base_coordinates = [
            torch.linspace(-1, 1, n, device=self.grid.device, dtype=self.grid.dtype) for n in spatial_shape
        ]
        base_grid = torch.meshgrid(*base_coordinates, indexing='ij')

        scales = [0.0 if n == 1 else (n - 1) / 2 for n in spatial_shape]
        if dim == 3:
            return torch.stack(
                (
                    (self.grid[..., 2] - base_grid[0]) * scales[0],
                    (self.grid[..., 1] - base_grid[1]) * scales[1],
                    (self.grid[..., 0] - base_grid[2]) * scales[2],
                ),
                dim=-4,
            )

        return torch.stack(
            (
                (self.grid[..., 1] - base_grid[0]) * scales[0],
                (self.grid[..., 0] - base_grid[1]) * scales[1],
            ),
            dim=-3,
        )

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the GridSampling operator.

        This operator samples an input tensor `x` at locations specified by a grid.
        The grid coordinates are normalized to `[-1, 1]`. The output tensor's spatial
        dimensions are determined by the grid's dimensions. Interpolation is used
        if grid points do not fall exactly on input tensor elements.

        Parameters
        ----------
        x
            Input tensor to be sampled.

        Returns
        -------
            Output tensor containing sampled values.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply forward of GridSamplingOp.

        .. note::
            Prefer calling the instance of the GridSamplingOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        if (
            (x.shape[-1] != self.input_shape.x)
            or (x.shape[-2] != self.input_shape.y)
            or (x.shape[-3] != self.input_shape.z and self.grid.shape[-1] == 3)
        ):
            warnings.warn(
                'Mismatch between x.shape and input shape. Adjoint on the result will return the wrong shape.',
                stacklevel=1,
            )
        return self.__reshape_wrapper(x, self._forward_implementation)

    def _adjoint_implementation(
        self, x_flatbatch_flatchannel: torch.Tensor, grid_flatbatch: torch.Tensor
    ) -> torch.Tensor:
        """Apply the actual adjoint after reshaping."""
        dim = self.grid.shape[-1]
        shape = (*x_flatbatch_flatchannel.shape[:-dim], *self.input_shape.zyx[-dim:])
        sampled = AdjointGridSample.apply(
            x_flatbatch_flatchannel,
            grid_flatbatch,
            shape,
            self.interpolation_mode,
            self.padding_mode,
            self.align_corners,
        )[0]
        return sampled

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the adjoint of the GridSampling operator.

        This operation is the adjoint of the forward grid sampling. It effectively
        "scatters" the values from the input tensor `x` (which is in the grid's domain)
        back to a tensor in the original input domain of the forward operation,
        using the same grid and interpolation settings.

        Parameters
        ----------
        x
            Input tensor, corresponding to the output of the forward operation.

        Returns
        -------
            Output tensor in the original input domain of the forward operation.
        """
        return self.__reshape_wrapper(x, self._adjoint_implementation)
