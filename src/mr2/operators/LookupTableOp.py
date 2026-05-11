"""Lookup table operator."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import torch
from typing_extensions import TypeVarTuple, Unpack

from mr2.operators.Operator import Operator
from mr2.utils.interpolate import _interp_along_axis
from mr2.utils.reshape import normalize_index, unsqueeze_right
from mr2.utils.TensorList import TensorList

Tin = TypeVarTuple('Tin')


class LookupTableOp(Operator[Unpack[Tin], tuple[torch.Tensor,]]):
    r"""Interpolated lookup-table operator.

    This operator approximates a tensor-valued generating function on a regular
    rectangular grid and evaluates it by multilinear interpolation. It is
    intended as a surrogate for expensive signal models such as sequence-
    specific MRF simulations.

    If ``index_of_scaling_parameter`` is provided, that parameter is excluded
    from the lookup grid. The lookup table is then built for unit scale, and the
    interpolated result is multiplied by the scaling parameter in the forward
    pass.
    """

    def __init__(
        self,
        generating_function: Callable[[Unpack[Tin]], tuple[torch.Tensor,]],
        parameter_ranges: Sequence[tuple[float, float, int]],
        index_of_scaling_parameter: int | None = None,
    ) -> None:
        """Initialize the lookup-table operator.

        Parameters
        ----------
        generating_function
            Function mapping parameters to exactly one output tensor.
        parameter_ranges
            Regular rectangular grid specification for the non-scaling
            parameters. Each entry is ``(minimum, maximum, n_steps)`` and is
            interpreted via ``torch.linspace(minimum, maximum, n_steps)``.
            If ``index_of_scaling_parameter`` is not ``None``, the scaling
            parameter must be omitted from this list.
        index_of_scaling_parameter
            Optional index of a multiplicative scaling parameter. If provided,
            the lookup table is built for unit scale and the forward pass
            multiplies the interpolated result by the supplied scaling tensor.
        """
        super().__init__()
        if not parameter_ranges:
            raise ValueError('parameter_ranges must not be empty.')
        if any(n_steps < 2 for _, _, n_steps in parameter_ranges):
            raise ValueError('Each parameter range must contain at least two grid points.')

        self._n_lookup_parameters = len(parameter_ranges)
        self._scaling_position = (
            None
            if index_of_scaling_parameter is None
            else normalize_index(self._n_lookup_parameters + 1, index_of_scaling_parameter)
        )

        grid_axes = [
            torch.linspace(minimum, maximum, n_steps, dtype=torch.float32)
            for minimum, maximum, n_steps in parameter_ranges
        ]
        grid_sizes = tuple(len(axis) for axis in grid_axes)
        self._grid_axes = TensorList(grid_axes)

        mesh = torch.meshgrid(*grid_axes, indexing='ij')
        if self._scaling_position is None:
            function_args = mesh
        else:
            function_args = (
                *mesh[: self._scaling_position],
                torch.tensor(1.0),
                *mesh[self._scaling_position :],
            )

        (lut,) = generating_function(*cast(tuple[Unpack[Tin]], function_args))
        if lut.shape[-self._n_lookup_parameters :] != grid_sizes:
            raise ValueError(
                'LookupTableOp expects the generating_function output to broadcast over the lookup grid as trailing '
                f'dimensions {grid_sizes}, got shape {tuple(lut.shape)}.'
            )

        self._value_shape = lut.shape[: -self._n_lookup_parameters]
        self._value_ndim = len(self._value_shape)
        lut = lut.permute(
            *range(lut.ndim - self._n_lookup_parameters, lut.ndim),
            *range(lut.ndim - self._n_lookup_parameters),
        ).contiguous()
        self._lut = lut

    def __call__(self, *parameters: Unpack[Tin]) -> tuple[torch.Tensor,]:
        """Evaluate the interpolated lookup table.

        Parameters
        ----------
        *parameters
            Parameter tensors for the generating function. The tensors are
            broadcast to a common batch shape before interpolation. If a
            scaling parameter was configured at initialization, it is excluded
            from the lookup grid and applied multiplicatively to the
            interpolated unit-scale output.

        Returns
        -------
            interpolated output tensor.
        """
        return super().__call__(*parameters)

    def forward(self, *parameters: Unpack[Tin]) -> tuple[torch.Tensor,]:
        """Apply forward of LookupTableOp.

        .. note::
            Prefer calling the instance of the LookupTableOp as ``operator(x)``
            over directly calling this method.
        """
        n_total_parameters = self._n_lookup_parameters + (self._scaling_position is not None)
        if len(parameters) != n_total_parameters:
            raise ValueError(f'LookupTableOp expects {n_total_parameters} input parameters, got {len(parameters)}.')
        if not all(isinstance(p, torch.Tensor) for p in parameters):
            raise ValueError('All input parameters must be tensors.')

        parameters_broadcast = torch.broadcast_tensors(*parameters)
        batch_shape = parameters_broadcast[0].shape

        if self._scaling_position is None:
            lookup_parameters = parameters_broadcast
            scale = None
        else:
            scale = parameters_broadcast[self._scaling_position]
            lookup_parameters = tuple(p for i, p in enumerate(parameters_broadcast) if i != self._scaling_position)

        flattened_parameters = tuple(p.flatten() for p in lookup_parameters)
        scale_flat = None if scale is None else scale.flatten()
        n_points = flattened_parameters[0].numel()
        result_flat = self._lut.unsqueeze(0).expand(n_points, *self._lut.shape)

        for query, axis in zip(flattened_parameters, self._grid_axes, strict=True):
            result_flat = _interp_along_axis(query, axis, result_flat, axis=1)

        if scale_flat is not None:
            result_flat = unsqueeze_right(scale_flat, self._value_ndim) * result_flat

        result = result_flat.reshape(*batch_shape, *self._value_shape)
        if self._value_ndim:
            result = torch.moveaxis(
                result,
                tuple(range(-self._value_ndim, 0)),
                tuple(range(self._value_ndim)),
            ).contiguous()
        return (result,)
