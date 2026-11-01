"""Expose network weights as the inputs of an MR2 operator."""

from collections.abc import Sequence

import torch
from torch import nn
from torch.func import functional_call
from typing_extensions import Unpack

from mr2.operators.Operator import Operator


class _FunctionalModuleOp(Operator[Unpack[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]):
    """Evaluate a module using explicit trainable weights and forward inputs."""

    def __init__(self, module: nn.Module, names: Sequence[str]) -> None:
        """Store the module, parameter order and initial buffers."""
        super().__init__()
        self.module = module
        self.names = tuple(names)
        self._buffer_names = tuple(name for name, _ in module.named_buffers())
        self._buffer_state = nn.Module()
        for index, buffer in enumerate(module.buffers()):
            self._buffer_state.register_buffer(str(index), buffer.clone())

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Read weights first, then pass the remaining arguments to the module."""
        n_params = len(self.names)
        if len(args) < n_params:
            raise ValueError(f'Expected at least {n_params} parameter tensors, got {len(args)} arguments.')
        params, inputs = args[:n_params], args[n_params:]
        replacements = dict(zip(self.names, params, strict=True))
        buffers = {
            name: value.clone() for name, value in zip(self._buffer_names, self._buffer_state.buffers(), strict=True)
        }
        result = functional_call(self.module, (replacements, buffers), inputs)
        if isinstance(result, torch.Tensor):
            return (result,)
        if isinstance(result, tuple) and all(isinstance(value, torch.Tensor) for value in result):
            return result
        raise TypeError('The module must return a tensor or a tuple of tensors.')


def make_functional(
    module: nn.Module,
) -> tuple[Operator[Unpack[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]:
    """Expose module weights and forward inputs as explicit operator arguments.

    The returned operator represents ``G(*params, *inputs) = module(*inputs)``
    evaluated with the supplied weights. It composes with MR2 operators and
    functionals, so an optimizer can fit a network without manually translating
    between named module parameters and tuples of tensors.

    Parameters
    ----------
    module
        Module whose parameters with ``requires_grad=True`` are exposed.
        Frozen parameters remain part of the stored module. Parameter sharing
        and PyTorch parametrizations are preserved by ``functional_call``.

    Returns
    -------
    operator
        Operator taking the trainable weights first, followed by the module's
        positional tensor inputs. The parameter count determines the split.
        A tensor output is wrapped in a tuple; tuple outputs are retained.
    parameters
        Initial trainable parameters in the operator's argument order. These
        are references to the original tensors, without cloning or detaching.
        A module with no trainable parameters returns an empty tuple.

    Notes
    -----
    Buffers are copied when this adapter is created and cloned for every
    evaluation. Running-statistic updates therefore affect neither the original
    module nor subsequent evaluations. The module's training/evaluation mode is
    retained; random operations such as dropout remain random.

    Inputs are supplied on each evaluation and are not captured by this helper.
    Passing ``(*params, z)`` as an optimizer's initial values optimizes both the
    weights and the input. To keep ``z`` fixed, use a closure such as
    ``lambda *p: objective(*p, z)`` and optimize only ``params``.

    The module is held by reference. Set its device, dtype and mode before
    creating the adapter, and keep the parameter structure unchanged. Supplied
    inputs must use compatible devices and dtypes. In-place mutation of weights
    or inputs in ``forward`` is not supported. This function does not change
    whether an optimizer differentiates through its iterations.

    Examples
    --------
    Fit a linear generator and its input jointly:

    >>> from mr2.algorithms.optimizers import adam
    >>> from mr2.operators.functionals import L2NormSquared
    >>> net = nn.Linear(2, 1)
    >>> z = torch.ones(1, 2)
    >>> target = torch.ones(1, 1)
    >>> generator, params = make_functional(net)
    >>> objective = 0.5 * L2NormSquared(target) @ generator
    >>> fitted = adam(objective, (*params, z))
    >>> image, = generator(*fitted)
    """
    named = [(name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad]
    names = tuple(name for name, _ in named)
    params = tuple(parameter for _, parameter in named)
    return _FunctionalModuleOp(module, names), params
