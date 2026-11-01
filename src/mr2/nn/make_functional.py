"""Expose network weights as the inputs of an MR2 operator."""

from collections.abc import Sequence

import torch
from torch import nn
from torch.func import functional_call
from typing_extensions import Unpack

from mr2.operators.Operator import Operator


class _FunctionalModuleOp(Operator[Unpack[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]):
    """Evaluate a module at bound inputs using explicit trainable weights."""

    def __init__(
        self, module: nn.Module, names: Sequence[str], args: tuple[object, ...], kwargs: dict[str, object]
    ) -> None:
        """Bind the module, parameter names and forward arguments."""
        super().__init__()
        self.module = module
        self.names = tuple(names)
        self.args = args
        self.kwargs = kwargs
        self._buffer_names = tuple(name for name, _ in module.named_buffers())
        self._buffer_state = nn.Module()
        for index, buffer in enumerate(module.buffers()):
            self._buffer_state.register_buffer(str(index), buffer.clone())

    def forward(self, *params: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Evaluate with explicit weights and fresh copies of the initial buffers."""
        replacements = dict(zip(self.names, params, strict=True))
        buffers = {
            name: value.clone() for name, value in zip(self._buffer_names, self._buffer_state.buffers(), strict=True)
        }
        result = functional_call(self.module, (replacements, buffers), self.args, self.kwargs)
        if isinstance(result, torch.Tensor):
            return (result,)
        if isinstance(result, tuple) and all(isinstance(value, torch.Tensor) for value in result):
            return result
        raise TypeError('The module must return a tensor or a tuple of tensors.')


def make_functional(
    module: nn.Module, *args: object, **kwargs: object
) -> tuple[Operator[Unpack[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]:
    """Bind module inputs and expose its trainable weights as operator arguments.

    The returned operator represents ``G(*params) = module(*args, **kwargs)``
    evaluated with the supplied weights. It composes with MR2 operators and
    functionals, so an optimizer can fit a network without manually translating
    between named module parameters and tuples of tensors.

    Parameters
    ----------
    module
        Module whose parameters with ``requires_grad=True`` are exposed.
        Frozen parameters remain part of the stored module. Parameter sharing
        and PyTorch parametrizations are preserved by ``functional_call``.
    *args
        Positional inputs bound to the module's forward call.
    **kwargs
        Keyword inputs bound to the module's forward call.

    Returns
    -------
    operator
        Operator taking the trainable weights as positional tensor arguments.
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

    The module and bound inputs are held by reference. Set their device, dtype
    and training mode before calling this function, and keep the parameter
    structure unchanged while using the returned operator. In-place mutation of
    weights or bound inputs in ``forward`` is not supported. This function does
    not change whether an optimizer differentiates through its iterations.

    Examples
    --------
    Fit a linear generator while keeping its input fixed:

    >>> from mr2.algorithms.optimizers import adam
    >>> from mr2.operators.functionals import L2NormSquared
    >>> net = nn.Linear(2, 1)
    >>> z = torch.ones(1, 2)
    >>> target = torch.ones(1, 1)
    >>> generator, params = make_functional(net, z)
    >>> objective = 0.5 * L2NormSquared(target) @ generator
    >>> fitted = adam(objective, params)
    >>> image, = generator(*fitted)
    """
    named = [(name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad]
    names = tuple(name for name, _ in named)
    params = tuple(parameter for _, parameter in named)
    return _FunctionalModuleOp(module, names, args, kwargs), params
