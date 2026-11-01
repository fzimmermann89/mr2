"""Expose network weights as the inputs of an MR2 operator."""

import torch
from torch import nn
from torch.func import functional_call
from typing_extensions import Unpack

from mr2.operators.Operator import Operator, OperatorType


class _FunctionalModuleOp(Operator[Unpack[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]):
    """Evaluate a module using explicit trainable weights and forward inputs.

    Buffers are captured at construction and copied for each call so updates
    do not accumulate or change the original module.
    """

    def __init__(self, module: nn.Module, names: tuple[str, ...]) -> None:
        """Store the module and capture its buffers.

        Parameters
        ----------
        module
            Module to evaluate with replacement weights.
        names
            Names of trainable parameters, in operator input order.
        """
        super().__init__()
        self.module = module
        self.names = names
        self._buffer_names = tuple(name for name, _ in module.named_buffers())
        self._buffer_state = nn.Module()
        for index, buffer in enumerate(module.buffers()):
            self._buffer_state.register_buffer(str(index), buffer.clone())

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Evaluate with replacement weights followed by positional inputs.

        Parameters
        ----------
        args
            Trainable parameter tensors in the stored order, then module inputs.

        Returns
        -------
            Module outputs as a tuple of tensors.
        """
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
) -> tuple[OperatorType, tuple[torch.Tensor, ...]]:
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
    Buffers are captured once and copied per call; updates do not persist.
    The module is held by reference: prepare its device, dtype and mode first,
    and keep its parameter structure unchanged. In-place changes to weights
    or inputs are unsupported. Disable checkpointing inside the module: backward
    would use the original weights. Checkpoint the returned operator instead.

    Examples
    --------
    Fit weights and input jointly, or keep the input fixed with a closure:

    >>> from mr2.algorithms.optimizers import adam
    >>> from mr2.operators.functionals import L2NormSquared
    >>> generator, params = make_functional(nn.Linear(2, 1))
    >>> z = torch.ones(1, 2)
    >>> objective = L2NormSquared(torch.ones(1, 1)) @ generator
    >>> fitted = adam(objective, (*params, z))  # fit weights and input
    >>> fitted_params = adam(lambda *p: objective(*p, z), params)  # fit weights only
    """
    named = [(name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad]
    names = tuple(name for name, _ in named)
    params = tuple(parameter for _, parameter in named)
    return _FunctionalModuleOp(module, names), params
