"""Dictionary Matching Operator."""

from collections.abc import Callable
from itertools import pairwise
from typing import Literal, cast, overload

import torch
from typing_extensions import Self, TypeVarTuple, Unpack

from mr2.operators.Operator import Operator
from mr2.utils import normalize_index
from mr2.utils.TensorList import TensorList

Tin = TypeVarTuple('Tin')


class DifferentiableDictionaryMatchOp(Operator[torch.Tensor, tuple[Unpack[Tin]]]):
    r"""Differentiable Dictionary Matching Operator.

    This operator can be used for dictionary matching, for example in
    magnetic resonance fingerprinting.

    It performs absolute normalized dot product matching between a dictionary of signals,
    i.e. find the entry :math:`d^*` in the dictionary maximizing
    :math:`\left|\frac{d}{\|d\|} \cdot \frac{y}{\|y\|}\right|` and uses  the
    associated signal model parameters :math:`x` generating the matching signal :math:`d^*=d(x)`
    as support points for a linear approximation of the signal model.

    At initialization, a signal model needs to be provided.
    Afterwards `append` with different `x` values should be called to add entries to the dictionary.
    This operator then calculates for each `x` value the signal returned by the model and
    precalculates the Jacobians of the signal model at these support points.
    To perform a match, use `__call__` and supply some `y` values. The operator will then perform the
    dot product matching and return the associated `x` values.

    .. note::
            The Gauss-Newton refinement step is differentiable in ``input_signal``.
            The dictionary lookup itself (argmax) is non-differentiable.
    """

    def __init__(
        self,
        generating_function: Callable[[Unpack[Tin]], tuple[torch.Tensor,]],
        index_of_scaling_parameter: int | None = None,
        batch_size: int = 64 * 1024,
    ):
        """Initialize DictionaryMatchOp.

        Parameters
        ----------
        generating_function
            signal model that takes n inputs and returns a signal y.
        index_of_scaling_parameter
            Normalized dot product matching is insensitive to overall signal scaling.
            A scaling factor (e.g. the equilibrium magnetization `m0` in `~mr2.operators.models.InversionRecovery`)
            is calculated after the dictionary matching if `index_of_scaling_parameter` is not `None`.
            `index_of_scaling_parameter` should set to the index of the scaling parameter in the signal model.

            Example:
                For ~mr2.operators.models.InversionRecovery the parameters are ``[m0, t1]`` and therefore
                `index_of_scaling_parameter` should be set to 0. The operator will then return `t1` estimated
                via dictionary matching and `m0` via a post-processing step.
                If `index_of_scaling_parameter` is None, the value returned for `m0` will be meaningless.

        batch_size
            Size of the chunks to split the input signal into for batch processing. Reduce to save memory.
        """
        super().__init__()
        self._f = generating_function
        self.x = TensorList()
        self.y = torch.tensor([])
        self.partials = TensorList()
        self._index_of_scaling_parameter = index_of_scaling_parameter
        self.norm_y = torch.tensor([])
        self.batch_size = batch_size

    def append(self, *x: Unpack[Tin]) -> Self:
        """Append `x` values to the dictionary.

        Parameters
        ----------
        x
            points where the signal model will be evaluated. For signal models
            with n inputs, n tensors should be provided. Broadcasting is supported.

        Returns
        -------
            Self

        """
        scaling_position: int | None
        x_stored: tuple[torch.Tensor, ...]
        primals: tuple[torch.Tensor, ...]
        if self._index_of_scaling_parameter is not None:
            scaling_position = normalize_index(len(x), self._index_of_scaling_parameter)
            x_stored = (*x[:scaling_position], *x[scaling_position + 1 :])  # type: ignore[arg-type]
            primals = (*x[:scaling_position], torch.tensor(1.0), *x[scaling_position + 1 :])  # type: ignore[arg-type]
        else:
            x_stored = x  # type: ignore[assignment]
            primals = x  # type: ignore[assignment]
            scaling_position = None

        partials: list[torch.Tensor] = []
        zeros = tuple(torch.zeros_like(t) for t in x_stored)
        for i in range(len(x_stored)):
            tangents = (*zeros[:i], torch.ones_like(x_stored[i]), *zeros[i + 1 :])
            if scaling_position is not None:
                tangents = (
                    *tangents[:scaling_position],
                    torch.tensor(0.0),
                    *tangents[scaling_position:],
                )
            (y,), (dy_i,), *_ = torch.func.jvp(self._f, primals, tangents)
            partials.append(dy_i.flatten(start_dim=1))

        y = y.flatten(start_dim=1)

        x_list = [t.flatten() for t in torch.broadcast_tensors(*x_stored)]
        norm_y = y.norm(dim=0)
        if (norm_y < 1e-12).any():
            raise ValueError('Dictionary entries must have non-zero norm.')
        y = y / norm_y

        if not self.x:  # first append
            self.x = TensorList(x_list)
            self.y = cast(torch.Tensor, y)
            self.partials = TensorList(partials)
            self.norm_y = cast(torch.Tensor, norm_y)
            return self

        self.x = TensorList([torch.cat((old, new)) for old, new in zip(self.x, x_list, strict=True)])
        self.y = torch.cat((self.y, y), dim=-1)
        self.partials = TensorList(
            torch.cat((old, new), dim=-1) for old, new in zip(self.partials, partials, strict=True)
        )
        self.norm_y = torch.cat((self.norm_y, norm_y))
        return self

    @overload
    def __call__(
        self,
        input_signal: torch.Tensor,
        *,
        prior: tuple[Unpack[Tin]] | None = None,
        prior_precision: tuple[Unpack[Tin]] | torch.Tensor | None = None,
        return_signal: Literal[False] = False,
    ) -> tuple[Unpack[Tin]]: ...

    @overload
    def __call__(
        self,
        input_signal: torch.Tensor,
        *,
        prior: tuple[Unpack[Tin]] | None = None,
        prior_precision: tuple[Unpack[Tin]] | torch.Tensor | None = None,
        return_signal: Literal[True] = ...,
    ) -> tuple[tuple[Unpack[Tin]], torch.Tensor]: ...

    def __call__(
        self,
        input_signal: torch.Tensor,
        *,
        prior: tuple[Unpack[Tin]] | None = None,
        prior_precision: tuple[Unpack[Tin]] | torch.Tensor | None = None,
        return_signal: bool = False,
    ) -> tuple[Unpack[Tin]] | tuple[tuple[Unpack[Tin]], torch.Tensor]:
        """Perform dot-product matching.

        Performs dictionary matching, optionally with a prior, followed by a
        single Gauss-Newton refinement step using precomputed Jacobians.
        The solution step is differentiable in ``input_signal``.


        Parameters
        ----------
        input_signal
            Input signal(s) to match against the dictionary.
            Expected shape is `(m, ...)`, where `m` is the signal dimension
            (e.g., number of time points) and `(...)` are batch dimensions.

        prior
            Prior means, one tensor per returned parameter. Each tensor must be
            broadcastable to the batch/image shape ``input_signal.shape[1:]``.
        prior_precision
            Diagonal prior precisions, one tensor per returned parameter, or a single
            tensor to use for all parameters. Each tensor must be broadcastable to
            ``input_signal.shape[1:]`` and contain finite, non-negative values.
            ``prior`` and ``prior_precision`` must be provided together.
        return_signal
            If True, return the signal approximation as well.

        Returns
        -------
            A tuple of tensors representing the parameters `x` from the dictionary
            that best matched the input signal(s). Each tensor in the tuple corresponds
            to a parameter, and their shapes will match the batch dimensions of `input_signal`.
        """
        return super(Operator, self).__call__(
            input_signal, prior=prior, prior_precision=prior_precision, return_signal=return_signal
        )

    def forward(  # type: ignore[override]
        self,
        input_signal: torch.Tensor,
        *,
        prior: tuple[Unpack[Tin]] | None = None,
        prior_precision: tuple[Unpack[Tin]] | torch.Tensor | None = None,
        return_signal: bool = False,
    ) -> tuple[tuple[Unpack[Tin]], torch.Tensor] | tuple[Unpack[Tin]]:
        """Apply forward of DictionaryMatchOp.

        .. note::
            Prefer calling the instance as ``operator(x)`` over  directly calling this method.
        """
        if not self.x:
            raise KeyError('No keys in the dictionary. Please first add some x values using `append`.')

        dtype = torch.result_type(input_signal, self.y)

        if self._index_of_scaling_parameter is not None:
            n_x = len(self.x) + 1
            scaling_position = self._index_of_scaling_parameter % n_x
        else:
            n_x = len(self.x)
            scaling_position = None

        norm_y = self.norm_y
        norm_y_sq = norm_y.square()
        batch_shape = input_signal.shape[1:]
        prior_flat: list[torch.Tensor] = []
        precision_flat: list[torch.Tensor] = []

        if (prior is None) != (prior_precision is None):
            raise ValueError('prior and prior_precision must either both be provided or both be None.')
        if prior is not None and prior_precision is not None:
            if len(prior) != n_x or not all(isinstance(p, torch.Tensor) for p in prior):
                raise ValueError('Prior must be a tuple of tensors matching the number of parameters.')
            prior_ = tuple(p for p in prior if isinstance(p, torch.Tensor))
            if isinstance(prior_precision, torch.Tensor):
                prior_precision = cast(tuple[Unpack[Tin]], (prior_precision,) * n_x)
            if len(prior_precision) != n_x or not all(
                isinstance(lam, torch.Tensor)
                and torch.isfinite(lam).all()
                and not (lam.is_complex() or (lam < 0).any())
                for lam in prior_precision
            ):
                raise ValueError(
                    'prior_precision must be a tuple of finite non-negative real tensors matching the number '
                    'of parameters.'
                )
            prior_precision_ = tuple(lam for lam in prior_precision if isinstance(lam, torch.Tensor))
            prior_flat = [p.broadcast_to(batch_shape).flatten() for p in prior_]
            precision_flat = [lam.broadcast_to(batch_shape).to(dtype.to_real()).flatten() for lam in prior_precision_]

        signal = input_signal.flatten(1).mT  # (batch, m)
        idx_chunks: list[torch.Tensor] = []

        if prior_flat:
            with torch.no_grad():
                for start, stop in pairwise(range(0, signal.shape[0] + self.batch_size, self.batch_size)):
                    signal_chunk = signal[start:stop]
                    prior_chunk = [p[start:stop] for p in prior_flat]
                    precision_chunk = [lam[start:stop] for lam in precision_flat]

                    similarity = signal_chunk.to(dtype) @ self.y.conj().to(dtype)
                    if scaling_position is not None:
                        scale_precision = precision_chunk[scaling_position]
                        numerator = (similarity * norm_y) + scale_precision[:, None] * prior_chunk[scaling_position][
                            :, None
                        ]
                        scores = numerator.abs().square() / (norm_y_sq[None, :] + scale_precision[:, None])
                    else:
                        scores = similarity.abs().square()

                    prior_chunk_stored = [prior_chunk[i] for i in range(n_x) if i != scaling_position]
                    precision_chunk_stored = [precision_chunk[i] for i in range(n_x) if i != scaling_position]
                    for x_k, p_k, lam_k in zip(self.x, prior_chunk_stored, precision_chunk_stored, strict=True):
                        scores = scores - lam_k[:, None] * (x_k[None, :] - p_k[:, None]).abs().square()

                    idx_chunks.append(scores.argmax(dim=1))

        else:  # no prior
            with torch.no_grad():
                for signal_chunk in signal.split(self.batch_size, dim=0):
                    similarity = signal_chunk.to(dtype) @ self.y.conj().to(dtype)
                    scores = similarity.abs().square()
                    idx_chunks.append(scores.argmax(dim=1))

        idx = torch.cat(idx_chunks)
        y = (self.y[:, idx] * norm_y[idx]).mT
        x = [x_k[idx] for x_k in self.x]
        j_cols = [p[:, idx].mT for p in self.partials]

        if scaling_position is not None:
            scale = (y.conj() * signal).sum(1)
            if not prior_flat:
                scale = scale / norm_y_sq[idx]
            else:
                scale_precision = precision_flat[scaling_position]
                scale = (scale + scale_precision * prior_flat[scaling_position]) / (norm_y_sq[idx] + scale_precision)
            j_cols = [scale[:, None] * col for col in j_cols]
            x.insert(scaling_position, scale)
            j_cols.insert(scaling_position, y)  # y is the partial deriv. wrt scale
            y = scale[:, None] * y

        j = torch.stack(j_cols, dim=2)
        residual = signal - y

        lhs = j.mH @ j
        rhs = (j.mH @ residual.unsqueeze(-1)).squeeze(-1)
        if prior_flat:
            lam = torch.stack(precision_flat, dim=1)
            lhs = lhs + torch.diag_embed(lam).to(j.dtype)
            rhs = rhs + lam * torch.stack([(p_k - x_k) for p_k, x_k in zip(prior_flat, x, strict=True)], dim=1)
        delta = torch.linalg.solve(lhs, rhs)
        result = [
            (x + (d if x.is_complex() else d.real)).reshape(batch_shape)
            for x, d in zip(x, delta.unbind(1), strict=True)
        ]
        params = cast(tuple[Unpack[Tin]], tuple(result))
        if return_signal:
            y_approx = (y + (j @ delta.unsqueeze(-1)).squeeze(-1)).mT.reshape(input_signal.shape)
            return params, y_approx
        return params
