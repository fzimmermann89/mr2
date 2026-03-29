"""Bloch-McConnell simulation."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from mr2.data.Dataclass import Dataclass
from mr2.utils.reshape import unsqueeze_right
from mr2.utils.TensorAttributeMixin import TensorAttributeMixin


@dataclass
class MTSaturation(ABC):
    """Base class for MT lineshape models."""

    pool_index: int
    """Index of the MT pool in the pool dimension."""

    t2: torch.Tensor
    """Transverse relaxation time in seconds."""

    @abstractmethod
    def __call__(self, delta_omega: torch.Tensor) -> torch.Tensor:
        r"""Evaluate \(G(\Delta)\) [s]."""


@dataclass
class LorentzianMT(MTSaturation):
    """Lorentzian lineshape for MT saturation."""

    def __call__(self, delta_omega: torch.Tensor) -> torch.Tensor:
        r"""Evaluate \(G(\Delta)\) [s].

        Parameters
        ----------
        delta_omega
            Detuning in rad/s.
        t2
            Transverse relaxation time in seconds.

        Returns
        -------
        g
            Lineshape value in seconds.
        """
        t2 = self.t2.to(delta_omega)
        x = delta_omega * t2
        return t2 / (1 + x * x)


@dataclass
class SuperLorentzianMT(MTSaturation):
    """Super-Lorentzian lineshape for MT saturation."""

    samples: int = 101
    """Quadrature samples for numerical integration."""

    def __call__(self, delta_omega: torch.Tensor) -> torch.Tensor:
        r"""Evaluate \(G(\Delta)\) [s].

        Parameters
        ----------
        delta_omega
            Detuning in rad/s.
        t2
            Transverse relaxation time in seconds.
        """
        t2 = self.t2.to(delta_omega)
        u = torch.linspace(0.0, 1.0, self.samples, device=delta_omega.device, dtype=delta_omega.dtype)
        du = u[1] - u[0]
        denom = (3 * u * u - 1).abs().clamp_min(1e-12)
        x = (delta_omega[..., None] * t2[..., None]) / denom
        integrand = (2.0 / torch.pi) ** 0.5 * t2[..., None] / denom * torch.exp(-2 * x * x)
        return integrand.sum(dim=-1) * torch.pi * du


class Parameters(Dataclass):
    """Parameters for Bloch-McConnell simulation.

    Shapes
    ------
    - poolwise: ``(..., pools)``

    Notes
    -----
    Hyperpolarization is handled by setting a non-equilibrium initial state
    (via ``initial_state(..., mz=...)`` or ``ResetBlock(state=...)``) and by
    choosing ``equilibrium_magnetization`` appropriately.
    """

    equilibrium_magnetization: torch.Tensor
    """Equilibrium magnetization."""
    t1: torch.Tensor
    """T1 relaxation time in seconds.
    Shape ``(..., pools)``.
    """
    t2: torch.Tensor
    """T2 relaxation time in seconds. Shape ``(..., pools)``."""

    exchange_rate: torch.Tensor
    """Exchange rate in 1/s.
    Shape ``(..., pools, pools)``
    where element ``[..., i, j]`` is the ratefrom pool j to pool i."""

    chemical_shift: torch.Tensor | None = None
    """Chemical shift in Hz. Shape ``(..., pools)``."""

    static_off_resonance: torch.Tensor | None = None
    """Delta B0 in rad/s. Shape ``(...)`` (global per voxel/batch)."""

    relative_b1: torch.Tensor | None = None
    """Relative B1 scaling factor. Shape ``(...)`` (global per voxel/batch)."""

    mt_saturation: MTSaturation | None = None
    """MT saturation model. Shape ``(..., pools)``. """

    @property
    def n_pools(self) -> int:
        """Number of pools."""
        return int(self.equilibrium_magnetization.shape[-1])

    @property
    def ndim(self) -> int:
        """Broadcast ndim of parameter batch dimensions."""
        ndim = max(
            self.equilibrium_magnetization.ndim,
            self.t1.ndim,
            self.t2.ndim,
            self.exchange_rate.ndim - 1,
        )
        if self.chemical_shift is not None:
            ndim = max(ndim, self.chemical_shift.ndim)
        if self.static_off_resonance is not None:
            ndim = max(ndim, self.static_off_resonance.ndim + 1)
        if self.relative_b1 is not None:
            ndim = max(ndim, self.relative_b1.ndim + 1)
        return ndim


def system_recovery_vector(parameters: Parameters) -> torch.Tensor:
    """Build the affine recovery vector."""
    m0, t1 = parameters.equilibrium_magnetization, parameters.t1
    batch_shape = torch.broadcast_shapes(
        m0.shape[:-1],
        t1.shape[:-1],
        parameters.t2.shape[:-1],
        parameters.exchange_rate.shape[:-2],
    )
    if parameters.chemical_shift is not None:
        batch_shape = torch.broadcast_shapes(batch_shape, parameters.chemical_shift.shape[:-1])
    if parameters.static_off_resonance is not None:
        batch_shape = torch.broadcast_shapes(batch_shape, parameters.static_off_resonance.shape)
    if parameters.relative_b1 is not None:
        batch_shape = torch.broadcast_shapes(batch_shape, parameters.relative_b1.shape)
    c = torch.zeros(*batch_shape, 3 * parameters.n_pools, device=m0.device, dtype=m0.dtype)
    c[..., 2 * parameters.n_pools :] = (1.0 / t1) * m0
    return c


def initial_state(parameters: Parameters, mz: torch.Tensor | None = None) -> torch.Tensor:
    """Create an initial magnetization state.

    Parameters
    ----------
    parameters
        Simulation parameters.
    mz
        Optional initial longitudinal magnetization, shape ``(..., pools)``.
        If omitted, uses equilibrium_magnetization.

    Returns
    -------
    state
        Tensor with shape ``(..., pools, 3)`` holding ``(Mx, My, Mz)``.
    """
    mz0 = parameters.equilibrium_magnetization if mz is None else mz
    mz0 = mz0.to(parameters.equilibrium_magnetization)
    z = torch.zeros_like(mz0)
    return torch.stack([z, z, mz0], dim=-1)


def exchange_generator(exchange_rate: torch.Tensor) -> torch.Tensor:
    r"""Construct exchange generator \(Q\) for \(dM/dt = Q M\).

    Parameters
    ----------
    exchange_rate
        Shape ``(..., pools, pools)`` with element ``[..., i, j]`` the rate
        from pool j to pool i.

    Returns
    -------
    q
        Shape ``(..., pools, pools)``.
    """
    exchange_rate = torch.as_tensor(exchange_rate)
    out_rate = exchange_rate.sum(dim=-2)
    return exchange_rate - torch.diag_embed(out_rate)


def system_base_matrix(
    parameters: Parameters,
    rf_frequency: torch.Tensor | float,
) -> torch.Tensor:
    """Build the RF-amplitude independent Bloch-McConnell matrix."""
    m0, t1, t2, exchange = parameters.equilibrium_magnetization, parameters.t1, parameters.t2, parameters.exchange_rate
    freq = torch.as_tensor(rf_frequency, device=m0.device, dtype=m0.dtype)

    if parameters.chemical_shift is not None:
        shift = parameters.chemical_shift.to(m0)
    else:
        shift = m0.new_zeros(*m0.shape[:-1], parameters.n_pools)

    if parameters.static_off_resonance is not None:
        dw0 = parameters.static_off_resonance.to(m0)
    else:
        dw0 = m0.new_zeros(m0.shape[:-1])

    batch = torch.broadcast_shapes(
        m0.shape[:-1],
        t1.shape[:-1],
        t2.shape[:-1],
        exchange.shape[:-2],
        freq.shape,
        shift.shape[:-1],
        dw0.shape,
    )
    t1 = torch.broadcast_to(t1, (*batch, parameters.n_pools))
    t2 = torch.broadcast_to(t2, (*batch, parameters.n_pools))
    exchange = torch.broadcast_to(exchange, (*batch, parameters.n_pools, parameters.n_pools))
    shift = torch.broadcast_to(shift, (*batch, parameters.n_pools))
    dw0 = torch.broadcast_to(dw0, batch)
    freq = torch.broadcast_to(freq, batch)

    r1 = 1.0 / t1
    r2 = 1.0 / t2

    qz = exchange_generator(exchange)
    qxy = qz
    if parameters.mt_saturation is not None:
        if not (0 <= parameters.mt_saturation.pool_index < parameters.n_pools):
            raise ValueError('mt_saturation.pool_index out of bounds.')
        qxy = qz.clone()
        qxy[..., parameters.mt_saturation.pool_index, :] = 0
        qxy[..., :, parameters.mt_saturation.pool_index] = 0

    delta_omega = dw0[..., None] - 2 * torch.pi * freq[..., None] + 2 * torch.pi * shift

    a_xx = qxy - torch.diag_embed(r2)
    a_zz = qz - torch.diag_embed(r1)
    a_xy = -torch.diag_embed(delta_omega)

    n = 3 * parameters.n_pools
    matrix = torch.zeros(*batch, n, n, device=m0.device, dtype=m0.dtype)
    matrix[..., : parameters.n_pools, : parameters.n_pools] = a_xx
    matrix[..., parameters.n_pools : 2 * parameters.n_pools, parameters.n_pools : 2 * parameters.n_pools] = a_xx
    matrix[..., 2 * parameters.n_pools :, 2 * parameters.n_pools :] = a_zz
    matrix[..., : parameters.n_pools, parameters.n_pools : 2 * parameters.n_pools] += a_xy
    matrix[..., parameters.n_pools : 2 * parameters.n_pools, : parameters.n_pools] -= a_xy
    return matrix


def system_rf_matrix(
    parameters: Parameters,
    rf_amplitude: torch.Tensor | float,
    rf_phase: torch.Tensor | float,
    rf_frequency: torch.Tensor | float,
) -> torch.Tensor:
    """Build the RF-amplitude dependent Bloch-McConnell matrix contribution."""
    m0 = parameters.equilibrium_magnetization

    amp = torch.as_tensor(rf_amplitude, device=m0.device, dtype=m0.dtype)
    phase = torch.as_tensor(rf_phase, device=m0.device, dtype=m0.dtype)
    freq = torch.as_tensor(rf_frequency, device=m0.device, dtype=m0.dtype)

    if parameters.relative_b1 is not None:
        rb1 = parameters.relative_b1.to(amp)
        if rb1.is_complex():
            phase = phase + rb1.angle()
            amp = amp * rb1.abs()
        else:
            amp = amp * rb1

    if parameters.chemical_shift is not None:
        shift = parameters.chemical_shift.to(m0)
    else:
        shift = m0.new_zeros(*m0.shape[:-1], parameters.n_pools)

    if parameters.static_off_resonance is not None:
        dw0 = parameters.static_off_resonance.to(m0)
    else:
        dw0 = m0.new_zeros(m0.shape[:-1])

    batch = torch.broadcast_shapes(
        m0.shape[:-1],
        amp.shape,
        phase.shape,
        freq.shape,
        shift.shape[:-1],
        dw0.shape,
    )
    shift = torch.broadcast_to(shift, (*batch, parameters.n_pools))
    dw0 = torch.broadcast_to(dw0, batch)
    amp = torch.broadcast_to(amp, batch)
    phase = torch.broadcast_to(phase, batch)
    freq = torch.broadcast_to(freq, batch)

    delta_omega = dw0[..., None] - 2 * torch.pi * freq[..., None] + 2 * torch.pi * shift

    w1 = 2 * torch.pi * amp
    w1x = w1 * torch.cos(phase)
    w1y = w1 * torch.sin(phase)

    eye_rf = torch.eye(parameters.n_pools, device=m0.device, dtype=m0.dtype)
    if parameters.mt_saturation is not None:
        eye_rf = eye_rf.clone()
        eye_rf[parameters.mt_saturation.pool_index, parameters.mt_saturation.pool_index] = 0.0

    a_xz = -w1y[..., None, None] * eye_rf
    a_yz = w1x[..., None, None] * eye_rf

    n = 3 * parameters.n_pools
    matrix = torch.zeros(*batch, n, n, device=m0.device, dtype=m0.dtype)
    matrix[..., : parameters.n_pools, 2 * parameters.n_pools :] += a_xz
    matrix[..., parameters.n_pools : 2 * parameters.n_pools, 2 * parameters.n_pools :] += a_yz
    matrix[..., 2 * parameters.n_pools :, : parameters.n_pools] -= a_xz
    matrix[..., 2 * parameters.n_pools :, parameters.n_pools : 2 * parameters.n_pools] -= a_yz

    if parameters.mt_saturation is not None:
        g = parameters.mt_saturation(delta_omega[..., parameters.mt_saturation.pool_index])
        one_hot = torch.zeros(parameters.n_pools, device=m0.device, dtype=m0.dtype)
        one_hot[parameters.mt_saturation.pool_index] = 1.0
        mt_diag = torch.diag_embed(((w1 * w1) * g)[..., None] * one_hot)
        matrix[..., 2 * parameters.n_pools :, 2 * parameters.n_pools :] -= mt_diag

    return matrix


def system_matrix(
    parameters: Parameters,
    rf_amplitude: torch.Tensor | float,
    rf_phase: torch.Tensor | float,
    rf_frequency: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Build affine Bloch-McConnell system \(dm/dt = A m + c\).

    Parameters
    ----------
    parameters
        Simulation parameters.
    rf_amplitude
        RF amplitude in Hz, broadcastable to batch. Shape ``(...)``.
    rf_phase
        RF phase in rad, broadcastable to batch. Shape ``(...)``.
    rf_frequency
        RF carrier offset in Hz, broadcastable to batch. Shape ``(...)``.

    Returns
    -------
    A
        System matrix with shape ``(..., 3*pools, 3*pools)``.
    c
        Inhomogeneity vector with shape ``(..., 3*pools)``.
    """
    matrix = system_base_matrix(parameters, rf_frequency) + system_rf_matrix(
        parameters, rf_amplitude, rf_phase, rf_frequency
    )
    c = system_recovery_vector(parameters)
    return matrix, c


def propagate(
    state: torch.Tensor,
    matrix: torch.Tensor,
    c: torch.Tensor,
    duration: torch.Tensor | float,
) -> torch.Tensor:
    r"""Propagate dynamics \(dm/dt = A m + c\) via exact affine evolution."""
    step = propagation_step(matrix, c, duration)
    return apply_propagation_step(state, step)


def propagation_step(
    matrix: torch.Tensor,
    c: torch.Tensor,
    duration: torch.Tensor | float,
) -> torch.Tensor:
    """Build exact affine propagation steps for constant-system evolution."""
    duration = torch.as_tensor(duration, device=matrix.device, dtype=matrix.dtype)
    linear_step = torch.matrix_exp(matrix * duration[..., None, None])
    identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    offset_rhs = ((linear_step - identity) @ c.unsqueeze(-1)).squeeze(-1)
    offset, info = torch.linalg.solve_ex(matrix, offset_rhs.unsqueeze(-1))
    if torch.any(info != 0):
        augmented = torch.zeros(
            *matrix.shape[:-2],
            matrix.shape[-1] + 1,
            matrix.shape[-1] + 1,
            device=matrix.device,
            dtype=matrix.dtype,
        )
        augmented[..., :-1, :-1] = matrix
        augmented[..., :-1, -1] = c
        augmented_step = torch.matrix_exp(augmented * duration[..., None, None])
        return augmented_step[..., :-1, :]
    return torch.cat([linear_step, offset], dim=-1)


def apply_propagation_step(state: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    """Apply a precomputed exact affine propagation step to a state."""
    pools = int(state.shape[-2])
    n = 3 * pools
    batch = torch.broadcast_shapes(state.shape[:-2], step.shape[:-2])
    state = torch.broadcast_to(state, (*batch, pools, 3))
    m = state.transpose(-1, -2).reshape(*batch, n)
    step = step.to(state)
    linear_step, offset = step[..., :n], step[..., n]
    m_next = (linear_step @ m.unsqueeze(-1)).squeeze(-1) + offset
    return m_next.reshape(*batch, 3, pools).transpose(-1, -2)


def transverse_readout(state: torch.Tensor) -> torch.Tensor:
    """Complex transverse readout per pool."""
    return torch.complex(state[..., 0], state[..., 1])


class BMCBlock(TensorAttributeMixin, ABC):
    """Base class for Bloch-McConnell blocks."""

    def __call__(
        self, parameters: Parameters, state: torch.Tensor | None = None, **kwargs
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the block."""
        if state is None:
            state = initial_state(parameters)
        return super().__call__(parameters, state, **kwargs)

    @abstractmethod
    def forward(
        self, parameters: Parameters, state: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the block."""
        raise NotImplementedError

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return torch.as_tensor(0.0)


class ConstantRFBlock(BMCBlock):
    """Constant RF block for a duration."""

    def __init__(
        self,
        duration: torch.Tensor | float,
        rf_amplitude: torch.Tensor | float,
        rf_phase: torch.Tensor | float = 0.0,
        rf_frequency: torch.Tensor | float = 0.0,
    ) -> None:
        """Initialize the block.

        Parameters
        ----------
        duration
            Duration in seconds. Shape ``(..., pools)``.
        rf_amplitude
            RF amplitude in Hz. Shape ``(..., pools)``.
        rf_phase
            RF phase in rad. Shape ``(..., pools)``.
        rf_frequency
            RF frequency in Hz. Shape ``(..., pools)``.
        """
        super().__init__()
        self._duration = torch.as_tensor(duration)
        self.rf_amplitude = torch.as_tensor(rf_amplitude)
        self.rf_phase = torch.as_tensor(rf_phase)
        self.rf_frequency = torch.as_tensor(rf_frequency)

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return self._duration

    def forward(
        self,
        parameters: Parameters,
        state: torch.Tensor,
        **kwargs,  # noqa: ARG002
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the block."""
        matrix, c = system_matrix(
            parameters,
            self.rf_amplitude.to(state),
            self.rf_phase.to(state),
            self.rf_frequency.to(state),
        )
        state = propagate(state, matrix, c, self._duration.to(state))
        return state, ()


class PiecewiseRFBlock(BMCBlock):
    """Piecewise-constant RF block."""

    def __init__(
        self,
        rf_amplitude: torch.Tensor,
        rf_phase: torch.Tensor | float = 0.0,
        rf_frequency: torch.Tensor | float = 0.0,
        dt: torch.Tensor | float = 0.0,
    ) -> None:
        """Initialize the block.

        Parameters
        ----------
        rf_amplitude
            RF amplitude. Shape ``(time, ...)``.
        rf_phase
            RF phase in rad. Shape ``(time, ...)``, ``(1, ...)`` or scalar.
        rf_frequency
            RF frequency in Hz. Shape ``(time, ...)``, ``(1, ...)`` or scalar.
        dt
            Sample duration in seconds. Shape ``(time, ...)``, ``(1, ...)`` or scalar.
        """
        super().__init__()
        self.rf_amplitude = torch.as_tensor(rf_amplitude)
        self.rf_phase = torch.as_tensor(rf_phase)
        self.rf_frequency = torch.as_tensor(rf_frequency)
        self.dt = torch.as_tensor(dt)

        if self.rf_amplitude.ndim < 1:
            raise ValueError('rf_amplitude must have a leading time dimension.')

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        if self.dt.ndim == 0:
            return self.dt * self.rf_amplitude.shape[0]
        if self.dt.shape[0] == 1:
            return self.dt.squeeze(0) * self.rf_amplitude.shape[0]
        return self.dt.sum(dim=0)

    def forward(
        self, parameters: Parameters, state: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the block."""
        del kwargs
        amp = self.rf_amplitude.to(state)
        ph = self.rf_phase.to(state)
        fr = self.rf_frequency.to(state)
        dt = self.dt.to(state)

        time = amp.shape[0]
        if ph.ndim == 0:
            ph = ph.reshape(1)
        if fr.ndim == 0:
            fr = fr.reshape(1)
        if dt.ndim == 0:
            dt = dt.reshape(1)
        if ph.shape[0] not in (1, time):
            raise ValueError('rf_phase must have leading dimension 1 or match rf_amplitude.')
        if fr.shape[0] not in (1, time):
            raise ValueError('rf_frequency must have leading dimension 1 or match rf_amplitude.')
        if dt.shape[0] not in (1, time):
            raise ValueError('dt must have leading dimension 1 or match rf_amplitude.')

        ndim = max(amp.ndim, ph.ndim, fr.ndim, dt.ndim, state.ndim - 1, parameters.ndim)

        amp = unsqueeze_right(amp, ndim - amp.ndim)
        ph = unsqueeze_right(ph, ndim - ph.ndim)
        fr = unsqueeze_right(fr, ndim - fr.ndim)
        dt = unsqueeze_right(dt, ndim - dt.ndim)

        if ph.shape[0] == 1 and time != 1:
            ph = ph.expand(time, *ph.shape[1:])
        if fr.shape[0] == 1 and time != 1:
            fr = fr.expand(time, *fr.shape[1:])
        if dt.shape[0] == 1 and time != 1:
            dt = dt.expand(time, *dt.shape[1:])

        c = system_recovery_vector(parameters)
        same_frequency = bool(torch.all(fr == fr[:1]))
        base_matrix = system_base_matrix(parameters, fr[0]) if same_frequency else None

        work = state[..., 0, 0].numel() * time
        if work <= 128_000:
            chunk_size = time
        else:
            chunk_size = 16 if same_frequency else 8

        def run_chunk(
            state: torch.Tensor,
            amp_chunk: torch.Tensor,
            ph_chunk: torch.Tensor,
            fr_chunk: torch.Tensor,
            dt_chunk: torch.Tensor,
        ) -> torch.Tensor:
            if same_frequency:
                assert base_matrix is not None
                matrices = base_matrix + system_rf_matrix(parameters, amp_chunk, ph_chunk, fr_chunk)
            else:
                matrices = system_base_matrix(parameters, fr_chunk) + system_rf_matrix(
                    parameters, amp_chunk, ph_chunk, fr_chunk
                )
            steps = propagation_step(matrices, c, dt_chunk)
            for step in steps:
                state = apply_propagation_step(state, step)
            return state

        use_checkpoint = torch.is_grad_enabled() and chunk_size < time

        for start in range(0, time, chunk_size):
            stop = min(start + chunk_size, time)
            amp_chunk = amp[start:stop]
            ph_chunk = ph[start:stop]
            fr_chunk = fr[start:stop]
            dt_chunk = dt[start:stop]
            if use_checkpoint:
                state = activation_checkpoint(
                    run_chunk,
                    state,
                    amp_chunk,
                    ph_chunk,
                    fr_chunk,
                    dt_chunk,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                state = run_chunk(state, amp_chunk, ph_chunk, fr_chunk, dt_chunk)
        return state, ()


class DelayBlock(BMCBlock):
    """Delay without RF."""

    def __init__(self, duration: torch.Tensor | float) -> None:
        """Initialize the block.

        Parameters
        ----------
        duration
            Duration in seconds. Shape ``(..., pools)``.
        """
        super().__init__()
        self._duration = torch.as_tensor(duration)

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return self._duration

    def forward(
        self,
        parameters: Parameters,
        state: torch.Tensor,
        *,
        zero_matrix: torch.Tensor | None = None,
        zero_c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the block.

        Parameters
        ----------
        parameters
            Simulation parameters.
        state
            State tensor. Shape ``(..., pools, 3)``.
        zero_matrix
            Cached no-RF system matrix for the current sequence execution, if available.
        zero_c
            Cached no-RF recovery vector for the current sequence execution, if available.

        Returns
        -------
        state
            State tensor. Shape ``(..., pools, 3)``.
        """
        if zero_matrix is None or zero_c is None:
            matrix, c = system_matrix(parameters, 0.0, 0.0, 0.0)
        else:
            matrix, c = zero_matrix, zero_c
        state = propagate(state, matrix, c, self._duration.to(state))
        return state, ()


class SpoilBlock(DelayBlock):
    """Perfect spoiling with non-zero duration."""

    def forward(
        self,
        parameters: Parameters,
        state: torch.Tensor,
        *,
        zero_matrix: torch.Tensor | None = None,
        zero_c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the block.

        Parameters
        ----------
        parameters
            Simulation parameters.
        state
            State tensor. Shape ``(..., pools, 3)``.
        zero_matrix
            Cached no-RF system matrix for the current sequence execution, if available.
        zero_c
            Cached no-RF recovery vector for the current sequence execution, if available.

        Returns
        -------
        state
            State tensor. Shape ``(..., pools, 3)``.
        """
        state, out = super().forward(parameters, state, zero_matrix=zero_matrix, zero_c=zero_c)
        mx, _, mz = state.unbind(-1)
        z = torch.zeros_like(mx)
        return torch.stack([z, z, mz], dim=-1), out


class AcquisitionBlock(BMCBlock):
    """Acquisition block that emits a readout."""

    def forward(
        self,
        parameters: Parameters,  # noqa: ARG002
        state: torch.Tensor,
        **kwargs,  # noqa: ARG002
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the block.

        Parameters
        ----------
        parameters
            Simulation parameters.
        state
            State tensor. Shape ``(..., pools, 3)``.

        Returns
        -------
        state
            State tensor. Shape ``(..., pools, 3)``.
        """
        return state, (transverse_readout(state),)


class LongitudinalReadoutBlock(BMCBlock):
    """Read out longitudinal magnetization of a selected pool."""

    def __init__(self, pool_index: int = 0) -> None:
        """Initialize the block.

        Parameters
        ----------
        pool_index
            Pool index to read out.
        """
        super().__init__()
        self.pool_index = pool_index

    def forward(
        self,
        parameters: Parameters,
        state: torch.Tensor,
        **kwargs,  # noqa: ARG002
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the block."""
        if not (0 <= self.pool_index < parameters.n_pools):
            raise ValueError('pool_index out of bounds.')
        return state, (state[..., self.pool_index, 2],)


class ResetBlock(BMCBlock):
    """Reset state to equilibrium or to a provided state."""

    def __init__(self, state: torch.Tensor | None = None) -> None:
        """Initialize the block.

        Parameters
        ----------
        state
            State tensor. Shape ``(..., pools, 3)``.
        """
        super().__init__()
        self.state = state

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return torch.as_tensor(0.0)

    def forward(
        self,
        parameters: Parameters,
        state: torch.Tensor,
        **kwargs,  # noqa: ARG002
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the block.

        Parameters
        ----------
        parameters
            Simulation parameters.
        state
            State tensor. Shape ``(..., pools, 3)``.
        kwargs
            Unused extension point for subclasses following the ``BMCBlock`` interface.

        Returns
        -------
        state
            State tensor. Shape ``(..., pools, 3)``.
        """
        if self.state is None:
            return initial_state(parameters).to(state), ()
        return self.state.to(state), ()


class BMCSequence(torch.nn.ModuleList, BMCBlock):
    """Sequence of Bloch-McConnell blocks."""

    def __init__(self, blocks: Sequence[BMCBlock] = ()) -> None:
        """Initialize the sequence.

        Parameters
        ----------
        blocks
            Sequence of Bloch-McConnell blocks.
        """
        torch.nn.ModuleList.__init__(self, blocks)

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the sequence."""
        return sum(
            (b.duration for b in self if isinstance(b, BMCBlock)),
            start=torch.as_tensor(0.0),
        )

    def forward(
        self,
        parameters: Parameters,
        state: torch.Tensor,
        **kwargs,  # noqa: ARG002
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the sequence of blocks.

        Parameters
        ----------
        parameters
            Simulation parameters.
        state
            State tensor. Shape ``(..., pools, 3)``.
        kwargs
            Unused extension point for subclasses following the ``BMCBlock`` interface.

        Returns
        -------
        state
            State tensor. Shape ``(..., pools, 3)``.
        outputs
            List of output tensors.
        """
        parameters = parameters.to(state, copy=False)
        zero_matrix: torch.Tensor | None = None
        zero_c: torch.Tensor | None = None
        outputs: list[torch.Tensor] = []
        for block in self:
            assert isinstance(block, BMCBlock)  # noqa: S101
            if isinstance(block, DelayBlock | SpoilBlock):
                if zero_matrix is None or zero_c is None:
                    zero_matrix, zero_c = system_matrix(parameters, 0.0, 0.0, 0.0)
                state, out = block(parameters, state, zero_matrix=zero_matrix, zero_c=zero_c)
            else:
                state, out = block(parameters, state)
            outputs.extend(out)
        return state, tuple(outputs)
