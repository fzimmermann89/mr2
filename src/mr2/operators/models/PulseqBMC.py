"""Pulseq-driven Bloch-McConnell signal model."""

from pathlib import Path
from typing import Literal

import torch

from mr2.operators.SignalModel import SignalModel
from mr2.operators.models.BMC import (
    AcquisitionBlock,
    BMCSequence,
    ConstantRFBlock,
    DelayBlock,
    LongitudinalReadoutBlock,
    MTSaturation,
    Parameters,
    PiecewiseRFBlock,
    ResetBlock,
    SpoilBlock,
)


class PulseqBMCModel(
    SignalModel[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        MTSaturation | None,
    ]
):
    """Bloch-McConnell signal model built from a Pulseq sequence."""

    def __init__(
        self,
        seq: str | Path | object,
        *,
        readout: Literal['longitudinal', 'transverse'] = 'longitudinal',
        readout_pool: int = 0,
        reset_between_readouts: bool = False,
    ) -> None:
        """Initialize the model from a Pulseq sequence."""
        super().__init__()
        import pypulseq as pp

        if isinstance(seq, pp.Sequence):
            pulseq = seq
        else:
            pulseq = pp.Sequence()
            pulseq.read(str(seq))

        self.sequence = BMCSequence()
        accum_phase = 0.0

        for block_id in range(1, len(pulseq.block_events) + 1):
            block = pulseq.get_block(block_id)
            has_adc = block.adc is not None
            has_rf = block.rf is not None
            has_gz = getattr(block, 'gz', None) is not None
            has_gx = getattr(block, 'gx', None) is not None
            has_gy = getattr(block, 'gy', None) is not None

            if has_adc and (has_rf or has_gz or has_gx or has_gy):
                raise ValueError('Mixed ADC blocks are not supported.')
            if has_rf and (has_gx or has_gy or has_gz):
                raise ValueError('Simultaneous RF and gradient blocks are not supported.')

            if has_adc:
                delay_before = float(block.adc.delay)
                if delay_before > 0:
                    self.sequence.append(DelayBlock(delay_before))
                self.sequence.append(
                    LongitudinalReadoutBlock(pool_index=readout_pool)
                    if readout == 'longitudinal'
                    else AcquisitionBlock()
                )
                delay_after = float(pp.calc_duration(block)) - delay_before
                if delay_after > 0:
                    self.sequence.append(DelayBlock(delay_after))
                if reset_between_readouts:
                    self.sequence.append(ResetBlock())
                accum_phase = 0.0
                continue

            if has_rf:
                signal = torch.as_tensor(block.rf.signal)
                amplitude = signal.abs()
                phase = float(block.rf.phase_offset) - accum_phase - torch.angle(signal)
                is_constant_block = torch.allclose(amplitude, amplitude[:1]) and torch.allclose(phase, phase[:1])
                if signal.numel() == 1:
                    dt = torch.as_tensor(float(block.rf.shape_dur), dtype=amplitude.dtype)
                else:
                    t = torch.as_tensor(block.rf.t, dtype=amplitude.dtype)
                    dt = t[1] - t[0]
                    if not torch.allclose(t[1:] - t[:-1], torch.full_like(t[1:], dt)):
                        raise ValueError('Only uniformly sampled RF blocks are supported.')

                active = amplitude > 1e-6
                if active.any():
                    active_index = torch.nonzero(active, as_tuple=False).squeeze(-1)
                    start = int(active_index[0])
                    stop = int(active_index[-1]) + 1
                    active_amplitude = amplitude[start:stop]
                    active_phase = phase[start:stop]
                    active_duration = (
                        torch.as_tensor(float(block.rf.shape_dur), dtype=amplitude.dtype)
                        if is_constant_block
                        else dt * active_amplitude.shape[0]
                    )

                    delay_before = float(block.rf.delay) + float(start * dt)
                    if delay_before > 0:
                        self.sequence.append(DelayBlock(delay_before))

                    rf_frequency = torch.as_tensor(float(block.rf.freq_offset), dtype=active_amplitude.dtype)
                    if is_constant_block:
                        self.sequence.append(
                            ConstantRFBlock(
                                duration=active_duration,
                                rf_amplitude=active_amplitude[0],
                                rf_phase=active_phase[0],
                                rf_frequency=rf_frequency,
                            )
                        )
                    else:
                        self.sequence.append(
                            PiecewiseRFBlock(
                                rf_amplitude=active_amplitude,
                                rf_phase=active_phase,
                                rf_frequency=rf_frequency,
                                dt=dt,
                            )
                        )

                    delay_after = float(pp.calc_duration(block)) - delay_before - float(active_duration)
                    if delay_after > 0:
                        self.sequence.append(DelayBlock(delay_after))

                    accum_phase = (
                        accum_phase + float(block.rf.freq_offset) * 2 * torch.pi * float(active_duration)
                    ) % (2 * torch.pi)
                else:
                    duration = float(pp.calc_duration(block))
                    if duration > 0:
                        self.sequence.append(DelayBlock(duration))
                continue

            if has_gx or has_gy or has_gz:
                duration = float(pp.calc_duration(block))
                if duration > 0:
                    self.sequence.append(SpoilBlock(duration))
                continue

            duration = float(getattr(block, 'block_duration', 0.0) or pp.calc_duration(block))
            if duration > 0:
                self.sequence.append(DelayBlock(duration))

        definitions = getattr(pulseq, 'definitions', {})
        offsets_ppm = definitions.get('offsets_ppm')
        self.offsets_ppm = None if offsets_ppm is None else torch.as_tensor(offsets_ppm)

    def forward(
        self,
        equilibrium_magnetization: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
        exchange_rate: torch.Tensor,
        chemical_shift: torch.Tensor,
        static_off_resonance: torch.Tensor | None = None,
        relative_b1: torch.Tensor | None = None,
        mt_saturation: MTSaturation | None = None,
    ) -> tuple[torch.Tensor]:
        """Simulate the Pulseq-defined BMC signal."""
        parameters = Parameters(
            equilibrium_magnetization,
            t1,
            t2,
            exchange_rate,
            chemical_shift,
            static_off_resonance=static_off_resonance,
            relative_b1=relative_b1,
            mt_saturation=mt_saturation,
        )
        _, signals = self.sequence(parameters)
        if len(signals) == 1:
            return (signals[0],)
        return (torch.stack(signals, dim=0),)
