import pypulseq as pp
import pytest
import torch
from mr2.operators.models.BMC import (
    BMCSequence,
    ConstantRFBlock,
    DelayBlock,
    LongitudinalReadoutBlock,
    Parameters,
    PiecewiseRFBlock,
    ResetBlock,
    SpoilBlock,
)
from mr2.operators.models.PulseqBMC import PulseqBMCModel


def test_pulseq_bmc_constant_rf_matches_explicit_sequence() -> None:
    system = pp.Opts(rf_ringdown_time=0, rf_dead_time=0, rf_raster_time=1e-6)
    pulse = pp.make_block_pulse(flip_angle=0.9, duration=12e-3, phase_offset=0.3, freq_offset=23.0, system=system)
    adc = pp.make_adc(num_samples=1, duration=1e-3, delay=2e-3)
    recovery = pp.make_delay(8e-3)

    pulseq = pp.Sequence(system=system)
    pulseq.add_block(pulse)
    pulseq.add_block(recovery)
    pulseq.add_block(adc)

    model = PulseqBMCModel(pulseq)

    explicit = BMCSequence()
    explicit.append(
        ConstantRFBlock(
            duration=float(pulse.shape_dur),
            rf_amplitude=torch.as_tensor(abs(pulse.signal[0])),
            rf_phase=torch.as_tensor(float(pulse.phase_offset) - torch.angle(torch.as_tensor(pulse.signal[0]))),
            rf_frequency=torch.as_tensor(float(pulse.freq_offset)),
        )
    )
    explicit.append(DelayBlock(float(pp.calc_duration(recovery))))
    explicit.append(DelayBlock(float(adc.delay)))
    explicit.append(LongitudinalReadoutBlock())
    explicit.append(DelayBlock(float(pp.calc_duration(adc)) - float(adc.delay)))

    parameters = Parameters(
        equilibrium_magnetization=torch.tensor([1.0, 0.08]),
        t1=torch.tensor([1.4, 1.0]),
        t2=torch.tensor([0.08, 0.04]),
        exchange_rate=torch.tensor([[0.0, 8.0], [0.64, 0.0]]),
        chemical_shift=torch.tensor([0.0, 40.0]),
    )

    (parsed_signal,) = model(
        parameters.equilibrium_magnetization,
        parameters.t1,
        parameters.t2,
        parameters.exchange_rate,
        parameters.chemical_shift,
    )
    _, explicit_signals = explicit(parameters)

    torch.testing.assert_close(parsed_signal, explicit_signals[0])
    torch.testing.assert_close(model.sequence.duration.to(torch.float64), explicit.duration.to(torch.float64))


def test_pulseq_bmc_shaped_sequence_matches_explicit_sequence() -> None:
    reset_between_readouts = True
    system = pp.Opts(rf_ringdown_time=0, rf_dead_time=0, rf_raster_time=2.5e-4)
    pulse_signal = torch.tensor([0.2, 0.7, 1.1, 0.9, 0.4], dtype=torch.float32)
    dt = 2.5e-4
    freq_offset = 31.0
    pulse_duration = dt * pulse_signal.numel()
    accum_phase = float(2 * torch.pi * freq_offset * pulse_duration)

    pulse_1 = pp.make_arbitrary_rf(
        signal=pulse_signal.numpy(),
        flip_angle=1.0,
        dwell=dt,
        delay=7.5e-4,
        freq_offset=freq_offset,
        no_signal_scaling=True,
        phase_offset=0.0,
        system=system,
        use='saturation',
    )
    pulse_2 = pp.make_arbitrary_rf(
        signal=pulse_signal.numpy(),
        flip_angle=1.0,
        dwell=dt,
        delay=0.0,
        freq_offset=freq_offset,
        no_signal_scaling=True,
        phase_offset=accum_phase,
        system=system,
        use='saturation',
    )
    interpulse_delay = pp.make_delay(3e-3)
    interreadout_delay = pp.make_delay(0.12)
    gx_spoil = pp.make_trapezoid(channel='x', amplitude=0.03, flat_time=4e-3, rise_time=8e-4, system=system)
    adc = pp.make_adc(num_samples=1, duration=1e-3, delay=6e-4)

    pulseq = pp.Sequence(system=system)
    pulseq.add_block(pulse_1)
    pulseq.add_block(interpulse_delay)
    pulseq.add_block(pulse_2)
    pulseq.add_block(gx_spoil)
    pulseq.add_block(adc)
    pulseq.add_block(interreadout_delay)
    pulseq.add_block(pulse_1)
    pulseq.add_block(interpulse_delay)
    pulseq.add_block(pulse_2)
    pulseq.add_block(gx_spoil)
    pulseq.add_block(adc)

    model = PulseqBMCModel(pulseq, reset_between_readouts=reset_between_readouts)

    explicit = BMCSequence()
    explicit.append(DelayBlock(float(pulse_1.delay)))
    explicit.append(PiecewiseRFBlock(rf_amplitude=pulse_signal, rf_phase=0.0, rf_frequency=freq_offset, dt=dt))
    explicit.append(DelayBlock(float(pp.calc_duration(interpulse_delay))))
    explicit.append(PiecewiseRFBlock(rf_amplitude=pulse_signal, rf_phase=0.0, rf_frequency=freq_offset, dt=dt))
    explicit.append(SpoilBlock(float(pp.calc_duration(gx_spoil))))
    explicit.append(DelayBlock(float(adc.delay)))
    explicit.append(LongitudinalReadoutBlock())
    explicit.append(DelayBlock(float(pp.calc_duration(adc)) - float(adc.delay)))
    if reset_between_readouts:
        explicit.append(ResetBlock())
    explicit.append(DelayBlock(float(pp.calc_duration(interreadout_delay))))
    explicit.append(DelayBlock(float(pulse_1.delay)))
    explicit.append(PiecewiseRFBlock(rf_amplitude=pulse_signal, rf_phase=0.0, rf_frequency=freq_offset, dt=dt))
    explicit.append(DelayBlock(float(pp.calc_duration(interpulse_delay))))
    explicit.append(PiecewiseRFBlock(rf_amplitude=pulse_signal, rf_phase=0.0, rf_frequency=freq_offset, dt=dt))
    explicit.append(SpoilBlock(float(pp.calc_duration(gx_spoil))))
    explicit.append(DelayBlock(float(adc.delay)))
    explicit.append(LongitudinalReadoutBlock())
    explicit.append(DelayBlock(float(pp.calc_duration(adc)) - float(adc.delay)))
    if reset_between_readouts:
        explicit.append(ResetBlock())

    parameters = Parameters(
        equilibrium_magnetization=torch.tensor([1.0, 0.11]),
        t1=torch.tensor([1.2, 0.9]),
        t2=torch.tensor([0.09, 0.05]),
        exchange_rate=torch.tensor([[0.0, 10.0], [1.1, 0.0]]),
        chemical_shift=torch.tensor([0.0, 55.0]),
    )

    (parsed_signal,) = model(
        parameters.equilibrium_magnetization,
        parameters.t1,
        parameters.t2,
        parameters.exchange_rate,
        parameters.chemical_shift,
    )
    _, explicit_signals = explicit(parameters)

    torch.testing.assert_close(parsed_signal, torch.stack(explicit_signals, dim=0))
    torch.testing.assert_close(model.sequence.duration.to(torch.float64), explicit.duration.to(torch.float64))
