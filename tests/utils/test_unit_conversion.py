"""Tests of unit conversion."""

from collections.abc import Callable
from typing import TypeVar

import numpy as np
import pytest
import torch
from mr2.utils import RandomGenerator
from mr2.utils.unit_conversion import (
    GYROMAGNETIC_RATIO_PROTON,
    deg_to_rad,
    lamor_frequency_to_magnetic_field,
    m_to_micrometer,
    m_to_mm,
    magnetic_field_to_lamor_frequency,
    micrometer_to_m,
    mm_to_m,
    ms_to_s,
    rad_to_deg,
    s_to_ms,
)

ConversionInput = TypeVar('ConversionInput', float, torch.Tensor, list[float], tuple[float, ...])

CONVERSIONS = pytest.mark.parametrize(
    ('function', 'reference'),
    [
        (mm_to_m, lambda x: x / 1000),
        (m_to_mm, lambda x: x * 1000),
        (micrometer_to_m, lambda x: x / 1.0e6),
        (m_to_micrometer, lambda x: x * 1.0e6),
        (ms_to_s, lambda x: x / 1000),
        (s_to_ms, lambda x: x * 1000),
        (rad_to_deg, torch.rad2deg),
        (deg_to_rad, torch.deg2rad),
        (lamor_frequency_to_magnetic_field, lambda x: x / GYROMAGNETIC_RATIO_PROTON),
        (magnetic_field_to_lamor_frequency, lambda x: x * GYROMAGNETIC_RATIO_PROTON),
    ],
    ids=[
        'mm_to_m',
        'm_to_mm',
        'micrometer_to_m',
        'm_to_micrometer',
        'ms_to_s',
        's_to_ms',
        'rad_to_deg',
        'deg_to_rad',
        'lamor_frequency_to_magnetic_field',
        'magnetic_field_to_lamor_frequency',
    ],
)

INPUTS = pytest.mark.parametrize(
    'x',
    [
        pytest.param(torch.tensor([-2.5, 0.0, 3.0], dtype=torch.float64), id='tensor'),
        pytest.param([-2.5, 0.0, 3.0], id='list'),
        pytest.param((-2.5, 0.0, 3.0), id='tuple'),
        pytest.param(1.25, id='float'),
    ],
)


@CONVERSIONS
@INPUTS
def test_unit_conversion(
    function: Callable[[ConversionInput], ConversionInput],
    reference: Callable[[torch.Tensor], torch.Tensor],
    x: ConversionInput,
) -> None:
    """Verify conversion values and preservation of the input type."""
    result = function(x)
    assert type(result) is type(x)
    if isinstance(result, (list, tuple)):
        assert all(isinstance(value, float) for value in result)
    torch.testing.assert_close(
        torch.as_tensor(result, dtype=torch.float64),
        reference(torch.as_tensor(x, dtype=torch.float64)),
        atol=0,
        rtol=1e-7,
    )


@CONVERSIONS
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize(
    'device',
    [
        'cpu',
        pytest.param('cuda', marks=pytest.mark.cuda),
    ],
)
def test_unit_conversion_tensor(
    function: Callable[[torch.Tensor], torch.Tensor],
    reference: Callable[[torch.Tensor], torch.Tensor],
    dtype: torch.dtype,
    device: str,
) -> None:
    """Verify tensor values, shape, dtype, and device are preserved."""
    x = RandomGenerator(9).float32_tensor((3, 4, 5)).to(dtype=dtype, device=device)
    result = function(x)
    assert isinstance(result, torch.Tensor)
    assert result.shape == x.shape
    assert result.dtype == x.dtype
    assert result.device == x.device
    torch.testing.assert_close(result, reference(x), atol=0, rtol=1e-6)


@pytest.mark.parametrize(
    ('function', 'reference'),
    [
        (lamor_frequency_to_magnetic_field, lambda x, gamma: x / gamma),
        (magnetic_field_to_lamor_frequency, lambda x, gamma: x * gamma),
    ],
    ids=['lamor_frequency_to_magnetic_field', 'magnetic_field_to_lamor_frequency'],
)
@pytest.mark.parametrize('gyromagnetic_ratio', [2 * GYROMAGNETIC_RATIO_PROTON, -4.316e6])
@INPUTS
def test_unit_conversion_custom_gyromagnetic_ratio(
    function: Callable[..., ConversionInput],
    reference: Callable[[torch.Tensor, float], torch.Tensor],
    gyromagnetic_ratio: float,
    x: ConversionInput,
) -> None:
    """Verify every input type uses the supplied gyromagnetic ratio."""
    result = function(x, gyromagnetic_ratio=gyromagnetic_ratio)
    assert type(result) is type(x)
    torch.testing.assert_close(
        torch.as_tensor(result, dtype=torch.float64),
        reference(torch.as_tensor(x, dtype=torch.float64), gyromagnetic_ratio),
        atol=0,
        rtol=1e-7,
    )


@pytest.mark.parametrize(
    ('forward', 'inverse'),
    [
        (mm_to_m, m_to_mm),
        (micrometer_to_m, m_to_micrometer),
        (ms_to_s, s_to_ms),
        (rad_to_deg, deg_to_rad),
        (lamor_frequency_to_magnetic_field, magnetic_field_to_lamor_frequency),
    ],
    ids=['length_mm', 'length_micrometer', 'time', 'angle', 'magnetic_field'],
)
@INPUTS
def test_unit_conversion_round_trip(
    forward: Callable[[ConversionInput], ConversionInput],
    inverse: Callable[[ConversionInput], ConversionInput],
    x: ConversionInput,
) -> None:
    """Verify round trips in both directions preserve values and types."""
    for result in (inverse(forward(x)), forward(inverse(x))):
        assert type(result) is type(x)
        torch.testing.assert_close(result, x)


@pytest.mark.parametrize('gyromagnetic_ratio', [2 * GYROMAGNETIC_RATIO_PROTON, -4.316e6])
@INPUTS
def test_unit_conversion_custom_gyromagnetic_ratio_round_trip(gyromagnetic_ratio: float, x: ConversionInput) -> None:
    """Verify frequency and field round trips with a custom gyromagnetic ratio."""
    field = lamor_frequency_to_magnetic_field(x, gyromagnetic_ratio=gyromagnetic_ratio)
    frequency = magnetic_field_to_lamor_frequency(x, gyromagnetic_ratio=gyromagnetic_ratio)
    torch.testing.assert_close(magnetic_field_to_lamor_frequency(field, gyromagnetic_ratio=gyromagnetic_ratio), x)
    torch.testing.assert_close(lamor_frequency_to_magnetic_field(frequency, gyromagnetic_ratio=gyromagnetic_ratio), x)


def test_rad_to_deg_float() -> None:
    """Verify radians to degree conversion."""
    assert rad_to_deg(np.pi / 2) == 90.0


def test_deg_to_rad_float() -> None:
    """Verify degree to radians conversion."""
    assert deg_to_rad(180.0) == np.pi


def test_lamor_frequency_to_magnetic_field() -> None:
    """Verify conversion of lamor frequency to magnetic field."""
    proton_gyromagnetic_ratio = 42.58 * 1e6
    proton_lamor_frequency_at_3tesla = 127.74 * 1e6
    assert lamor_frequency_to_magnetic_field(proton_lamor_frequency_at_3tesla, proton_gyromagnetic_ratio) == 3.0


def test_magnetic_field_to_lamor_frequency() -> None:
    """Verify conversion of magnetic field to lamor frequency."""
    proton_gyromagnetic_ratio = 42.58 * 1e6
    magnetic_field_strength = 3.0
    assert magnetic_field_to_lamor_frequency(magnetic_field_strength, proton_gyromagnetic_ratio) == 127.74 * 1e6
