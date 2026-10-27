"""Tests for acquisition filters."""

import ismrmrd
import pytest
from mr2.data.acq_filters import is_image_acquisition


@pytest.mark.parametrize(
    'flags',
    [
        (),
        (ismrmrd.ACQ_FIRST_IN_ENCODE_STEP2,),
        (ismrmrd.ACQ_LAST_IN_ENCODE_STEP2,),
        (ismrmrd.ACQ_FIRST_IN_AVERAGE,),
        (ismrmrd.ACQ_LAST_IN_AVERAGE,),
        (ismrmrd.ACQ_FIRST_IN_ENCODE_STEP2, ismrmrd.ACQ_LAST_IN_AVERAGE),
    ],
)
def test_is_image_acquisition_keeps_boundary_flags(flags: tuple[int, ...]) -> None:
    """Encoding-step and average boundary flags do not exclude imaging readouts."""
    acquisition = ismrmrd.Acquisition()
    for flag in flags:
        acquisition.setFlag(flag)
    assert is_image_acquisition(acquisition)


@pytest.mark.parametrize('flag', [ismrmrd.ACQ_IS_PHASE_STABILIZATION, ismrmrd.ACQ_IS_PHASE_STABILIZATION_REFERENCE])
def test_is_image_acquisition_rejects_phase_stabilization(flag: int) -> None:
    """Phase-stabilization readouts are excluded from imaging."""
    acquisition = ismrmrd.Acquisition()
    acquisition.setFlag(flag)
    assert not is_image_acquisition(acquisition)
