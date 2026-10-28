"""Tests for acquisition enums."""

import ismrmrd
import pytest
from mr2.data.enums import AcqFlags


@pytest.mark.parametrize('name', AcqFlags.__members__)
def test_acq_flags_match_ismrmrd(name: str) -> None:
    """Every acquisition flag matches the corresponding ISMRMRD bitmask."""
    # ISMRMRD constants are one-based bit positions; it has no ACQ_NO_FLAG constant.
    expected = 0 if name == 'ACQ_NO_FLAG' else 1 << (getattr(ismrmrd, name) - 1)
    assert AcqFlags[name].value == expected
