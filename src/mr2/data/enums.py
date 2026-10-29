"""Acquisition and data layout enums."""

from enum import Enum, Flag, IntEnum, IntFlag, auto


class BartDimension(IntEnum):
    """BART array axis indices, matching ``src/misc/mri.h``.

    Values are zero-based axis indices, not bitmasks. For example,
    ``other_dims=(BartDimension.TE, BartDimension.SLICE)`` maps two MR2
    leading axes to BART's echo and slice axes. Support for processing
    each axis depends on the BART command.

    References
    ----------
    .. [BARTdims] https://codeberg.org/mrirecon/bart/src/branch/master/src/misc/mri.h
    """

    READ = 0
    """Readout / first spatial axis; coordinate components for non-Cartesian trajectories."""
    PHS1 = 1
    """First phase encoding axis; samples per readout for non-Cartesian data."""
    PHS2 = 2
    """Second phase encoding axis; readouts for non-Cartesian data."""
    COIL = 3
    """Receive coils."""
    MAPS = 4
    """ESPIRiT maps."""
    TE = 5
    """Echo dimension."""
    COEFF = 6
    """First coefficient dimension."""
    COEFF2 = 7
    """Second coefficient dimension."""
    ITER = 8
    """Iteration dimension, also used for motion."""
    CSHIFT = 9
    """Chemical shift dimension."""
    TIME = 10
    """First time dimension."""
    TIME2 = 11
    """Second time dimension."""
    LEVEL = 12
    """Level dimension."""
    SLICE = 13
    """Slices."""
    AVG = 14
    """Averages."""
    BATCH = 15
    """Batches. Not supported by ``bart pics``, use BART's loop option ``bart -l 32768``."""
    MOTION = ITER
    """Alias for the iteration dimension, matching BART's ``MOTION_DIM``."""


class AcqFlags(Flag):
    """Acquisition flags.

    NOTE: values in enum ISMRMRD_AcquisitionFlags start at 1 and not 0, but
    1 << (val-1) is used in 'ismrmrd_is_flag_set' function to calc bitmask value [ISMb]_.

    References
    ----------
    .. [ISMb] ISMRMRD https://github.com/ismrmrd/ismrmrd/blob/master/include/ismrmrd/ismrmrd.h
    """

    ACQ_NO_FLAG = 0
    ACQ_FIRST_IN_ENCODE_STEP1 = auto()
    ACQ_LAST_IN_ENCODE_STEP1 = auto()
    ACQ_FIRST_IN_ENCODE_STEP2 = auto()
    ACQ_LAST_IN_ENCODE_STEP2 = auto()
    ACQ_FIRST_IN_AVERAGE = auto()
    ACQ_LAST_IN_AVERAGE = auto()
    ACQ_FIRST_IN_SLICE = auto()
    ACQ_LAST_IN_SLICE = auto()
    ACQ_FIRST_IN_CONTRAST = auto()
    ACQ_LAST_IN_CONTRAST = auto()
    ACQ_FIRST_IN_PHASE = auto()
    ACQ_LAST_IN_PHASE = auto()
    ACQ_FIRST_IN_REPETITION = auto()
    ACQ_LAST_IN_REPETITION = auto()
    ACQ_FIRST_IN_SET = auto()
    ACQ_LAST_IN_SET = auto()
    ACQ_FIRST_IN_SEGMENT = auto()
    ACQ_LAST_IN_SEGMENT = auto()
    ACQ_IS_NOISE_MEASUREMENT = auto()
    ACQ_IS_PARALLEL_CALIBRATION = auto()
    ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING = auto()
    ACQ_IS_REVERSE = auto()
    ACQ_IS_NAVIGATION_DATA = auto()
    ACQ_IS_PHASECORR_DATA = auto()
    ACQ_LAST_IN_MEASUREMENT = auto()
    ACQ_IS_HPFEEDBACK_DATA = auto()
    ACQ_IS_DUMMYSCAN_DATA = auto()
    ACQ_IS_RTFEEDBACK_DATA = auto()
    ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA = auto()
    ACQ_IS_PHASE_STABILIZATION_REFERENCE = auto()
    ACQ_IS_PHASE_STABILIZATION = 1 << 30
    ACQ_COMPRESSION1 = 1 << 52  # 53 (on purpose!)
    ACQ_COMPRESSION2 = auto()
    ACQ_COMPRESSION3 = auto()
    ACQ_COMPRESSION4 = auto()
    ACQ_USER1 = auto()
    ACQ_USER2 = auto()
    ACQ_USER3 = auto()
    ACQ_USER4 = auto()
    ACQ_USER5 = auto()
    ACQ_USER6 = auto()
    ACQ_USER7 = auto()
    ACQ_USER8 = auto()


class InterleavingDimension(Enum):
    """Interleaving dimension."""

    PHASE = 'phase'
    REPETITION = 'repetition'
    CONTRAST = 'contrast'
    AVERAGE = 'average'
    OTHER = 'other'


class MultibandCalibration(Enum):
    """Multiband calibration."""

    SEPARABLE2_D = 'separable2D'
    FULL3_D = 'full3D'
    OTHER = 'other'


class PatientPosition(Enum):
    """Patient position."""

    HFP = 'HFP'
    HFS = 'HFS'
    HFDR = 'HFDR'
    HFDL = 'HFDL'
    FFP = 'FFP'
    FFS = 'FFS'
    FFDR = 'FFDR'
    FFDL = 'FFDL'
    OTHER = 'OTHER'


class TrajectoryType(Enum):
    """Trajectory type."""

    CARTESIAN = 'cartesian'
    EPI = 'epi'
    RADIAL = 'radial'
    GOLDENANGLE = 'goldenangle'
    SPIRAL = 'spiral'
    OTHER = 'other'


class WaveformInformation(Enum):
    """Waveform information."""

    ECG = 'ecg'
    PULSE = 'pulse'
    RESPIRATORY = 'respiratory'
    TRIGGER = 'trigger'
    GRADIENTWAVEFORM = 'gradientwaveform'
    OTHER = 'other'


class CalibrationMode(Enum):
    """Calibration mode."""

    EMBEDDED = 'embedded'
    INTERLEAVED = 'interleaved'
    SEPARATE = 'separate'
    EXTERNAL = 'external'
    OTHER = 'other'


class TrajType(IntFlag):
    """Special Properties of the Trajectory."""

    SINGLEVALUE = auto()
    ONGRID = auto()
