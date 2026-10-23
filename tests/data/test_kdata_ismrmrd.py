"""Object and serialized ISMRMRD readers share the HDF5 normalization pipeline."""

import copy
import datetime
import io
import weakref

import h5py
import ismrmrd
import ismrmrd.serialization
import pytest
import torch
from mr2.data import KData, KTrajectory
from mr2.data.acq_filters import is_image_acquisition
from mr2.data.traj_calculators import KTrajectoryCartesian, KTrajectoryIsmrmrd
from mr2.data.traj_calculators.KTrajectoryCalculator import DummyTrajectory

from tests.data import IsmrmrdRawTestData


@pytest.fixture
def raw_data(tmp_path):
    """A tiny real HDF5 dataset, with noise and unsorted imaging acquisitions."""
    filename = tmp_path / 'raw.h5'
    IsmrmrdRawTestData(filename, matrix_size=4, n_coils=2, oversampling=1, sampling_order='high_low')
    with ismrmrd.File(filename, 'r') as file:
        header = file['dataset'].header
        acquisitions = file['dataset'].acquisitions[:]
    return filename, header, acquisitions


def serialize_measurements(measurements) -> io.BytesIO:
    """Serialize headers and acquisitions, including ignorable protocol objects."""
    stream = io.BytesIO()
    with ismrmrd.serialization.ProtocolSerializer(stream) as writer:
        writer.serialize('scanner information')
        writer.serialize(ismrmrd.Waveform())
        for header, acquisitions in measurements:
            writer.serialize(header)
            for acq in acquisitions:
                writer.serialize(acq)
    assert not stream.closed
    stream.seek(0)
    return stream


def assert_kdata_equal(actual: KData, expected: KData) -> None:
    """Compare data, trajectory, acquisition metadata and physical header fields."""
    torch.testing.assert_close(actual.data, expected.data)
    torch.testing.assert_close(actual.traj.as_tensor(), expected.traj.as_tensor())
    torch.testing.assert_close(actual.header.acq_info.idx.k1, expected.header.acq_info.idx.k1)
    torch.testing.assert_close(
        actual.header.acq_info.acquisition_time_stamp, expected.header.acq_info.acquisition_time_stamp
    )
    torch.testing.assert_close(
        actual.header.acq_info.orientation.as_matrix(), expected.header.acq_info.orientation.as_matrix()
    )
    assert actual.header.recon_matrix == expected.header.recon_matrix
    assert actual.header.encoding_fov == expected.header.encoding_fov
    assert actual.header.datetime == expected.header.datetime
    assert actual.header.protocol_name == expected.header.protocol_name


@pytest.mark.parametrize('trajectory', [DummyTrajectory(), KTrajectoryCartesian(), KTrajectoryIsmrmrd()])
def test_reader_parity(raw_data, trajectory, tmp_path) -> None:
    """Paths, HDF5 BytesIO, open files, serialized streams and objects agree."""
    filename, header, acquisitions = raw_data
    overwrites = {'datetime': datetime.datetime(2020, 1, 2), 'protocol_name': 'overwritten'}
    expected = KData.from_ismrmrd_hdf5(filename, trajectory, header_overwrites=overwrites)
    assert_kdata_equal(KData.from_ismrmrd(header, acquisitions, trajectory, header_overwrites=overwrites), expected)
    with filename.open('rb') as file:
        assert_kdata_equal(KData.from_file(file, trajectory, header_overwrites=overwrites), expected)
        assert not file.closed
    buffer = io.BytesIO(filename.read_bytes())
    assert_kdata_equal(KData.from_file(buffer, trajectory, header_overwrites=overwrites), expected)
    assert not buffer.closed
    stream = serialize_measurements([(header, acquisitions)])
    assert_kdata_equal(KData.from_ismrmrd_stream(stream, trajectory, header_overwrites=overwrites), expected)
    assert not stream.closed
    stream_path = tmp_path / 'serialized.ismrmrd'
    stream_path.write_bytes(stream.getvalue())
    assert_kdata_equal(KData.from_file(stream_path, trajectory, header_overwrites=overwrites), expected)
    with stream_path.open('rb') as file:
        assert_kdata_equal(KData.from_ismrmrd_stream(file, trajectory, header_overwrites=overwrites), expected)
        assert not file.closed
    # HDF5 detection uses content even for an unfamiliar suffix.
    hdf5_path = tmp_path / 'raw.bin'
    hdf5_path.write_bytes(filename.read_bytes())
    assert_kdata_equal(KData.from_file(hdf5_path, trajectory, header_overwrites=overwrites), expected)


@pytest.mark.parametrize('index', [0, 1, 2, -1, -2, -3])
def test_multiple_measurements(raw_data, index) -> None:
    """Both index directions select the right measurement without mixing headers."""
    _, header, acquisitions = raw_data
    measurements = []
    for i in range(3):
        selected = copy.deepcopy(acquisitions)
        for acq in selected:
            acq.data[:] = complex(i + 1, i + 2)
        measurements.append((header, selected))
    stream = serialize_measurements(measurements)
    result = KData.from_ismrmrd_stream(stream, DummyTrajectory(), dataset_idx=index)
    torch.testing.assert_close(result.data, torch.full_like(result.data, complex(index % 3 + 1, -(index % 3 + 2))))
    if index >= 0 and index < 2:
        assert stream.tell() < len(stream.getvalue()) - 2
    else:
        assert stream.tell() == len(stream.getvalue())
    assert not stream.closed


def test_positive_selection_does_not_read_later_payload(raw_data) -> None:
    """A subsequent malformed acquisition cannot affect an already complete measurement."""
    _, header, acquisitions = raw_data
    prefix = serialize_measurements([(header, acquisitions)]).getvalue()[:-2]
    next_header = io.BytesIO()
    with ismrmrd.serialization.ProtocolSerializer(next_header) as writer:
        writer.serialize(header)
    truncated = io.BytesIO(prefix + next_header.getvalue()[:-2] + b'\x08\x00')
    result = KData.from_ismrmrd_stream(truncated, DummyTrajectory(), dataset_idx=0)
    assert result.data.shape[-2] == 4
    assert truncated.read() == b'\x08\x00'


@pytest.mark.parametrize('index', [3, -4])
def test_out_of_range(raw_data, index) -> None:
    """Invalid measurement indices raise IndexError and leave caller streams open."""
    _, header, acquisitions = raw_data
    stream = serialize_measurements([(header, acquisitions)] * 3)
    with pytest.raises(IndexError, match='index out of range'):
        KData.from_ismrmrd_stream(stream, DummyTrajectory(), dataset_idx=index)
    assert not stream.closed


class Nonseekable(io.BytesIO):
    """A binary input that rejects any attempt to seek or inspect its position."""

    def seek(self, *_args) -> int:
        """Reject seeking."""
        raise io.UnsupportedOperation('not seekable')

    def tell(self) -> int:
        """Reject position inspection."""
        raise io.UnsupportedOperation('not seekable')


def test_nonseekable(raw_data) -> None:
    """The explicit stream reader needs only sequential reads."""
    _, header, acquisitions = raw_data
    source = Nonseekable(serialize_measurements([(header, acquisitions)]).getvalue())
    result = KData.from_ismrmrd_stream(source, DummyTrajectory())
    assert result.data.shape[-2:] == (4, 4)
    assert not source.closed


@pytest.mark.parametrize(
    ('objects', 'message'), [([], 'No ISMRMRD header'), (['acquisition'], 'before the first header')]
)
def test_invalid_grouping(raw_data, objects, message) -> None:
    """Missing headers and acquisitions before headers produce actionable errors."""
    _, _, acquisitions = raw_data
    stream = io.BytesIO()
    with ismrmrd.serialization.ProtocolSerializer(stream) as writer:
        for _ in objects:
            writer.serialize(acquisitions[-1])
    stream.seek(0)
    with pytest.raises(ValueError, match=message):
        KData.from_ismrmrd_stream(stream, DummyTrajectory())
    assert not stream.closed


@pytest.mark.parametrize('empty_group', [False, True])
def test_empty_selection(raw_data, empty_group) -> None:
    """Empty measurements and filters excluding all acquisitions use the existing error."""
    _, header, acquisitions = raw_data
    stream = serialize_measurements([(header, [] if empty_group else acquisitions)])
    with pytest.raises(ValueError, match='No acquisitions meeting'):
        KData.from_ismrmrd_stream(stream, DummyTrajectory(), acquisition_filter_criterion=lambda _: False)
    assert not stream.closed


@pytest.mark.parametrize('trim', [1, 2, 8, 100])
def test_truncation(raw_data, trim) -> None:
    """Missing CLOSE, partial IDs and partial payloads keep their corruption errors."""
    _, header, acquisitions = raw_data
    data = serialize_measurements([(header, acquisitions)]).getvalue()
    stream = io.BytesIO(data[:-trim])
    with pytest.raises((EOFError, ValueError)):
        KData.from_ismrmrd_stream(stream, DummyTrajectory())
    assert not stream.closed


def test_unknown_message() -> None:
    """Unknown protocol IDs are not swallowed as EOF."""
    stream = io.BytesIO(b'\xff\xff')
    with pytest.raises(ValueError, match='Unknown MessageID'):
        KData.from_ismrmrd_stream(stream, DummyTrajectory())
    assert not stream.closed


def test_hdf5_errors(tmp_path) -> None:
    """Corrupt HDF5 and missing paths preserve backend errors."""
    path = tmp_path / 'broken.h5'
    path.write_bytes(b'broken HDF5')
    with pytest.raises(OSError):
        KData.from_file(path, DummyTrajectory())
    with pytest.raises(FileNotFoundError):
        KData.from_file(tmp_path / 'missing.ismrmrd', DummyTrajectory())
    stream = io.BytesIO(path.read_bytes())
    with pytest.raises(OSError):
        KData.from_file(stream, DummyTrajectory())
    assert not stream.closed


def test_disable_filter_and_timestamp(raw_data) -> None:
    """Explicit None supports preselected non-image acquisitions and timestamp fallback."""
    filename, header, acquisitions = raw_data
    imaging = [acq for acq in acquisitions if is_image_acquisition(acq)]
    for acq in imaging:
        acq.setFlag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)
    fallback = datetime.datetime(2001, 2, 3)
    result = KData.from_ismrmrd(
        header, imaging, DummyTrajectory(), acquisition_filter_criterion=None, modification_time=fallback
    )
    assert result.header.datetime == fallback
    assert result.data.shape[-2] == 4
    stream_path = filename.with_suffix('.ismrmrd')
    stream_path.write_bytes(serialize_measurements([(header, imaging)]).getvalue())
    result = KData.from_file(stream_path, DummyTrajectory(), acquisition_filter_criterion=None)
    assert result.header.datetime == datetime.datetime.fromtimestamp(stream_path.stat().st_mtime)


@pytest.mark.parametrize('trajectory', [KTrajectoryCartesian(), KTrajectoryIsmrmrd()])
def test_mixed_readouts_keep_metadata_aligned(raw_data, trajectory) -> None:
    """Dropping a shorter post-discard readout also drops its metadata and trajectory."""
    _, header, acquisitions = raw_data
    acquisitions = [acq for acq in acquisitions if is_image_acquisition(acq)]
    acquisitions[1].discard_post = 1
    retained = [acquisitions[i] for i in (0, 2, 3)]
    expected = KData.from_ismrmrd(header, retained, trajectory)
    with pytest.warns(UserWarning, match='different shape'):
        result = KData.from_ismrmrd(header, acquisitions, trajectory)
    torch.testing.assert_close(result.data, expected.data)
    torch.testing.assert_close(result.traj.as_tensor(), expected.traj.as_tensor())
    torch.testing.assert_close(result.header.acq_info.idx.k1, expected.header.acq_info.idx.k1)
    torch.testing.assert_close(
        result.header.acq_info.acquisition_time_stamp, expected.header.acq_info.acquisition_time_stamp
    )
    assert result.data.shape[-2] == 3


def test_precomputed_trajectory(raw_data) -> None:
    """A precomputed trajectory takes the same path for object and file inputs."""
    filename, header, acquisitions = raw_data
    expected = KData.from_file(filename, KTrajectoryIsmrmrd())
    traj = KTrajectoryIsmrmrd()(
        [acq for acq in acquisitions if is_image_acquisition(acq)], expected.header.encoding_matrix
    )
    assert isinstance(traj, KTrajectory)
    result = KData.from_ismrmrd(header, acquisitions, traj, modification_time=expected.header.datetime)
    assert_kdata_equal(result, expected)


@pytest.mark.parametrize('index', [-1, -3])
def test_trailing_selection_releases_old_acquisitions(raw_data, index, monkeypatch) -> None:
    """Reading many measurements retains only a bounded number of acquisition objects."""
    _, header, acquisitions = raw_data
    image = next(acq for acq in acquisitions if is_image_acquisition(acq))
    stream = serialize_measurements([(header, [image])] * 20)
    deserialize = ismrmrd.Acquisition.deserialize_from
    references = []
    peak_retained = 0

    def tracked_deserialize(read):
        nonlocal peak_retained
        acq = deserialize(read)
        references.append(weakref.ref(acq))
        peak_retained = max(peak_retained, sum(ref() is not None for ref in references))
        return acq

    monkeypatch.setattr(ismrmrd.Acquisition, 'deserialize_from', staticmethod(tracked_deserialize))
    result = KData.from_ismrmrd_stream(stream, DummyTrajectory(), dataset_idx=index)
    assert result.data.shape[-2] == 1
    assert len(references) == 20
    # The trailing groups, the current group, and the next deserialized acquisition.
    assert peak_retained <= abs(index) + 2


@pytest.mark.parametrize('reader', ['from_file', 'from_ismrmrd_hdf5', 'from_ismrmrd_stream', 'from_ismrmrd'])
def test_custom_filter(raw_data, reader) -> None:
    """All entry points forward a custom predicate and crop data before sorting."""
    filename, header, acquisitions = raw_data
    image = next(acq for acq in acquisitions if is_image_acquisition(acq))
    criterion = lambda acq: is_image_acquisition(acq) and acq.idx.kspace_encode_step_1 == image.idx.kspace_encode_step_1  # noqa: E731
    if reader == 'from_ismrmrd':
        result = KData.from_ismrmrd(header, acquisitions, DummyTrajectory(), acquisition_filter_criterion=criterion)
    elif reader == 'from_ismrmrd_stream':
        result = KData.from_ismrmrd_stream(
            serialize_measurements([(header, acquisitions)]), DummyTrajectory(), acquisition_filter_criterion=criterion
        )
    else:
        result = getattr(KData, reader)(filename, DummyTrajectory(), acquisition_filter_criterion=criterion)
    torch.testing.assert_close(result.data.flatten(), torch.as_tensor(image.data).conj().flatten())
    assert result.header.acq_info.idx.k1.item() == image.idx.kspace_encode_step_1


@pytest.mark.parametrize('index', [0, 1, -1, -2])
def test_hdf5_measurements(raw_data, index) -> None:
    """The compatibility reader keeps dataset indexing for HDF5 paths and buffers."""
    filename, _, _ = raw_data
    with h5py.File(filename, 'r+') as file:
        file.copy('dataset', 'dataset_1')
        data = file['dataset_1/data'][:]
        data['data'] = [row * 2 for row in data['data']]
        file['dataset_1/data'][:] = data
    original = KData.from_file(filename, DummyTrajectory(), dataset_idx=0)
    expected = original.data * (1 if index % 2 == 0 else 2)
    result = KData.from_file(io.BytesIO(filename.read_bytes()), DummyTrajectory(), dataset_idx=index)
    torch.testing.assert_close(result.data, expected)


def test_unknown_timestamp_stays_unknown(raw_data) -> None:
    """Loading undated in-memory data does not invent a measurement time."""
    _, header, acquisitions = raw_data
    first = KData.from_ismrmrd(header, acquisitions, DummyTrajectory())
    second = KData.from_ismrmrd(header, acquisitions, DummyTrajectory())
    assert first.header.datetime is None
    assert second.header.datetime is None
    stream = serialize_measurements([(header, acquisitions)])
    assert KData.from_ismrmrd_stream(stream, DummyTrajectory()).header.datetime is None
