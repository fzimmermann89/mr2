"""Object and serialized ISMRMRD readers share the HDF5 normalization pipeline."""

import copy
import datetime
import io
from contextlib import closing

import ismrmrd
import ismrmrd.serialization
import pytest
import torch
from mr2.data import KData, KTrajectory
from mr2.data.acq_filters import is_image_acquisition
from mr2.data.traj_calculators import KTrajectoryCartesian, KTrajectoryIsmrmrd, KTrajectorySunflowerGoldenRpe
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


def test_reader_parity(raw_data, tmp_path) -> None:
    """Paths, HDF5 BytesIO, open files, serialized streams and objects agree."""
    filename, header, acquisitions = raw_data
    trajectory = KTrajectoryIsmrmrd()
    overwrites = {'datetime': datetime.datetime(2020, 1, 2), 'protocol_name': 'overwritten'}
    expected = KData.from_file(filename, trajectory, header_overwrites=overwrites)
    assert_kdata_equal(KData.from_ismrmrd(header, acquisitions, trajectory, header_overwrites=overwrites), expected)
    precomputed = trajectory(
        [acq for acq in acquisitions if is_image_acquisition(acq)], expected.header.encoding_matrix
    )
    assert_kdata_equal(KData.from_ismrmrd(header, acquisitions, precomputed, header_overwrites=overwrites), expected)
    with filename.open('rb') as file:
        assert_kdata_equal(KData.from_file(file, trajectory, header_overwrites=overwrites), expected)
        assert not file.closed
    buffer = io.BytesIO(filename.read_bytes())
    assert_kdata_equal(KData.from_file(buffer, trajectory, header_overwrites=overwrites), expected)
    assert not buffer.closed
    stream = serialize_measurements([(header, acquisitions)])
    assert_kdata_equal(
        KData.from_file(stream, trajectory, header_overwrites=overwrites, file_format='ismrmrd-stream'), expected
    )
    assert not stream.closed
    stream_path = tmp_path / 'serialized.mrd'
    stream_path.write_bytes(stream.getvalue())
    assert_kdata_equal(KData.from_file(stream_path, trajectory, header_overwrites=overwrites), expected)
    # HDF5 detection uses content even for an unfamiliar suffix.
    hdf5_path = tmp_path / 'raw.bin'
    hdf5_path.write_bytes(filename.read_bytes())
    assert_kdata_equal(KData.from_file(hdf5_path, trajectory, header_overwrites=overwrites), expected)


@pytest.mark.parametrize('index', [0, 2, -1, -3])
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
    result = KData.from_file(stream, DummyTrajectory(), dataset_idx=index, file_format='ismrmrd-stream')
    torch.testing.assert_close(result.data, torch.full_like(result.data, complex(index % 3 + 1, -(index % 3 + 2))))
    assert not stream.closed


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
    assert_kdata_equal(result, expected)
    assert result.data.shape[-2] == 3


def test_custom_filter(raw_data) -> None:
    """Stream construction forwards a custom acquisition predicate."""
    _, header, acquisitions = raw_data
    image = next(acq for acq in acquisitions if is_image_acquisition(acq))
    criterion = lambda acq: is_image_acquisition(acq) and acq.idx.kspace_encode_step_1 == image.idx.kspace_encode_step_1  # noqa: E731
    result = KData.from_file(
        serialize_measurements([(header, acquisitions)]),
        DummyTrajectory(),
        acquisition_filter_criterion=criterion,
        file_format='ismrmrd-stream',
    )
    torch.testing.assert_close(result.data.flatten(), torch.as_tensor(image.data).conj().flatten())
    assert result.header.acq_info.idx.k1.item() == image.idx.kspace_encode_step_1


@pytest.mark.parametrize('header_format', ['xml', 'bytes'])
def test_server_acquisitions(raw_data, header_format) -> None:
    """Server metadata and decoded acquisitions construct KData without a file round trip."""
    _, header, acquisitions = raw_data
    acquisitions = [acq for acq in acquisitions if is_image_acquisition(acq)]
    expected = KData.from_ismrmrd(header, acquisitions, KTrajectoryCartesian())
    metadata = header.toXML()
    if header_format == 'bytes':
        metadata = metadata.encode()

    result = KData.from_ismrmrd(metadata, iter(acquisitions), KTrajectoryCartesian(), acquisition_filter_criterion=None)
    assert_kdata_equal(result, expected)
    assert result.header.datetime is None


class ShortReads(io.BytesIO):
    """A blocking nonseekable input delivering at most three bytes per read."""

    def seek(self, *_args) -> int:
        """Reject seeking."""
        raise io.UnsupportedOperation('not seekable')

    def tell(self) -> int:
        """Reject position inspection."""
        raise io.UnsupportedOperation('not seekable')

    def read(self, size: int | None = -1) -> bytes:
        """Return at most three bytes per read."""
        return super().read(3 if size is None or size < 0 else min(size, 3))


def test_short_reads(raw_data) -> None:
    """Chunked nonseekable transports preserve data and report premature EOF."""
    _, header, acquisitions = raw_data
    data = serialize_measurements([(header, acquisitions)]).getvalue()
    stream = ShortReads(data)
    expected = KData.from_ismrmrd(header, acquisitions, DummyTrajectory())
    assert_kdata_equal(KData.from_file(stream, DummyTrajectory(), file_format='ismrmrd-stream'), expected)
    assert not stream.closed
    truncated = ShortReads(data[:-1])
    with pytest.raises(EOFError):
        KData.from_file(truncated, DummyTrajectory(), file_format='ismrmrd-stream')
    assert not truncated.closed


def test_iter_acquisitions(raw_data) -> None:
    """Iteration preserves each readout, its counters and measurement header."""
    _, header, acquisitions = raw_data
    imaging = [acq for acq in acquisitions if is_image_acquisition(acq)]
    next_header = copy.deepcopy(header)
    next_header.measurementInformation = ismrmrd.xsd.measurementInformationType(
        patientPosition=ismrmrd.xsd.patientPositionType.HFS, protocolName='next measurement'
    )
    stream = serialize_measurements([(header, imaging), (next_header, imaging[:1])])
    with closing(KData.iter_ismrmrd(stream, KTrajectoryCartesian())) as readouts:
        for acq in imaging:
            result = next(readouts)
            assert_kdata_equal(result, KData.from_ismrmrd(header, [acq], KTrajectoryCartesian()))
        result = next(readouts)
        assert result.header.protocol_name == 'next measurement'
        with pytest.raises(StopIteration):
            next(readouts)
    assert not stream.closed


@pytest.mark.parametrize('trajectory', [KTrajectoryIsmrmrd(normalize=True), KTrajectorySunflowerGoldenRpe()])
def test_iter_rejects_measurement_trajectory(raw_data, trajectory) -> None:
    """Trajectories requiring all readouts cannot be evaluated independently."""
    _, header, acquisitions = raw_data
    stream = serialize_measurements([(header, acquisitions)])
    with pytest.raises(ValueError, match='complete measurement'):
        next(KData.iter_ismrmrd(stream, trajectory))
    assert stream.tell() == 0
    assert not stream.closed


def test_iter_rejects_full_measurement_trajectory(raw_data) -> None:
    """A single readout must not broadcast over a full measurement's trajectory."""
    _, header, acquisitions = raw_data
    coordinates = torch.arange(16).reshape(4, 1, 1, 1, 4).float()
    trajectory = KTrajectory(coordinates, coordinates, coordinates)
    stream = serialize_measurements([(header, acquisitions)])
    with pytest.raises(ValueError, match=r'Trajectory shape .* does not match data shape'):
        next(KData.iter_ismrmrd(stream, trajectory))
    assert not stream.closed


def test_iter_close_owned_file(raw_data, tmp_path, monkeypatch) -> None:
    """Closing iteration early closes the file opened by the iterator."""
    _, header, acquisitions = raw_data
    filename = tmp_path / 'measurements.ismrmrd'
    filename.write_bytes(serialize_measurements([(header, acquisitions)]).getvalue())
    stream = filename.open('rb')
    monkeypatch.setattr(type(filename), 'open', lambda *_args, **_kwargs: stream)
    with closing(KData.iter_ismrmrd(filename, DummyTrajectory())) as readouts:
        next(readouts)
        assert not stream.closed
    assert stream.closed
