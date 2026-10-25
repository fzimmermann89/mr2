"""Tests for the A4IM low-field ISMRMRD datasets."""

import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests
import torch
from mr2.data import KData
from mr2.data.traj_calculators import KTrajectoryCartesian
from mr2.phantoms.a4im_lowfield import (
    FreeMaxDataset,
    I3MDataset,
    IBTDataset,
    OSI2Dataset,
    download_free_max,
    download_i3m,
    download_ibt,
    download_osi2,
    download_zenodo_files,
    extract_zip,
)


def test_ibt_average_dataset(ismrmrd_cart, tmp_path) -> None:
    """Separate Philips average files become a leading average dimension."""
    for average in (2, 0, 1):
        shutil.copy2(ismrmrd_cart.filename, tmp_path / f'Subject01-Brain-T2_average{average}.h5')
    shutil.copy2(ismrmrd_cart.filename, tmp_path / 'Subject01-Brain-T2.h5')

    dataset = IBTDataset(tmp_path, acquisition='t2w', individual_averages=True)
    assert len(dataset) == 1
    kdata = dataset[0]
    assert kdata.shape[0] == 3
    torch.testing.assert_close(kdata.header.acq_info.idx.average[:, 0, 0, 0, 0, 0], torch.arange(3))


def test_ibt_average_dataset_rejects_duplicate_average(ismrmrd_cart, tmp_path) -> None:
    """A repeated path cannot silently create two copies of the same average."""
    filename = tmp_path / 'Subject01-Brain-T2_average0.h5'
    shutil.copy2(ismrmrd_cart.filename, filename)
    with pytest.raises(ValueError, match='Duplicate average indices'):
        IBTDataset([filename, filename], individual_averages=True)


def test_ibt_dataset_selects_acquisition_and_combined_data(ismrmrd_cart, tmp_path) -> None:
    """IBT selectors distinguish combined acquisitions from individual averages."""
    for name in ('Subject01-Brain-T1.h5', 'Subject01-Brain-T2.h5', 'Subject01-Brain-T2_average0.h5'):
        shutil.copy2(ismrmrd_cart.filename, tmp_path / name)
    dataset = IBTDataset(tmp_path, acquisition='t2w')
    assert [file.name for file in dataset.files] == ['Subject01-Brain-T2.h5']


def test_osi2_dataset_selectors(tmp_path) -> None:
    """OSI² selectors combine subset, contrast, noise correction, and subject."""
    filenames = (
        'LLR/noise_corr_on/9147/T2w_TSE_underS_R1.h5',
        'LLR/noise_corr_off/9147/T2w_TSE_underS_R1.h5',
        'LLR/noise_corr_on/9003/T2w_TSE_underS_R1p7.h5',
        'R1_R2/noise_corr_on/9147/T2w_TSE_R2.h5',
    )
    for filename in filenames:
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    dataset = OSI2Dataset(tmp_path, acquisition='t2w', noise_correction=True, subsets='llr', subjects=9147)
    assert [file.relative_to(tmp_path).as_posix() for file in dataset.files] == [
        'LLR/noise_corr_on/9147/T2w_TSE_underS_R1.h5'
    ]


@pytest.mark.parametrize(
    ('acquisition', 'filename'),
    [
        ('pdw', 'T1_3_4_0_Lab2.h5'),
        ('ir_t1w', 'T1w_IR_TSE_R1_Lab2.h5'),
        ('3t_t1w', '3T_T1w_TSE_PF.h5'),
    ],
)
def test_osi2_acquisition_names(tmp_path, acquisition, filename) -> None:
    """OSI² selectors cover the alternate names used in the archives."""
    for name in ('T1_3_4_0_Lab2.h5', 'T1w_IR_TSE_R1_Lab2.h5', '3T_T1w_TSE_PF.h5'):
        (tmp_path / name).touch()
    assert [file.name for file in OSI2Dataset(tmp_path, acquisition=acquisition).files] == [filename]


def test_i3m_dataset_selectors(tmp_path) -> None:
    """i3M mapping selectors recognize in-vivo and phantom filenames."""
    for name in (
        'Subject2-LeftKnee-TI40ms (415).h5',
        'Subject2-RightKnee-TI40ms (287).h5',
        'Subject2-RightKnee-TE50ms (473).h5',
        'Subject2-RightKnee-T1 (331).h5',
        'Lab_n01_TI40ms.h5',
    ):
        (tmp_path / name).touch()
    assert [file.name for file in I3MDataset(tmp_path, acquisition='t1map').files] == [
        'Lab_n01_TI40ms.h5',
        'Subject2-LeftKnee-TI40ms (415).h5',
        'Subject2-RightKnee-TI40ms (287).h5',
    ]


@pytest.mark.parametrize(
    ('kwargs', 'expected'),
    [
        ({'acquisition': 't2w'}, ['t2_tse_sag_Left.mrd', 't2_tse_sag_Right.mrd']),
        ({'acquisition': 't1map'}, ['t1map_MOLLI_Right.mrd']),
    ],
)
def test_free_max_dataset_selectors(tmp_path, kwargs, expected) -> None:
    """Free.Max selectors retain both sides and exclude Deep Resolve files."""
    for name in (
        't2_tse_sag_Left.mrd',
        't2_tse_sag_Left_DeepResolve.mrd',
        't2_tse_sag_Right.mrd',
        't1map_MOLLI_Right.mrd',
    ):
        (tmp_path / name).touch()
    dataset = FreeMaxDataset(tmp_path, **kwargs)
    assert [file.name for file in dataset.files] == expected


def test_download_zenodo_files(monkeypatch, tmp_path) -> None:
    """Downloads are verified, safely extracted, and cached by a manifest."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('dataset/scan.h5', b'raw data')
    content = buffer.getvalue()
    metadata = {
        'files': [
            {
                'key': 'data.zip',
                'size': len(content),
                'checksum': f'md5:{hashlib.md5(content, usedforsecurity=False).hexdigest()}',
                'links': {'self': 'https://example.test/data.zip'},
            }
        ]
    }
    metadata_response = Mock()
    metadata_response.json.return_value = metadata
    download_response = Mock()
    download_response.iter_content.return_value = [content]
    get = Mock(side_effect=(metadata_response, download_response, metadata_response))
    monkeypatch.setattr('mr2.phantoms.a4im_lowfield.requests.get', get)
    download_zenodo_files(1, ['data.zip', 'data.zip'], tmp_path)
    download_zenodo_files(1, ['data.zip'], tmp_path)

    assert (tmp_path / 'dataset/scan.h5').read_bytes() == b'raw data'
    assert not (tmp_path / 'data.zip').exists()
    assert get.call_count == 3


def test_extract_zip_rejects_parent_path(tmp_path) -> None:
    """ZIP members cannot escape the output directory."""
    archive = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(archive, 'w') as zip_file:
        zip_file.writestr('../outside', b'unsafe')
    with pytest.raises(ValueError, match='outside output directory'):
        extract_zip(archive, tmp_path / 'output')


@pytest.mark.parametrize(
    ('download', 'kwargs', 'record', 'filenames'),
    [
        (download_osi2, {'subsets': ('llr', 'loraks')}, 19661402, ['LLR.zip', 'R1_R2.zip']),
        (download_ibt, {'subjects': (10, 1, 1)}, 18847561, ['Subject 01.zip', 'Subject 10.zip']),
        (
            download_i3m,
            {'subjects': 2, 'phantom': True, 'documentation': True},
            20700288,
            [
                'Subject 2.zip',
                'ACRphantomData.zip',
                'Useful information for data users.pdf',
                'In vivo imaging with MRILab\N{RIGHT SINGLE QUOTATION MARK}s Physio 1 Scanner.pdf',
            ],
        ),
        (download_free_max, {'subjects': 3, 'phantoms': True}, 20516472, ['Subject3.zip', 'PHANTOMS.zip']),
    ],
)
def test_dataset_download_selection(monkeypatch, tmp_path, download, kwargs, record, filenames) -> None:
    """Dataset selectors map to the correct Zenodo archives."""
    selected = {}

    def mock_download(selected_record, selected_filenames, output_directory, **_):
        selected['record'] = selected_record
        selected['filenames'] = selected_filenames
        return Path(output_directory)

    monkeypatch.setattr('mr2.phantoms.a4im_lowfield.download_zenodo_files', mock_download)
    assert download(output_directory=tmp_path, **kwargs) == tmp_path
    assert selected == {'record': record, 'filenames': filenames}


def test_free_max_known_corrupt_file(tmp_path) -> None:
    """Only the documented truncated subject/scan is excluded, with a warning."""
    for subject in (9, 10):
        folder = tmp_path / f'Subject{subject}' / 'raw'
        folder.mkdir(parents=True)
        (folder / 'pd_tse_sag_Right.mrd').touch()
    with pytest.warns(UserWarning, match='known truncated'):
        dataset = FreeMaxDataset(tmp_path)
    assert len(dataset) == 1
    assert dataset.files[0].parent.parent.name == 'Subject10'


def test_ibt_missing_averages_and_metadata(ismrmrd_cart, tmp_path) -> None:
    """Noncontiguous average labels survive; shared timing is not duplicated."""
    for average in (10, 2):
        shutil.copy2(ismrmrd_cart.filename, tmp_path / f'Brain-T2_average{average}.h5')
    dataset = IBTDataset(tmp_path, individual_averages=True)
    first = dataset.load(dataset.files[0])
    assert first.header.acq_info.idx.average.unique().tolist() == [0]
    result = dataset[0]
    torch.testing.assert_close(result.data.broadcast_to(result.shape)[0], first.data)
    torch.testing.assert_close(result.data.broadcast_to(result.shape)[1], first.data)
    assert result.header.te == first.header.te
    assert result.header.recon_fov == first.header.recon_fov
    assert result.header.acq_info.idx.average.flatten().tolist() == [2, 10]


def test_ibt_incompatible_geometry(ismrmrd_cart, tmp_path, monkeypatch) -> None:
    """Files with matching names cannot silently mix different geometries."""
    for average in (0, 1):
        (tmp_path / f'Brain-T2_average{average}.h5').touch()
    first = KData.from_file(ismrmrd_cart.filename, KTrajectoryCartesian())
    second = first.clone()
    second.header.recon_fov.x *= 2
    dataset = IBTDataset(tmp_path, individual_averages=True)
    monkeypatch.setattr(dataset, 'load', Mock(side_effect=[first, second]))
    with pytest.raises(ValueError, match='Incompatible individual averages'):
        dataset[0]


@pytest.fixture
def zenodo_mock(monkeypatch):
    """Provide a tiny ZIP record and independent responses for every request."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('dataset/scan.h5', b'raw data')
    content = buffer.getvalue()
    metadata = {
        'files': [
            {
                'key': 'data.zip',
                'size': len(content),
                'checksum': f'md5:{hashlib.md5(content, usedforsecurity=False).hexdigest()}',
                'links': {'self': 'https://example.test/data.zip'},
            }
        ]
    }

    def response(url, **_):
        result = Mock()
        if '/api/' in url:
            result.json.return_value = metadata
        else:
            result.iter_content.return_value = [content]
        return result

    get = Mock(side_effect=response)
    monkeypatch.setattr('mr2.phantoms.a4im_lowfield.requests.get', get)
    return get, metadata, content


@pytest.mark.parametrize('manifest', ['{', 'null', '[]', '{}', '{"members": []}'])
def test_invalid_cache_manifest(tmp_path, zenodo_mock, manifest) -> None:
    """Malformed or incomplete manifests trigger a clean download."""
    download_zenodo_files(1, 'data.zip', tmp_path)
    (tmp_path / '.mr2-downloads/1-data.zip.json').write_text(manifest)
    download_zenodo_files(1, 'data.zip', tmp_path)
    assert zenodo_mock[0].call_count == 4


@pytest.mark.parametrize('damage', ['missing', 'truncated', 'same_size'])
def test_damaged_cached_member(tmp_path, zenodo_mock, damage) -> None:
    """Missing, truncated and hash-mismatched extracted files can be recovered."""
    download_zenodo_files(1, 'data.zip', tmp_path)
    member = tmp_path / 'dataset/scan.h5'
    if damage == 'missing':
        member.unlink()
    else:
        member.write_bytes(b'bad' if damage == 'truncated' else b'bad data')
    download_zenodo_files(1, 'data.zip', tmp_path, verify=True)
    assert member.read_bytes() == b'raw data'
    assert zenodo_mock[0].call_count == 4


@pytest.mark.parametrize(('record', 'filename'), [(2, 'data.zip'), (1, 'other.zip')])
@pytest.mark.parametrize('member_name', ['dataset/scan.h5', 'dataset/./scan.h5'])
def test_overlapping_archive_cache(tmp_path, zenodo_mock, record, filename, member_name) -> None:
    """Installing overlapping archives cannot leave a stale size-only cache hit."""
    get, metadata, original_content = zenodo_mock
    download_zenodo_files(1, 'data.zip', tmp_path)
    # Unrelated malformed manifests must not prevent installing valid downloads.
    (tmp_path / '.mr2-downloads/invalid.json').write_text('null')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr(member_name, b'new data')
    replacement_content = buffer.getvalue()
    replacement_metadata = {
        'files': [
            {
                'key': filename,
                'size': len(replacement_content),
                'checksum': f'md5:{hashlib.md5(replacement_content, usedforsecurity=False).hexdigest()}',
                'links': {'self': 'https://example.test/replacement.zip'},
            }
        ]
    }
    replacement_response = Mock()
    replacement_response.iter_content.return_value = [replacement_content]
    replacement_record_response = Mock()
    replacement_record_response.json.return_value = replacement_metadata
    original_response = Mock()
    original_response.iter_content.return_value = [original_content]
    original_record_response = Mock()
    original_record_response.json.return_value = metadata
    get.side_effect = [
        replacement_record_response,
        replacement_response,
        original_record_response,
        original_response,
        original_record_response,
    ]
    download_zenodo_files(record, filename, tmp_path)
    assert (tmp_path / 'dataset/scan.h5').read_bytes() == b'new data'
    download_zenodo_files(1, 'data.zip', tmp_path)
    assert (tmp_path / 'dataset/scan.h5').read_bytes() == b'raw data'
    download_zenodo_files(1, 'data.zip', tmp_path)
    assert get.call_count == 7  # The final call reuses the restored cache.


def test_checksum_failure_and_retry(tmp_path, zenodo_mock) -> None:
    """A failed checksum publishes neither extracted members nor a manifest."""
    get, metadata, _ = zenodo_mock
    checksum = metadata['files'][0]['checksum']
    metadata['files'][0]['checksum'] = 'md5:incorrect'
    with pytest.raises(OSError, match='Checksum verification failed'):
        download_zenodo_files(1, 'data.zip', tmp_path)
    assert list((tmp_path / '.mr2-downloads').iterdir()) == []
    assert not (tmp_path / 'dataset').exists()
    metadata['files'][0]['checksum'] = checksum
    download_zenodo_files(1, 'data.zip', tmp_path)
    assert get.call_count == 4


@pytest.mark.parametrize('failure', ['interrupted', 'http', 'extraction'])
def test_failed_download_cleanup(tmp_path, zenodo_mock, monkeypatch, failure) -> None:
    """Interrupted responses, HTTP errors and bad archives leave a retryable cache."""
    get, metadata, content = zenodo_mock
    if failure == 'extraction':
        content = b'not a ZIP'
        metadata['files'][0]['checksum'] = f'md5:{hashlib.md5(content, usedforsecurity=False).hexdigest()}'
    response = Mock()
    if failure == 'interrupted':

        def chunks(**_):
            yield content[:5]
            raise requests.ConnectionError('interrupted')

        response.iter_content.side_effect = chunks
    else:
        response.iter_content.return_value = [content]
    if failure == 'http':
        response.raise_for_status.side_effect = requests.HTTPError('failed')
    metadata_response = Mock()
    metadata_response.json.return_value = metadata
    get.side_effect = [metadata_response, response]
    error = zipfile.BadZipFile if failure == 'extraction' else requests.RequestException
    with pytest.raises(error):
        download_zenodo_files(1, 'data.zip', tmp_path)
    assert not (tmp_path / 'dataset').exists()
    assert list((tmp_path / '.mr2-downloads').iterdir()) == []
    response.close.assert_called_once()


def test_concurrent_download_rejected(tmp_path, zenodo_mock) -> None:
    """A second downloader cannot modify a cache while its lock is held."""
    lock = tmp_path / '.mr2-downloads/lock'
    lock.mkdir(parents=True)
    with pytest.raises(FileExistsError, match='already in progress'):
        download_zenodo_files(1, 'data.zip', tmp_path)
    assert lock.is_dir()
    zenodo_mock[0].assert_not_called()


def test_cache_record_provenance(tmp_path, zenodo_mock) -> None:
    """A different record with the same filename does not reuse the old manifest."""
    download_zenodo_files(1, 'data.zip', tmp_path)
    download_zenodo_files(2, 'data.zip', tmp_path)
    assert zenodo_mock[0].call_count == 4


@pytest.mark.parametrize('name', ['../outside', '/absolute', 'folder/file', '..', '', 'folder\\file'])
def test_download_invalid_filename(tmp_path, zenodo_mock, name) -> None:
    """Invalid filenames are rejected before network access."""
    with pytest.raises(ValueError, match='directory components'):
        download_zenodo_files(1, name, tmp_path)
    zenodo_mock[0].assert_not_called()


def test_extract_zip_rejects_symlink(tmp_path) -> None:
    """Archive symlinks cannot redirect extraction outside its staging area."""
    archive = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(archive, 'w') as zip_file:
        info = zipfile.ZipInfo('link')
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        zip_file.writestr(info, '../outside')
    with pytest.raises(ValueError, match='symbolic link'):
        extract_zip(archive, tmp_path / 'output')


@pytest.mark.parametrize('members', [{}, [], {'../outside': {'size': 8}}, {'dataset/scan.h5': None}])
def test_invalid_cache_members(tmp_path, zenodo_mock, members) -> None:
    """Matching archive provenance cannot validate malformed or unsafe members."""
    download_zenodo_files(1, 'data.zip', tmp_path)
    manifest_path = tmp_path / '.mr2-downloads/1-data.zip.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['members'] = members
    manifest_path.write_text(json.dumps(manifest))
    download_zenodo_files(1, 'data.zip', tmp_path)
    assert zenodo_mock[0].call_count == 4


def test_non_archive_download(tmp_path, zenodo_mock) -> None:
    """Documentation files are stored directly and cached without ZIP extraction."""
    get, metadata, content = zenodo_mock
    metadata['files'][0]['key'] = 'documentation.pdf'
    download_zenodo_files(1, 'documentation.pdf', tmp_path)
    download_zenodo_files(1, 'documentation.pdf', tmp_path, verify=True)
    assert (tmp_path / 'documentation.pdf').read_bytes() == content
    assert get.call_count == 3


@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize(
    ('root_name', 'kwargs'),
    [('1', {'subjects': 1}), ('LLR', {'subsets': 'llr'}), ('noise_corr_on', {'noise_correction': True})],
)
def test_osi2_filters_ignore_ancestors(tmp_path, explicit, root_name, kwargs) -> None:
    """Ancestor names cannot select a different subject, subset or correction."""
    root = tmp_path / root_name
    scan = root / 'R1_R2/noise_corr_off/9147/T2w_scan.h5'
    scan.parent.mkdir(parents=True)
    scan.touch()
    assert len(OSI2Dataset([scan] if explicit else root, **kwargs)) == 0


@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('layout', ['LLR/noise_corr_on/9147', 'wrapper/LLR/noise_corr_on/9147/Lab2'])
def test_osi2_nested_layout(tmp_path, explicit, layout) -> None:
    """Filters work below wrappers and with directories below the subject ID."""
    scan = tmp_path / layout / 'T2w_scan.h5'
    scan.parent.mkdir(parents=True)
    scan.touch()
    dataset = OSI2Dataset([scan] if explicit else tmp_path, subjects=9147, subsets='llr', noise_correction=True)
    assert dataset.files == (scan,)


@pytest.mark.parametrize('layout', ['noise_corr_on/9147', '9147'])
def test_osi2_explicit_files_without_subset(tmp_path, layout) -> None:
    """Explicit selections can omit the subset level without using outer IDs."""
    scan = tmp_path / '1' / layout / 'T2w_scan.h5'
    scan.parent.mkdir(parents=True)
    scan.touch()
    assert OSI2Dataset(scan, subjects=9147).files == (scan,)
    assert len(OSI2Dataset([scan], subjects=1)) == 0


def test_ibt_individual_averages_exclude_unsuffixed_mapping(tmp_path) -> None:
    """Individual mode excludes T1 maps but keeps each T2 mapping contrast."""
    for name in ('Brain-T1map_set1.h5', 'Brain-T2map_contrast1_average0.h5', 'Brain-T2map_contrast2_average0.h5'):
        (tmp_path / name).touch()
    assert len(IBTDataset(tmp_path, acquisition='t1map', individual_averages=True)) == 0
    assert len(IBTDataset(tmp_path, acquisition='t1map')) == 1
    assert len(IBTDataset(tmp_path, acquisition='t2map', individual_averages=True)) == 2


def test_ibt_rejects_multiple_in_file_averages(ismrmrd_cart, tmp_path, monkeypatch) -> None:
    """Filename labels cannot hide a file containing multiple acquisition averages."""
    scan = tmp_path / 'Brain-T2_average2.h5'
    scan.touch()
    item = KData.from_file(ismrmrd_cart.filename, KTrajectoryCartesian())
    item.header.acq_info.idx.average = torch.tensor([0, 1])
    dataset = IBTDataset([scan], individual_averages=True)
    monkeypatch.setattr(dataset, 'load', Mock(return_value=item))
    with pytest.raises(ValueError, match='single acquisition average'):
        dataset[0]
