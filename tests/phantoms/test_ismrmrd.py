"""Tests for the generic ISMRMRD dataset wrapper."""

import pytest
import torch
from mr2.data import KData
from mr2.data.traj_calculators import KTrajectoryCartesian
from mr2.phantoms.ismrmrd import IsmrmrdDataset
from torch.utils.data import DataLoader


def test_ismrmrd_dataset_natural_sort(tmp_path) -> None:
    """ISMRMRD files are found recursively and naturally sorted."""
    (tmp_path / 'nested').mkdir()
    for name in ('Subject10.mrd', 'Subject3.HDF5', 'Subject2.h5', 'ignore.txt', 'Subject1.h5'):
        (tmp_path / 'nested' / name).touch()
    dataset = IsmrmrdDataset(tmp_path)
    assert [file.name for file in dataset.files] == ['Subject1.h5', 'Subject2.h5', 'Subject3.HDF5', 'Subject10.mrd']


def test_ismrmrd_dataset_getitem(ismrmrd_cart) -> None:
    """The generic wrapper returns KData."""
    dataset = IsmrmrdDataset([ismrmrd_cart.filename])
    assert isinstance(dataset[0], KData)


def test_missing_path(tmp_path) -> None:
    """Missing paths fail during selection, rather than during iteration."""
    with pytest.raises(FileNotFoundError, match='regular file'):
        IsmrmrdDataset(tmp_path / 'missing')
    with pytest.raises(FileNotFoundError):
        IsmrmrdDataset([tmp_path])


def test_empty_and_directory_suffix(tmp_path) -> None:
    """Empty collections are valid and directories with file suffixes are ignored."""
    (tmp_path / 'not_a_file.h5').mkdir()
    assert len(IsmrmrdDataset(tmp_path)) == 0
    assert len(IsmrmrdDataset([])) == 0


def test_loading_options_and_metadata(ismrmrd_cart) -> None:
    """The wrapper preserves KData loading, metadata and negative indexing."""
    trajectory = KTrajectoryCartesian()
    expected = KData.from_file(ismrmrd_cart.filename, trajectory, header_overwrites={'te': [0.012]})
    dataset = IsmrmrdDataset(ismrmrd_cart.filename, trajectory, header_overwrites={'te': [0.012]})
    actual = dataset[-1]
    torch.testing.assert_close(actual.data, expected.data)
    assert actual.header.te == [0.012]
    assert actual.traj == expected.traj
    assert actual.header.recon_fov == expected.header.recon_fov
    with pytest.raises(IndexError):
        dataset[1]


def test_dataloader(ismrmrd_cart) -> None:
    """No handles are retained and unbatched DataLoader iteration returns KData."""
    dataset = IsmrmrdDataset(ismrmrd_cart.filename)
    assert isinstance(next(iter(DataLoader(dataset, batch_size=None, num_workers=1))), KData)
