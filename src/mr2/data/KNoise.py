"""MR noise measurements class."""

from collections.abc import Callable

import h5py
import ismrmrd
import torch
from einops import repeat
from typing_extensions import Self

from mr2.data.acq_filters import is_noise_acquisition
from mr2.data.Dataclass import Dataclass
from mr2.utils.typing import FileOrPath


class KNoise(Dataclass):
    """MR raw data / k-space data class for noise measurements."""

    data: torch.Tensor
    """K-space data of noise measurements. Shape `(*other coils k2 k1 k0)`"""

    @classmethod
    def from_file(
        cls, filename: FileOrPath, dataset_idx: int = -1, acquisition_filter_criterion: Callable = is_noise_acquisition
    ) -> Self:
        """Load noise measurements from an MR2 or ISMRMRD HDF5 file.

        Parameters
        ----------
        filename
            Path or seekable binary file containing MR2 or ISMRMRD data.
        dataset_idx
            Index of the dataset to load (converter creates dataset, dataset_1, ...)
        acquisition_filter_criterion
            function which returns True if an acquisition should be included in KNoise
        """
        if cls._is_mr2_file(filename):
            return super().from_file(filename)
        with h5py.File(filename, 'r') as hdf5_file:
            file = ismrmrd.file.Folder(hdf5_file)
            ds = file[list(file.keys())[dataset_idx]]
            acquisitions = ds.acquisitions[:]

        # Read out noise measurements
        acquisitions = [acq for acq in acquisitions if acquisition_filter_criterion(acq)]
        if len(acquisitions) == 0:
            raise ValueError(f'No noise measurements found in {filename}')
        noise_data = torch.stack([torch.as_tensor(acq.data, dtype=torch.complex64) for acq in acquisitions])

        # Reshape to standard dimensions
        noise_data = repeat(noise_data, '... coils k0->... coils k2 k1 k0', k1=1, k2=1)

        return cls(noise_data)
