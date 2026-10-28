"""MR raw data / k-space data class."""

import copy
import datetime
import warnings
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from math import prod
from types import EllipsisType
from typing import TYPE_CHECKING, Literal, cast
from xml.etree import ElementTree as ET

import h5py
import ismrmrd
import numpy as np
import torch
from einops import rearrange
from typing_extensions import Self, TypeVar, overload

if TYPE_CHECKING:
    from mr2.operators.LinearOperator import LinearOperator

from mr2.data.acq_filters import has_n_coils, is_image_acquisition
from mr2.data.AcqInfo import (
    AcqInfo,
    convert_time_stamp_from_osi2,
    convert_time_stamp_from_siemens,
    convert_time_stamp_to_osi2,
    convert_time_stamp_to_siemens,
    write_acqinfo_to_ismrmrd_acquisition_,
)
from mr2.data.Dataclass import Dataclass
from mr2.data.EncodingLimits import EncodingLimits
from mr2.data.enums import AcqFlags
from mr2.data.KHeader import KHeader
from mr2.data.KNoise import KNoise
from mr2.data.KTrajectory import KTrajectory
from mr2.data.Rotation import Rotation
from mr2.data.SpatialDimension import SpatialDimension
from mr2.data.traj_calculators.KTrajectoryCalculator import KTrajectoryCalculator
from mr2.data.traj_calculators.KTrajectoryIsmrmrd import KTrajectoryIsmrmrd
from mr2.utils.filters import gaussian_window, hann_window, radial_hann_window
from mr2.utils.reshape import normalize_index, normalize_indices
from mr2.utils.summarize import summarize_object
from mr2.utils.typing import FileOrPath

RotationOrTensor = TypeVar('RotationOrTensor', bound=torch.Tensor | Rotation)

KDIM_SORT_LABELS = (
    'k1',
    'k2',
    'average',
    'slice',
    'contrast',
    'phase',
    'repetition',
    'set',
    'user0',
    'user1',
    'user2',
    'user3',
    'user4',
    'user7',
)

OTHER_LABELS = (
    'average',
    'slice',
    'contrast',
    'phase',
    'repetition',
    'set',
    'user0',
    'user1',
    'user2',
    'user3',
    'user4',
    'user7',
)

T = TypeVar('T', Rotation, torch.Tensor, Rotation | torch.Tensor)


class KData(Dataclass):
    """MR raw data / k-space data class."""

    header: KHeader
    """Header information for k-space data"""

    data: torch.Tensor
    """K-space data. Shape `(*other coils k2 k1 k0)`"""

    traj: KTrajectory
    """K-space trajectory along kz, ky and kx. Shape `(*other 1 k2 k1 k0)`"""

    @classmethod
    def from_file(
        cls,
        filename: FileOrPath,
        trajectory: KTrajectoryCalculator | KTrajectory | KTrajectoryIsmrmrd,
        header_overwrites: Mapping[str, object] | None = None,
        dataset_idx: int = -1,
        acquisition_filter_criterion: Callable = is_image_acquisition,
    ) -> Self:
        """Load k-space data from an ISMRMRD file.

        Parameters
        ----------
        filename
            path to the ISMRMRD file or file-like object
        trajectory
            KTrajectoryCalculator to calculate the k-space trajectory or an already calculated KTrajectory
            If a KTrajectory is given, the shape should be `(acquisisions 1 1 k0)` in the same order as the acquisitions
            in the ISMRMRD file.
        header_overwrites
            dictionary of key-value pairs to overwrite the header
        dataset_idx
            index of the ISMRMRD dataset to load (converter creates dataset, dataset_1, ...)
        acquisition_filter_criterion
            function which returns True if an acquisition should be included in KData
        """
        # Can raise FileNotFoundError
        with ismrmrd.File(filename, 'r') as file:
            dataset = file[list(file.keys())[dataset_idx]]
            try:
                ismrmrd_header = dataset.header
            except TypeError:
                root = ET.fromstring(dataset._contents['xml'][0])  # noqa: S314
                prefix = root.tag.removesuffix('ismrmrdHeader')
                measurement = root.find(f'{prefix}measurementInformation')
                if measurement is not None and measurement.find(f'{prefix}patientPosition') is None:
                    ET.SubElement(measurement, f'{prefix}patientPosition').text = 'HFS'
                for encoding in root.findall(f'{prefix}encoding'):
                    if encoding.find(f'{prefix}trajectory') is None:
                        ET.SubElement(encoding, f'{prefix}trajectory').text = 'other'
                ismrmrd_header = ismrmrd.xsd.CreateFromDocument(ET.tostring(root))
            acquisitions = dataset.acquisitions[:]
            try:
                mtime: int = h5py.h5g.get_objinfo(dataset['data']._contents.id).mtime
            except AttributeError:
                mtime = 0
            modification_time = datetime.datetime.fromtimestamp(mtime)

        acquisitions = [acq for acq in acquisitions if acquisition_filter_criterion(acq)]

        # we need the same number of receiver coils for all acquisitions
        n_coils_available = {acq.data.shape[0] for acq in acquisitions}
        if len(n_coils_available) > 1:
            if (
                ismrmrd_header.acquisitionSystemInformation is not None
                and ismrmrd_header.acquisitionSystemInformation.receiverChannels is not None
            ):
                n_coils = int(ismrmrd_header.acquisitionSystemInformation.receiverChannels)
            else:
                # most likely, highest number of elements are the coils used for imaging
                n_coils = int(max(n_coils_available))

            warnings.warn(
                f'Acquisitions with different number {n_coils_available} of receiver coil elements detected. '
                f'Data with {n_coils} receiver coil elements will be used.',
                stacklevel=1,
            )
            acquisitions = [acq for acq in acquisitions if has_n_coils(n_coils, acq)]

        if not acquisitions:
            raise ValueError('No acquisitions meeting the given filter criteria were found.')

        if ismrmrd_header.acquisitionSystemInformation is not None and isinstance(
            ismrmrd_header.acquisitionSystemInformation.systemVendor, str
        ):
            match ismrmrd_header.acquisitionSystemInformation.systemVendor.lower():
                case 'siemens':
                    convert_time_stamp = convert_time_stamp_from_siemens  # 2.5ms time steps
                case 'osi2':
                    convert_time_stamp = convert_time_stamp_from_osi2  # 1ms time steps
                case str(vendor):
                    warnings.warn(
                        f'Unknown vendor {vendor}. '
                        'Assuming Siemens time stamp format. If this is wrong, consider opening an Issue.',
                        stacklevel=1,
                    )
                    convert_time_stamp = convert_time_stamp_from_siemens  # 2.5ms time steps
        else:
            warnings.warn('No vendor information found. Assuming Siemens time stamp format.', stacklevel=1)
            convert_time_stamp = convert_time_stamp_from_siemens

        acq_info, (k0_center, n_k0_tensor, discard_pre, discard_post) = AcqInfo.from_ismrmrd_acquisitions(
            acquisitions,
            additional_fields=('center_sample', 'number_of_samples', 'discard_pre', 'discard_post'),
            convert_time_stamp=convert_time_stamp,
        )
        discard_pre_1d = rearrange(discard_pre, 'n_readouts 1 1 1 1 -> n_readouts')
        discard_post_1d = rearrange(discard_post, 'n_readouts 1 1 1 1 -> n_readouts')

        if len(torch.unique(acq_info.idx.user5)) > 1:
            warnings.warn(
                'The Siemens to ismrmrd converter currently (ab)uses '
                'the user 5 indices for storing the kspace center line number.\n'
                'User 5 indices will be ignored',
                stacklevel=1,
            )

        if len(torch.unique(acq_info.idx.user6)) > 1:
            warnings.warn(
                'The Siemens to ismrmrd converter currently (ab)uses '
                'the user 6 indices for storing the kspace center partition number.\n'
                'User 6 indices will be ignored',
                stacklevel=1,
            )

        shapes = (
            torch.as_tensor([acq.data.shape[-1] for acq in acquisitions]) - discard_pre_1d - discard_post_1d
        ).unique()
        if len(shapes) > 1:
            warnings.warn(
                f'Acquisitions have different shape. Got {list(shapes)}. '
                f'Keeping only acquisistions with {shapes[-1]} data samples. Note: discard_pre and discard_post '
                f'{"have been applied. " if discard_pre_1d.any() or discard_post_1d.any() else "were empty. "}'
                'Please open an issue of you need to handle this kind of data.',
                stacklevel=1,
            )
        data = torch.stack(
            [
                torch.as_tensor(acq.data[..., pre : acq.data.shape[-1] - post], dtype=torch.complex64)
                for acq, pre, post in zip(acquisitions, discard_pre_1d, discard_post_1d, strict=True)
                if acq.data.shape[-1] - pre - post == shapes[-1]
            ]
        )
        n_k0_tensor = n_k0_tensor - discard_pre - discard_post
        data = rearrange(data, 'acquisitions coils k0 -> acquisitions coils 1 1 k0')

        # Raises ValueError if required fields are missing in the header
        header = KHeader.from_ismrmrd(
            ismrmrd_header,
            acq_info,
            defaults={
                'datetime': modification_time,  # use the modification time of the dataset as fallback
                'trajectory': trajectory,
            },
            overwrite=header_overwrites,
        )
        if header.vendor.lower() == 'siemens':
            # Siemens assumes a fft to go from k-space to image space
            data = data.conj_physical()

        # Calculate trajectory and check if it matches the kdata shape
        match trajectory:
            case KTrajectoryIsmrmrd():
                trajectory_ = trajectory(acquisitions, encoding_matrix=header.encoding_matrix)
            case KTrajectoryCalculator():
                reversed_readout_mask = (header.acq_info.flags[..., 0] & AcqFlags.ACQ_IS_REVERSE.value).bool()
                n_k0_unique = torch.unique(n_k0_tensor)
                if len(n_k0_unique) > 1:
                    raise ValueError(
                        'Trajectory can only be calculated for constant number of readout samples.\n'
                        f'Got unique values {list(n_k0_unique)}'
                    )
                encoding_limits = EncodingLimits.from_ismrmrd_header(ismrmrd_header)
                trajectory_ = trajectory(
                    n_k0=int(n_k0_unique[0]),
                    k0_center=k0_center,
                    k1_idx=header.acq_info.idx.k1,
                    k1_center=encoding_limits.k1.center,
                    k2_idx=header.acq_info.idx.k2,
                    k2_center=encoding_limits.k2.center,
                    reversed_readout_mask=reversed_readout_mask,
                    encoding_matrix=header.encoding_matrix,
                )
            case KTrajectory():
                try:
                    torch.broadcast_shapes(trajectory.shape, (data.shape[0], *data.shape[-3:]))
                except RuntimeError:
                    raise ValueError(
                        f'Trajectory shape {trajectory.shape} does not match data shape {data.shape}.'
                    ) from None
                trajectory_ = trajectory
            case _:
                raise TypeError(
                    'ktrajectory must be KTrajectoryIsmrmrd, KTrajectory or KTrajectoryCalculator'
                    f'not {type(trajectory)}',
                )

        kdata = cls(header, data, trajectory_)
        kdata = kdata.reshape_by_idx()
        return kdata

    def reshape_by_idx(self) -> Self:
        """Sort and reshape according to the acquisition indices.

        Reshapes the data to ("all other labels", coils, k2, k1, k0).
        Within "all other labels", the order is determined by `KDIM_SORT_LABELS`.
        """
        shape = self.shape

        def expand(x: T) -> T:
            x = x.expand(*shape[:-4], -1, shape[-3], shape[-2], -1)
            return x

        # First, determine if we can split into k2 and k1 and how large these should be
        acq_indices_other = torch.stack(
            [expand(getattr(self.header.acq_info.idx, label)).ravel() for label in OTHER_LABELS],
            dim=0,
        )
        _, n_acqs_per_other = torch.unique(acq_indices_other, dim=1, return_counts=True)
        # unique counts of acquisitions for each combination of the label values in "other"
        n_acqs_per_other = torch.unique(n_acqs_per_other)

        acq_indices_other_k2 = torch.cat(
            (acq_indices_other, expand(self.header.acq_info.idx.k2).ravel().unsqueeze(0)), dim=0
        )
        _, n_acqs_per_other_and_k2 = torch.unique(acq_indices_other_k2, dim=1, return_counts=True)
        # unique counts of acquisitions for each combination of other **and k2**
        n_acqs_per_other_and_k2 = torch.unique(n_acqs_per_other_and_k2)

        if len(n_acqs_per_other_and_k2) == 1:
            # This is the most common case:
            # All other and k2 combinations have the same number of acquisitions which we can use as k1
            # to reshape to (other, coils, k2, k1, k0)
            n_k1 = n_acqs_per_other_and_k2[0]
            n_k2 = n_acqs_per_other[0] // n_k1
        elif len(n_acqs_per_other) == 1:
            # We cannot split the data along phase encoding steps into k2 and k1, as there are different numbers of k1.
            # But all "other labels" combinations have the same number of acquisitions,
            # so we can reshape to (other, coils, k, 1, k0)
            n_k1 = 1
            n_k2 = n_acqs_per_other[0]  # total number of k encoding steps
        else:
            # For different other combinations we have different numbers of acquisitions,
            # so we can only reshape to (acquisitions, coils, 1, 1, k0)
            # This might be an user error.
            warnings.warn(
                f'There are different numbers of acquisitions for combinations of labels {"/".join(OTHER_LABELS)}.\n'
                f'Found {n_acqs_per_other.tolist()}.\n'
                'The data will be reshaped to (acquisitions, coils, 1, 1, k0).\n'
                'For reconstruction, subset the data or fix the indices so it can be reshaped to '
                '(other, coils, k2, k1, k0).\n'
                'If unintentional, check the labels in the ISMRMRD file and the acquisition flag filter.\n'
                'Then call reshape_by_idx and consider recalculating the trajectory.',
                stacklevel=1,
            )
            n_k1 = 1
            n_k2 = 1

        # Second, determine the sorting order
        acq_indices = np.stack(
            [expand(getattr(self.header.acq_info.idx, label)).ravel() for label in KDIM_SORT_LABELS],
            axis=0,
        )
        sort_idx = torch.as_tensor(np.lexsort(acq_indices))  # torch has no lexsort as of pytorch 2.6 (March 2025)

        # Finally, reshape and sort the tensors in acqinfo and acqinfo.idx, and kdata.
        def sort(x: T) -> T:
            flat = cast(T, rearrange(expand(x), '... coils k2 k1 k0 -> (... k2 k1) coils k0'))
            return cast(
                T, rearrange(flat[sort_idx], '(other k2 k1) coils k0 -> other coils k2 k1 k0', k1=n_k1, k2=n_k2)
            )

        header = self.header.apply(lambda field: sort(field) if isinstance(field, torch.Tensor | Rotation) else field)
        data = sort(self.data)
        kz, ky, kx = (sort(t) for t in (self.traj.kz, self.traj.ky, self.traj.kx))
        traj = KTrajectory(kz, ky, kx, self.traj.grid_detection_tolerance, self.traj.repeat_detection_tolerance)
        return type(self)(header=header._reduce_repeats_(), data=data, traj=traj._reduce_repeats_())

    def squeeze(self, dim: int | Sequence[int] | None = None) -> Self:
        """Squeeze KData along the specified dimensions."""
        shape = self.shape
        n_dim = len(shape)
        if dim is None:
            dims_set = {i for i, size in enumerate(shape) if size == 1}
        else:
            dims_set = set(normalize_indices(n_dim, dim))
        if dims_set and n_dim - len(dims_set) < 5:
            warnings.warn(
                'Squeezing leaves fewer dimensions than the usual (*other, coil, k2, k1, k0).',
                stacklevel=2,
            )

        for d in dims_set:
            if d >= n_dim - 4:
                warnings.warn(
                    f'Squeezing removes the assumed {("coil", "k2", "k1", "k0")[d - (n_dim - 4)]} dimension. '
                    'This can cause misalignment of other dimensions with the assumed '
                    '(*other, coil, k2, k1, k0) alignment.',
                    stacklevel=2,
                )
        return super().squeeze(dim)

    def to_file(self, filename: FileOrPath) -> None:
        """Save KData as ISMRMRD dataset to file.

        Parameters
        ----------
        filename
            path to the ISMRMRD file
        """
        ismrmrd_version = int(version('ismrmrd').split('.')[0])
        header = self.header.to_ismrmrd()
        header.acquisitionSystemInformation.receiverChannels = self.data.shape[-4]

        # For the k-space center of k1 and k2 we can only make an educated guess on where it is:
        # k-space point closest to 0
        encoding_limits = header.encoding[0].encodingLimits
        center_idx = self.traj.as_tensor(-1).abs().sum(dim=-1).argmin()
        encoding_limits.kspace_encoding_step_1.center = int(
            self.header.acq_info.idx.k1.broadcast_to(self.traj.shape).flatten()[center_idx]
        )
        encoding_limits.kspace_encoding_step_2.center = int(
            self.header.acq_info.idx.k2.broadcast_to(self.traj.shape).flatten()[center_idx]
        )

        # Vendors use different units for time stamps
        if self.header.vendor.lower() == 'osi2':
            convert_time_stamp = convert_time_stamp_to_osi2  # 1ms time steps
        else:
            convert_time_stamp = convert_time_stamp_to_siemens  # 2.5ms time steps

        with ismrmrd.Dataset(filename, 'dataset', create_if_needed=True) as dataset:
            dataset.write_xml_header(header.toXML('utf-8'))

            flattened = self.rearrange('... coils k2 k1 k0 -> (... k2 k1) coils 1 1 k0')
            for scan_counter, acq in enumerate(flattened):
                ismrmrd_acq = ismrmrd.Acquisition()
                ismrmrd_acq.resize(
                    number_of_samples=acq.shape[-1], active_channels=acq.shape[-4], trajectory_dimensions=3
                )
                ismrmrd_acq.available_channels = acq.shape[-4]
                ismrmrd_acq.version = ismrmrd_version
                ismrmrd_acq.scan_counter = 0
                write_acqinfo_to_ismrmrd_acquisition_(acq.header.acq_info, ismrmrd_acq, convert_time_stamp)
                ismrmrd_acq.traj[:] = acq.traj.as_tensor(-1).squeeze().cpu().numpy()[:, ::-1]  # zyx -> xyz
                ismrmrd_acq.center_sample = np.argmin(np.abs(ismrmrd_acq.traj[:, 0]))
                acq_data = acq.data.squeeze().cpu().numpy()
                # Siemens assumes a fft to go from k-space to image space
                ismrmrd_acq.data[:] = acq_data.conj() if self.header.vendor.lower() == 'siemens' else acq_data
                ismrmrd_acq.scan_counter = scan_counter
                dataset.append_acquisition(ismrmrd_acq)

    def __repr__(self):
        """Representation method for KData class."""
        traj_info = '\n   '.join(repr(self.traj).splitlines())
        header_info = '\n   '.join(repr(self.header).splitlines())
        representation = '\n'.join(
            [
                super().__repr__().splitlines()[0],
                f'  data: {summarize_object(self.data)}',
                f'  traj: {traj_info}',
                f'  header:  {header_info}',
            ]
        )
        return representation

    @overload
    def compress_coils(
        self,
        n_compressed_coils: int,
        batch_dims: Sequence[int] | None = None,
        joint_dims: Sequence[int] | EllipsisType = ...,
        rotate: bool = False,
        return_compression_op: Literal[False] = False,
    ) -> Self: ...

    @overload
    def compress_coils(
        self,
        n_compressed_coils: int,
        batch_dims: Sequence[int] | None = None,
        joint_dims: Sequence[int] | EllipsisType = ...,
        rotate: bool = False,
        return_compression_op: Literal[True] = ...,
    ) -> tuple[Self, 'LinearOperator']: ...

    def compress_coils(
        self,
        n_compressed_coils: int,
        batch_dims: Sequence[int] | None = None,
        joint_dims: Sequence[int] | EllipsisType = ...,
        rotate: bool = False,
        return_compression_op: bool = False,
    ) -> Self | tuple[Self, 'LinearOperator']:
        """Reduce the number of coils based on a PCA compression.

        A PCA is carried out along the coil dimension and the n_compressed_coils virtual coil elements are selected. For
        more information on coil compression please see [BUE2007]_, [DON2008]_ and [HUA2008]_.

        Returns a copy of the data.

        Parameters
        ----------
        kdata
            K-space data
        n_compressed_coils
            Number of compressed coils
        batch_dims
            Dimensions which are treated as batched, i.e. separate coil compression matrizes (e.g. different slices).
            Default is to do one coil compression matrix for the entire k-space data. Only batch_dim or joint_dim can
            be defined. If batch_dims is not None then joint_dims has to be ...
        joint_dims
            Dimensions which are combined to calculate single coil compression matrix (e.g. k0, k1, contrast). Default
            is that all dimensions (except for the coil dimension) are joint_dims. Only batch_dim or joint_dim can
            be defined. If joint_dims is not ... batch_dims has to be None
        rotate
            Apply a varimax rotation to the compression matrix to distribute the signal more equally across the
            compressed coils.
        return_compression_op
            If True, return the compression operator used for compression along with the compressed data.

        Returns
        -------
            Copy of K-space data with compressed coils.
            (Optional) Compression operator used for compression along with the compressed data.

        Raises
        ------
        `ValueError`
            If both batch_dims and joint_dims are defined.
        `ValuerError`
            If coil dimension is part of joint_dims or batch_dims.

        References
        ----------
        .. [BUE2007] Buehrer M, Pruessmann KP, Boesiger P, Kozerke S (2007) Array compression for MRI with large coil
           arrays. MRM 57. https://doi.org/10.1002/mrm.21237
        .. [DON2008] Doneva M, Boernert P (2008) Automatic coil selection for channel reduction in SENSE-based parallel
           imaging. MAGMA 21. https://doi.org/10.1007/s10334-008-0110-x
        .. [HUA2008] Huang F, Vijayakumar S, Li Y, Hertel S, Duensing GR (2008) A software channel compression
           technique for faster reconstruction with many channels. MRM 26. https://doi.org/10.1016/j.mri.2007.04.010

        """
        from mr2.operators import PCACompressionOp, RearrangeOp

        coil_dim = normalize_index(self.data.ndim, -4)

        if n_compressed_coils > (n_current_coils := self.data.shape[coil_dim]):
            raise ValueError(
                f'Number of compressed coils ({n_compressed_coils}) cannot be greater '
                f'than the number of current coils ({n_current_coils}).'
            )

        if batch_dims is not None and joint_dims is not Ellipsis:
            raise ValueError('Either batch_dims or joint_dims can be defined not both.')

        all_dims = tuple(range(self.data.ndim))
        dim_names = (*[f'other{dim}' for dim in all_dims[:-4]], 'coils', 'kz', 'ky', 'kx')

        if joint_dims is not Ellipsis:
            joint_dims_normalized = normalize_indices(self.data.ndim, joint_dims)
            if coil_dim in joint_dims_normalized:
                raise ValueError('Coil dimension must not be in joint_dims')
            batch_dims_normalized = tuple(
                dim for dim in all_dims if dim not in joint_dims_normalized and dim != coil_dim
            )
        else:
            batch_dims_normalized = normalize_indices(self.data.ndim, batch_dims)
            if coil_dim in batch_dims_normalized:
                raise ValueError('Coil dimension must not be in batch_dims')
            joint_dims_normalized = tuple(
                dim for dim in all_dims if dim not in batch_dims_normalized and dim != coil_dim
            )

        batch_pattern = ' '.join(dim_names[dim] for dim in batch_dims_normalized)
        joint_pattern = ' '.join(dim_names[dim] for dim in joint_dims_normalized)
        input_pattern = ' '.join(dim_names)
        to_pca_layout = RearrangeOp(
            f'{input_pattern} -> {batch_pattern} ({joint_pattern}) coils',
            {dim_names[dim]: self.data.shape[dim] for dim in joint_dims_normalized},
        )

        (kdata_flattened,) = to_pca_layout(self.data)
        pca_compression_op = PCACompressionOp(data=kdata_flattened, n_components=n_compressed_coils, rotate=rotate)
        compression_op = to_pca_layout.H @ pca_compression_op @ to_pca_layout
        (kdata_coil_compressed,) = compression_op(self.data)
        del kdata_flattened

        kdata = type(self)(self.header.clone(), kdata_coil_compressed, self.traj.clone())
        if return_compression_op:
            return kdata, compression_op
        return kdata

    def prewhiten(self, knoise: KNoise | torch.Tensor, scale_factor: float | torch.Tensor = 1.0) -> Self:
        """Calculate noise prewhitening matrix and decorrelate coils.

        This is a convenience wrapper around :func:`mr2.algorithms.prewhiten_kspace`.

        Parameters
        ----------
        knoise
            Noise measurements. Either a KNoise object or a tensor of shape (coils, k0).
        scale_factor
            Square root is applied on the noise covariance matrix. Used to adjust for effective noise bandwidth
            and difference in sampling rate between noise calibration and actual measurement:
            ``scale_factor = (T_acq_dwell/T_noise_dwell)*NoiseReceiverBandwidthRatio``

        Returns
        -------
            Prewhitened copy of k-space data.
        """
        from mr2.algorithms.prewhiten_kspace import prewhiten_kspace

        if not isinstance(knoise, KNoise):
            if knoise.ndim == 2:
                knoise = rearrange(knoise, 'coils k0 -> 1 coils 1 1 k0')
            else:
                knoise = rearrange(knoise, '... coils k0 -> ... coils 1 1 k0')
            knoise = KNoise(knoise)

        whitened = prewhiten_kspace(self, knoise, scale_factor)
        return type(self)(whitened.header, whitened.data, whitened.traj)

    def apodize(
        self,
        width: float | SpatialDimension[float] | None = None,
        *,
        window: Literal['hann', 'gaussian'] = 'hann',
        radial: bool = False,
    ) -> Self:
        """Return a copy with a centered k-space window applied at trajectory coordinates.

        Suppressing high spatial frequencies smooths the reconstructed image and
        reduces ringing at the cost of spatial resolution. Weights are shared
        across coils, with unit gain at DC and no energy/noise normalization.
        Trajectory, header and acquisition metadata are copied unchanged.

        Parameters
        ----------
        width
            Full window width in trajectory units: one unit normally corresponds
            to one Cartesian encoding step, or ``1 / encoding_fov`` in spatial
            frequency. A scalar applies to all spatial directions; a
            `SpatialDimension` specifies z, y and x separately. Defaults to the
            header's encoding matrix sizes, independent of acquired sample count.
            All widths must be finite and positive.
        window
            ``'hann'`` has compact support of the specified width.
            ``'gaussian'`` uses width as its full width at half maximum,
            with no hard cutoff. See :func:`mr2.utils.filters.hann_window` and
            :func:`mr2.utils.filters.gaussian_window` for the formulas.
        radial
            If False, multiply the windows along kz, ky and kx (separable).
            If True, evaluate one window at the radius formed from coordinates
            scaled by their respective widths (ellipsoidal support for Hann).
            Gaussian weights coincide for both choices.

        Returns
        -------
            Apodized copy with the same dtype and device. Repeated data dimensions
            are reduced to singletons following the usual KData convention.
            Density compensation is not applied or modified.
        """
        widths: tuple[float, float, float]
        if width is None:
            widths = self.header.encoding_matrix.zyx
        elif isinstance(width, SpatialDimension):
            widths = width.zyx
        else:
            widths = (width, width, width)
        coordinates = tuple(
            coordinate.to(device=self.data.device, dtype=self.data.real.dtype)
            for coordinate in (self.traj.kz, self.traj.ky, self.traj.kx)
        )
        match window, radial:
            case 'hann', True:
                weight = radial_hann_window(coordinates, widths)
            case 'hann', False:
                weight = prod(
                    (hann_window(k, w) for k, w in zip(coordinates, widths, strict=True)),
                    start=self.data.real.new_ones(()),
                )
            case 'gaussian', _:
                weight = prod(
                    (gaussian_window(k, w) for k, w in zip(coordinates, widths, strict=True)),
                    start=self.data.real.new_ones(()),
                )
            case _:
                raise ValueError("window must be 'hann' or 'gaussian'.")
        return type(self)(self.header.clone(), self.data * weight, self.traj.clone())

    def remove_readout_os(self: Self) -> Self:
        """Remove any oversampling along the readout direction.

        Removes oversampling along the readout direction by cropping the data
        to the size of the reconstruction matrix in image space [GAD]_.

        Returns a copy of the data.

        Parameters
        ----------
        kdata
            K-space data

        Returns
        -------
            Copy of K-space data with oversampling removed.

        Raises
        ------
        `ValueError`
            If the recon matrix along x is larger than the encoding matrix along x.

        References
        ----------
        .. [GAD] Gadgetron https://github.com/gadgetron/gadgetron-python
        """
        from mr2.operators.FastFourierOp import FastFourierOp

        # Ratio of k0/x between encoded and recon space
        x_ratio = self.header.recon_matrix.x / self.header.encoding_matrix.x
        if x_ratio == 1:
            # If the encoded and recon space is the same we don't have to do anything
            return self
        elif x_ratio > 1:
            raise ValueError('Recon matrix along x should be equal or larger than encoding matrix along x.')

        # Starting and end point of image after removing oversampling
        start_cropped_readout = (self.header.encoding_matrix.x - self.header.recon_matrix.x) // 2
        end_cropped_readout = start_cropped_readout + self.header.recon_matrix.x

        def crop_readout(data_to_crop: torch.Tensor) -> torch.Tensor:
            # returns a cropped copy
            return data_to_crop[..., start_cropped_readout:end_cropped_readout].clone()

        # Transform to image space along readout, crop to reconstruction matrix size and transform back
        fourier_k0_op = FastFourierOp(dim=(-1,))
        (cropped_data,) = fourier_k0_op(crop_readout(*fourier_k0_op.H(self.data)))

        # Adapt trajectory
        ks = [self.traj.kz, self.traj.ky, self.traj.kx]
        # only cropped ks that are not broadcasted/singleton along k0
        cropped_ks = [crop_readout(k) if k.shape[-1] > 1 else k.clone() for k in ks]
        cropped_traj = KTrajectory(cropped_ks[0], cropped_ks[1], cropped_ks[2])

        # Adapt header parameters
        header = copy.deepcopy(self.header)
        header.encoding_matrix.x = cropped_data.shape[-1]
        header.encoding_fov.x = self.header.recon_fov.x

        return type(self)(header, cropped_data, cropped_traj)

    def partial_fourier(
        self: Self,
        dim: int = -2,
        method: Literal['pocs', 'homodyne'] = 'pocs',
        *,
        n_iterations: int = 10,
    ) -> Self:
        """Complete one Cartesian partial Fourier direction using trajectory-derived support.

        Requires separable Cartesian axes with shared sampling across batches
        and a contiguous acquired interval touching either edge.

        Parameters
        ----------
        dim
            Partial Fourier axis within ``(k2, k1, k0)``; accepts positive or negative indices.
        method
            ``'pocs'`` preserves acquired values; ``'homodyne'`` uses ramp weighting.
        n_iterations
            POCS iteration count; unused for homodyne.

        Returns
        -------
            KData with completed k-space, a full Cartesian trajectory and updated
            encoding indices. Metadata for synthesized lines inherits the nearest
            acquired line's values. The input is unchanged.
        """
        from mr2.algorithms.partial_fourier import homodyne, pocs
        from mr2.data.enums import TrajectoryType
        from mr2.operators.CartesianSamplingOp import CartesianSamplingOp

        data = self.data.expand(torch.broadcast_shapes(self.data.shape, self.traj.shape))
        axis = normalize_index(data.ndim, dim) - data.ndim
        if axis not in (-3, -2, -1):
            raise ValueError('dim must select one of the last three KData dimensions.')
        if method not in ('pocs', 'homodyne'):
            raise ValueError("method must be 'pocs' or 'homodyne'.")

        # Align metadata with the sorted grid, extending edge metadata to missing lines.
        indices = []
        coordinates = []
        for d, coord, size in zip(
            (-3, -2, -1), (self.traj.kz, self.traj.ky, self.traj.kx), self.header.encoding_matrix.zyx, strict=True
        ):
            if coord.numel() != coord.shape[d] or coord.shape[d] != data.shape[d]:
                raise ValueError('Trajectory must be separable and aligned with the Cartesian data axes.')
            values = coord.reshape(-1)
            if not torch.allclose(values, values.round(), atol=self.traj.grid_detection_tolerance, rtol=0):
                raise ValueError('Trajectory must lie on the integer Cartesian grid.')
            values = values.round()
            grid = torch.arange(size, device=coord.device, dtype=coord.dtype) - size // 2
            if values.min() < grid[0] or values.max() > grid[-1]:
                raise ValueError('Trajectory lies outside the encoding matrix.')
            sorted_values, order = values.sort()
            if (sorted_values.diff() == 0).any():
                raise ValueError('Repeated Cartesian coordinates are unsupported.')
            source = torch.searchsorted(sorted_values, grid).clamp(max=values.numel() - 1)
            indices.append(order[source])
            shape = [1] * data.ndim
            shape[d] = size
            coordinates.append(grid.reshape(shape))

        sampling = CartesianSamplingOp(self.header.encoding_matrix, self.traj)
        (grid_data,) = sampling.H(data)
        (mask,) = sampling.H(torch.ones_like(data.real[..., :1, :, :, :]))
        mask = mask.bool()
        if method == 'pocs':
            completed = pocs(grid_data, mask, (-3, -2, -1), axis, n_iterations=n_iterations)
        else:
            completed = homodyne(grid_data, mask, (-3, -2, -1), axis)
        if completed.shape[-3:] != self.header.encoding_matrix.zyx:
            raise ValueError('Trajectory must cover every non-partial encoded Cartesian axis.')

        header = copy.deepcopy(self.header)

        def extend_metadata(field: object) -> object:
            if isinstance(field, torch.Tensor | Rotation):
                for d, index in zip((-3, -2, -1), indices, strict=True):
                    if field.shape[d] > 1:
                        indexer: list[slice | torch.Tensor] = [slice(None)] * len(field.shape)
                        indexer[d] = index.to(field.device)
                        field = field[tuple(indexer)]
            return field

        header.acq_info.apply_(extend_metadata)
        header.trajectory_type = TrajectoryType.CARTESIAN
        for d, label in ((-3, 'k2'), (-2, 'k1')):
            size = completed.shape[d]
            shape = [1] * data.ndim
            shape[d] = size
            setattr(header.acq_info.idx, label, torch.arange(size, device=data.device).reshape(shape))
        trajectory = KTrajectory(coordinates[0], coordinates[1], coordinates[2])
        return type(self)(header, completed, trajectory)

    def select_other_subset(
        self: Self,
        subset_idx: torch.Tensor,
        subset_label: Literal['average', 'slice', 'contrast', 'phase', 'repetition', 'set'],
    ) -> Self:
        """Select a subset from the other dimension of KData.

        .. warning::
            This function will be deprecated in the future.

        Parameters
        ----------
        kdata
            K-space data `(other coils k2 k1 k0)`
        subset_idx
            Index which elements of the other subset to use, e.g. phase 0,1,2 and 5
        subset_label
            Name of the other label, e.g. phase

        Returns
        -------
            K-space data `(other_subset coils k2 k1 k0)`

        Raises
        ------
        `ValueError`
            If the subset indices are not available in the data
        """
        # Flatten multi-dimensional other
        n_other = self.data.shape[:-4]  # Assume that data is not broadcasted along other
        header = self.header.apply(
            lambda field: (
                rearrange(field.expand(*n_other, *field.shape[-4:]), '... coils k2 k1 k0->(...) coils k2 k1 k0')
                if isinstance(field, torch.Tensor | Rotation)
                else field
            )
        )
        traj = self.traj.as_tensor()
        traj = torch.broadcast_to(traj, (traj.shape[0], *self.data.shape[:-4], *traj.shape[-4:]))  # broadcast "other"
        traj = traj.flatten(start_dim=1, end_dim=-5)  # flatten "other" dimensions
        data = self.data.flatten(end_dim=-5)

        # Find elements in the subset
        label_idx = getattr(header.acq_info.idx, subset_label)
        if not all(el in torch.unique(label_idx) for el in subset_idx):
            raise ValueError('Subset indices are outside of the available index range')
        other_idx = torch.cat([torch.where(idx == label_idx[:, 0, 0, 0, 0])[0] for idx in subset_idx], dim=0)

        # Select subset
        header.acq_info.apply_(lambda field: field[other_idx] if isinstance(field, torch.Tensor | Rotation) else field)
        data = data[other_idx]
        traj = traj[:, other_idx]
        return type(self)(header, data, type(self.traj).from_tensor(traj))
