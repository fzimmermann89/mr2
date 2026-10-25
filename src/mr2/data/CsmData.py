"""Class for coil sensitivity maps (csm)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

from mr2.data.IData import IData
from mr2.data.IHeader import IHeader
from mr2.data.SpatialDimension import SpatialDimension
from mr2.utils.interpolate import apply_lowres
from mr2.utils.smap import smap

if TYPE_CHECKING:
    from mr2.data.KData import KData
    from mr2.data.KNoise import KNoise
    from mr2.operators.SensitivityOp import SensitivityOp


def get_downsampled_size(
    data_size: torch.Size, downsampled_size: int | SpatialDimension[int] | None
) -> tuple[int, int, int]:
    """Make sure downsampled_size is available for z,y,x and is not larger than data size.

    Parameters
    ----------
    data_size
        Size of data.
    downsampled_size
        Desired size of downsampled data

    Returns
    -------
        Size of downsampled data
    """
    if downsampled_size is None:
        return (data_size[-3], data_size[-2], data_size[-1])  # needed for mypy

    if isinstance(downsampled_size, int):
        downsampled_size = SpatialDimension(z=downsampled_size, y=downsampled_size, x=downsampled_size)

    return (
        min(downsampled_size.z, data_size[-3]),
        min(downsampled_size.y, data_size[-2]),
        min(downsampled_size.x, data_size[-1]),
    )


class CsmData(IData, init=False):
    """Coil sensitivity map class."""

    @classmethod
    def from_kdata_espirit(
        cls,
        acs: KData,
        singular_value_threshold: float = 0.02,
        kernel_width: int = 6,
        crop_threshold: float = 0.3,
        n_iterations: int = 10,
        calibration_size: int | SpatialDimension[int] | None = None,
    ) -> Self:
        """ESPIRiT sensitivity estimation.

        Prepare calibration data with `~mr2.data.KData.acs`, then estimate maps
        on the encoded image grid and crop to the reconstruction grid.

        Parameters
        ----------
        acs
            Acquisition or separate calibration data. Cartesian data may be
            undersampled outside its central calibration region. Non-Cartesian
            calibration is approximately gridded using density compensation.
        singular_value_threshold
            threshold for the singular value decomposition
        kernel_width
            width of the kernel for the espirit algorithm
        crop_threshold
            Eigenvalue threshold for cropping the maps. Defaults to 0.3 here;
            the tensor-level `~mr2.algorithms.csm.espirit` defaults to 0.1.
        n_iterations
            Number of power iterations.
        calibration_size
            Requested ACS width or per-axis widths. If omitted, infer a common
            centered square or cube using `~mr2.data.KData.acs`.

        Raises
        ------
        ValueError
            If the requested calibration region has insufficient coverage or is
            smaller than the kernel on an active axis, or the reconstruction
            geometry is not a crop of the encoded image grid.
        """
        from mr2.algorithms.csm.espirit import espirit
        from mr2.operators.FourierOp import FourierOp

        # FourierOp uses encoding-grid voxel spacing and centered image cropping.
        expected_fov = acs.header.encoding_fov * (acs.header.recon_matrix / acs.header.encoding_matrix)
        if any(
            abs(actual - expected) > 1e-6 * max(abs(actual), abs(expected), 1e-12)
            for actual, expected in zip(acs.header.recon_fov.zyx, expected_fov.zyx, strict=True)
        ) or any(r > e for r, e in zip(acs.header.recon_matrix.zyx, acs.header.encoding_matrix.zyx, strict=True)):
            raise ValueError('ESPIRiT requires reconstruction geometry matching a crop of the encoded image grid.')
        kspace_data = acs.acs(calibration_size)
        if any(
            n < kernel_width and encoded > 1
            for n, encoded in zip(kspace_data.shape[-3:], acs.header.encoding_matrix.zyx, strict=True)
        ):
            raise ValueError('The ACS region must be at least kernel_width along each active encoding axis.')
        csm_data = smap(
            lambda c: espirit(
                c,
                img_shape=acs.header.encoding_matrix,
                singular_value_threshold=singular_value_threshold,
                kernel_width=kernel_width,
                crop_threshold=crop_threshold,
                n_iterations=n_iterations,
            ),
            kspace_data,
            passed_dimensions=(-4, -3, -2, -1),  # coils, z, y, x
        )
        # Match the actual Fourier routing, including mixed trajectories that
        # fall back to NUFFT on all active axes.
        nufft_dims = FourierOp.from_kdata(acs)._nufft_dims
        slices = []
        for dim, (encoded, reconstructed) in enumerate(
            zip(acs.header.encoding_matrix.zyx, acs.header.recon_matrix.zyx, strict=True), start=-3
        ):
            start = encoded // 2 - reconstructed // 2 if dim in nufft_dims else (encoded - reconstructed + 1) // 2
            slices.append(slice(start, start + reconstructed))
        csm_data = csm_data[..., slices[0], slices[1], slices[2]]
        csm = cls(header=IHeader.from_kheader(acs.header), data=csm_data)
        return csm

    @classmethod
    def from_kdata_walsh(
        cls,
        kdata: KData,
        noise: KNoise | None = None,
        smoothing_width: int | SpatialDimension[int] = 5,
        align_phase: bool = True,
        chunk_size_otherdim: int | None = None,
        downsampled_size: int | SpatialDimension[int] | None = None,
    ) -> Self:
        """Create csm object from k-space data using Walsh method.

        See also `~mr2.algorithms.csm.walsh`.

        Parameters
        ----------
        kdata
            k-space data
        noise, optional
            Noise measurement for prewhitening.
        smoothing_width
            Width of smoothing filter.
        align_phase
            If `True`, resolve the global phase ambiguity using the phase alignment
            described in [INA2013]_.
        chunk_size_otherdim
            How many elements of the other dimensions should be processed at once.
            Default is `None`, which means that all elements are processed at once.
        downsampled_size
            IData will be downsampled to this size before calculating the csm to speed up the calculation and
            reduce memory requirements. The final csm will be upsampled to the original size. If set to `None` no
            downsampling will be performed.

        Returns
        -------
            Coil sensitivity maps
        """
        from mr2.algorithms.reconstruction import DirectReconstruction

        return cls.from_idata_walsh(
            DirectReconstruction(kdata, noise=noise, csm=None)(kdata),
            smoothing_width,
            align_phase,
            chunk_size_otherdim,
            downsampled_size,
        )

    @classmethod
    def from_idata_walsh(
        cls,
        idata: IData,
        smoothing_width: int | SpatialDimension[int] = 5,
        align_phase: bool = True,
        chunk_size_otherdim: int | None = None,
        downsampled_size: int | SpatialDimension[int] | None = None,
    ) -> Self:
        """Create csm object from image data using Walsh method.

        See also `~mr2.algorithms.csm.walsh`.

        Parameters
        ----------
        idata
            IData object containing the images for each coil element.
        smoothing_width
            Width of smoothing filter.
        align_phase
            If `True`, resolve the global phase ambiguity using the phase alignment
            described in [INA2013]_.
        chunk_size_otherdim:
            How many elements of the other dimensions should be processed at once.
            Default is `None`, which means that all elements are processed at once.
        downsampled_size
            IData will be downsampled to this size before calculating the csm to speed up the calculation and
            reduce memory requirements. The final csm will be upsampled to the original size. If set to `None` no
            downsampling will be performed.


        Returns
        -------
            Coil sensitivity maps
        """
        from mr2.algorithms.csm.walsh import walsh

        csm_fun = torch.vmap(
            lambda img: apply_lowres(
                lambda x: walsh(x, smoothing_width, align_phase=align_phase),
                size=get_downsampled_size(idata.data.shape, downsampled_size),
                dim=(-3, -2, -1),
            )(img),
            chunk_size=chunk_size_otherdim,
        )
        csm_tensor = csm_fun(idata.data.flatten(end_dim=-5)).reshape(idata.data.shape)
        # upsampled csm requires normalization
        csm_tensor = torch.nn.functional.normalize(csm_tensor, p=2, dim=-4, eps=1e-9)
        csm = cls(header=idata.header, data=csm_tensor)
        return csm

    @classmethod
    def from_kdata_inati(
        cls,
        kdata: KData,
        noise: KNoise | None = None,
        smoothing_width: int | SpatialDimension[int] = 5,
        n_iterations: int = 10,
        chunk_size_otherdim: int | None = None,
        downsampled_size: int | SpatialDimension[int] | None = None,
    ) -> Self:
        """Create csm object from k-space data using the iterative Inati method.

        See also `~mr2.algorithms.csm.inati`.

        Parameters
        ----------
        kdata
            k-space data
        noise, optional
            Noise measurement for prewhitening.
        smoothing_width
            Width of smoothing filter.
        n_iterations
            Number of Inati iterations used to refine the combined image and the
            coil sensitivity maps.
        chunk_size_otherdim
            How many elements of the other dimensions should be processed at once.
            Default is `None`, which means that all elements are processed at once.
        downsampled_size
            IData will be downsampled to this size before calculating the csm to speed up the calculation and
            reduce memory requirements. The final csm will be upsampled to the original size. If set to `None` no
            downsampling will be performed.

        Returns
        -------
            Coil sensitivity maps
        """
        from mr2.algorithms.reconstruction import DirectReconstruction

        return cls.from_idata_inati(
            DirectReconstruction(kdata, noise=noise, csm=None)(kdata),
            smoothing_width,
            n_iterations,
            chunk_size_otherdim,
            downsampled_size,
        )

    @classmethod
    def from_idata_inati(
        cls,
        idata: IData,
        smoothing_width: int | SpatialDimension[int] = 5,
        n_iterations: int = 10,
        chunk_size_otherdim: int | None = None,
        downsampled_size: int | SpatialDimension[int] | None = None,
    ) -> Self:
        """Create csm object from image data using the iterative Inati method.

        See also `~mr2.algorithms.csm.inati`.

        Parameters
        ----------
        idata
            IData object containing the images for each coil element.
        smoothing_width
            Size of the smoothing kernel.
        n_iterations
            Number of Inati iterations used to refine the combined image and the
            coil sensitivity maps.
        chunk_size_otherdim:
            How many elements of the other dimensions should be processed at once.
            Default is `None`, which means that all elements are processed at once.
        downsampled_size
            IData will be downsampled to this size before calculating the csm to speed up the calculation and
            reduce memory requirements. The final csm will be upsampled to the original size. If set to `None` no
            downsampling will be performed.

        Returns
        -------
            Coil sensitivity maps
        """
        from mr2.algorithms.csm.inati import inati

        csm_fun = torch.vmap(
            lambda img: apply_lowres(
                lambda x: inati(x, smoothing_width, n_iterations=n_iterations),
                size=get_downsampled_size(idata.data.shape, downsampled_size),
                dim=(-3, -2, -1),
            )(img),
            chunk_size=chunk_size_otherdim,
        )
        csm_tensor = csm_fun(idata.data.flatten(end_dim=-5)).reshape(idata.data.shape)
        # upsampled csm requires normalization
        csm_tensor = torch.nn.functional.normalize(csm_tensor, p=2, dim=-4, eps=1e-9)
        csm = cls(header=idata.header, data=csm_tensor)
        return csm

    def as_operator(self) -> SensitivityOp:
        """Create SensitivityOp using a copy of the CSMs."""
        from mr2.operators.SensitivityOp import SensitivityOp

        return SensitivityOp(self)
