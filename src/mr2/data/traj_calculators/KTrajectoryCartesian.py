"""Cartesian trajectory class."""

from collections.abc import Sequence

import torch

from mr2.data.KTrajectory import KTrajectory
from mr2.data.SpatialDimension import SpatialDimension
from mr2.data.traj_calculators.KTrajectoryCalculator import KTrajectoryCalculator
from mr2.utils import RandomGenerator
from mr2.utils.reshape import unsqueeze_tensors_left


class KTrajectoryCartesian(KTrajectoryCalculator):
    """Cartesian trajectory."""

    def __call__(
        self,
        *,
        n_k0: int,
        k0_center: int | torch.Tensor,
        k1_idx: torch.Tensor,
        k1_center: int | torch.Tensor,
        k2_idx: torch.Tensor,
        k2_center: int | torch.Tensor,
        reversed_readout_mask: torch.Tensor | None = None,
        **_,
    ) -> KTrajectory:
        """Calculate Cartesian trajectory for given KHeader.

        Parameters
        ----------
        n_k0
            number of samples in k0
        k0_center
            position of k-space center in k0
        k1_idx
            indices of k1
        k1_center
            position of k-space center in k1
        k2_idx
            indices of k2
        k2_center
            position of k-space center in k2
        reversed_readout_mask
            boolean tensor indicating reversed readout

        Returns
        -------
            Cartesian trajectory for given KHeader
        """
        # K-space locations along readout lines
        kx = self._readout(n_k0, k0_center, reversed_readout_mask=reversed_readout_mask)

        # Trajectory along phase and slice encoding
        ky = (k1_idx - k1_center).to(torch.float32)
        kz = (k2_idx - k2_center).to(torch.float32)

        kz, ky, kx = unsqueeze_tensors_left(kz, ky, kx, ndim=5)
        return KTrajectory(kz, ky, kx)

    @classmethod
    def fullysampled(cls, encoding_matrix: SpatialDimension[int]) -> KTrajectory:
        """Generate fully sampled Cartesian trajectory.

        Parameters
        ----------
        encoding_matrix
            Encoded K-space size.

        Returns
        -------
            Cartesian trajectory.
        """
        return cls()(
            n_k0=encoding_matrix.x,
            k0_center=encoding_matrix.x // 2,
            k1_idx=torch.arange(encoding_matrix.y)[:, None],
            k1_center=encoding_matrix.y // 2,
            k2_idx=torch.arange(encoding_matrix.z)[:, None, None],
            k2_center=encoding_matrix.z // 2,
        )

    @classmethod
    def gaussian_variable_density(
        cls,
        encoding_matrix: SpatialDimension[int] | int,
        acceleration: float = 2.0,
        n_center: int = 10,
        fwhm_ratio: float = 1.0,
        n_other: Sequence[int] = (1,),
        seed: int | None = None,
    ) -> KTrajectory:
        """
        Generate k-space Gaussian weighted variable density sampling.

        Parameters
        ----------
        encoding_matrix
            Encoded K-space size, must have ``encoding_matrix.z=1``.
            If a single integer, a square k-space is considered.
        acceleration
            Acceleration factor (undersampling rate).
        n_center
            Number of fully-sampled center lines to always include.
        fwhm_ratio
            Full-width at half-maximum of the Gaussian relative to encoding_matrix.y.
            Larger values approach uniform sampling. Set to infinity for uniform sampling.
        n_other
            Batch size(s). The trajectory is different for each batch sample.
        seed
            Random seed for reproducibility.


        Returns
        -------
            Cartesian trajectory.

        Raises
        ------
        ValueError
            If `n_center` exceeds the total number of lines to keep given the acceleration.
        NotImplementedError
            If called with a 3D encoding matrix.
        """
        if isinstance(encoding_matrix, int):
            encoding_matrix = SpatialDimension(1, encoding_matrix, encoding_matrix)
        elif encoding_matrix.z > 1:
            raise NotImplementedError('Only 2D trajectories can be created this way.')

        return cls.gaussian_variable_density_nd(
            encoding_matrix=encoding_matrix,
            acceleration=SpatialDimension(1, acceleration, 1),
            n_center=SpatialDimension(1, n_center, 1),
            fwhm_ratio=SpatialDimension(1.0, fwhm_ratio, 1.0),
            n_other=n_other,
            seed=seed,
        )

    @staticmethod
    def gaussian_variable_density_nd(
        encoding_matrix: SpatialDimension[int],
        acceleration: SpatialDimension[float],
        n_center: SpatialDimension[int] | int = 10,
        fwhm_ratio: SpatialDimension[float] | float = 1.0,
        n_other: Sequence[int] = (1,),
        seed: int | None = None,
    ) -> KTrajectory:
        """Generate Gaussian weighted variable density Cartesian undersampling.

        Undersampling is applied independently in each dimension where acceleration > 1.
        """
        if not isinstance(n_center, SpatialDimension):
            n_center = SpatialDimension(n_center, n_center, n_center)
        if not isinstance(fwhm_ratio, SpatialDimension):
            fwhm_ratio = SpatialDimension(fwhm_ratio, fwhm_ratio, fwhm_ratio)

        for name, n, acc, center in zip(
            ['z', 'y', 'x'], encoding_matrix.zyx, acceleration.zyx, n_center.zyx, strict=True
        ):
            if acc <= 0:
                raise ValueError(f'acceleration.{name} must be > 0, got {acc}.')
            if center < 0 or center > n:
                raise ValueError(f'n_center.{name} must be in [0, {n}], got {center}.')
            if acc > 1:
                n_keep = min(int(n / acc), n)
                if center > n_keep:
                    raise ValueError(
                        f'Number of center lines in {name} ({center}) exceeds number of lines to keep ({n_keep}).'
                    )

        rng = RandomGenerator(seed)

        def sample_axis(n: int, acc: float, center: int, fwhm_rel: float) -> torch.Tensor:
            low = -(n // 2)
            high = low + n
            if acc <= 1:
                return torch.arange(low, high).broadcast_to(*n_other, -1)
            n_keep = min(int(n / acc), n)
            return rng.gaussian_variable_density_samples(
                (*n_other, n_keep),
                low=low,
                high=high,
                fwhm=fwhm_rel * n,
                always_sample=range(-center // 2, center // 2),
            )

        kz_idx = sample_axis(encoding_matrix.z, acceleration.z, n_center.z, fwhm_ratio.z)
        ky_idx = sample_axis(encoding_matrix.y, acceleration.y, n_center.y, fwhm_ratio.y)
        kx_idx = sample_axis(encoding_matrix.x, acceleration.x, n_center.x, fwhm_ratio.x)

        kz = kz_idx[..., None, :, None, None].to(torch.float32)
        ky = ky_idx[..., None, None, :, None].to(torch.float32)
        kx = kx_idx[..., None, None, None, :].to(torch.float32)
        return KTrajectory(kz, ky, kx)

    @classmethod
    def uniform_undersampling(
        cls,
        encoding_matrix: SpatialDimension[int] | int,
        acceleration: float = 2.0,
        n_center: int = 10,
    ) -> KTrajectory:
        """
        Generate deterministic uniformly undersampled Cartesian sampling.

        Every ``acceleration``-th line in k1 is sampled, and center k-space lines
        are always sampled via ``n_center``.

        Parameters
        ----------
        encoding_matrix
            Encoded K-space size, must have ``encoding_matrix.z=1``.
            If a single integer, a square k-space is considered.
        acceleration
            Uniform undersampling factor. Must be an integer value >= 1.
            For example, ``acceleration=4`` samples every 4th line.
        n_center
            Number of fully-sampled center lines to always include.

            Note: if ``n_center > 0``, the effective acceleration is lower than
            ``acceleration`` because ACS lines are added on top of regular samples.

        Returns
        -------
            Cartesian trajectory.

        Raises
        ------
        ValueError
            If acceleration is invalid or ``n_center`` is outside ``[0, n_k1]``.
        NotImplementedError
            If called with a 3D encoding matrix.
        """
        if isinstance(encoding_matrix, int):
            n_k1 = encoding_matrix
            n_k0 = encoding_matrix
        elif encoding_matrix.z > 1:
            raise NotImplementedError('Only 2D trajectories can be created this way.')
        else:
            n_k1, n_k0 = encoding_matrix.y, encoding_matrix.x

        if acceleration < 1 or int(acceleration) != acceleration:
            raise ValueError(f'acceleration must be an integer >= 1, got {acceleration}.')
        acceleration = int(acceleration)

        if not 0 <= n_center <= n_k1:
            raise ValueError(f'n_center must be in [0, {n_k1}], got {n_center}.')

        low = -n_k1 // 2
        high = n_k1 // 2
        uniform_lines = torch.arange(low, high, acceleration)
        center_lines = torch.arange(-n_center // 2, n_center // 2)
        k1_idx = torch.cat([uniform_lines, center_lines]).unique(sorted=True)

        return cls()(
            n_k0=n_k0,
            k0_center=n_k0 // 2,
            k1_idx=k1_idx[..., None, None, :, None],
            k1_center=0,
            k2_idx=torch.tensor(0),
            k2_center=0,
        )
