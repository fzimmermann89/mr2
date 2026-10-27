"""Density compensation data (DcfData) class."""

from functools import reduce
from itertools import product
from math import isfinite
from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

from mr2.algorithms.dcf._trajectory import _trajectory_groups
from mr2.algorithms.dcf.dcf_pipe import dcf_pipe
from mr2.algorithms.dcf.dcf_voronoi import dcf_1d, dcf_2d3d_voronoi
from mr2.data.Dataclass import Dataclass
from mr2.data.KTrajectory import KTrajectory
from mr2.data.SpatialDimension import SpatialDimension
from mr2.utils import reduce_repeat, smap

if TYPE_CHECKING:
    from mr2.operators.DensityCompensationOp import DensityCompensationOp


class DcfData(Dataclass):
    """Density compensation data (DcfData) class."""

    data: torch.Tensor
    """Density compensation values. Shape `(... other, coils, k2, k1, k0)`"""

    @classmethod
    def from_traj_voronoi(cls, traj: KTrajectory) -> Self:
        """Calculate dcf using voronoi approach for 2D or 3D trajectories.

        Parameters
        ----------
        traj
            Trajectory to calculate the density compensation for. Can be broadcasted or dense.
        """
        dcfs = []

        ks = [traj.kz, traj.ky, traj.kx]
        spatial_dims = (-3, -2, -1)
        ks_needing_voronoi = set()
        for dim in spatial_dims:
            non_singleton_ks = [ax for ax in ks if ax.shape[dim] != 1]
            if len(non_singleton_ks) == 1:
                # Found a dimension with only one non-singleton axes in ks
                # --> Can handle this as a 1D trajectory
                dcfs.append(smap(dcf_1d, non_singleton_ks.pop(), (dim,)))
            elif len(non_singleton_ks) > 0:
                # More than one of the ks is non-singleton
                # --> A full dimension needing voronoi
                ks_needing_voronoi |= set(non_singleton_ks)
            else:
                # A dimension in which each of ks is singleton
                # --> Don't need to do anything
                pass

        if ks_needing_voronoi:
            # Handle full dimensions needing voronoi
            dcfs.append(smap(dcf_2d3d_voronoi, torch.stack(torch.broadcast_tensors(*ks_needing_voronoi), -4), 4))

        if dcfs:
            # Multiply all dcfs together
            dcf = reduce(torch.mul, dcfs)
        else:
            # Edgecase: Only singleton spatial dimensions
            dcf = traj.kx.new_ones((*traj.shape[:-4], 1, 1, 1, 1))

        return cls(data=dcf)

    @classmethod
    def from_traj_pipe(
        cls,
        traj: KTrajectory,
        encoding_matrix: SpatialDimension[int],
        n_iterations: int = 10,
        oversampling: float = 2.0,
    ) -> Self:
        """Calculate gridding-based Pipe density compensation.

        Parameters
        ----------
        traj
            Trajectory in encoded-grid units. Dense and broadcast layouts are
            supported. Leading dimensions describe independent geometries; the
            coil dimension must be singleton. Weights broadcast over data coils.
        encoding_matrix
            Encoded matrix sizes in z, y, x. Constant coordinate directions are
            ignored. Independent sample dimensions are handled as separable
            factors, including Cartesian readouts in mixed trajectories.
        n_iterations
            Number of Pipe iterations, at least one.
        oversampling
            Explicit gridding oversampling, at least one.

        Notes
        -----
        See `~mr2.algorithms.dcf.dcf_pipe` for kernel, normalization and gradient
        conventions. Each connected group of coordinate directions is normalized
        to its encoded volume. Singleton sample axes contribute a factor of one.
        """
        if traj.shape[-4] != 1:
            raise ValueError('Pipe DCF requires a singleton trajectory coil dimension.')
        if any(size < 1 for size in encoding_matrix.zyx):
            raise ValueError('encoding_matrix sizes must be positive.')
        if n_iterations < 1 or not isfinite(oversampling) or oversampling < 1:
            raise ValueError('n_iterations must be positive and oversampling must be finite and at least one.')
        if any(not k.isfinite().all() for k in (traj.kz, traj.ky, traj.kx)):
            raise ValueError('traj must contain finite coordinates.')

        coordinates = torch.broadcast_tensors(traj.kz.detach(), traj.ky.detach(), traj.kx.detach())
        results = []
        for index in product(*(range(size) for size in traj.shape[:-4])):
            ks = [reduce_repeat(k[index], tol=0, dim=(-3, -2, -1)) for k in coordinates]
            dcf = traj.kx.new_ones(1, 1, 1, 1)
            for axes in _trajectory_groups(ks):
                group = torch.cat(torch.broadcast_tensors(*(ks[i] for i in axes)), dim=-4)
                dcf = dcf * dcf_pipe(group, [encoding_matrix.zyx[i] for i in axes], n_iterations, oversampling)
            results.append(dcf.expand(1, *traj.shape[-3:]))
        return cls(data=torch.stack(results).reshape(traj.shape))

    def as_operator(self) -> 'DensityCompensationOp':
        """Create a density compensation operator using a copy of the DCF."""
        from mr2.operators.DensityCompensationOp import DensityCompensationOp

        return DensityCompensationOp(self.data.clone())
