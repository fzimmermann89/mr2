"""Density compensation data (DcfData) class."""

from functools import partial, reduce
from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

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
            dcf = torch.ones(*traj.shape[-3:], 1, 1, 1, device=traj.kx.device)

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
        ks = [reduce_repeat(k.detach(), tol=0, dim=(-3, -2, -1)) for k in (traj.kz, traj.ky, traj.kx)]
        # As in the Voronoi constructor, group coordinates by the sample axes
        # they share. Merge transitively to keep coupled coordinates together.
        groups: list[set[int]] = []
        for axis, k in enumerate(ks):
            if all(size == 1 for size in k.shape[-3:]):
                continue
            group = {axis}
            for previous in groups.copy():
                if any(ks[i].shape[d] > 1 and k.shape[d] > 1 for i in previous for d in (-3, -2, -1)):
                    group |= previous
                    groups.remove(previous)
            groups.append(group)

        def calculate(k: torch.Tensor, sizes: list[int]) -> torch.Tensor:
            return dcf_pipe(k, sizes, n_iterations, oversampling).unsqueeze(0)

        dcf = torch.ones((*traj.shape[:-4], 1, 1, 1, 1), dtype=traj.kx.dtype, device=traj.kx.device)
        for group in groups:
            axes = sorted(group)
            coordinates = torch.cat(torch.broadcast_tensors(*(ks[i] for i in axes)), dim=-4)
            dimensions = (-4, *(d for d in (-3, -2, -1) if coordinates.shape[d] > 1))
            dcf = dcf * smap(partial(calculate, sizes=[encoding_matrix.zyx[i] for i in axes]), coordinates, dimensions)
        return cls(data=dcf)

    def as_operator(self) -> 'DensityCompensationOp':
        """Create a density compensation operator using a copy of the DCF."""
        from mr2.operators.DensityCompensationOp import DensityCompensationOp

        return DensityCompensationOp(self.data.clone())
