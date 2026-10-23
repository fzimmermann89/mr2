"""Density compensation data (DcfData) class."""

from functools import reduce
from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

from mr2.algorithms.dcf.dcf_voronoi import _dcf_1d_with_geometry, _dcf_2d3d_with_geometry
from mr2.data.Dataclass import Dataclass
from mr2.data.KTrajectory import KTrajectory
from mr2.utils import smap

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
        return cls._from_traj_voronoi_with_geometry(traj)[0]

    @classmethod
    def _from_traj_voronoi_with_geometry(cls, traj: KTrajectory) -> tuple[Self, torch.Tensor]:
        """Calculate weights and unsplit, uncorrected geometric volumes together."""
        dcfs = []
        geometries = []

        ks = [traj.kz, traj.ky, traj.kx]
        spatial_dims = (-3, -2, -1)
        ks_needing_voronoi = set()
        for dim in spatial_dims:
            non_singleton_ks = [ax for ax in ks if ax.shape[dim] != 1]
            if len(non_singleton_ks) == 1:
                # Found a dimension with only one non-singleton axes in ks
                # --> Can handle this as a 1D trajectory
                paired = smap(lambda x: torch.cat(_dcf_1d_with_geometry(x), dim=-1), non_singleton_ks.pop(), (dim,))
                weights, geometry = paired.chunk(2, dim=dim)
                dcfs.append(weights)
                geometries.append(geometry)
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
            paired = smap(
                lambda x: torch.stack(_dcf_2d3d_with_geometry(x), dim=-4),
                torch.stack(torch.broadcast_tensors(*ks_needing_voronoi), -4),
                4,
            )
            weights, geometry = paired.unbind(dim=-4)
            dcfs.append(weights)
            geometries.append(geometry)

        if dcfs:
            # Multiply all dcfs together
            dcf = reduce(torch.mul, dcfs)
        else:
            # Edgecase: Only singleton spatial dimensions
            dcf = torch.ones_like(traj.kx)

        geometry = reduce(torch.mul, geometries) if geometries else torch.ones_like(dcf)
        return cls(data=dcf), geometry

    def as_operator(self) -> 'DensityCompensationOp':
        """Create a density compensation operator using a copy of the DCF."""
        from mr2.operators.DensityCompensationOp import DensityCompensationOp

        return DensityCompensationOp(self.data.clone())
