"""Density compensation data (DcfData) class."""

from collections.abc import Sequence
from functools import reduce
from itertools import product
from math import isfinite, prod
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from einops import rearrange
from typing_extensions import Self

from mr2.algorithms.dcf._trajectory import _trajectory_groups
from mr2.algorithms.dcf.dcf_pipe import dcf_pipe
from mr2.algorithms.dcf.dcf_voronoi import dcf_1d, dcf_2d3d_voronoi
from mr2.data.bart import BART_NONCARTESIAN_DIMS, from_bart_layout, is_bart_file, read_bart, to_bart_layout, write_bart
from mr2.data.Dataclass import Dataclass
from mr2.data.enums import BartDimension
from mr2.data.KTrajectory import KTrajectory
from mr2.data.SpatialDimension import SpatialDimension
from mr2.utils import reduce_repeat, smap
from mr2.utils.typing import FileOrPath

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

        ks = [reduce_repeat(k, tol=0, dim=(-3, -2, -1)) for k in (traj.kz, traj.ky, traj.kx)]
        for axes in _trajectory_groups(ks):
            if len(axes) == 1:
                dcfs.append(smap(dcf_1d, ks[axes[0]], 3))
            else:
                group = torch.stack(torch.broadcast_tensors(*(ks[i] for i in axes)), dim=-4)
                dcfs.append(smap(dcf_2d3d_voronoi, group, 4))

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
            repeats = prod(traj.shape[d] for d in (-3, -2, -1) if all(k.shape[d] == 1 for k in ks))
            results.append((dcf / repeats).expand(1, *traj.shape[-3:]))
        return cls(data=torch.stack(results).reshape(traj.shape))

    @classmethod
    def from_file(cls, filename: FileOrPath, *, other_dims: Sequence[BartDimension] | None = None) -> Self:
        """Load density compensation data from MR2 storage or a BART file.

        BART files are expected in the non-Cartesian sample layout, see `to_bart`.
        All readouts are restored along `k1` with singleton `k2`.

        Parameters
        ----------
        filename
            Path to MR2 file or BART file (``.cfl``, ``.hdr`` or ``.ra``), or seekable binary input with MR2 data.
        other_dims
            BART dimensions that become the `other` dimensions, in this order. Only used for BART files.
            `None` uses the BART batch dimension.
        """
        if isinstance(filename, (str, PathLike)) and is_bart_file(filename):
            data = from_bart_layout(read_bart(filename), BART_NONCARTESIAN_DIMS, other_dims)
            if data.numel() and data.imag.abs().max() > 1e-6 * data.abs().max():
                raise ValueError(f'Density compensation in {filename} has a non-zero imaginary part.')
            return cls(data=data.real.unsqueeze(-3))
        if other_dims is not None:
            raise ValueError('other_dims is only supported for BART files.')
        return super().from_file(filename)

    def to_bart(
        self,
        filename: str | Path,
        *,
        other_dims: Sequence[BartDimension] | None = None,
        shape: Sequence[int] | None = None,
    ) -> None:
        """Save density compensation data as BART array.

        Uses the BART layout of non-Cartesian k-space samples, as written by `KData.to_bart`:
        `k0` is stored in `PHS1`, the readouts `(k2 k1)` in `PHS2` and coils in `COIL`.
        Singleton dimensions are kept and are broadcast by BART.
        The values are saved unchanged, i.e., without any normalization.
        ``bart pics -p`` applies its weights inside the data consistency term, so the DCF-weighted
        least squares problem of MR2 corresponds to the square root of the DCF, e.g. ``bart spow 0.5 dcf weights``.

        Parameters
        ----------
        filename
            A ``.cfl`` or ``.hdr`` suffix writes a ``.hdr``/``.cfl`` pair, a ``.ra`` suffix writes a single file.
        other_dims
            BART dimensions of the `other` dimensions, in tensor order. Should match the k-space data.
            `None` flattens all `other` dimensions into `BartDimension.BATCH`.
        shape
            Shape of the k-space data, e.g. ``kdata.shape``. If given, broadcast dimensions except coils
            are expanded to it, so readouts and `other` dimensions are flattened in the same way as the k-space data.
            Required if the DCF is broadcast along only some of the dimensions that are flattened together.
        """
        data = self.data
        if shape is not None:
            data = data.broadcast_to((*shape[:-4], data.shape[-4], *shape[-3:]))
        data = rearrange(data, '... coils k2 k1 k0 -> ... coils (k2 k1) k0')
        write_bart(filename, to_bart_layout(data, BART_NONCARTESIAN_DIMS, other_dims))

    def as_operator(self) -> 'DensityCompensationOp':
        """Create a density compensation operator using a copy of the DCF."""
        from mr2.operators.DensityCompensationOp import DensityCompensationOp

        return DensityCompensationOp(self.data.clone())
