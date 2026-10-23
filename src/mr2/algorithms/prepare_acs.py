"""Prepare centered Cartesian calibration data from acquired k-space samples."""

from __future__ import annotations

from math import prod, sqrt
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.spatial import ConvexHull, QhullError, cKDTree

from mr2.data.enums import TrajType
from mr2.data.SpatialDimension import SpatialDimension

if TYPE_CHECKING:
    from mr2.data.KData import KData


def _center_crop(tensor: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    """Crop while keeping zero frequency at size // 2 for every parity combination."""
    slices = tuple(
        slice(old // 2 - new // 2, old // 2 - new // 2 + new) for old, new in zip(tensor.shape[-3:], shape, strict=True)
    )
    return tensor[..., slices[0], slices[1], slices[2]]


def _requested_shape(size: int | SpatialDimension[int], encoded: tuple[int, int, int]) -> tuple[int, int, int]:
    """Expand scalar widths on active axes and validate explicit extents."""
    if not isinstance(size, int | SpatialDimension):
        raise ValueError('ACS size must be an integer or SpatialDimension of integers.')
    shape = tuple(size if n > 1 else 1 for n in encoded) if isinstance(size, int) else size.zyx
    if any(
        isinstance(n, bool) or not isinstance(n, int) or n < 1 or n > limit
        for n, limit in zip(shape, encoded, strict=True)
    ):
        raise ValueError('ACS size must contain positive integer widths no larger than the encoding matrix.')
    if isinstance(size, int) and (isinstance(size, bool) or size < 1):
        raise ValueError('ACS size must be a positive integer.')
    return shape[0], shape[1], shape[2]


def _select_shape(
    available: torch.Tensor,
    dense: torch.Tensor,
    encoded: tuple[int, int, int],
    size: int | SpatialDimension[int] | None,
) -> tuple[int, int, int]:
    """Choose one centered box supported by every batch."""
    widths = range(min(n for n in encoded if n > 1), 0, -1) if any(n > 1 for n in encoded) else (1,)
    candidates = (
        [_requested_shape(size, encoded)]
        if size is not None
        else (_requested_shape(width, encoded) for width in widths)
    )
    for shape in candidates:
        supported = _center_crop(available, shape)
        density = _center_crop(dense, shape).flatten(start_dim=-3).float().mean(-1)
        if supported.all() and (density >= 0.95).all():
            return shape[0], shape[1], shape[2]
    raise ValueError('No adequately covered centered ACS region of the requested size is available.')


def _noncartesian_coverage(
    coordinates: torch.Tensor,
    geometry: torch.Tensor,
    encoded: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Check geometric density, distance and convex-hull support on the target grid."""
    active = [i for i, n in enumerate(encoded) if n > 1]
    points = coordinates[active].movedim(0, -1).detach().cpu().numpy()
    volumes = geometry.detach().cpu().numpy()
    available, dense = [], []
    for batch_points, batch_volumes in zip(points, volumes, strict=True):
        tree = cKDTree(batch_points)
        try:
            hull = ConvexHull(batch_points) if len(active) > 1 else None
        except QhullError as error:
            raise ValueError('Non-Cartesian ACS requires samples spanning the active spatial dimensions.') from error
        supported_parts, dense_parts = [], []
        for start in range(0, prod(encoded), 65536):
            indices = np.arange(start, min(start + 65536, prod(encoded)))
            query = np.stack(np.unravel_index(indices, encoded), axis=-1)[:, active]
            query = query - np.asarray(encoded)[active] // 2
            distance, nearest = tree.query(query)
            inside = np.ones(len(query), dtype=bool)
            if hull is None:
                inside &= (query[:, 0] >= batch_points[:, 0].min()) & (query[:, 0] <= batch_points[:, 0].max())
            else:
                # Bound intermediate storage for trajectories with many hull facets.
                for equation in hull.equations:
                    inside &= query @ equation[:-1] + equation[-1] <= 1e-6
            supported_parts.append(inside & (distance <= sqrt(len(active)) / 2 + 1e-6))
            local_volume = batch_volumes[nearest]
            dense_parts.append(np.isfinite(local_volume) & (local_volume > 0) & (local_volume <= 1 + 1e-6))
        available.append(np.concatenate(supported_parts).reshape(encoded))
        dense.append(np.concatenate(dense_parts).reshape(encoded))
    return torch.from_numpy(np.stack(available)), torch.from_numpy(np.stack(dense))


def prepare_acs(kdata: KData, size: int | SpatialDimension[int] | None = None) -> torch.Tensor:
    """Prepare centered Cartesian calibration k-space.

    Parameters
    ----------
    kdata
        Acquired data and trajectory, in encoding-grid coordinate units. Samples
        must already use the coil basis intended for reconstruction, including any
        prewhitening. Batches are processed separately and are never averaged.
    size
        Calibration widths in z, y, x. An integer requests a square or cube on
        active encoding axes, with singleton axes unchanged. If omitted, choose
        the largest centered square or cube supported in every batch.

    Returns
    -------
    torch.Tensor
        Centered Cartesian calibration data, shape ``(..., coils, kz, ky, kx)``.
        Frequency spacing corresponds to the encoding FOV, not the reconstruction
        matrix. Cartesian samples are extracted exactly. Non-Cartesian samples
        are approximately gridded using a density-compensated adjoint and FFT.

    Notes
    -----
    Automatic non-Cartesian sizing is a geometric heuristic: all target points
    must lie within the sampled convex hull and within half a Cartesian cell
    diagonal of a sample. At least 95% must have nearest-sample Voronoi volume
    at most one in encoding-grid units. Original volumes are used before boundary
    corrections or splitting repeated samples. The same density calculation
    supplies the weights for reconstruction, which uses the full trajectory.
    This criterion does not guarantee calibration accuracy for every trajectory.

    Raises
    ------
    ValueError
        If the size is invalid or the requested region has insufficient coverage.
    """
    from mr2.data.DcfData import DcfData
    from mr2.operators.FastFourierOp import FastFourierOp
    from mr2.operators.FourierOp import FourierOp

    encoded = kdata.header.encoding_matrix.zyx
    if size is not None:
        _requested_shape(size, encoded)
    # Expand only the logical acquisition dimensions; repeated tensor storage is valid.
    acquisition_shape = kdata.shape
    trajectory_shape = (*acquisition_shape[:-4], 1, *acquisition_shape[-3:])
    coords = torch.stack([k.expand(trajectory_shape) for k in (kdata.traj.kz, kdata.traj.ky, kdata.traj.kx)])
    coords = coords.flatten(start_dim=-3).reshape(3, -1, prod(acquisition_shape[-3:]))
    if not torch.isfinite(coords).all():
        raise ValueError('ACS preparation requires finite trajectory coordinates.')
    batch_shape = acquisition_shape[:-4]
    cartesian = all(t & TrajType.ONGRID for t in kdata.traj.type_along_kzyx)
    if cartesian:
        integer = coords.round().long()
        index = torch.zeros_like(integer[0])
        inside = torch.ones_like(index, dtype=torch.bool)
        for coordinate, n in zip(integer, encoded, strict=True):
            offset = coordinate + n // 2
            inside &= (offset >= 0) & (offset < n)
            index = index * n + offset
        counts = torch.zeros(index.shape[0], prod(encoded), dtype=torch.int64, device=index.device)
        counts.scatter_add_(-1, index.clamp(0, prod(encoded) - 1), inside.long())
        complete = (counts == 1).reshape(-1, *encoded)
        shape = _select_shape(complete, complete, encoded, size)
        from mr2.data.KTrajectory import KTrajectory
        from mr2.operators.CartesianSamplingOp import CartesianSamplingOp

        # Reuse sampling adjoints on the selected block, avoiding a full encoded coil grid.
        selected = inside.clone()
        for coordinate, n in zip(integer, shape, strict=True):
            offset = coordinate + n // 2
            selected &= (offset >= 0) & (offset < n)
        data = kdata.data.expand(acquisition_shape).reshape(index.shape[0], acquisition_shape[-4], -1)
        batches = []
        for values, keep, points in zip(data, selected, integer.movedim(0, 1), strict=True):
            trajectory = KTrajectory(*(point[keep].reshape(1, 1, 1, 1, -1) for point in points))
            sampling = CartesianSamplingOp(SpatialDimension(*shape), trajectory)
            (block,) = sampling.H(values[:, keep].reshape(1, acquisition_shape[-4], 1, 1, -1))
            batches.append(block.reshape(acquisition_shape[-4], *shape))
        return torch.stack(batches).reshape(*batch_shape, acquisition_shape[-4], *shape)

    try:
        dcf, geometry = DcfData._from_traj_voronoi_with_geometry(kdata.traj)
    except QhullError as error:
        raise ValueError('Non-Cartesian ACS requires a non-degenerate sampling trajectory.') from error
    geometry = geometry.expand(trajectory_shape).reshape(coords.shape[1:])
    available, dense = _noncartesian_coverage(coords, geometry, encoded)
    shape = _select_shape(available, dense, encoded, size)
    # The encoded image grid keeps the frequency spacing and FOV consistent with ACS.
    fourier = FourierOp.from_kdata(kdata, recon_shape=kdata.header.encoding_matrix)
    (coil_images,) = fourier.H(*dcf.as_operator()(kdata.data.expand(acquisition_shape)))
    (grid,) = FastFourierOp(dim=(-3, -2, -1))(coil_images)
    return _center_crop(grid, shape)
