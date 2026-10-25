"""Prepare centered Cartesian calibration data from acquired k-space samples."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import torch
from scipy.spatial import QhullError

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
        if supported.all():
            return shape[0], shape[1], shape[2]
    raise ValueError('No adequately covered centered ACS region of the requested size is available.')


def _noncartesian_coverage(
    coordinates: torch.Tensor,
    window: tuple[int, int, int],
) -> torch.Tensor:
    """Mark nearest Cartesian bins, ignoring repeated samples and points outside the window."""
    index = torch.zeros_like(coordinates[0], dtype=torch.int64)
    inside = torch.ones_like(index, dtype=torch.bool)
    for coordinate, n in zip(coordinates, window, strict=True):
        offset = coordinate.round().long() + n // 2
        inside &= (offset >= 0) & (offset < n)
        index = index * n + offset
    occupied = torch.zeros(index.shape[0], prod(window), dtype=torch.uint8, device=index.device)
    occupied.scatter_reduce_(-1, index.clamp(0, prod(window) - 1), inside.to(torch.uint8), reduce='amax')
    return occupied.bool().reshape(-1, *window)


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
    Automatic non-Cartesian sizing rounds samples to the nearest encoding-grid
    bin and chooses the largest centered square or cube with no empty bins in
    any trajectory batch. Repeated samples do not improve occupancy. This is an
    alignment-sensitive heuristic and can underestimate useful calibration size;
    it does not guarantee calibration quality. Explicit non-Cartesian sizes skip
    this estimate and are only checked against the encoding matrix. A single
    Voronoi density calculation supplies weights for the full-trajectory adjoint.

    Raises
    ------
    ValueError
        If the size or trajectory is invalid, automatic sizing finds no covered
        center, or an explicit Cartesian block has missing or repeated samples.
    """
    from mr2.data.DcfData import DcfData
    from mr2.operators.FastFourierOp import FastFourierOp
    from mr2.operators.FourierOp import FourierOp

    encoded = kdata.header.encoding_matrix.zyx
    if size is not None:
        _requested_shape(size, encoded)
    # Expand only the logical acquisition dimensions; repeated tensor storage is valid.
    acquisition_shape = kdata.shape
    cartesian = all(t & TrajType.ONGRID for t in kdata.traj.type_along_kzyx)
    # Keep broadcast trajectory batches compact for non-Cartesian coverage checks.
    trajectory_shape = (*acquisition_shape[:-4], 1, *acquisition_shape[-3:]) if cartesian else tuple(kdata.traj.shape)
    coords = torch.stack([k.expand(trajectory_shape) for k in (kdata.traj.kz, kdata.traj.ky, kdata.traj.kx)])
    coords = coords.flatten(start_dim=-3).reshape(3, -1, prod(trajectory_shape[-3:]))
    if not torch.isfinite(coords).all():
        raise ValueError('ACS preparation requires finite trajectory coordinates.')
    batch_shape = acquisition_shape[:-4]
    if cartesian:
        integer = coords.round().long()
        index = torch.zeros_like(integer[0])
        inside = torch.ones_like(index, dtype=torch.bool)
        for coordinate, n in zip(integer, encoded, strict=True):
            offset = coordinate + n // 2
            inside &= (offset >= 0) & (offset < n)
            index = index * n + offset
        counts = torch.zeros(index.shape[0], prod(encoded), dtype=torch.int32, device=index.device)
        counts.scatter_add_(-1, index.clamp(0, prod(encoded) - 1), inside.to(counts.dtype))
        complete = (counts == 1).reshape(-1, *encoded)
        shape = _select_shape(complete, encoded, size)
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

    if size is None:
        max_width = min(n for n in encoded if n > 1)
        window = _requested_shape(max_width, encoded)
        available = _noncartesian_coverage(coords, window)
        shape = _select_shape(available, encoded, None)
    else:
        shape = _requested_shape(size, encoded)
    try:
        dcf = DcfData.from_traj_voronoi(kdata.traj)
    except QhullError as error:
        raise ValueError('Non-Cartesian ACS requires a non-degenerate sampling trajectory.') from error
    # The encoded image grid keeps the frequency spacing and FOV consistent with ACS.
    fourier = FourierOp.from_kdata(kdata, recon_shape=kdata.header.encoding_matrix)
    (coil_images,) = fourier.H(*dcf.as_operator()(kdata.data.expand(acquisition_shape)))
    (grid,) = FastFourierOp(dim=(-3, -2, -1))(coil_images)
    return _center_crop(grid, shape)
