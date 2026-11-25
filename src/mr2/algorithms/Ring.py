"""Radial spoke intersections for gradient delay estimation (RING)."""

import warnings
from typing import overload

import torch
from einops import einsum, rearrange

from mr2.data.KData import KData
from mr2.data.KTrajectory import KTrajectory
from mr2.utils.vector_norm import vector_norm


class Ring:
    r"""Gradient delay correction of 2D radial acquisitions with RING.

    Gradient delays shift each radial spoke by :math:`S n`, where :math:`n` is the unit vector along the
    readout direction of the spoke and :math:`S` is a symmetric matrix. In radial spoke intersections for
    gradient delay estimation (RING) [ROS2019]_, :math:`S` is estimated from the intersection points of
    (nearly) orthogonal spokes:

    - Each spoke is low-pass filtered and interpolated to a fine grid around the nominal k-space center.
    - For each spoke and its partner spokes, the pair of samples with the most similar signal in all coils
      is considered as the intersection of both spokes.
    - As both spokes are at the same true k-space position at the intersection, each intersection
      gives two linear equations for :math:`S`, which are solved in the least squares sense.

    All readouts of `kdata` are used to estimate a single :math:`S`. The readouts have to be straight,
    equidistantly sampled lines within the same :math:`k_x`-:math:`k_y` plane passing close to the k-space
    center and should have the same contrast. Few spokes are sufficient [ROS2019]_, but the signal has to
    differ between the coils. The estimate can be inaccurate for single-coil data.

    Call with k-space data or a trajectory to obtain a copy with the corrected trajectory. The estimate can be obtained
    from a subset of the readouts, e.g. ``Ring(kdata[subset])(kdata)``.

    References
    ----------
    .. [ROS2019] Rosenzweig S, Holme HCM, Uecker M (2019) Simple auto-calibrated gradient delay estimation
       from few spokes using Radial Intersections (RING). MRM 81(3) https://doi.org/10.1002/mrm.27506
    """

    def __init__(
        self,
        kdata: KData,
        *,
        n_intersections: int = 1,
        search_radius: float = 1.0,
        resolution: float = 0.01,
        crop_factor: float = 0.6,
    ) -> None:
        """Estimate the gradient delays.

        Parameters
        ----------
        kdata
            K-space data of a 2D radial acquisition with the nominal trajectory.
        n_intersections
            Number of partner spokes used for each spoke. The spokes closest to orthogonal are chosen.
        search_radius
            Maximum distance of an intersection from the nominal k-space center along a spoke,
            in units of the trajectory. Should be larger than the expected shifts.
        resolution
            Target spacing of the fine grid along each spoke, in units of the trajectory.
        crop_factor
            Fraction of the field-of-view along the readout that is kept by the low-pass filter.
        """
        data = rearrange(kdata.data, '... coils k2 k1 k0 -> (... k2 k1) coils k0')
        n_spokes, _, n_k0 = data.shape
        traj_shape = (*kdata.data.shape[:-4], 1, *kdata.data.shape[-3:])
        kz, ky, kx = (k.to(data.real).broadcast_to(traj_shape).reshape(n_spokes, n_k0) for k in kdata.traj.as_tensor())
        if not torch.allclose(kz, kz[0, 0]):
            raise ValueError('RING requires all readouts to be within the same kx-ky plane.')

        # Nominal positions of each spoke are offset + spacing * (k0_idx - center_idx) * direction
        positions = torch.stack((ky, kx), dim=-1)
        direction = positions[:, -1] - positions[:, 0]
        spacing = vector_norm(direction, dim=-1) / (n_k0 - 1)
        direction = torch.nn.functional.normalize(direction, dim=-1)
        center_idx = -einsum(positions[:, 0], direction, 'spokes yx, spokes yx -> spokes') / spacing
        offset = positions[:, 0] + (spacing * center_idx)[:, None] * direction
        k0_idx = torch.arange(n_k0).to(spacing)
        nominal = offset[:, None] + (spacing[:, None] * (k0_idx - center_idx[:, None]))[..., None] * direction[:, None]
        if not torch.allclose(positions, nominal, atol=1e-2 * float(spacing.min())):
            raise ValueError('RING requires readouts that are straight, equidistantly sampled lines.')

        # Fine grid around the sample of each spoke closest to the nominal k-space center, in units of samples
        n_fine = 2 * round(search_radius / resolution) + 1
        fine_idx = torch.linspace(-1, 1, n_fine).to(spacing) * search_radius / spacing.mean()
        if (center_idx + fine_idx[0] < 0).any() or (center_idx + fine_idx[-1] > n_k0 - 1).any():
            raise ValueError('The search region around the k-space center is not covered by all readouts.')

        # Low-pass filter by cropping the 1D projections and interpolate the spokes to the fine grid
        n_crop = round(crop_factor * n_k0)
        x = torch.arange(-(n_crop // 2), n_crop - n_crop // 2, device=data.device)
        projection = torch.fft.ifft(data, dim=-1)[..., x % n_k0]
        phase = 2 * torch.pi / n_k0 * x
        projection = projection * torch.exp(-1j * center_idx[:, None, None] * phase)
        fine = einsum(
            projection, torch.exp(-1j * phase[:, None] * fine_idx), 'spokes coils x, x fine -> spokes coils fine'
        )
        fine = fine.to(torch.complex128)  # differences between neighboring fine samples are small

        # The intersection of each spoke with its most orthogonal partners is at the most similar fine samples
        sine = (direction[:, None, 0] * direction[:, 1] - direction[:, None, 1] * direction[:, 0]).abs()
        spoke_j = sine.topk(min(n_intersections, n_spokes), dim=-1).indices.flatten()
        spoke_i = torch.arange(n_spokes, device=data.device).repeat_interleave(len(spoke_j) // n_spokes)
        energy = einsum(fine.abs().square(), 'spokes coils fine -> spokes fine')
        cross = einsum(
            fine[spoke_i], fine[spoke_j].conj(), 'pairs coils fine_i, pairs coils fine_j -> pairs fine_i fine_j'
        )
        squared_difference = energy[spoke_i, :, None] + energy[spoke_j, None, :] - 2 * cross.real
        fine_i, fine_j = torch.unravel_index(squared_difference.flatten(1).argmin(-1), (n_fine, n_fine))
        if ((fine_i == 0) | (fine_i == n_fine - 1) | (fine_j == 0) | (fine_j == n_fine - 1)).any():
            warnings.warn(
                'Intersections found at the border of the search region. Increase search_radius.', stacklevel=2
            )

        # At the intersection: position_i + S direction_i = position_j + S direction_j. Solve for (Syy, Sxx, Sxy).
        position_i = offset[spoke_i] + (spacing[spoke_i] * fine_idx[fine_i])[:, None] * direction[spoke_i]
        position_j = offset[spoke_j] + (spacing[spoke_j] * fine_idx[fine_j])[:, None] * direction[spoke_j]
        lhs_y, lhs_x = (direction[spoke_i] - direction[spoke_j]).unbind(-1)
        zero = torch.zeros_like(lhs_y)
        matrix = torch.cat((torch.stack((lhs_y, zero, lhs_x), dim=-1), torch.stack((zero, lhs_x, lhs_y), dim=-1)))
        if torch.linalg.matrix_rank(matrix, rtol=1e-3) < 3:
            raise ValueError('RING requires intersecting spokes with at least three different directions.')
        s_yy, s_xx, s_xy = torch.linalg.lstsq(matrix, (position_j - position_i).T.flatten()).solution
        self.delay = torch.stack((s_yy, s_xy, s_xy, s_xx)).reshape(2, 2)
        """Symmetric matrix :math:`S` with elements :math:`((S_{yy}, S_{xy}), (S_{xy}, S_{xx}))`."""

    @overload
    def __call__(self, kdata_or_traj: KData) -> KData: ...
    @overload
    def __call__(self, kdata_or_traj: KTrajectory) -> KTrajectory: ...
    def __call__(self, kdata_or_traj: KData | KTrajectory) -> KData | KTrajectory:
        """Correct a nominal 2D radial trajectory for the gradient delays.

        Parameters
        ----------
        kdata_or_traj
            K-space data of a 2D radial acquisition with the nominal trajectory or the nominal trajectory.

        Returns
        -------
            Corrected trajectory or copy of the k-space data with the corrected trajectory.
        """
        if isinstance(kdata_or_traj, KData):
            return KData(kdata_or_traj.header.clone(), kdata_or_traj.data.clone(), self(kdata_or_traj.traj))
        ky, kx = torch.broadcast_tensors(kdata_or_traj.ky, kdata_or_traj.kx)
        direction = torch.stack((ky[..., -1:] - ky[..., :1], kx[..., -1:] - kx[..., :1]))
        direction = torch.nn.functional.normalize(direction, dim=0)
        shift_y, shift_x = einsum(self.delay.to(direction), direction, 'shift yx, yx ... -> shift ...')
        return KTrajectory(
            kdata_or_traj.kz,
            ky + shift_y,
            kx + shift_x,
            grid_detection_tolerance=kdata_or_traj.grid_detection_tolerance,
            repeat_detection_tolerance=kdata_or_traj.repeat_detection_tolerance,
        )
