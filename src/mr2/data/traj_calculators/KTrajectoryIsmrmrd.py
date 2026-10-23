"""Returns the trajectory saved in an ISMRMRD raw data file."""

from collections.abc import Sequence

import ismrmrd
import torch

from mr2.data.KTrajectory import KTrajectory
from mr2.data.SpatialDimension import SpatialDimension
from mr2.utils.reshape import unsqueeze_at


class KTrajectoryIsmrmrd:
    """Get trajectory in ISMRMRD raw data file.

    Use an instance of this class to tell `mr2.data.KData.from_file` to read in the trajectory
    from the ISMRMRD file [TRA]_.

    The trajectory is optionally normalized to fit in the encoding matrix.

    References
    ----------
    .. [TRA] ISMRMRD trajectory https://ismrmrd.readthedocs.io/en/latest/mrd_raw_data.html#k-space-trajectory
    """

    def __init__(self, normalize: bool = False) -> None:
        """Initialize KTrajectoryIsmrmrd.

        The ISMRMRD specification deos not define a normalization for the trajectory. If the trajectory is not
        normalized to the encoding_matrix, please set ``normalize=True`` `to enable normalization. Otherwise,
        the trajectory is read in as-is.

        Parameters
        ----------
        normalize
            normalize the trajectory to the encoding matrix.
        """
        self.normalize = normalize

    def __call__(self, acquisitions: Sequence[ismrmrd.Acquisition], encoding_matrix: SpatialDimension) -> KTrajectory:
        """Read out the trajectory from the ISMRMRD data file.

        Parameters
        ----------
        acquisitions
            List of ISMRMRD acquisitions to read from. Needs at least one acquisition.
            Trajectories must have one, two or three coordinate dimensions in x, y, z order.
            Missing y and z coordinates are set to zero.
        encoding_matrix
            encoding matrix, used to normalize the trajectory.

        Returns
        -------
            trajectory in the shape of the original raw data.

        Raises
        ------
        ValueError
            If trajectory information is missing or has more than three coordinate dimensions.
        """
        trajectories = [torch.as_tensor(acq.traj, dtype=torch.float32) for acq in acquisitions]
        for traj in trajectories:
            if not traj.numel():
                raise ValueError('No trajectory information available in the ISMRMRD file.')
            if traj.shape[-1] > 3:
                raise ValueError('Trajectory must have 1, 2 or 3 coordinate dimensions.')
        traj = torch.stack(trajectories)

        if traj.shape[-1] < 3:  # pad missing y and z coordinates
            zero = torch.zeros_like(traj[..., :1])
            traj = torch.cat([traj, *([zero] * (3 - traj.shape[-1]))], dim=-1)

        traj = unsqueeze_at(traj, dim=-3, n=5 - traj.ndim + 1)  # +1 due to stack dim
        scaling = encoding_matrix if self.normalize else None
        return KTrajectory.from_tensor(traj, stack_dim=-1, axes_order='xyz', scaling_matrix=scaling)
