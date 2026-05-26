"""A pytorch implementation of scipy.spatial.transform.Rotation.

A container for proper and improper Rotations, that can be created from quaternions, euler angles, rotation vectors,
rotation matrices, etc, can be applied to torch.Tensors and SpatialDimensions, multiplied, and can be converted
to quaternions, euler angles, etc.

see also https://github.com/scipy/scipy/blob/main/scipy/spatial/transform/_rotation.pyx
"""

# based on Scipy implementation, which has the following copyright:
# Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Literal, cast

import h5py
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from einops._backends import AbstractBackend
from typing_extensions import Self, Unpack, overload

from mr2.data.SpatialDimension import SpatialDimension
from mr2.utils import RandomGenerator
from mr2.utils.indexing import Indexer
from mr2.utils.reduce_repeat import reduce_repeat
from mr2.utils.reshape import broadcasted_rearrange, normalize_indices
from mr2.utils.typing import NestedSequence, TorchIndexerType
from mr2.utils.vmf import sample_vmf

AXIS_ORDER = 'zyx'  # This can be modified
QUAT_AXIS_ORDER = AXIS_ORDER + 'w'  # Do not modify
assert QUAT_AXIS_ORDER[:3] == AXIS_ORDER, 'Quaternion axis order has to match axis order'


def _compose_quaternions_single(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Calculate p * q."""
    cross = torch.linalg.cross(p[:3], q[:3])
    product = torch.stack(
        (
            p[3] * q[0] + q[3] * p[0] + cross[0],
            p[3] * q[1] + q[3] * p[1] + cross[1],
            p[3] * q[2] + q[3] * p[2] + cross[2],
            p[3] * q[3] - p[0] * q[0] - p[1] * q[1] - p[2] * q[2],
        ),
        0,
    )
    return product


def _compose_quaternions(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Calculate p * q, with p and q batched quaternions."""
    p, q = torch.broadcast_tensors(p, q)
    product = torch.vmap(_compose_quaternions_single)(p.reshape(-1, 4), q.reshape(-1, 4)).reshape(p.shape)
    return product


def _canonical_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert to canonical form, i.e. positive w."""
    x, y, z, w = (quaternion[..., QUAT_AXIS_ORDER.index(axis)] for axis in 'xyzw')
    needs_inversion = (w < 0) | ((w == 0) & ((x < 0) | ((x == 0) & ((y < 0) | ((y == 0) & (z < 0))))))
    canonical_quaternion = torch.where(needs_inversion.unsqueeze(-1), -quaternion, quaternion)
    return canonical_quaternion


def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert matrix to quaternion."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f'Invalid rotation matrix shape {matrix.shape}.')

    batch_shape = matrix.shape[:-2]
    # matrix elements
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.flatten(start_dim=-2), -1)
    # q,r,s are some permutation of x,y,z
    qrsw = torch.nn.functional.relu(
        torch.stack(
            [
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
                1.0 + m00 + m11 + m22,
            ],
            dim=-1,
        )
    )
    q, r, s, w = qrsw.unbind(-1)
    # all these are the same except in edge cases.
    # we will choose the one that is most numerically stable.
    # we calculate all choices as this is faster
    candidates = torch.stack(
        (
            *(q, m10 + m01, m02 + m20, m21 - m12),
            *(m10 + m01, r, m12 + m21, m02 - m20),
            *(m20 + m02, m21 + m12, s, m10 - m01),
            *(m21 - m12, m02 - m20, m10 - m01, w),
        ),
        dim=-1,
    ).reshape(*batch_shape, 4, 4)
    # now we make the choice.
    # the choice will not influence the gradients.
    choice = qrsw.argmax(dim=-1)
    quaternion = candidates.take_along_dim(choice[..., None, None], -2).squeeze(-2) / (
        qrsw.take_along_dim(choice[..., None], -1).sqrt() * 2
    )
    return quaternion


def _make_elementary_quat(axis: str, angle: torch.Tensor):
    """Make a quaternion for the rotation around one of the axes."""
    quat = torch.zeros(*angle.shape, 4, device=angle.device, dtype=angle.dtype)
    axis_index = QUAT_AXIS_ORDER.index(axis)
    w_index = QUAT_AXIS_ORDER.index('w')
    quat[..., w_index] = torch.cos(angle / 2)
    quat[..., axis_index] = torch.sin(angle / 2)
    return quat


def _quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix."""
    # use same order for quaternions as for matrix. this saves two index lookups.
    # we use q, r, s for a permutation of x, y, z
    # as this function will be used for the application of the rotatoin matrix, it should be fast.
    q, r, s, w = quaternion.unbind(-1)
    qq = q.square()
    rr = r.square()
    ss = s.square()
    ww = w.square()
    qr = q * r
    sw = s * w
    qs = q * s
    rw = r * w
    rs = r * s
    qw = q * w

    matrix = torch.stack(
        (
            *(qq - rr - ss + ww, 2 * (qr - sw), 2 * (qs + rw)),
            *(2 * (qr + sw), -qq + rr - ss + ww, 2 * (rs - qw)),
            *(2 * (qs - rw), 2 * (rs + qw), -qq - rr + ss + ww),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)
    return matrix


def _quaternion_to_axis_angle(quaternion: torch.Tensor, degrees: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert quaternion to rotation axis and angle.

    Parameters
    ----------
    quaternion
        The batched quaternions, shape (..., 4)
    degrees
        If True, the angle is returned in degrees, otherwise in radians.

    Returns
    -------
    axis
        The rotation axis, shape (..., 3)
    angle
        The rotation angle, shape (...)
    """
    quaternion = _canonical_quaternion(quaternion)
    angle = 2 * torch.atan2(torch.linalg.vector_norm(quaternion[..., :3], dim=-1), quaternion[..., 3])
    axis = quaternion[..., :3] / torch.linalg.vector_norm(quaternion[..., :3], dim=-1, keepdim=True)
    if degrees:
        angle = torch.rad2deg(angle)
    return axis, angle


def _quaternion_to_euler(quaternion: torch.Tensor, seq: str, extrinsic: bool):
    """Convert quaternion to euler angles.

    Parameters
    ----------
    quaternion
        The batched quaternions
    seq
        The axes sequence, lower case. For example 'xyz'
    extrinsic
        If the rotations are extrinsic (True) or intrinsic (False)
    """
    # The algorithm assumes extrinsic frame transformations. The algorithm
    # in the paper is formulated for rotation quaternions, which are stored
    # directly by Rotation.
    # Adapt the algorithm for our case by reversing both axis sequence and
    # angles for intrinsic rotations when needed

    if not extrinsic:
        seq = seq[::-1]
    q, r, s = (QUAT_AXIS_ORDER.index(axis) for axis in seq)  # one of x,y,z
    w = QUAT_AXIS_ORDER.index('w')

    # proper angles, with first and last axis the same
    if symmetric := q == s:
        s = 3 - q - r  # get third axis

    # Check if permutation is even (+1) or odd (-1)
    sign = (q - r) * (r - s) * (s - q) // 2

    if symmetric:
        a = quaternion[..., w]
        b = quaternion[..., q]
        c = quaternion[..., r]
        d = quaternion[..., s] * sign
    else:
        a = quaternion[..., w] - quaternion[..., r]
        b = quaternion[..., q] + quaternion[..., s] * sign
        c = quaternion[..., r] + quaternion[..., w]
        d = quaternion[..., s] * sign - quaternion[..., q]

    # Compute angles
    angles_1 = 2 * torch.atan2(torch.hypot(c, d), torch.hypot(a, b))
    half_sum = torch.atan2(b, a)
    half_diff = torch.atan2(d, c)

    angles_0 = half_sum - half_diff
    angles_2 = half_sum + half_diff

    if not symmetric:
        angles_2 *= sign
        angles_1 -= torch.pi / 2
    if not extrinsic:
        # flip first and last rotation
        angles_2, angles_0 = angles_0, angles_2

    # Check if angles_1 is equal to is 0 (case=1) or pi (case=2), causing a singularity,
    # i.e. a gimble lock. case=0 is the normal.
    case = 1 * (torch.abs(angles_1) <= 1e-7) + 2 * (torch.abs(angles_1 - torch.pi) <= 1e-7)
    # if Gimbal lock, sett last angle to 0 and use 2 * half_sum / 2 * half_diff for first angle.
    angles_2 = (case == 0) * angles_2
    angles_0 = (
        (case == 0) * angles_0 + (case == 1) * 2 * half_sum + (case == 2) * 2 * half_diff * (-1 if extrinsic else 1)
    )

    angles = torch.stack((angles_0, angles_1, angles_2), -1)
    angles += (angles < -torch.pi) * 2 * torch.pi
    angles -= (angles > torch.pi) * 2 * torch.pi
    return angles


def _align_vectors(
    a: torch.Tensor,
    b: torch.Tensor,
    weights: torch.Tensor,
    return_sensitivity: bool = False,
    allow_improper: bool = False,
):
    """Estimate a rotation to optimally align two sets of vectors."""
    n_vecs = a.shape[0]
    if a.shape != b.shape:
        raise ValueError(f'Expected inputs to have same shapes, got {a.shape} and {b.shape}')
    if a.shape[-1] != 3:
        raise ValueError(f'Expected inputs to have shape (..., 3), got {a.shape} and {b.shape}')
    if weights.shape != (n_vecs,) or (weights < 0).any():
        raise ValueError(f'Invalid weights: expected shape ({n_vecs},) with non-negative values')
    if (a.norm(dim=-1) < 1e-6).any() or (b.norm(dim=-1) < 1e-6).any():
        raise ValueError('Cannot align zero length primary vectors')
    dtype = torch.result_type(a, b)
    # we require double precision for the calculations to match scipy results
    weights = weights.double()
    a = a.double()
    b = b.double()

    inf_mask = torch.isinf(weights)
    if inf_mask.sum() > 1:
        raise ValueError('Only one infinite weight is allowed')

    if inf_mask.any() or n_vecs == 1:
        # special case for one vector pair or one infinite weight

        if return_sensitivity:
            raise ValueError('Cannot return sensitivity matrix with an infinite weight or one vector pair')

        a_primary, b_primary = (a[0], b[0]) if n_vecs == 1 else (a[inf_mask][0], b[inf_mask][0])
        a_primary, b_primary = F.normalize(a_primary, dim=0), F.normalize(b_primary, dim=0)
        cross = torch.linalg.cross(b_primary, a_primary, dim=0)
        angle = torch.atan2(torch.norm(cross), torch.dot(a_primary, b_primary))
        rot_primary = _axisangle_to_matrix(cross, angle)

        if n_vecs == 1:
            return rot_primary.to(dtype), torch.tensor(0.0, device=a.device, dtype=dtype)

        a_secondary, b_secondary = a[~inf_mask], b[~inf_mask]
        sec_w = weights[~inf_mask]
        rot_sec_b = (rot_primary @ b_secondary.T).T
        sin_term = torch.einsum('ij,j->i', torch.linalg.cross(rot_sec_b, a_secondary, dim=1), a_primary)
        cos_term = torch.einsum('ij,ij->i', rot_sec_b, a_secondary) - torch.einsum(
            'ij,j->i', rot_sec_b, a_primary
        ) * torch.einsum('ij,j->i', a_secondary, a_primary)

        phi = torch.atan2((sec_w * sin_term).sum(), (sec_w * cos_term).sum())
        rot_secondary = _axisangle_to_matrix(a_primary, phi)
        rot_optimal = rot_secondary @ rot_primary
        rssd_w = weights.clone()
        rssd_w[inf_mask] = 0
        est_a = (rot_optimal @ b.T).T
        rssd = torch.sqrt(torch.sum(rssd_w * torch.sum((a - est_a) ** 2, dim=1)))
        return rot_optimal.to(dtype), rssd.to(dtype)

    corr_mat = torch.einsum('i j, i k, i -> j k', a, b, weights)
    u, s, vt = cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.linalg.svd(corr_mat))
    if s[1] + s[2] < 1e-16 * s[0]:
        warnings.warn('Optimal rotation is not uniquely or poorly defined for the given sets of vectors.', stacklevel=2)

    if (u @ vt).det() < 0 and not allow_improper:
        u[:, -1] *= -1

    rot_optimal = (u @ vt).to(dtype)
    rssd = ((weights * (b**2 + a**2).sum(dim=1)).sum() - 2 * s.sum()).clamp_min(0.0).sqrt().to(dtype)

    if return_sensitivity:
        zeta = (s[0] + s[1]) * (s[1] + s[2]) * (s[2] + s[0])
        kappa = s[0] * s[1] + s[1] * s[2] + s[2] * s[0]
        sensitivity = (
            weights.mean() / zeta * (kappa * torch.eye(3, device=a.device, dtype=torch.float64) + corr_mat @ corr_mat.T)
        ).to(dtype)
        return rot_optimal, rssd, sensitivity

    return rot_optimal, rssd


def _axisangle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Compute a rotation matrix using Rodrigues' rotation formula."""
    axis = F.normalize(axis, dim=-1, eps=1e-6)
    cos, sin = torch.cos(angle), torch.sin(angle)
    t = 1 - cos
    q, r, s = axis.unbind(-1)
    matrix = rearrange(
        torch.stack(
            [
                t * q * q + cos,
                t * q * r - s * sin,
                t * q * s + r * sin,
                t * q * r + s * sin,
                t * r * r + cos,
                t * r * s - q * sin,
                t * q * s - r * sin,
                t * r * s + q * sin,
                t * s * s + cos,
            ],
            dim=-1,
        ),
        '... (row col) -> ... row col',
        row=3,
    )
    return matrix


class Rotation(torch.nn.Module, Iterable['Rotation']):
    """A container for Rotations.

    A pytorch implementation of scipy.spatial.transform.Rotation.
    For more information see the scipy documentation:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html

    Differences compared to scipy.spatial.transform.Rotation:

    - torch.nn.Module based, the quaternions are a Parameter
    - not all features are implemented. Notably, mrp, davenport, and reduce are missing.
    - arbitrary number of batching dimensions
    - support for improper rotations (rotoinversion), i.e., rotations with an coordinate inversion
      or a reflection about a plane perpendicular to the rotation axis.
    """

    def __init__(
        self,
        quaternions: torch.Tensor | NestedSequence[float],
        normalize: bool = True,
        copy: bool = True,
        inversion: torch.Tensor | NestedSequence[bool] | bool = False,
        reflection: torch.Tensor | NestedSequence[bool] | bool = False,
    ) -> None:
        """Initialize a new Rotation.

        Instead of calling this method, also consider the different `from_*` class methods to construct a Rotation.

        Parameters
        ----------
        quaternions
            Rotatation quaternions. If these requires_grad, the resulting Rotation will require gradients
        normalize
            If the quaternions should be normalized. Only disable if you are sure the quaternions are already
            normalized.
            Will keep a possible negative w to represent improper rotations.
        copy
            Always ensure that a copy of the quaternions is created. If both normalize and copy are False,
            the quaternions Parameter of this instance will be a view if the quaternions passed in.
        inversion
            If the rotation should contain an inversion of the coordinate system, i.e. a reflection of all three axes,
            resulting in a rotoinversion (improper rotation).
            If a boolean tensor is given, it should broadcast with the quaternions.
        reflection
            If the rotation should contain a reflection about a plane perpendicular to the rotation axis.
            This will result in a rotoflexion (improper rotation).
            If a boolean tensor is given, it should broadcast with the quaternions.
        """
        super().__init__()

        quaternions_ = torch.as_tensor(quaternions)
        if torch.is_complex(quaternions_):
            raise ValueError('quaternions should be real numbers')
        if not torch.is_floating_point(quaternions_):
            # integer or boolean dtypes
            quaternions_ = quaternions_.float()
        if quaternions_.shape[-1] != 4:
            raise ValueError(f'Expected `quaternions` to have shape (..., 4), got {quaternions_.shape}.')

        reflection_ = torch.as_tensor(reflection, device=quaternions_.device)
        inversion_ = torch.as_tensor(inversion, device=quaternions_.device)
        if reflection_.any():
            axis, angle = _quaternion_to_axis_angle(quaternions_)
            angle = (angle + torch.pi * reflection_.float()).unsqueeze(-1)
            is_improper = inversion_ ^ reflection_
            quaternions_ = torch.cat((torch.sin(angle / 2) * axis, torch.cos(angle / 2)), -1)
        elif inversion_.any():
            is_improper = inversion_
        else:
            is_improper = torch.zeros_like(quaternions_[..., 0], dtype=torch.bool)

        batchsize = torch.broadcast_shapes(quaternions_.shape[:-1], is_improper.shape)
        is_improper = is_improper.expand(batchsize)

        # If a single quaternion is given, convert it to a 2D 1 x 4 matrix but
        # set self._single to True so that we can return appropriate objects
        # in the `to_...` methods
        if quaternions_.shape == (4,):
            quaternions_ = quaternions_[None, :]
            is_improper = is_improper[None]
            self._single = True
        else:
            self._single = False

        if normalize:
            norms = torch.linalg.vector_norm(quaternions_, dim=-1, keepdim=True)
            if torch.any(torch.isclose(norms.float(), torch.tensor(0.0))):
                raise ValueError('Found zero norm quaternion in `quaternions`.')
            quaternions_ = quaternions_ / norms
        elif copy:
            # no need to clone if we are normalizing
            quaternions_ = quaternions_.clone()
        if copy:
            is_improper = is_improper.clone()

        if is_improper.requires_grad:
            warnings.warn('Rotation is not differentiable in the improper parameter.', stacklevel=2)

        self._quaternions = torch.nn.Parameter(quaternions_, quaternions_.requires_grad)
        self._is_improper = torch.nn.Parameter(is_improper, False)

    @property
    def single(self) -> bool:
        """Returns true if this a single rotation."""
        return self._single

    @property
    def is_improper(self) -> torch.Tensor:
        """Returns a true boolean tensor if the rotation is improper."""
        return self._is_improper

    @is_improper.setter
    def is_improper(self, improper: torch.Tensor | NestedSequence[bool] | bool) -> None:
        """Set the improper parameter."""
        self._is_improper[:] = torch.as_tensor(improper, dtype=torch.bool, device=self._is_improper.device)

    @property
    def det(self) -> torch.Tensor:
        """Returns the determinant of the rotation matrix.

        Will be 1. for proper rotations and -1. for improper rotations.
        """
        return self._is_improper.float() * -2 + 1

    @classmethod
    def from_quat(
        cls,
        quaternions: torch.Tensor | NestedSequence[float],
        inversion: torch.Tensor | NestedSequence[bool] | bool = False,
        reflection: torch.Tensor | NestedSequence[bool] | bool = False,
    ) -> Self:
        """Initialize from quaternions.

        3D rotations can be represented using unit-norm quaternions [QUAa]_.
        As an extension to the standard, this class also supports improper rotations,
        i.e. rotations with reflection with respect to the plane perpendicular to the rotation axis
        or inversion of the coordinate system.

        .. note::
            If ``inversion != reflection``, the rotation will be improper and saved
            as a rotation followed by an inversion inversion of the coordinate system.

        Parameters
        ----------
        quaternions
            shape `(..., 4)`
            Each row is a (possibly non-unit norm) quaternion representing an
            active rotation, in scalar-last `(x, y, z, w)` format. Each
            quaternion will be normalized to unit norm.
        inversion
            if the rotation should contain an inversion of the coordinate system, i.e. a reflection
            of all three axes. If a boolean tensor is given, it should broadcast with the quaternions.
        reflection
            if the rotation should contain a reflection about a plane perpendicular to the rotation axis.


        Returns
        -------
        rotation
            Object containing the rotations represented by input quaternions.

        References
        ----------
        .. [QUAa] Quaternions and spatial rotation https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        """
        return cls(quaternions, normalize=True, copy=True, inversion=inversion, reflection=reflection)

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor | NestedSequence[float], allow_improper: bool = True) -> Self:
        """Initialize from rotation matrix.

        Rotations in 3 dimensions can be represented with 3 x 3 proper
        orthogonal matrices [ROTa]_. If the input is not proper orthogonal,
        an approximation is created using the method described in [MAR2008]_.
        If the input matrix has a negative determinant, the rotation is considered
        as improper, i.e. containing a reflection. The resulting rotation
        will include this reflection [ROTb]_.

        Parameters
        ----------
        matrix
            A single matrix or a stack of matrices, shape `(..., 3, 3)`
        allow_improper
            If true, the rotation is considered as improper if the determinant of the matrix is negative.
            If false, an ValueError is raised if the determinant is negative.

        Returns
        -------
        rotation
            Object containing the rotations represented by the rotation
            matrices.

        References
        ----------
        .. [ROTa] Rotation matrix https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        .. [ROTb] Improper Rotation https://en.wikipedia.org/wiki/Improper_rotation
        .. [MAR2008] Landis Markley F (2008) Unit Quaternion from Rotation Matrix, Journal of guidance, control, and
           dynamics 31(2),440-442.
        """
        matrix_ = torch.as_tensor(matrix)
        if matrix_.shape[-2:] != (3, 3):
            raise ValueError(f'Expected `matrix` to have shape (..., 3, 3), got {matrix_.shape}')
        if torch.is_complex(matrix_):
            raise ValueError('matrix should be real, not complex.')
        if not torch.is_floating_point(matrix_):
            # integer or boolean dtypes
            matrix_ = matrix_.float()

        det = torch.linalg.det(matrix_)
        improper = det < 0
        if improper.any():
            if not allow_improper:
                raise ValueError(
                    'Found negative determinant in `matrix`. '
                    'This would result in an improper rotation, but allow_improper is False.'
                )
            matrix_ = matrix_ * det.unsqueeze(-1).unsqueeze(-1).sign()

        quaternions = _matrix_to_quaternion(matrix_)

        return cls(quaternions, normalize=True, copy=False, inversion=improper, reflection=False)

    @classmethod
    def from_directions(
        cls, *basis: Unpack[tuple[SpatialDimension, SpatialDimension, SpatialDimension]], allow_improper: bool = True
    ):
        """Initialize from basis vectors as SpatialDimensions.

        Parameters
        ----------
        *basis
            3 Basis vectors of the new coordinate system, i.e. the columns of the rotation matrix
        allow_improper
            If true, the rotation is considered as improper if the determinant of the matrix is negative
            and the sign will be preserved. If false, a `ValueError` is raised if the determinant is negative.


        Returns
        -------
        rotation
            Object containing the rotations represented by the basis vectors.
        """
        b1, b2, b3 = (
            torch.stack(torch.broadcast_tensors(*(torch.as_tensor(getattr(v_, ax)) for ax in AXIS_ORDER)), dim=-1)
            for v_ in basis
        )
        matrix = torch.stack(torch.broadcast_tensors(b1, b2, b3), -1)
        det = torch.linalg.det(matrix)
        if not allow_improper and (det < 0).any():
            raise ValueError('The given basis vectors do not form a proper rotation matrix.')
        if ((1 - det.abs()) > 0.1).any():
            raise ValueError('The given basis vectors do not form a rotation matrix.')

        return cls.from_matrix(matrix, allow_improper=allow_improper)

    def as_directions(
        self,
    ) -> tuple[SpatialDimension[torch.Tensor], SpatialDimension[torch.Tensor], SpatialDimension[torch.Tensor]]:
        """Represent as the basis vectors of the new coordinate system as SpatialDimensions.

        Returns the three basis vectors of the new coordinate system after rotation,
        i.e. the columns of the rotation matrix, as `~mr2.data.SpatialDimensions`.

        Returns
        -------
        basis
            The basis vectors of the new coordinate system.
        """
        matrix = self.as_matrix()
        ret = (
            SpatialDimension(**dict(zip(AXIS_ORDER, matrix[..., 0].unbind(-1), strict=True))),
            SpatialDimension(**dict(zip(AXIS_ORDER, matrix[..., 1].unbind(-1), strict=True))),
            SpatialDimension(**dict(zip(AXIS_ORDER, matrix[..., 2].unbind(-1), strict=True))),
        )
        return ret

    @classmethod
    def from_rotvec(
        cls,
        rotvec: torch.Tensor | NestedSequence[float],
        degrees: bool = False,
        reflection: torch.Tensor | NestedSequence[bool] | bool = False,
        inversion: torch.Tensor | NestedSequence[bool] | bool = False,
    ) -> Self:
        """
        Construct a Rotation from rotation vectors (axis-angle representation).
        
        A rotation vector is a 3D vector whose direction is the rotation axis and whose norm is the rotation angle.
        
        Parameters:
            rotvec (torch.Tensor | Sequence[float]):
                Rotation vectors with shape (..., 3).
            degrees (bool):
                If True, interpret rotation magnitudes in degrees; otherwise in radians.
            reflection (torch.Tensor | Sequence[bool] | bool):
                If True for an entry, produce a rotoflection (a rotation combined with a reflection
                about the plane perpendicular to the rotation axis). Can be a scalar or broadcastable mask.
            inversion (torch.Tensor | Sequence[bool] | bool):
                If True for an entry, produce a rotoinversion (a rotation combined with an inversion
                of the coordinate system). Can be a scalar or broadcastable mask.
        
        Returns:
            Rotation:
                Rotation object representing the provided rotation vectors, with improperness
                encoded from `reflection` and `inversion`.
        """
        rotvec_ = torch.as_tensor(rotvec)
        reflection_ = torch.as_tensor(reflection, device=rotvec_.device)
        inversion_ = torch.as_tensor(inversion, device=rotvec_.device)
        if rotvec_.is_complex():
            raise ValueError('rotvec should be real numbers')
        if not rotvec_.is_floating_point():
            # integer or boolean dtypes
            rotvec_ = rotvec_.float()
        if degrees:
            rotvec_ = torch.deg2rad(rotvec_)

        if rotvec_.shape[-1] != 3:
            raise ValueError(f'Expected `rot_vec` to have shape (..., 3), got {rotvec_.shape}')

        angles = torch.linalg.vector_norm(rotvec_, dim=-1, keepdim=True)
        scales = torch.special.sinc(angles / (2 * torch.pi)) / 2
        quaternions = torch.cat((scales * rotvec_, torch.cos(angles / 2)), -1)
        if reflection_.any():
            # we can do it here and avoid the extra of converting to quaternions,
            # back to axis-angle and then to quaternions.
            inversion_ = reflection_ ^ inversion_
            scales = torch.cos(0.5 * angles) / angles
            reflected_quaternions = torch.cat((scales * rotvec_, -torch.sin(angles / 2)), -1)
            quaternions = torch.where(reflection_.unsqueeze(-1), reflected_quaternions, quaternions)

        return cls(quaternions, normalize=False, copy=False, inversion=inversion_, reflection=False)

    @classmethod
    def from_euler(
        cls,
        seq: str,
        angles: torch.Tensor | NestedSequence[float] | float,
        degrees: bool = False,
        inversion: torch.Tensor | NestedSequence[bool] | bool = False,
        reflection: torch.Tensor | NestedSequence[bool] | bool = False,
    ) -> Self:
        """Initialize from Euler angles.

        Rotations in 3-D can be represented by a sequence of 3
        rotations around a sequence of axes. In theory, any three axes spanning
        the 3-D Euclidean space are enough. In practice, the axes of rotation are
        chosen to be the basis vectors.

        The three rotations can either be in a global frame of reference
        (extrinsic) or in a body centered frame of reference (intrinsic), which
        is attached to, and moves with, the object under rotation [EULa]_.

        Parameters
        ----------
        seq
            Specifies sequence of axes for rotations. Up to 3 characters
            belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or
            {'x', 'y', 'z'} for extrinsic rotations. Extrinsic and intrinsic
            rotations cannot be mixed in one function call.
        angles
            (..., [1 or 2 or 3]), matching the number of axes in seq.
            Euler angles specified in radians (`degrees` is False) or degrees
            (`degrees` is True).
        degrees
            If True, then the given angles are assumed to be in degrees.
            Otherwise they are assumed to be in radians
        inversion
            If True, the resulting transformation will contain an inversion of the coordinate system,
            resulting in a rotoinversion (improper rotation).
        reflection
            If True, the resulting transformation will contain a reflection
            about a plane perpendicular to the rotation axis, resulting in an
            improper rotation.

        Returns
        -------
        rotation
            Object containing the rotation represented by the sequence of
            rotations around given axes with given angles.

        References
        ----------
        .. [EULa] Euler angles https://en.wikipedia.org/wiki/Euler_angles#Definition_by_intrinsic_rotations
        """
        n_axes = len(seq)
        if n_axes < 1 or n_axes > 3:
            raise ValueError(f'Expected axis specification to be a non-empty string of upto 3 characters, got {seq}')

        intrinsic = re.match(r'^[XYZ]{1,3}$', seq) is not None
        extrinsic = re.match(r'^[xyz]{1,3}$', seq) is not None
        if not (intrinsic or extrinsic):
            raise ValueError(f"Expected axes from `seq` to be from ['x', 'y', 'z'] or ['X', 'Y', 'Z'], got {seq}")

        if any(seq[i] == seq[i + 1] for i in range(n_axes - 1)):
            raise ValueError(f'Expected consecutive axes to be different, got {seq}')
        seq = seq.lower()

        angles = torch.as_tensor(angles)
        if degrees:
            angles = torch.deg2rad(angles)
        if n_axes == 1 and angles.ndim == 0:
            angles = angles.reshape((1, 1))
            is_single = True
        elif angles.ndim == 1:
            angles = angles[None, :]
            is_single = True
        else:
            is_single = False
        if angles.ndim < 2 or angles.shape[-1] != n_axes:
            raise ValueError(f'Expected angles to have shape (..., n_axes), got {angles.shape}.')

        quaternions = _make_elementary_quat(seq[0], angles[..., 0])
        for axis, angle in zip(seq[1:], angles[..., 1:].unbind(-1), strict=False):
            if intrinsic:
                quaternions = _compose_quaternions(quaternions, _make_elementary_quat(axis, angle))
            else:
                quaternions = _compose_quaternions(_make_elementary_quat(axis, angle), quaternions)

        if is_single:
            return cls(quaternions[0], normalize=False, copy=False, inversion=inversion, reflection=reflection)
        else:
            return cls(quaternions, normalize=False, copy=False, inversion=inversion, reflection=reflection)

    @classmethod
    def from_davenport(cls, axes: torch.Tensor, order: str, angles: torch.Tensor, degrees: bool = False):
        """Not implemented."""
        raise NotImplementedError

    @classmethod
    def from_mrp(cls, mrp: torch.Tensor) -> Self:
        """Not implemented."""
        raise NotImplementedError

    @overload
    def as_quat(
        self, canonical: bool = ..., *, improper: Literal['warn'] | Literal['ignore'] = 'warn'
    ) -> torch.Tensor: ...
    @overload
    def as_quat(
        self, canonical: bool = ..., *, improper: Literal['reflection'] | Literal['inversion']
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def as_quat(
        self,
        canonical: bool = False,
        *,
        improper: Literal['reflection'] | Literal['inversion'] | Literal['ignore'] | Literal['warn'] = 'warn',
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Represent as quaternions.

        Active rotations in 3 dimensions can be represented using unit norm
        quaternions [QUAb]_. The mapping from quaternions to rotations is
        two-to-one, i.e. quaternions `q` and `-q`, where `-q` simply
        reverses the sign of each component, represent the same spatial
        rotation. The returned value is in scalar-last (x, y, z, w) format.

        Parameters
        ----------
        canonical
            Whether to map the redundant double cover of rotation space to a
            unique "canonical" single cover. If True, then the quaternion is
            chosen from {q, -q} such that the w term is positive. If the w term
            is 0, then the quaternion is chosen such that the first nonzero
            term of the x, y, and z terms is positive.
        improper
            How to handle improper rotations. If 'warn', a warning is raised if
            the rotation is improper. If 'ignore', the reflection information is
            discarded. If 'reflection' or 'inversion', additional information is
            returned in the form of a boolean tensor indicating if the rotation
            is improper.
            If 'reflection', the boolean tensor indicates if the rotation contains
            a reflection about a plane perpendicular to the rotation axis.
            Note that this required additional computation.
            If 'inversion', the boolean tensor indicates if the rotation contains
            an inversion of the coordinate system.
            The quaternion is adjusted to represent the rotation to be performed
            before the reflection or inversion.

        Returns
        -------
        quaternions
            shape `(..., 4,)`, depends on shape of inputs used for initialization.
        (optional) reflection (if improper is 'reflection') or inversion (if improper is 'inversion')
            boolean tensor of shape `(...,)`, indicating if the rotation is improper
            and if a reflection or inversion should be performed after the rotation.

        References
        ----------
        .. [QUAb] Quaternions https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        """
        quaternions: torch.Tensor = self._quaternions
        is_improper: torch.Tensor = self._is_improper

        if improper == 'warn':
            if is_improper.any():
                warnings.warn(
                    'Rotation contains improper rotations. Set `improper="reflection"` or `improper="inversion"` '
                    'to get reflection or inversion information.',
                    stacklevel=2,
                )
        elif improper == 'ignore' or improper == 'inversion':
            ...
        elif improper == 'reflection':
            axis, angle = _quaternion_to_axis_angle(quaternions)
            angle = (angle + torch.pi * is_improper.float()).unsqueeze(-1)
            quaternions = torch.cat((torch.sin(angle / 2) * axis, torch.cos(angle / 2)), -1)
        else:
            raise ValueError(f'Invalid improper value: {improper}')

        if self.single:
            quaternions = quaternions[0]
            is_improper = is_improper[0]

        if canonical:
            quaternions = _canonical_quaternion(quaternions)
        else:
            quaternions = quaternions.clone()

        if improper == 'reflection' or improper == 'inversion':
            return quaternions, is_improper
        else:
            return quaternions

    def as_matrix(self) -> torch.Tensor:
        """Represent as rotation matrix.

        3D rotations can be represented using rotation matrices, which
        are 3 x 3 real orthogonal matrices with determinant equal to +1 [ROT]_
        for proper rotations and -1 for improper rotations.

        Returns
        -------
        matrix
            shape `(..., 3, 3)`, depends on shape of inputs used for initialization.

        References
        ----------
        .. [ROT] Rotation matrix https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        """
        quaternions = self._quaternions
        matrix = _quaternion_to_matrix(quaternions)
        if self._is_improper.any():
            matrix = matrix * self.det.unsqueeze(-1).unsqueeze(-1)

        if self._single:
            return matrix[0]
        else:
            return matrix

    @overload
    def as_rotvec(
        self, degrees: bool = ..., *, improper: Literal['ignore'] | Literal['warn'] = 'warn'
    ) -> torch.Tensor: ...
    @overload
    def as_rotvec(
        self, degrees: bool = ..., *, improper: Literal['reflection'] | Literal['inversion']
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def as_rotvec(
        self,
        degrees: bool = False,
        improper: Literal['reflection'] | Literal['inversion'] | Literal['ignore'] | Literal['warn'] = 'warn',
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Represent as rotation vectors.

        A rotation vector is a 3 dimensional vector which is co-directional to
        the axis of rotation and whose norm gives the angle of rotation [ROTc]_.

        Parameters
        ----------
        degrees
            Returned magnitudes are in degrees if this flag is True, else they are in radians
        improper
            How to handle improper rotations. If 'warn', a warning is raised if
            the rotation is improper. If 'ignore', the reflection information is
            discarded. If 'reflection' or 'inversion', additional information is
            returned in the form of a boolean tensor indicating if the rotation
            is improper.
            If 'reflection', the boolean tensor indicates if the rotation contains
            a reflection about a plane perpendicular to the rotation axis.
            If 'inversion', the boolean tensor indicates if the rotation contains
            an inversion of the coordinate system.
            The quaternion is adjusted to represent the rotation to be performed
            before the reflection or inversion.

        Returns
        -------
        rotvec
            Shape `(..., 3)`, depends on shape of inputs used for initialization.
        (optional) reflection (if improper is 'reflection') or inversion (if improper is 'inversion')
            boolean tensor of shape `(...,)`, indicating if the rotation is improper
            and if a reflection or inversion should be performed after the rotation.


        References
        ----------
        .. [ROTc] Rotation vector https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Rotation_vector
        """
        if improper == 'reflection' or improper == 'inversion':
            quaternions, is_improper = self.as_quat(canonical=True, improper=improper)
        else:
            quaternions, is_improper = self.as_quat(canonical=True, improper=improper), None
        angles = 2 * torch.atan2(torch.linalg.vector_norm(quaternions[..., :3], dim=-1), quaternions[..., 3])
        scales = 2 / (torch.special.sinc(angles / (2 * torch.pi)))
        rotvec = scales[..., None] * quaternions[..., :3]
        if degrees:
            rotvec = torch.rad2deg(rotvec)
        if is_improper is not None:
            return rotvec, is_improper
        else:
            return rotvec

    @overload
    def as_euler(
        self,
        seq: str,
        degrees: bool = ...,
        *,
        improper: Literal['ignore'] | Literal['warn'] = 'warn',
    ) -> torch.Tensor: ...
    @overload
    def as_euler(
        self,
        seq: str,
        degrees: bool = ...,
        *,
        improper: Literal['reflection'] | Literal['inversion'],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def as_euler(
        self,
        seq: str,
        degrees: bool = False,
        *,
        improper: Literal['reflection'] | Literal['inversion'] | Literal['ignore'] | Literal['warn'] = 'warn',
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Represent as Euler angles.

        Any orientation can be expressed as a composition of 3 elementary
        rotations. Once the axis sequence has been chosen, Euler angles define
        the angle of rotation around each respective axis [EULb]_.

        The algorithm from [BER2022]_ has been used to calculate Euler angles for the
        rotation about a given sequence of axes.

        Euler angles suffer from the problem of gimbal lock [GIM]_, where the
        representation loses a degree of freedom and it is not possible to
        determine the first and third angles uniquely. In this case,
        a warning is raised, and the third angle is set to zero. Note however
        that the returned angles still represent the correct rotation.

        Parameters
        ----------
        seq
            3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic
            rotations, or {'x', 'y', 'z'} for extrinsic rotations [EULb]_.
            Adjacent axes cannot be the same.
            Extrinsic and intrinsic rotations cannot be mixed in one function
            call.
        degrees
            Returned angles are in degrees if this flag is True, else they are
            in radians
        improper
            How to handle improper rotations. If 'warn', a warning is raised if
            the rotation is improper. If 'ignore', the reflection information is
            discarded. If 'reflection' or 'inversion', additional information is
            returned in the form of a boolean tensor indicating if the rotation
            is improper.
            If 'reflection', the boolean tensor indicates if the rotation contains
            a reflection about a plane perpendicular to the rotation axis.
            If 'inversion', the boolean tensor indicates if the rotation contains
            an inversion of the coordinate system.
            The quaternion is adjusted to represent the rotation to be performed
            before the reflection or inversion.

        Returns
        -------
        angles
            shape `(3,)` or `(..., 3)`, depending on shape of inputs used to initialize object.
            The returned angles are in the range:

            - First angle belongs to ``[-180, 180]`` degrees (both inclusive)
            - Third angle belongs to ``[-180, 180]`` degrees (both inclusive)
            - Second angle belongs to:

             + ``[-90, 90]`` degrees if all axes are different (like xyz)
             + ``[0, 180]`` degrees if first and third axes are the same (like zxz)

        References
        ----------
        .. [EULb] Euler Angles https://en.wikipedia.org/wiki/Euler_angles#Definition_by_intrinsic_rotations
        .. [BER2022] Bernardes E, Viollet S (2022) Quaternion to Euler angles conversion: A direct, general and
           computationally efficient method. PLoS ONE 17(11) https://doi.org/10.1371/journal.pone.0276302
        .. [GIM] Gimbal lock https://en.wikipedia.org/wiki/Gimbal_lock#In_applied_mathematics
        """
        if len(seq) != 3:
            raise ValueError(f'Expected 3 axes, got {seq}.')

        intrinsic = re.match(r'^[XYZ]{1,3}$', seq) is not None
        extrinsic = re.match(r'^[xyz]{1,3}$', seq) is not None
        if not (intrinsic or extrinsic):
            raise ValueError(f"Expected axes from `seq` to be from ['x', 'y', 'z'] or ['X', 'Y', 'Z'], got {seq}")

        if any(seq[i] == seq[i + 1] for i in range(2)):
            raise ValueError(f'Expected consecutive axes to be different, got {seq}')

        seq = seq.lower()
        if improper == 'reflection' or improper == 'inversion':
            quat, is_improper = self.as_quat(canonical=True, improper=improper)
        else:
            quat, is_improper = self.as_quat(improper=improper), None

        if quat.ndim == 1:
            quat = quat[None, :]

        angles = _quaternion_to_euler(quat, seq, extrinsic)
        if degrees:
            angles = torch.rad2deg(angles)

        angles_ = angles[0] if self._single else angles

        if is_improper is not None:
            return angles_, is_improper
        else:
            return angles_

    def as_davenport(self, axes: torch.Tensor, order: str, degrees: bool = False) -> torch.Tensor:
        """Not implemented."""
        raise NotImplementedError

    def as_mrp(self) -> torch.Tensor:
        """Not implemented."""
        raise NotImplementedError

    def concatenate(
        self: Rotation | Sequence[Rotation], *rotations: Rotation | Sequence[Rotation], dim: int = 0
    ) -> Rotation:
        """Concatenate a sequence of `Rotation` objects into a single object.

        Parameters
        ----------
        rotations
            The rotations to concatenate.
        dim
            The dimension to concatenate along.

        Returns
        -------
            The concatenated rotations.
        """
        # In scipy, this is a classmethod. We mimic this behavior, but also support calling it on an instance.
        rotations_ = []
        for el in rotations:
            if isinstance(el, Rotation):
                rotations_.append(el)
            else:
                rotations_.extend(el)
        if isinstance(self, Rotation):
            rotations_ = [self, *rotations_]
            cls = type(self)
        else:
            rotations_ = [*self, *rotations_]
            cls = type(self[0])

        if not all(isinstance(x, Rotation) for x in rotations_):
            raise TypeError('input must contain Rotation objects only')

        quats = torch.cat([torch.atleast_2d(x.as_quat(improper='ignore')) for x in rotations_], dim=dim)
        inversions = torch.cat([torch.atleast_1d(x._is_improper) for x in rotations_], dim=dim)
        return cls(quats, normalize=False, copy=False, inversion=inversions, reflection=False)

    @overload
    def apply(self, fn: NestedSequence[float] | torch.Tensor, inverse: bool) -> torch.Tensor: ...

    @overload
    def apply(
        self, fn: SpatialDimension[torch.Tensor] | SpatialDimension[float], inverse: bool
    ) -> SpatialDimension[torch.Tensor]: ...

    @overload
    def apply(self, fn: Callable[[torch.nn.Module], None]) -> Self: ...

    def apply(
        self,
        fn: NestedSequence[float]
        | torch.Tensor
        | SpatialDimension[torch.Tensor]
        | SpatialDimension[float]
        | Callable[[torch.nn.Module], None],
        inverse: bool = False,
    ) -> torch.Tensor | SpatialDimension[torch.Tensor] | Self:
        """Either apply a function to the Rotation module or apply the rotation to a vector.

        This is a hybrid method that matches the signature of both `torch.nn.Module.apply` and
        `scipy.spatial.transform.Rotation.apply`.
        If a callable is passed, it is assumed to be a function that will be applied to the Rotation module.
        For applying the rotation to a vector, consider using ``rotation(vector)`` instead of
        ``rotation.apply(vector)``.
        """
        if callable(fn):
            # torch.nn.Module.apply
            return super().apply(fn)
        else:
            # scipy.spatial.transform.Rotation.apply
            warnings.warn('Consider using Rotation(vector) instead of Rotation.apply(vector).', stacklevel=2)
            return self(fn, inverse)

    @overload
    def __call__(self, vectors: NestedSequence[float] | torch.Tensor, inverse: bool = False) -> torch.Tensor: ...

    @overload
    def __call__(
        self, vectors: SpatialDimension[torch.Tensor] | SpatialDimension[float], inverse: bool = False
    ) -> SpatialDimension[torch.Tensor]: ...

    def __call__(
        self,
        vectors: NestedSequence[float] | torch.Tensor | SpatialDimension[torch.Tensor] | SpatialDimension[float],
        inverse: bool = False,
    ) -> torch.Tensor | SpatialDimension[torch.Tensor]:
        """Apply this rotation to a set of vectors."""
        # Only for type hinting
        return super().__call__(vectors, inverse)

    def forward(
        self,
        vectors: NestedSequence[float] | torch.Tensor | SpatialDimension[torch.Tensor] | SpatialDimension[float],
        inverse: bool = False,
    ) -> torch.Tensor | SpatialDimension[torch.Tensor]:
        """
        Apply this rotation to 3D vector(s).
        
        Accepts a tensor of shape (..., 3), a single 3-vector (shape (3,) or (1, 3)), or a SpatialDimension holding x/y/z components. If `inverse` is True, the inverse rotation is applied. Broadcasting between the rotation batch and the input vectors follows PyTorch broadcasting rules.
        
        Parameters:
            vectors: Input vectors to rotate; shape (..., 3) or a SpatialDimension with three components.
            inverse: If True, apply the inverse of this rotation.
        
        Returns:
            Rotated vectors. If the input was a SpatialDimension the same type is returned with components in the module's fixed axis order. If this Rotation represents a single rotation and the input was a single vector of shape (3,), a 1-D tensor of shape (3,) is returned; otherwise a tensor of shape (..., 3) is returned, where ... is the broadcast batch shape.
        """
        matrix = self.as_matrix()
        if inverse:
            matrix = matrix.mT
        if self._single:
            matrix = matrix.unsqueeze(0)

        if input_is_spatialdimension := isinstance(vectors, SpatialDimension):
            # sort the axis by AXIS_ORDER
            vectors_tensor = torch.stack(
                [torch.as_tensor(getattr(vectors, axis), device=matrix.device) for axis in AXIS_ORDER], -1
            )
        else:
            vectors_tensor = torch.as_tensor(vectors, device=matrix.device)
        if vectors_tensor.shape[-1] != 3:
            raise ValueError(f'Expected input of shape (..., 3), got {vectors_tensor.shape}.')
        if vectors_tensor.is_complex():
            raise ValueError('Complex vectors are not supported. The coordinates to rotate should be real numbers.')
        if vectors_tensor.dtype != matrix.dtype:
            dtype = torch.promote_types(matrix.dtype, vectors_tensor.dtype)
            matrix = matrix.to(dtype=dtype)
            vectors_tensor = vectors_tensor.to(dtype=dtype)

        try:
            result = (matrix @ vectors_tensor.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            raise ValueError(
                f'The batch-shape of the rotation, {list(matrix.shape[:-2])}, '
                f'is not compatible with the input batch shape {list(vectors_tensor.shape[:-1])}'
            ) from None

        if self._single and vectors_tensor.shape == (3,):
            # a single rotation and a single vector
            result = result[0]

        if input_is_spatialdimension:
            return SpatialDimension(
                x=result[..., AXIS_ORDER.index('x')],
                y=result[..., AXIS_ORDER.index('y')],
                z=result[..., AXIS_ORDER.index('z')],
            )
        else:
            return result

    @classmethod
    def random(
        cls,
        num: int | Sequence[int] | None = None,
        random_state: int | RandomGenerator | None = None,
        improper: bool | Literal['random'] = False,
        *,
        device: torch.device | str | None = None,
    ):
        """
        Generate uniformly distributed rotations.
        
        Parameters:
            num (int | Sequence[int] | None):
                Number of rotations to generate. If `None`, a single rotation is returned; if an int, returns that many rotations; if a sequence, returns an array shaped `(*num, 4)`.
            random_state (int | RandomGenerator | None):
                Seed or RNG to use. If `None`, a fresh RandomGenerator is created; if an int, a new RandomGenerator is created with that seed; if a RandomGenerator instance is provided, it is used and its state is advanced.
            improper (bool | Literal['random']):
                Controls inversion/reflection flags. `False` produces only proper rotations, `True` produces only improper rotations, and `"random"` samples proper/improper per rotation uniformly at random.
            device (torch.device | str | None):
                Device where the generated tensors are allocated. If `None`, the default device is used.
        
        Returns:
            Rotation or Rotation batch:
                A Rotation containing the generated quaternion(s). Returns a single-rotation instance when `num` is `None`, otherwise a batched Rotation with shape corresponding to `num`.
        """
        if random_state is None:
            rng = RandomGenerator(device=device)
        elif isinstance(random_state, RandomGenerator):
            rng = random_state
        else:
            rng = RandomGenerator(seed=random_state, device=device)
        if num is None:
            random_sample = rng.randn_tensor((4,), torch.float32, device=device)
        elif isinstance(num, int):
            random_sample = rng.randn_tensor((num, 4), torch.float32, device=device)
        else:
            random_sample = rng.randn_tensor((*num, 4), torch.float32, device=device)
        if improper == 'random':
            inversion: torch.Tensor | bool = rng.bool_tensor(random_sample.shape[:-1], device=device)
        elif isinstance(improper, bool):
            inversion = improper
        else:
            raise ValueError('improper should be a boolean or "random"')
        return cls(random_sample, inversion=inversion, reflection=False, normalize=True, copy=False)

    @classmethod
    def random_vmf(
        cls,
        num: int | None = None,
        mean_axis: torch.Tensor | None = None,
        kappa: float = 0.0,
        sigma: float = math.inf,
        random_state: int | RandomGenerator | None = None,
        *,
        device: torch.device | str | None = None,
    ):
        """
        Sample rotations whose axes follow a von Mises–Fisher distribution and whose rotation angles follow a 2π-wrapped Gaussian.
        
        Parameters:
            num (int | None): Number of samples to generate. If `None`, a single rotation is returned.
            mean_axis (torch.Tensor | None): Tensor of shape `(..., 3)` giving the mean direction for the von Mises–Fisher distribution; defaults to (1,0,0) when `None`.
            kappa (float): Concentration parameter for the von Mises–Fisher distribution (small → near-uniform, large → concentrated about `mean_axis`).
            sigma (float): Standard deviation (radians) of the wrapped Gaussian used to sample rotation angles; use `math.inf` to draw angles uniformly in [0, 2π).
            random_state (int | RandomGenerator | None): Seed or `RandomGenerator` to control randomness. If `None`, a fresh generator is created.
            device (torch.device | str | None): Device for the generated tensors; if `None`, the device of `mean_axis` is used.
        
        Returns:
            Rotation: A single Rotation when `num` is `None`, otherwise a batch of `num` sampled Rotations with shape `(num, ...)`.
        """
        n = 1 if num is None else num
        mu = torch.as_tensor((1.0, 0.0, 0.0) if mean_axis is None else mean_axis, device=device)
        if random_state is None:
            rng = RandomGenerator(device=mu.device)
        elif isinstance(random_state, RandomGenerator):
            rng = random_state
        else:
            rng = RandomGenerator(seed=random_state, device=mu.device)
        rot_axes = sample_vmf(mu=mu, kappa=kappa, n_samples=n, rng=rng)
        if sigma == math.inf:
            rot_angle = rng.rand_tensor((n, *mu.shape[:-1]), mu.dtype, device=mu.device) * 2 * math.pi
        else:
            rot_angle = (rng.randn_tensor((n, *mu.shape[:-1]), mu.dtype, device=mu.device) * sigma) % (2 * math.pi)
        return cls.from_rotvec(rot_axes * rot_angle.unsqueeze(-1))

    def __mul__(self, other: Rotation) -> Self:
        """
        Deprecated compatibility wrapper for composing two rotations using the legacy `*` operator.
        
        Warns about deprecation and returns the composition of `self` and `other` (equivalent to `self @ other`).
        """
        warnings.warn(
            'Using Rotation*Rotation is deprecated, consider Rotation@Rotation', DeprecationWarning, stacklevel=2
        )
        return self @ other

    def __matmul__(self, other: Rotation) -> Self:
        """Compose this rotation with the other.

        If `p` and `q` are two rotations, then the composition of 'q followed
        by p' is equivalent to ``p @ q``. In terms of rotation matrices,
        the composition can be expressed as
        ``p.as_matrix() @ q.as_matrix()``.

        Parameters
        ----------
        other
            Object containing the rotations to be composed with this one. Note
            that rotation compositions are not commutative, so ``p @ q`` is
            generally different from ``q @ p``.

        Returns
        -------
        composition
            This function supports composition of multiple rotations at a time.
            The following cases are possible:

            - Either `p` or `q` contains a single rotation. In this case
              `composition` contains the result of composing each rotation in
              the other object with the single rotation.
            - Both `p` and `q` contain `N` rotations. In this case each
              rotation `p[i]` is composed with the corresponding rotation
              `q[i]` and `output` contains `N` rotations.
        """
        if not isinstance(other, Rotation):
            return NotImplemented

        p = self._quaternions
        q = other._quaternions
        p, q = torch.broadcast_tensors(p, q)
        result_quaternions = _compose_quaternions(p, q)
        result_improper = self._is_improper ^ other._is_improper

        if self._single and other._single:
            result_quaternions = result_quaternions[0]
            result_improper = result_improper[0]
        return self.__class__(result_quaternions, normalize=True, copy=False, inversion=result_improper)

    def __pow__(self, n: float, modulus: None = None):
        """
        Raise this rotation to a real power by scaling its rotation angle about the rotation axis.
        
        For real `n`, the operation scales the rotation angle by `n` (equivalently `Rotation.from_rotvec(n * self.as_rotvec())`). Special cases: `n == 0` returns the identity rotation, `n == -1` returns the inverse rotation, and `n == 1` returns a copy of this rotation. When the rotation is marked improper (includes a reflection), the reflection is preserved only for integer, odd powers; non-integer powers produce a proper rotation (no reflection).
        
        Parameters:
            n (float): Power to raise the rotation to; may be non-integer or negative.
            modulus (None): Must be `None`; provided for API compatibility.
        
        Returns:
            Rotation: A new Rotation where each element in the batch is the corresponding input rotation raised to power `n`.
        
        Raises:
            NotImplementedError: If `modulus` is not `None`.
        """
        if modulus is not None:
            raise NotImplementedError('modulus not supported')

        # Exact short-cuts
        if n == 0:
            shape = None if self._single else self._quaternions.shape[:-1]
            return self.__class__.identity(shape, device=self.device).to(dtype=self._quaternions.dtype)
        elif n == -1:
            return self.inv()
        elif n == 1:
            if self._single:
                return self.__class__(self._quaternions[0], inversion=self._is_improper[0], copy=True)
            else:
                return self.__class__(self._quaternions, inversion=self._is_improper, copy=True)
        elif math.isclose(round(n), n) and round(n) % 2:
            improper: torch.Tensor | bool = self._is_improper
        else:
            improper = False

        return Rotation.from_rotvec(n * self.as_rotvec(), reflection=improper)

    def inv(self) -> Self:
        """
        Return the inverse of this rotation.
        
        The returned Rotation represents the transformation that composes with the original to produce an identity transformation. The improper/inversion flag is preserved on the result.
        
        Returns:
            inverse (Rotation): Rotation instance containing the inverses of the rotations in this object.
        """
        quaternions = self._quaternions * self._quaternions.new_tensor([-1, -1, -1, 1])
        improper = self._is_improper.clone()

        if self._single:
            quaternions = quaternions[0]
            improper = self._is_improper[0]

        return self.__class__(quaternions, inversion=improper, copy=False)

    def reflect(self) -> Self:
        """Reflect this rotation.

        Converts a proper rotation to an improper one, or vice versa
        by reflecting the rotation about a plane perpendicular to the rotation axis.

        Returns
        -------
        reflected
            Object containing the reflected rotations.
        """
        if self._single:
            quaternions = self._quaternions[0]
            is_improper = self._is_improper[0]
        else:
            quaternions = self._quaternions
            is_improper = self._is_improper

        return self.__class__(quaternions, copy=False, inversion=is_improper, reflection=True)

    def invert_axes(self) -> Self:
        """Invert the axes of the coordinate system.

        Converts a proper rotation to an improper one, or vice versa
        by inversion of the coordinate system.

        .. note::
           This is not the same as the inverse of the rotation.
           See `inv` an inverse.

        Returns
        -------
        inverted_axes
            Object containing the rotation with inverted axes.
        """
        quaternions = self._quaternions.clone()
        improper = ~self._is_improper
        if self._single:
            quaternions = quaternions[0]
            improper = improper[0]
        return self.__class__(quaternions, copy=False, inversion=improper)

    def magnitude(self) -> torch.Tensor:
        """Get the magnitude(s) of the rotation(s).

        Returns
        -------
        magnitude
            Angles in radians. The magnitude will always be in the range ``[0, pi]``.
        """
        angles = 2 * torch.atan2(
            torch.linalg.vector_norm(self._quaternions[..., :3], dim=-1), torch.abs(self._quaternions[..., 3])
        )
        if self._single:
            angles = angles[0]
        return angles

    def approx_equal(self, other: Rotation, atol: float = 1e-6, degrees: bool = False) -> torch.Tensor:
        """
        Check whether another rotation equals this one within an angular tolerance.
        
        Compares the rotation difference to `atol` (interpreted in degrees when `degrees=True`) and requires the improper/inversion flags to match for equality.
        
        Parameters:
            other (Rotation): Rotation(s) to compare against.
            atol (float): Absolute angular tolerance. Values below this are considered equal.
            degrees (bool): If True, interpret `atol` in degrees; otherwise in radians.
        
        Returns:
            bool or torch.Tensor: `True` (or elementwise `True`) if the angular difference is less than `atol` and the improper flags are equal, `False` otherwise.
        """
        if degrees:
            atol = np.deg2rad(atol)
        angles = (self @ other.inv()).magnitude()
        return (angles < atol) & (self._is_improper == other._is_improper)

    def __eq__(self, other: object) -> bool:
        """Check exact equality of two rotations.

        Tests equality up to broadcasting

        Parameters
        ----------
        other
            The other rotation to compare to.

        Returns
        -------
            True if the rotations are exactly equal
        """
        if not isinstance(other, type(self)):
            return False
        if self is other:
            return True
        try:
            if not torch.equal(*torch.broadcast_tensors(self._quaternions, other._quaternions)):
                return False
            if not torch.equal(*torch.broadcast_tensors(self._is_improper, other._is_improper)):
                return False
        except RuntimeError:
            return False
        return True

    def __getitem__(self, indexer: TorchIndexerType) -> Self:
        """
        Create a new Rotation containing the selected element(s) from this batch.
        
        Parameters:
            indexer: Index, slice, or tuple of indices selecting batch entries to extract.
        
        Returns:
            Rotation: A new Rotation containing the selected rotations; the improper/inversion flags are preserved for the selected entries.
        
        Raises:
            TypeError: If this instance represents a single rotation (not subscriptable).
        """
        if self._single:
            raise TypeError('Single rotation is not subscriptable.')

        indexer_quat = (*indexer, slice(None)) if isinstance(indexer, tuple) else (indexer, slice(None))
        batch_shape = torch.broadcast_shapes(self._quaternions.shape[:-1], self._is_improper.shape)
        quaternions = self._quaternions.expand(*batch_shape, 4)[indexer_quat]
        inversion = self._is_improper.expand(batch_shape)[indexer]
        return type(self)(quaternions, normalize=False, inversion=inversion)

    def __iter__(self) -> Iterator[Self]:
        """
        Iterate over the batch, yielding each Rotation element in order.
        
        Returns:
            Iterator[Self]: Yields individual Rotation instances until the batch is exhausted.
        """
        index = 0
        while True:
            try:
                yield self[index]
                index += 1
            except IndexError:
                break

    def _index(self, indexer: Indexer) -> Self:
        """Index using a custom indexer."""
        quaternions = torch.stack([indexer(q) for q in self._quaternions.unbind(-1)], -1)
        inversion = indexer(self._is_improper)
        return type(self)(quaternions, normalize=False, inversion=inversion)

    def _reduce_repeats_(self, tol: float = 1e-6, dim: Sequence[int] | None = None) -> Self:
        """Reduce repeated dimensions to singleton.

        Parameters
        ----------
        tol
            tolerance to apply to quaternions
        dim
            dimensions to try to reduce to singletons. `None` means all.
        """
        if dim is None:
            quaternion_dim: Sequence[int] = range(self._quaternions.ndim - 1)
        else:
            quaternion_dim = [
                d - 1 if d < 0 else d for d in dim if d > -self._quaternions.ndim + 1 and d < self._quaternions.ndim - 1
            ]
        self._quaternions.data = reduce_repeat(self._quaternions, tol, quaternion_dim)
        self._is_improper.data = reduce_repeat(self._is_improper, tol, dim)
        return self

    def _broadcasted_rearrange(
        self, pattern: str, broadcasted_shape: Sequence[int], reduce_views: bool = True, **axes_lengths: int
    ) -> Self:
        quaternions = [
            broadcasted_rearrange(q, pattern, broadcasted_shape, reduce_views=reduce_views, **axes_lengths)
            for q in self._quaternions.unbind(-1)
        ]
        inversion = broadcasted_rearrange(
            self._is_improper, pattern, broadcasted_shape=broadcasted_shape, reduce_views=reduce_views, **axes_lengths
        )
        return type(self)(torch.stack(quaternions, -1), False, False, inversion)

    @property
    def quaternion_x(self) -> torch.Tensor:
        """Get x component of the quaternion."""
        axis = QUAT_AXIS_ORDER.index('x')
        if self._single:
            return self._quaternions[0, axis]
        return self._quaternions[..., axis]

    @quaternion_x.setter
    def quaternion_x(self, quat_x: torch.Tensor | float):
        """Set x component of the quaternion."""
        axis = QUAT_AXIS_ORDER.index('x')
        self._quaternions[..., axis] = quat_x

    @property
    def quaternion_y(self) -> torch.Tensor:
        """Get y component of the quaternion."""
        axis = QUAT_AXIS_ORDER.index('y')
        if self._single:
            return self._quaternions[0, axis]
        return self._quaternions[..., axis]

    @quaternion_y.setter
    def quaternion_y(self, quat_y: torch.Tensor | float):
        """Set y component of the quaternion."""
        axis = QUAT_AXIS_ORDER.index('y')
        self._quaternions[..., axis] = quat_y

    @property
    def quaternion_z(self) -> torch.Tensor:
        """Get z component of the quaternion."""
        axis = QUAT_AXIS_ORDER.index('z')
        if self._single:
            return self._quaternions[0, axis]
        return self._quaternions[..., axis]

    @quaternion_z.setter
    def quaternion_z(self, quat_z: torch.Tensor | float):
        """Set z component of the quaternion."""
        axis = QUAT_AXIS_ORDER.index('z')
        self._quaternions[..., axis] = quat_z

    @property
    def quaternion_w(self) -> torch.Tensor:
        """Get w component of the quaternion."""
        axis = QUAT_AXIS_ORDER.index('w')
        if self._single:
            return self._quaternions[0, axis]
        return self._quaternions[..., axis]

    @quaternion_w.setter
    def quaternion_w(self, quat_w: torch.Tensor | float):
        """Set w component of the quaternion."""
        axis = QUAT_AXIS_ORDER.index('w')
        self._quaternions[..., axis] = quat_w

    def __setitem__(self, indexer: TorchIndexerType, value: Rotation):
        """
        Replace one or more rotations in this Rotation object at the specified index positions.
        
        Parameters:
            indexer (TorchIndexerType): Indexing expression selecting rotation slots to replace (same semantics as tensor indexing for the batch dimensions).
            value (Rotation): Rotation instance providing replacement rotations; its batch shape must be compatible with the selected positions.
        
        Raises:
            TypeError: If this Rotation represents a single rotation and therefore is not subscriptable.
            TypeError: If `value` is not a Rotation instance.
        """
        if self._single:
            raise TypeError('Single rotation is not subscriptable.')

        if not isinstance(value, Rotation):
            raise TypeError('value must be a Rotation object')

        if isinstance(indexer, tuple):
            indexer_quat = (*indexer, slice(None))
        else:
            indexer_quat = (indexer, slice(None))
        quat, inversion = value.as_quat(improper='inversion')
        self._quaternions[indexer_quat] = quat
        self._is_improper[indexer] = inversion

    @classmethod
    def identity(
        cls,
        shape: int | None | tuple[int, ...] = None,
        *,
        device: torch.device | str | None = None,
    ) -> Self:
        """
        Create identity rotation(s).
        
        Parameters:
            shape:
                If None, create a single identity rotation. If an int, create that many identities. If a tuple of ints, create a batch with the given shape.
            device:
                Device on which to allocate the underlying tensors; if None the default device is used.
        
        Returns:
            Identity rotation or batch of identity rotations with scalar part set to 1 and vector part set to 0.
        """
        match shape:
            case None:
                q = torch.zeros(4, device=device)
            case int():
                q = torch.zeros(shape, 4, device=device)
            case tuple():
                q = torch.zeros(*shape, 4, device=device)
        q[..., -1] = 1
        return cls(q, normalize=False)

    @overload
    @classmethod
    def align_vectors(
        cls,
        a: torch.Tensor | Sequence[torch.Tensor] | Sequence[float] | Sequence[Sequence[float]],
        b: torch.Tensor | Sequence[torch.Tensor] | Sequence[float] | Sequence[Sequence[float]],
        weights: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | None = None,
        *,
        return_sensitivity: Literal[False] = False,
        allow_improper: bool = ...,
    ) -> tuple[Rotation, torch.Tensor]: """
        Compute the optimal rotation that best aligns two sets of 3D vectors and return its residual sum of squared distances (and optional sensitivity).
        
        Parameters:
            a: First set of vectors. Shape must be (..., N, 3) or broadcastable to the same shape as `b`. May be a tensor or a sequence convertible to a float tensor.
            b: Second set of vectors, matching `a` in shape and semantics.
            weights: Optional per-vector nonnegative weights of shape (N,) or broadcastable to (..., N). If omitted, unit weights are used.
            return_sensitivity: If `True`, also return a sensitivity tensor describing how the optimal rotation changes with respect to perturbations of the weighted correlation matrix; otherwise only return the rotation and its RSSD.
            allow_improper: If `False`, force the returned rotation to be proper (determinant +1) by flipping sign when necessary; if `True`, allow improper (determinant -1) solutions.
        
        Returns:
            rotation: A Rotation instance that optimally aligns `a` to `b` under the provided weights.
            rssd_or_sensitivity: If `return_sensitivity` is `False`, a tensor containing the residual sum of squared distances for each batch. If `return_sensitivity` is `True`, a tensor containing the sensitivity matrix for each batch describing the derivative of the solution with respect to the weighted correlation matrix.
        
        Notes:
            - Inputs are validated for correct last dimension (3) and nonnegative weights.
            - Single-pair special cases and infinite-weight constraints are handled and reflected in the outputs.
        """
        ...

    @overload
    @classmethod
    def align_vectors(
        cls,
        a: torch.Tensor | Sequence[torch.Tensor] | Sequence[float] | Sequence[Sequence[float]],
        b: torch.Tensor | Sequence[torch.Tensor] | Sequence[float] | Sequence[Sequence[float]],
        weights: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | None = None,
        *,
        return_sensitivity: Literal[True],
        allow_improper: bool = ...,
    ) -> tuple[Rotation, torch.Tensor, torch.Tensor]: """
        Compute the optimal rotation that aligns weighted vector sets a -> b and also return the fit quality and sensitivity.
        
        Parameters:
            a: One or more source vectors with last dimension 3. Accepted shapes: (..., 3), (n_vecs, 3), or sequences convertible to a tensor; broadcasting across batch dimensions is supported.
            b: One or more target vectors with the same shape semantics as `a`; must correspond elementwise to `a`.
            weights: Optional per-vector nonnegative weights with shape (n_vecs,) or broadcastable to the batch; if `None`, equal weights are used.
            return_sensitivity: When `True`, also return a sensitivity tensor describing the linear response of the optimal rotation to small perturbations of the input (see returns).
            allow_improper: If `False`, force a proper rotation (determinant +1) by correcting improper solutions; if `True`, allow improper (determinant -1) solutions.
        
        Returns:
            A tuple (rotation, rssd, sensitivity) where:
            - rotation (Rotation): the optimal rotation aligning `a` to `b` under the provided weights.
            - rssd (torch.Tensor): residual sum of squared deviations for the fit; has the broadcasted batch shape.
            - sensitivity (torch.Tensor): tensor of sensitivity matrices that quantify how the optimal rotation changes under small perturbations of the inputs; its leading dimensions match the broadcasted batch and its trailing dimensions encode the per-fit sensitivity matrices.
        """
        ...

    @classmethod
    def align_vectors(
        cls,
        a: torch.Tensor | Sequence[torch.Tensor] | Sequence[float] | Sequence[Sequence[float]],
        b: torch.Tensor | Sequence[torch.Tensor] | Sequence[float] | Sequence[Sequence[float]],
        weights: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | None = None,
        *,
        return_sensitivity: bool = False,
        allow_improper: bool = False,
    ) -> tuple[Rotation, torch.Tensor] | tuple[Rotation, torch.Tensor, torch.Tensor]:
        R"""Estimate a rotation to optimally align two sets of vectors.

        Find a rotation between frames A and B which best aligns a set of
        vectors `a` and `b` observed in these frames. The following loss
        function is minimized to solve for the rotation matrix :math:`R`:

        .. math::
            L(R) = \frac{1}{2} \sum_{i = 1}^{n} w_i \| a_i - R b_i \|^2 ,

        where :math:`w_i`'s are the `weights` corresponding to each vector.

        The rotation is estimated with Kabsch algorithm [KAB]_, and solves what
        is known as the "pointing problem", or "Wahba's problem" [WAH]_.

        There are two special cases. The first is if a single vector is given
        for `a` and `b`, in which the shortest distance rotation that aligns
        `b` to `a` is returned. The second is when one of the weights is infinity.
        In this case, the shortest distance rotation between the primary infinite weight
        vectors is calculated as above. Then, the rotation about the aligned primary
        vectors is calculated such that the secondary vectors are optimally
        aligned per the above loss function. The result is the composition
        of these two rotations. The result via this process is the same as the
        Kabsch algorithm as the corresponding weight approaches infinity in
        the limit. For a single secondary vector this is known as the
        "align-constrain" algorithm [MAG2018]_.

        For both special cases (single vectors or an infinite weight), the
        sensitivity matrix does not have physical meaning and an error will be
        raised if it is requested. For an infinite weight, the primary vectors
        act as a constraint with perfect alignment, so their contribution to
        `rssd` will be forced to 0 even if they are of different lengths.

        Parameters
        ----------
        a
            Vector components observed in initial frame A. Each row of `a`
            denotes a vector.
        b
            Vector components observed in another frame B. Each row of `b`
            denotes a vector.
        weights
            Weights describing the relative importance of the vector
            observations. If `None`, then all values in `weights` are
            assumed to be 1. One and only one weight may be infinity, and
            weights must be positive.
        return_sensitivity
            Whether to return the sensitivity matrix.
        allow_improper
            If True, allow improper rotations to be returned. If False,
            then the rotation is restricted to be proper.

        Returns
        -------
        rotation
            Best estimate of the rotation that transforms `b` to `a`.
        rssd
            Square root of the weighted sum of the squared distances between the given sets of
            vectors
            after alignment.
        sensitivity_matrix
            Sensitivity matrix of the estimated rotation estimate as explained
            in Notes.

        References
        ----------
        .. [KAB] https://en.wikipedia.org/wiki/Kabsch_algorithm
        .. [WAH] https://en.wikipedia.org/wiki/Wahba%27s_problem
        .. [MAG2018] Magner R (2018), Extending target tracking capabilities through trajectory and momentum setpoint
           optimization. Small Satellite Conference.
        """
        a_tensor = torch.stack([torch.as_tensor(el) for el in a]) if isinstance(a, Sequence) else torch.as_tensor(a)
        b_tensor = torch.stack([torch.as_tensor(el) for el in b]) if isinstance(b, Sequence) else torch.as_tensor(b)
        dtype = torch.promote_types(a_tensor.dtype, b_tensor.dtype)
        if not dtype.is_floating_point:
            # boolean or integer inputs will result in float32
            dtype = torch.float32
        a_tensor = torch.atleast_2d(a_tensor).to(dtype=dtype)
        b_tensor = torch.atleast_2d(b_tensor).to(dtype=dtype)
        if weights is None:
            weights_tensor = a_tensor.new_ones(a_tensor.shape[:-1], dtype=dtype)
        else:
            weights_tensor = torch.atleast_1d(torch.as_tensor(weights, dtype=dtype, device=a_tensor.device))

        if a_tensor.ndim > 2 or b_tensor.ndim > 2 or weights_tensor.ndim > 1:
            raise NotImplementedError('Batched inputs are not supported.')

        if return_sensitivity:
            rot_matrix, rssd, sensitivity = _align_vectors(a_tensor, b_tensor, weights_tensor, True, allow_improper)
            return cls.from_matrix(rot_matrix), rssd, sensitivity
        else:
            rot_matrix, rssd = _align_vectors(a_tensor, b_tensor, weights_tensor, False, allow_improper)
            return cls.from_matrix(rot_matrix), rssd

    @property
    def shape(self) -> torch.Size:
        """Return the batch shape of the Rotation."""
        if self._single:
            return torch.Size()
        return self._quaternions.shape[:-1]

    def __bool__(self):
        """Comply with Python convention for objects to be True.

        Required because `Rotation.__len__()` is defined and not always
        truthy.
        """
        return True

    def __len__(self) -> int:
        """Return the leading dimensions size of the batched Rotation."""
        if self._single:
            raise TypeError('Single rotation has no len().')
        return self.shape[0]

    def __repr__(self):
        """Return String Representation of the Rotation."""
        if self._single and not self._is_improper:
            return f'Rotation({self._quaternions.tolist()})'
        elif self._single and self._is_improper:
            return f'improper Rotation({self._quaternions.tolist()})'
        elif self._is_improper.all():
            return f'{tuple(self.shape)}-batched improper Rotation()'
        elif self._is_improper.any():
            return f'{tuple(self.shape)}-batched (mixed proper/improper) Rotation()'
        else:
            return f'{tuple(self.shape)}-batched Rotation()'

    def mean(
        self,
        weights: torch.Tensor | NestedSequence[float] | None = None,
        dim: None | int | Sequence[int] = None,
        keepdim: bool = False,
    ) -> Self:
        """
        Compute the chordal L2 (projected) mean of the rotations over the specified batch dimensions.
        
        The mean minimizes the weighted sum of squared Frobenius norms between rotation matrices (the projected arithmetic mean). Improper rotations are excluded from the averaging; the returned rotation is marked improper (a reflection) when the weighted majority of inputs being averaged are improper.
        
        Parameters:
            weights (torch.Tensor | Sequence[float] | None):
                Per-rotation nonnegative weights broadcastable to the instance batch shape. If `None`, all rotations are weighted equally.
            dim (int | Sequence[int] | None):
                Batch dimension(s) to reduce. `None` reduces all batch dimensions and returns a single Rotation.
            keepdim (bool):
                If True, retained reduced dimensions as length-1 dimensions in the output batch shape.
        
        Returns:
            Rotation:
                Rotation object containing the mean quaternion(s). If `keepdim` is True, reduced dimensions are preserved with length 1; if the instance represented a single rotation, a single Rotation is returned.
        """
        if self._single:
            return self.__class__(self._quaternions[0], inversion=self._is_improper, normalize=False)

        if weights is None:
            weights = self._quaternions.new_ones(self.shape)
        else:
            weights = torch.as_tensor(weights, dtype=self._quaternions.dtype, device=self.device)
            weights = weights.expand(self.shape)

            if torch.any(weights < 0):
                raise ValueError('`weights` must be non-negative.')

        if isinstance(dim, Sequence):
            dim = tuple(dim)

        modal_improper = (weights * self._is_improper).sum(dim=dim, keepdim=keepdim) > 0.5 * weights.sum(
            dim=dim, keepdim=keepdim
        )

        quaternions = torch.as_tensor(self._quaternions)
        if dim is None:
            quaternions = quaternions.reshape(-1, 4)
            weights = weights.reshape(-1)
            dim = list(range(len(self.shape)))
        else:
            dim = normalize_indices(quaternions.ndim - 1, dim)
            batch_dims = [i for i in range(quaternions.ndim - 1) if i not in dim]
            permute_dims = (*batch_dims, *dim)
            quaternions = quaternions.permute(*permute_dims, -1).flatten(start_dim=len(batch_dims), end_dim=-2)
            weights = weights.permute(permute_dims).flatten(start_dim=len(batch_dims))
        k = (weights.unsqueeze(-2) * quaternions.mT) @ quaternions
        _, v = torch.linalg.eigh(k)
        mean_quaternions = v[..., -1]
        if keepdim:
            # unsqueeze the dimensions we removed in the reshape and product
            for d in sorted(dim):
                mean_quaternions = mean_quaternions.unsqueeze(d)

        return self.__class__(mean_quaternions, inversion=modal_improper, normalize=False)

    def reshape(self, *shape: int | Sequence[int]) -> Self:
        """
        Reshape the batch dimensions of this Rotation, preserving the quaternion component axis.
        
        The provided shape replaces the batch shape; internal quaternion and improper-mask tensors are first broadcast to a common batch shape and then reshaped. The final (component) dimension of the quaternion tensor remains size 4.
        
        Parameters:
            *shape (int | Sequence[int]): New batch shape as integers or sequences of integers; elements are concatenated.
        
        Returns:
            Rotation: A new Rotation instance with data copied and the requested batch shape.
        """
        newshape = []
        for s in shape:
            if isinstance(s, int):
                newshape.append(s)
            else:
                newshape.extend(s)
        batch_shape = torch.broadcast_shapes(self._quaternions.shape[:-1], self._is_improper.shape)
        return self.__class__(
            self._quaternions.expand(*batch_shape, 4).reshape(*newshape, 4),
            inversion=self._is_improper.expand(batch_shape).reshape(*newshape),
            copy=True,
        )

    def permute(self, dims: Sequence[int]) -> Self:
        """
        Reorder the Rotation's batch dimensions.
        
        Parameters:
            dims (Sequence[int]): New ordering of the batch axes. Negative indices are interpreted relative to the batch rank (like standard Python indexing). The quaternion component axis (last axis of size 4) is not included in these indices and remains the final axis.
        
        Returns:
            Rotation: A new Rotation whose batch dimensions have been permuted according to `dims`; quaternion components and improper flags are rearranged to match the new batch ordering.
        """
        batch_shape = torch.broadcast_shapes(self._quaternions.shape[:-1], self._is_improper.shape)
        inversion = self._is_improper.expand(batch_shape).permute(*dims)
        # negative dimensions should ignore the internal dimension
        batch_ndim = len(batch_shape)
        quaternion_dims = [d if d >= 0 else batch_ndim + d for d in dims]
        quaternions = self._quaternions.expand(*batch_shape, 4).permute(*quaternion_dims, -1)
        return self.__class__(quaternions, inversion=inversion, copy=True)

    def expand(self, *shape: int | Sequence[int]) -> Self:
        """
        Expand this Rotation's batch dimensions to the given shape.
        
        Parameters:
            shape (int | Sequence[int]): One or more integers or integer sequences specifying the target batch shape. The quaternion component dimension is preserved.
        
        Returns:
            Rotation: A Rotation with batch dimensions expanded to the requested shape.
        """
        newshape = []
        for s in shape:
            if isinstance(s, int):
                newshape.append(s)
            else:
                newshape.extend(s)
        return self.__class__(
            self._quaternions.expand(*newshape, 4),
            inversion=self._is_improper.expand(newshape),
            normalize=False,
            copy=False,
        )

    def unsqueeze(self, dim: int) -> Self:
        """Unsqueeze the Rotation object in a batch dimension.

        Add a new dimension to the Rotation object at the specified position.

        Parameters
        ----------
        dim
            The position where the new dimension is to be added.
        """
        quaternion_dim = dim if dim >= 0 else dim - 1  # last dimension are the quaternion components
        return self.__class__(
            self._quaternions.unsqueeze(quaternion_dim), inversion=self._is_improper.unsqueeze(dim), copy=True
        )

    @property
    def device(self) -> torch.device:
        """Get the device of the Rotation."""
        if self._quaternions.device != self._is_improper.device:
            raise RuntimeError('Quaternion and is_improper tensors are on different devices.')
        return self._quaternions.device

    def _mr2_save_to_group(self, group: h5py.Group):
        """Save the Rotation to an HDF5 group."""
        group.attrs['py_module'] = __name__
        group.attrs['class_name'] = self.__class__.__name__
        group.create_dataset('quaternions', data=self._quaternions.numpy(force=True)).attrs['py_type'] = 'torch.Tensor'
        group.create_dataset('inversion', data=self._is_improper.numpy(force=True)).attrs['py_type'] = 'torch.Tensor'


class RotationBackend(AbstractBackend):
    """Einops backend for Rotations."""

    framework_name = 'mr2.data.Rotation'

    def is_appropriate_type(self, x) -> bool:  # noqa: ANN001
        """Check if the object is a Rotation."""
        return isinstance(x, Rotation)

    def is_float_type(self, _: Rotation) -> bool:
        """Return True as Rotations are always float."""
        return True

    def reduce(self, x: Rotation, operation: str, reduced_axes: int) -> Rotation:
        """Perform reduction operation on the Rotation."""
        if operation == 'mean':
            return x.mean(dim=reduced_axes)
        raise NotImplementedError(f'Unknown reduction {operation} for Rotations')

    def transpose(self, x: Rotation, axes: Sequence[int]) -> Rotation:
        """Permute the axes of the Rotation."""
        return x.permute(axes)

    def stack_on_zeroth_dimension(self, x: Sequence[Rotation]) -> Rotation:
        """Stack the Rotations on the zeroth dimension."""
        return Rotation.concatenate([r.reshape(1, *r.shape) for r in x])

    def add_axis(self, x: Rotation, axis_position: int) -> Rotation:
        """Add a new axis to the Rotation."""
        return x.unsqueeze(axis_position)

    def add_axes(self, x: Rotation, n_axes: int, pos2len: dict[int, int]) -> Rotation:
        """Add multiple expanded axes to the Rotation."""
        repeats = [-1] * n_axes
        for axis_position, axis_length in pos2len.items():
            x = self.add_axis(x, axis_position)
            repeats[axis_position] = axis_length
        return x.expand(repeats)
