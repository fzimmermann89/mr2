"""1D, 2D and 3D density compensation function calculation with voronoi method."""

from itertools import permutations, product

import numpy as np
import torch
from scipy.spatial import ConvexHull, Delaunay, Voronoi, cKDTree

UNIQUE_ROUNDING_DECIMALS = 15


def _cell_volumes_3d(points: np.ndarray, block_size: int = 2**18) -> np.ndarray:
    """Calculate the volume of each 3D Voronoi cell from the Delaunay tessellation.

    Within a Delaunay tetrahedron `(a, b, c, d)`, the part of the Voronoi cell of `a` is the sum of the signed
    tetrahedra spanned by `a`, the midpoint of an edge `(a, b)`, a point on the Voronoi edge dual to a face `(a, b, c)`
    and the circumcenter of the tetrahedron. This is correct for any point on the line through the circumcenters of
    the two tetrahedra sharing the face, if both use the same point. The point is moved towards the circumcenter of
    the flatter tetrahedron, such that (nearly) flat tetrahedra, which Qhull creates for cospherical points and whose
    orientation is not defined, do not contribute. Cells touching exactly flat tetrahedra are rebuilt from
    the convex hull of their incident circumcenters. Volumes of points on the convex hull are not meaningful.
    """
    delaunay = Delaunay(points)
    all_points = torch.from_numpy(delaunay.points)
    blocks = [
        np.arange(start, min(start + block_size, len(delaunay.simplices)))
        for start in range(0, len(delaunay.simplices), block_size)
    ]

    def tetrahedra(block: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the vertex indices, the vertices and the signed volumes times 6 of tetrahedra."""
        simplices = torch.from_numpy(delaunay.simplices[block].astype(np.int64))
        vertices = all_points[simplices]
        return simplices, vertices, torch.linalg.det(vertices[:, 1:] - vertices[:, :1])

    def circumcenter(block: np.ndarray) -> torch.Tensor:
        """Get the circumcenters from the hyperplanes of the lifted points on the paraboloid."""
        equations = torch.from_numpy(delaunay.equations[block])
        return -equations[..., :3] / (2 * delaunay.paraboloid_scale * equations[..., 3:4])

    flatness = torch.cat([tetrahedra(block)[2].abs() for block in blocks])
    rebuild = np.zeros(len(points), dtype=bool)
    rebuild[np.unique(delaunay.simplices[flatness.numpy() == 0])] = True
    # All orders (a, b, c, d) of the vertices of a tetrahedron, with c < d, and their parity
    order = np.array([(a, b, *sorted({0, 1, 2, 3} - {a, b})) for a, b in permutations(range(4), 2)])
    parity = torch.from_numpy(np.linalg.det(np.eye(4)[order]))
    a, b, c, d = torch.from_numpy(order.T)
    volume = torch.zeros(len(points), dtype=torch.float64)
    for block in blocks:
        simplices, vertices, signed_volume = tetrahedra(block)
        # Neighbors opposite to each vertex. Faces on the convex hull have no neighbor.
        neighbors = np.where(delaunay.neighbors[block] >= 0, delaunay.neighbors[block], block[:, None])
        center, neighbor_center = circumcenter(block)[:, None], circumcenter(neighbors)
        weight, neighbor_weight = flatness[block][:, None, None], flatness[neighbors][..., None]
        face_point = (neighbor_weight * center + weight * neighbor_center) / (weight + neighbor_weight)
        face_point = torch.where(weight + neighbor_weight > 0, face_point, (center + neighbor_center) / 2)
        edge = (vertices[:, b] - vertices[:, a]) / 2
        spoke = center - vertices[:, a]
        piece = torch.linalg.vecdot(edge, torch.linalg.cross(face_point[:, d] - face_point[:, c], spoke))
        volume.index_add_(0, simplices[:, a].ravel(), (parity * signed_volume.sign()[:, None] * piece).ravel() / 6)
    # Qhull's triangulation of cospherical facets can overlap or contain flat tetrahedra. Rebuild only the
    # affected cells from their circumcenters, without relying on tetrahedron orientation or face connectivity.
    result = volume.numpy()
    rebuild |= (result <= 0) | ~np.isfinite(result)
    rebuild[np.unique(delaunay.convex_hull)] = False
    rebuild[delaunay.coplanar[:, 0]] = False
    rows, cols = np.nonzero(rebuild[delaunay.simplices])
    cell = delaunay.simplices[rows, cols]
    order = np.argsort(cell)
    cell, indices = cell[order], rows[order]
    for index in np.flatnonzero(rebuild):
        start, end = np.searchsorted(cell, [index, index + 1])
        equations = delaunay.equations[indices[start:end]]
        cell_vertices = -equations[:, :3] / (2 * delaunay.paraboloid_scale * equations[:, 3:4])
        result[index] = ConvexHull(cell_vertices - points[index]).volume
    # Qhull drops points that coincide with another point within its precision. The volume is shared equally, as for
    # exact repeats.
    owner = np.arange(len(points))
    owner[delaunay.coplanar[:, 0]] = delaunay.coplanar[:, 2]
    return np.bincount(owner, result)[owner] / np.bincount(owner)[owner]


def dcf_1d(traj: torch.Tensor) -> torch.Tensor:
    """Calculate sample density compensation function for 1D trajectory.

    This function operates on a single `other` sample.
    See also `~mr2.data.DcfData` and `~mr2.utils.smap`.

    Parameters
    ----------
    traj
        k-space positions, 1D tensor

    Returns
    -------
        Density compensation values with the trajectory dtype and device.
    """
    traj_sorted, inverse, counts = torch.unique(
        torch.round(traj, decimals=UNIQUE_ROUNDING_DECIMALS),
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )

    # For a sorted trajectory: x0 x1 x2 ... xN
    # we assign the point at x1 the area (x1 - x0) / 2  + (x2 - x1) / 2,
    # this is done by central differences (-1,0,1).
    # For the edges, we append/prepend the values afterwards, such that:
    # we assign the point at x0 the area (x1 - x0) / 2 + (x1 - x0) / 2, and
    # we assign the point at xN the area (xN - xN-1) / 2 + (xN - xN-1) / 2.

    kernel = traj.new_tensor([-1 / 2, 0, 1 / 2]).reshape(1, 1, 3)

    if (elements := len(traj_sorted)) >= 3:
        central_diff = torch.nn.functional.conv1d(traj_sorted[None, None, :], kernel)[0, 0]
        first = traj_sorted[1] - traj_sorted[0]
        last = traj_sorted[-1] - traj_sorted[-2]
        central_diff = torch.cat((first[None], central_diff, last[None]), -1)
    elif elements == 2:
        diff = traj_sorted[1] - traj_sorted[0]
        central_diff = torch.cat((diff[None], diff[None]), -1)
    else:
        central_diff = torch.ones_like(traj_sorted)

    # Repeated points are reduced by the number of repeats
    dcf = torch.nan_to_num(central_diff / counts)[inverse]
    return dcf


def dcf_2d3d_voronoi(traj: torch.Tensor) -> torch.Tensor:
    """Calculate sample density compensation function using Voronoi method.

    This function computes the DCF by determining the area around each point in k-space using the Voronoi tessellation.
    Points at the edge of k-space are detected as outliers and are assigned the area of the 1% largest DCF values.
    Nearly coincident 3D points are treated as repeated samples and share their cell weight.

    The Voronoi tessellation assigns each point in k-space a region based on the proximity to its nearest neighbors. The
    DCF is then computed based on the inverse of the area of these regions.

    This function operates on a single `other` sample.
    See also `~mr2.data.DcfData` and `~mr2.utils.smap`.

    Parameters
    ----------
    traj
        k-space positions `(2 or 3, k2, k1, k0)`

    Returns
    -------
        Density compensation values with the trajectory dtype and device, shaped `(k2, k1, k0)`.
    """
    # 2D and 3D trajectories supported
    dim = traj.shape[0]
    if dim not in (2, 3):
        raise ValueError(f'Only 2D or 3D trajectories supported, not {dim}D.')

    # Calculate dcf only for unique k-space positions
    traj_dim = traj.shape
    traj_numpy = np.round(traj.cpu().numpy(), decimals=UNIQUE_ROUNDING_DECIMALS)
    traj_numpy = traj_numpy.reshape(dim, -1)
    traj_unique, inverse, counts = np.unique(traj_numpy, return_inverse=True, return_counts=True, axis=1)
    inverse = inverse.reshape(-1)

    # Nearly coincident 3D points can cause Qhull topology errors. Merge small groups as repeated samples.
    # The tolerance is relative to the bounding corners used below. Keeping every member within this distance
    # of its representative bounds the group diameter to twice the tolerance and prevents merging long chains.
    furthest_corner = np.max(np.abs(traj_unique))
    if dim == 3:
        tolerance = np.sqrt(np.finfo(np.float64).eps) * furthest_corner * 10
        tree = cKDTree(traj_unique.T)
        nearest = tree.query(traj_unique.T, k=2, distance_upper_bound=tolerance)[0][:, 1]
        owner = np.arange(traj_unique.shape[1])
        assigned = np.zeros(len(owner), dtype=bool)
        for point in np.flatnonzero(np.isfinite(nearest)):
            if assigned[point]:
                continue
            neighbors = np.array(tree.query_ball_point(traj_unique[:, point], tolerance))
            neighbors = neighbors[~assigned[neighbors]]
            owner[neighbors] = point
            assigned[neighbors] = True
        representatives, cluster = np.unique(owner, return_inverse=True)
        traj_unique = traj_unique[:, representatives]
        inverse = cluster.reshape(-1)[inverse]
        counts = np.bincount(inverse)

    # Especially in 3D, errors in the calculation of the convex hull can occur for edge points. To avoid this,
    # the corner points of a cube bounding box are added here. The bounding box is chosen very large to ensure these
    # edge points of the trajectory can still be accurately detected in the outlier detection further down.
    corner_points = np.array(list(product([-1, 1], repeat=dim))) * furthest_corner * 10
    traj_extendend = np.concatenate((traj_unique, corner_points.transpose()), axis=1)

    if dim == 2:
        vdiagram = Voronoi(traj_extendend.transpose())
        regions = [vdiagram.regions[r] for r in vdiagram.point_region[: -len(corner_points)]]
        vertices = [vdiagram.vertices[region] for region in regions]

        def cross2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

        dcf = np.array([np.abs(cross2d(v[:-1], v[1:]).sum(0) + cross2d(v[-1], v[0])) / 2 for v in vertices])
    else:
        dcf = _cell_volumes_3d(traj_extendend.transpose())[: -len(corner_points)]

    # Get outliers (i.e. voronoi cell which are unbound) and set them to a reasonable value
    # Outliers are defined as values larger than 1.5 * inter quartile range of the values
    # Outliers are set to the average of the 1% largest values.
    dcf_sorted = np.sort(dcf)
    q1, q3 = np.percentile(dcf_sorted, [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    idx_outliers = np.nonzero(dcf > upper_bound)
    n_outliers = len(idx_outliers[0])
    high_values_start = int(0.99 * (len(dcf_sorted) - n_outliers))
    high_values_end = len(dcf_sorted) - n_outliers  # this works also for n_outliers==0
    fill_value = np.average(dcf_sorted[high_values_start:high_values_end])
    dcf[idx_outliers] = fill_value

    # Sort dcf values back into the original order (i.e. before calling unique)
    dcf = np.reshape((dcf / counts)[inverse], traj_dim[1:])

    return torch.tensor(dcf, dtype=traj.dtype, device=traj.device)
