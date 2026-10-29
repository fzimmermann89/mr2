"""1D, 2D and 3D density compensation function calculation with voronoi method."""

from itertools import permutations, product

import numpy as np
import torch
from scipy.spatial import ConvexHull, Delaunay, Voronoi, cKDTree

UNIQUE_ROUNDING_DECIMALS = 15


def _cell_volumes_3d(points: np.ndarray, block_size: int = 2**18) -> np.ndarray:
    """Calculate 3D Voronoi cell volumes from one Delaunay mesh.

    A sample's cell is the space closer to it than to any other sample. Its corners are the centers of spheres
    through the four vertices of nearby tetrahedra [1]_. We split each cell into small tetrahedra and add their
    volumes with plus or minus signs, so that pieces outside the cell cancel [2]_.

    Neighboring tetrahedra use the same point between their sphere centers. We place it closer to the flatter
    tetrahedron to reduce numerical errors. This weighting is specific to this implementation. Nonpositive or
    invalid volumes are recomputed from the cell's corners. A few positive volumes may remain approximate when
    points nearly coincide or lie on the same sphere. Points on the outer boundary have unbounded cells;
    their volumes are not used.

    References
    ----------
    .. [1] H. Si, "TetGen User's Manual", Section 1.1.1, Delaunay Triangulations, Voronoi Diagrams.
       https://www.wias-berlin.de/software/tetgen/1.5/doc/manual/manual002.html
    .. [2] A. N. Hirani, K. Kalyanaraman, E. B. VanderZee, "Delaunay Hodge Star", Section 2 (corrected preprint).
       https://arxiv.org/abs/1204.0747
    """
    delaunay = Delaunay(points)
    all_points = torch.from_numpy(delaunay.points)
    blocks = [
        np.arange(start, min(start + block_size, len(delaunay.simplices)))
        for start in range(0, len(delaunay.simplices), block_size)
    ]

    def tetrahedra(block: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the vertex indices and vertex positions of tetrahedra."""
        simplices = torch.from_numpy(delaunay.simplices[block].astype(np.int64))
        return simplices, all_points[simplices]

    def circumcenters(block: np.ndarray) -> np.ndarray:
        """Get the circumcenters of tetrahedra."""
        # Qhull lifts the points onto a paraboloid. The plane of each facet of the lifted hull gives the circumcenter.
        equations = delaunay.equations[block]
        return -equations[..., :3] / (2 * delaunay.paraboloid_scale * equations[..., 3:4])

    # Six times the signed volume. Its magnitude is used as the flatness of a tetrahedron.
    signed_volume = torch.cat([torch.linalg.det(v[:, 1:] - v[:, :1]) for _, v in map(tetrahedra, blocks)])
    flatness = signed_volume.abs()
    # All orders (a, b, c, d) of the vertices of a tetrahedron with c < d, and the sign of each permutation
    orders = np.array([(a, b, *sorted({0, 1, 2, 3} - {a, b})) for a, b in permutations(range(4), 2)])
    parity = torch.from_numpy(np.linalg.det(np.eye(4)[orders]))
    a, b, c, d = torch.from_numpy(orders.T)
    volume = torch.zeros(len(points), dtype=torch.float64)
    for block in blocks:
        simplices, vertices = tetrahedra(block)
        # Neighbor opposite to each vertex. Faces on the convex hull have no neighbor and use the tetrahedron itself.
        neighbors = np.where(delaunay.neighbors[block] >= 0, delaunay.neighbors[block], block[:, None])
        center = torch.from_numpy(circumcenters(block))[:, None]
        neighbor_center = torch.from_numpy(circumcenters(neighbors))
        weight, neighbor_weight = flatness[block][:, None, None], flatness[neighbors][..., None]
        # Point between the circumcenters, shared with the neighbor across the face opposite each vertex. It lies
        # close to the center of the flatter tetrahedron, as (nearly) flat tetrahedra have no reliable orientation.
        shared_point = (neighbor_weight * center + weight * neighbor_center) / (weight + neighbor_weight)
        shared_point = torch.where(weight + neighbor_weight > 0, shared_point, (center + neighbor_center) / 2)
        to_midpoint = (vertices[:, b] - vertices[:, a]) / 2
        to_center = center - vertices[:, a]
        # Six times the signed volume of (a, midpoint of (a, b), shared point across face (a, b, c), center) minus
        # the same with face (a, b, d). The shared point across face (a, b, c) is the one opposite to d.
        piece = torch.linalg.vecdot(to_midpoint, torch.linalg.cross(shared_point[:, d] - shared_point[:, c], to_center))
        piece = parity * signed_volume[block].sign()[:, None] * piece
        volume.index_add_(0, simplices[:, a].ravel(), piece.ravel() / 6)
    result = volume.numpy()

    # Recompute nonpositive or invalid volumes from the cell corners. Rebuilding every cell touching a flat
    # tetrahedron would repeat most of the work on Cartesian grids and stacked planes.
    rebuild = (result <= 0) | ~np.isfinite(result)
    rebuild[delaunay.convex_hull] = False
    rebuild[delaunay.coplanar[:, 0]] = False
    simplex, corner = np.nonzero(rebuild[delaunay.simplices])
    vertex = delaunay.simplices[simplex, corner]
    by_vertex = np.argsort(vertex)
    vertex, simplex = vertex[by_vertex], simplex[by_vertex]
    for index in np.flatnonzero(rebuild):
        start, end = np.searchsorted(vertex, [index, index + 1])
        result[index] = ConvexHull(circumcenters(simplex[start:end]) - points[index]).volume

    # Qhull drops points that coincide with another point within its precision. They share the volume of that point,
    # as exact repeats do.
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

    Each sample's weight is the area of its Voronoi cell in 2D or its volume in 3D.
    Large boundary cells are detected as outliers and assigned the mean of the largest 1% of non-outlier weights.
    Nearly coincident 3D points are treated as repeated samples and share their cell weight.

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

        # Shoelace equation for 2D
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
