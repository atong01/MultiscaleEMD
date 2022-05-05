from sklearn.metrics.pairwise import pairwise_distances

import numpy as np
import ot as pot  # Python Optimal Transport package
import scipy.sparse


def sinkhorn(
    p,
    q,
    p_weights,
    q_weights,
    metric="euclidean",
):
    """
    Returns the earth mover's distance between two point clouds
    Parameters
    ----------
    cloud1 : 2-D array
        First point cloud
    cloud2 : 2-D array
        Second point cloud
    Returns
    -------
    distance : float
        The distance between the two point clouds
    """
    if p_weights is None:
        p_weights = np.ones(len(p)) / len(p)
    if q_weights is None:
        q_weights = np.ones(len(q)) / len(q)
    p_weights = np.ones(len(p)) / len(p)
    q_weights = np.ones(len(q)) / len(q)

    pairwise_dist = np.ascontiguousarray(
        pairwise_distances(p, Y=q, metric=metric, n_jobs=-1)
    )

    result = pot.sinkhorn2(
        p_weights,
        q_weights,
        pairwise_dist,
        reg=1,
        # numItermax=100,
        return_matrix=False,
    )
    return np.sqrt(result)


def exact(
    p,
    q,
    p_weights=None,
    q_weights=None,
    metric="euclidean",
):
    """
    Returns the earth mover's distance between two point clouds
    Parameters
    ----------
    cloud1 : 2-D array
        First point cloud
    cloud2 : 2-D array
        Second point cloud
    Returns
    -------
    distance : float
        The distance between the two point clouds
    """
    if p_weights is None:
        p_weights = np.ones(len(p)) / len(p)
    if q_weights is None:
        q_weights = np.ones(len(q)) / len(q)
    pairwise_dist = np.ascontiguousarray(
        pairwise_distances(p, Y=q, metric=metric, n_jobs=-1)
    )
    result = pot.lp.emd2(
        p_weights, q_weights, pairwise_dist, numItermax=1e7, return_matrix=False
    )
    return result


def interpolate_with_ot(p0, p1, tmap, interp_frac, size):
    """
    Interpolate between p0 and p1 at fraction t_interpolate knowing a transport
    map from p0 to p1.

    Parameters
    ----------
    p0 : 2-D array
        The genes of each cell in the source population
    p1 : 2-D array
        The genes of each cell in the destination population
    tmap : 2-D array
        A transport map from p0 to p1
    t_interpolate : float
        The fraction at which to interpolate
    size : int
        The number of cells in the interpolated population
    Returns
    -------
    p05 : 2-D array
        An interpolated population of 'size' cells
    """
    p0 = p0.toarray() if scipy.sparse.isspmatrix(p0) else p0
    p1 = p1.toarray() if scipy.sparse.isspmatrix(p1) else p1
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    tmap = np.asarray(tmap, dtype=np.float64)
    if p0.shape[1] != p1.shape[1]:
        raise ValueError("Unable to interpolate. Number of genes do not match")
    if p0.shape[0] != tmap.shape[0] or p1.shape[0] != tmap.shape[1]:
        raise ValueError(
            "Unable to interpolate. Tmap size is {}, expected {}".format(
                tmap.shape, (len(p0), len(p1))
            )
        )
    # Assume growth is exponential and retrieve growth rate at t_interpolate
    # If all sums are the same then this does not change anything
    # This only matters if sum is not the same for all rows
    p = tmap / np.power(tmap.sum(axis=0), 1.0 - interp_frac)
    p = p.flatten(order="C")
    p = p / p.sum()
    choices = np.random.choice(len(p0) * len(p1), p=p, size=size)
    return np.asarray(
        [
            p0[i // len(p1)] * (1 - interp_frac) + p1[i % len(p1)] * interp_frac
            for i in choices
        ],
        dtype=np.float64,
    )


def interpolate_per_point_with_ot(p0, p1, tmap, interp_frac):
    """
    Interpolate between p0 and p1 at fraction t_interpolate knowing a transport
    map from p0 to p1.
    Parameters
    ----------
    p0 : 2-D array
        The genes of each cell in the source population
    p1 : 2-D array
        The genes of each cell in the destination population
    tmap : 2-D array
        A transport map from p0 to p1
    t_interpolate : float
        The fraction at which to interpolate
    Returns
    -------
    p05 : 2-D array
        An interpolated population of 'size' cells
    """
    assert len(p0) == len(p1)
    p0 = p0.toarray() if scipy.sparse.isspmatrix(p0) else p0
    p1 = p1.toarray() if scipy.sparse.isspmatrix(p1) else p1
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    tmap = np.asarray(tmap, dtype=np.float64)
    if p0.shape[1] != p1.shape[1]:
        raise ValueError("Unable to interpolate. Number of genes do not match")
    if p0.shape[0] != tmap.shape[0] or p1.shape[0] != tmap.shape[1]:
        raise ValueError(
            "Unable to interpolate. Tmap size is {}, expected {}".format(
                tmap.shape, (len(p0), len(p1))
            )
        )

    # Assume growth is exponential and retrieve growth rate at t_interpolate
    # If all sums are the same then this does not change anything
    # This only matters if sum is not the same for all rows
    p = tmap / (tmap.sum(axis=0) / 1.0 - interp_frac)
    # p = tmap / np.power(tmap.sum(axis=0), 1.0 - interp_frac)
    # p = p.flatten(order="C")
    p = p / p.sum(axis=0)
    choices = np.array([np.random.choice(len(p0), p=p[i]) for i in range(len(p0))])
    return np.asarray(
        [
            p0[i] * (1 - interp_frac) + p1[j] * interp_frac
            for i, j in enumerate(choices)
        ],
        dtype=np.float64,
    )
