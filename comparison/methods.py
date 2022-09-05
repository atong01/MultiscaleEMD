def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

from MultiscaleEMD import DiffusionCheb
from MultiscaleEMD import MetricTree
from MultiscaleEMD.emd import exact
from MultiscaleEMD.emd import sinkhorn
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import dataset
import graphtools
import numpy as np
import ot
import pandas as pd
import phate
import scipy
import time


def phemd(data, labels, n_neighbors=10, n_clusters=8, random_state=42):
    """Compute the PhEMD between distributions. As specified in Chen et al. 2019. Note
    that this reproduces the same steps but is not numerically the same.

    Args:
        data: 2-D array N x F points by features.
        labels: 2-D array N x M points by distributions.

    Returns:
        distance_matrix: 2-D M x M array with each cell representing the
        distance between each distribution of points.
    """
    start = time.time()
    phate_op = phate.PHATE(random_state=random_state, verbose=False)
    phate_op.fit(data)
    cluster_op = KMeans(n_clusters, random_state=random_state)
    cluster_ids = cluster_op.fit_predict(phate_op.diff_potential)
    cluster_centers = np.array(
        [
            np.average(
                data[(cluster_ids == c)],
                axis=0,
                weights=labels[cluster_ids == c].sum(axis=1),
            )
            for c in range(n_clusters)
        ]
    )
    # Compute the cluster histograms C x M
    cluster_counts = np.array(
        [labels[(cluster_ids == c)].sum(axis=0) for c in range(n_clusters)]
    )
    cluster_dists = np.ascontiguousarray(
        pairwise_distances(cluster_centers, metric="euclidean")
    )

    N, M = labels.shape
    assert data.shape[0] == N
    dists = np.empty((M, M))
    for i in range(M):
        for j in range(i, M):
            weights_a = np.ascontiguousarray(cluster_counts[:, i])
            weights_b = np.ascontiguousarray(cluster_counts[:, j])
            dists[i, j] = dists[j, i] = ot.emd2(weights_a, weights_b, cluster_dists)
    neigh = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm="auto", metric="precomputed"
    )
    neigh.fit(dists)
    adj = neigh.kneighbors_graph()
    end = time.time()
    tot = end - start
    return adj, dists, tot, tot


def diffusion_emd(data, labels, n_neighbors=10):
    start = time.time()
    dc = DiffusionCheb()
    graph = graphtools.Graph(data, use_pygsp=True)
    embeddings = dc.fit_transform(graph.W, labels)
    # Calculate all pairwise? Or nearest neighbors?
    neigh = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm="auto", metric="manhattan"
    )
    neigh.fit(embeddings)
    adj = neigh.kneighbors_graph()
    end = time.time()
    # Only count time towards nearest neighbors, but
    dists = pairwise_distances(embeddings, metric="manhattan")
    end2 = time.time()
    return adj, dists, end - start, end2 - start


def tree_emd(
    data,
    labels,
    n_neighbors=10,
    n_clusters=5,
    n_levels=6,
    n_trees=1,
    random_state=42,
    **kwargs
):
    start = time.time()
    embed_list = []
    for i in range(n_trees):
        tree_op = MetricTree(
            tree_type="cluster",
            n_clusters=n_clusters,
            n_levels=n_levels,
            random_state=random_state + i,
            **kwargs
        )
        embeddings = tree_op.fit_embed(data, labels)
        embed_list.append(embeddings)
    embeddings = np.concatenate(embed_list, axis=1)
    neigh = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm="auto", metric="manhattan"
    )
    neigh.fit(embeddings)
    adj = neigh.kneighbors_graph()
    end = time.time()
    # Only count time towards nearest neighbors, but
    dists = pairwise_distances(embeddings, metric="manhattan")
    end2 = time.time()
    return adj, dists, end - start, end2 - start


def pairwise_distribution_distance(data, labels, distance_fn, n_neighbors=10):
    """Computes the pairwise distances between distributions given a specified distance
    function.

    Args:
        data: 2-D array N x F points by features.
        labels: 2-D array N x M points by distributions.
        distance_fn: function of (p, q, p_weights, q_weights) --> Distance

    Returns:
        distance_matrix: 2-D M x M array with each cell representing the
        distance between each distribution of points.
    """
    start = time.time()
    N, M = labels.shape
    assert data.shape[0] == N
    dists = np.empty((M, M))

    for i in range(M):
        for j in range(i, M):
            mask_a = labels[:, i] > 0
            mask_b = labels[:, j] > 0
            points_a = data[mask_a]
            points_b = data[mask_b]
            weights_a = labels[:, i][mask_a]
            weights_b = labels[:, j][mask_b]
            dists[i, j] = dists[j, i] = distance_fn(
                points_a, points_b, weights_a, weights_b
            )

    neigh = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm="auto", metric="precomputed"
    )
    neigh.fit(dists)
    adj = neigh.kneighbors_graph()
    end = time.time()
    tot = end - start
    return adj, dists, tot, tot


def pairwise_emd(data, labels, n_neighbors=10):
    """Computes pairwise EMD using the exact method."""
    return pairwise_distribution_distance(data, labels, exact, n_neighbors)


def pairwise_sinkhorn(data, labels, n_neighbors=10):
    """Computes pairwise EMD using the Sinkhorn method."""
    return pairwise_distribution_distance(data, labels, sinkhorn, n_neighbors)


def pairwise_mean_diff(data, labels, n_neighbors=10):
    """Computes pairwise EMD as a difference in means."""

    def mean_approx(p, q, p_weights, q_weights):
        p_mean = np.average(p, axis=0, weights=p_weights)
        q_mean = np.average(q, axis=0, weights=q_weights)
        return np.linalg.norm(p_mean - q_mean)

    return pairwise_distribution_distance(data, labels, mean_approx, n_neighbors)


def precision_at_k(pred, true, k=10):
    assert np.all(np.sum(true, axis=1) == k)
    assert np.all(np.sum(pred, axis=1) == k)
    return np.sum((pred + true) == 2) / (k * true.shape[0])


def corrs(d1, d2):
    """Average spearman correlation accross points."""
    spearman_corrs = []
    for i in range(len(d1)):
        correlation, pval = scipy.stats.spearmanr(d1[i], d2[i])
        spearman_corrs.append(correlation)
    spearman_corrs = np.array(spearman_corrs)
    return np.mean(spearman_corrs)


def evaluate(pred, true, ks=[1, 5, 10, 100, 500]):
    """
    Args:
        pred: dists
        true: dists
    returns:
        results: (p@K, spearmancorr
    """
    M = true.shape[0]
    neigh_pred = NearestNeighbors(
        n_neighbors=min(M - 1, ks[-1]), algorithm="auto", metric="precomputed"
    )
    neigh_true = NearestNeighbors(
        n_neighbors=min(M - 1, ks[-1]), algorithm="auto", metric="precomputed"
    )
    neigh_pred.fit(pred)
    neigh_true.fit(true)
    ps = []
    for k in ks:
        if k >= M:
            continue
        adj_pred = neigh_pred.kneighbors_graph(n_neighbors=k)
        adj_true = neigh_true.kneighbors_graph(n_neighbors=k)
        ps.append(precision_at_k(adj_pred, adj_true, k))
    c = corrs(pred, true)
    return (c, *ps)
