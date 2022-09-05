from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

import numpy as np
import ot
import phate


def phemd(data, labels, n_clusters=8, random_state=42):
    """Compute the PhEMD between distributions. As specified in Chen et al. 2019. Note
    that this reproduces the same steps but is not numerically the same.

    Args:
        data: 2-D array N x F points by features.
        labels: 2-D array N x M points by distributions.

    Returns:
        distance_matrix: 2-D M x M array with each cell representing the
        distance between each distribution of points.
    """
    phate_op = phate.PHATE(random_state=random_state)
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
    return dists


if __name__ == "__main__":
    import dataset

    ds = dataset.SklearnDataset(
        name="s_curve", n_distributions=100, n_points_per_distribution=20
    )
    labels = ds.labels
    labels /= labels.sum(0)
    dists = phemd(ds.X, labels)
    print(dists)
