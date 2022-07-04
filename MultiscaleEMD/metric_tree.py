""" metric_tree.py
This file uses sklearn trees generally used for KNN calculation as an
approximate metric tree for wasserstein distance.  Further extensions are
quadtree, and one based on hierarchical clustering.  The idea is to use the
tree with edge lengths as the (L2) distance between means.  The distance
between any two points embedded in this tree is then the geodesic distance
along the tree.  Note that this is an offline algorithm, we do not support
adding points after the initial construction.
"""
from .tree import ClusterTree
from .tree import QuadTree
from scipy.sparse import coo_matrix
from sklearn.base import BaseEstimator
from sklearn.neighbors import BallTree
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import KDTree
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y

import itertools
import numpy as np


class MetricTree(BaseEstimator):
    def __init__(
        self,
        tree_type="ball",
        leaf_size=40,
        metric="euclidean",
        random_state=None,
        **kwargs
    ):
        self.tree_type = tree_type
        if tree_type == "ball":
            self.tree_cls = BallTree
        elif tree_type == "kd":
            self.tree_cls = KDTree
        elif tree_type == "cluster":
            self.tree_cls = ClusterTree
        elif tree_type == "quad":
            self.tree_cls = QuadTree
        else:
            raise NotImplementedError("Unknown tree type")
        self.kwargs = kwargs
        self.leaf_size = leaf_size
        self.metric = metric
        self.dist_fn = DistanceMetric.get_metric(metric)
        self.random_state = random_state

    def get_node_weights(self):
        """Takes the middle of the bounds as the node center for each node
        TODO (alex): This could be improved or at least experimented with
        """
        node_weights = self.tree.get_arrays()[-1]
        if self.tree_type == "ball":
            centers = node_weights[0]
            n = centers.shape[0]
            # Subtracts the child from the parent relying on the order of nodes in the tree
            lengths = np.linalg.norm(
                centers[np.insert(np.arange(n - 1) // 2, 0, 0)] - centers[np.arange(n)],
                axis=1,
            )
            return lengths
        elif self.tree_type == "kd":
            # Averages the two boundaries of the KD box
            centers = node_weights.mean(axis=0)
            n = centers.shape[0]
            # Subtracts the child from the parent relying on the order of nodes in the tree
            lengths = np.linalg.norm(
                centers[np.insert(np.arange(n - 1) // 2, 0, 0)] - centers[np.arange(n)],
                axis=1,
            )
            return lengths
        elif self.tree_type == "cluster":
            return node_weights
        elif self.tree_type == "quad":
            return node_weights
        else:
            raise NotImplementedError("Unknown tree type")

    def fit_transform(self, X, y):
        """
        X is data array (np array)
        y is one-hot encoded distribution index (np array of size # points x #
        distributions.
        """
        X, y = check_X_y(X, y, accept_sparse=True, multi_output=True)
        self.classes_ = y.shape[1]  # unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.tree = self.tree_cls(
            X,
            leaf_size=self.leaf_size,
            metric=self.metric,
            random_state=self.random_state,
            **self.kwargs
        )
        tree_indices = self.tree.get_arrays()[1]
        node_data = self.tree.get_arrays()[2]
        y_indices = y[tree_indices]  # reorders point labels by tree order.

        self.edge_weights = self.get_node_weights()
        counts = np.empty((len(node_data), y.shape[1]))
        for node_idx in reversed(range(len(node_data))):
            start, end, is_leaf, radius = node_data[node_idx]

            # Find the number of points present in this range from each distribution
            counts[node_idx] = np.sum(
                y_indices[start:end], axis=0
            )  # as y is a one-hot encoding, we just need to sum over the relevant bits.

        if np.issubdtype(y.dtype, np.floating):
            # if is floating then don't worry about the logic below
            self.counts_mtx = coo_matrix(counts).T
            return self.counts_mtx, self.edge_weights

        # convert to COO format
        dim = (self.classes_, len(node_data))
        dist_list = np.arange(1, self.classes_ + 1)
        self.counts_mtx = coo_matrix(dim, dtype=np.int32)
        for i, count in enumerate(counts):
            if np.sum(count) == 0:  # if no classes have signals in this region
                continue
            # get the signals with nonzero representation in the region
            # count is a list of the representation per distribution.
            # count_copy is used to eliminate distributions without representation
            count_copy = count.copy()
            count_copy[count_copy > 0] = 1
            dists_represented = np.multiply(dist_list, count_copy)
            j_list = (
                dists_represented[dists_represented != 0] - 1
            )  # we added 1 to the distribution numbers to do the zero trick.
            val_list = count[count != 0]
            i_list = [i] * len(j_list)
            self.counts_mtx += coo_matrix(
                (val_list, (j_list, i_list)), shape=dim, dtype=np.int32
            )

        return self.counts_mtx, self.edge_weights

    def transform(self, X):
        """Transforms datasets y to (L1) vector space.

        Returns vectors representing edge weights and weights over vector.
        """
        check_is_fitted(self, "X_")

        if X != self.X_:
            raise ValueError("X transformed must equal fitted X")


if __name__ == "__main__":
    mt = MetricTree(
        tree_type="cluster", cluster_method="random-kd", n_clusters=4, n_levels=4
    )
    gt = np.repeat(np.arange(10), 100)
    gt = (
        (np.repeat(np.arange(max(gt) + 1)[:, None], len(gt), axis=1) == gt)
        .astype(int)
        .T
    )
    for n in [100, 1000, 10000]:
        counts, edge_weights = mt.fit_transform(
            X=np.random.random_sample((1000, 30)), y=gt
        )
    # print(counts, edge_weights)
    print(counts.sum(axis=0))
    # print(counts.toarray()[:50])
    print(mt.tree.centers)
