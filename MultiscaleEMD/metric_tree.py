""" metric_tree.py
This file uses sklearn trees generally used for KNN calculation as an
approximate metric tree for wasserstein distance.  Further extensions are
quadtree, and one based on hierarchical clustering.  The idea is to use the
tree with edge lengths as the (L2) distance between means.  The distance
between any two points embedded in this tree is then the geodesic distance
along the tree.  Note that this is an offline algorithm, we do not support
adding points after the initial construction.
"""
from MultiscaleEMD.tree import BallTree
from MultiscaleEMD.tree import ClusterTree
from MultiscaleEMD.tree import KDTree
from MultiscaleEMD.tree import QuadTree
from scipy.sparse import coo_matrix
from scipy.sparse import hstack
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y
from typing import Optional

import numpy as np


class MetricTree(BaseEstimator):
    def __init__(
        self,
        tree_type="ball",
        metric="euclidean",
        return_sparse=False,
        random_state=None,
        **kwargs
    ):
        self.tree_type = tree_type
        self.tree_cls = self.parse_tree_cls()
        self.kwargs = kwargs
        self.metric = metric
        self.return_sparse = return_sparse
        self.random_state = random_state

    def parse_tree_cls(self):
        tree_type = self.tree_type
        if tree_type == "ball":
            return BallTree
        elif tree_type == "kd":
            return KDTree
        elif tree_type == "cluster":
            return ClusterTree
        elif tree_type == "quad":
            return QuadTree
        elif isinstance(tree_type, str):
            raise NotImplementedError("Unknown tree type")
        return tree_type

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True, multi_output=True)
        self.classes_ = y.shape[1]  # unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.tree = self.tree_cls(
            X, metric=self.metric, random_state=self.random_state, **self.kwargs
        )
        _, tree_indices, node_data, centers, dists = self.tree.get_arrays()
        y_indices = y[tree_indices]  # reorders point labels by tree order.
        self.edge_weights = dists
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
            if not self.return_sparse:
                self.counts_mtx = self.counts_mtx.toarray()
            return

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

        if not self.return_sparse:
            self.counts_mtx = self.counts_mtx.toarray()

    def fit_transform(self, X, y):
        """
        X is data array (np array)
        y is one-hot encoded distribution index (np array of size # points x #
        distributions.
        """
        self.fit(X, y)
        return self.transform(X, y)

    def fit_embed(self, X, y):
        print("return sparse", self.return_sparse)
        self.fit(X, y)
        return self.embed()

    def transform(self, X, y):
        """Transforms datasets y to (L1) vector space.

        Returns vectors representing edge weights and weights over vector.
        """
        check_is_fitted(self, ["X_", "y_"])
        if X[0, 0] != self.X_[0, 0]:
            raise ValueError("X transformed must equal fitted X")
        return self.counts_mtx, self.edge_weights

    def embed(self):
        check_is_fitted(self, ["X_", "y_"])
        if self.return_sparse:
            return self.counts_mtx.multiply(self.edge_weights)
        return self.counts_mtx * self.edge_weights

    def get_embeddings(self):
        return self.embed()

    def get_counts(self):
        return self.counts_mtx

    def get_weights(self):
        return self.edge_weights

    def get_arrays(self):
        return self.tree.get_arrays()


class MetricTreeCollection(MetricTree):
    def __init__(
        self,
        n_trees=1,
        tree_type="ball",
        metric="euclidean",
        manual_partition: Optional[np.ndarray] = None,
        return_sparse=False,
        random_state: int = 42,
        **kwargs
    ):
        # TODO allow non-integer random_states
        self.n_trees = n_trees
        self.manual_partition = None
        if self.manual_partition is not None:
            raise NotImplementedError("Top level partitioning is not yet implemented")
        self.unique_partitions = np.unique(manual_partition)
        super().__init__(tree_type, metric, return_sparse, random_state, **kwargs)

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True, multi_output=True)
        self.X_ = X
        self.y_ = y

        self.trees = [
            MetricTree(
                self.tree_type,
                self.metric,
                self.return_sparse,
                self.random_state + t,
                **self.kwargs
            )
            for t in range(self.n_trees)
        ]

        counts, weights = zip(*[mt.fit_transform(X, y) for mt in self.trees])
        if self.return_sparse:
            self.counts_mtx = hstack(counts)
        else:
            self.counts_mtx = np.hstack(counts)
        self.edge_weights = np.concatenate(weights)

    def get_node_data(self):
        """ Compute tree node metadata
        """
        check_is_fitted(self, ["X_", "y_"])
        arr = list(zip(*[tree.get_arrays() for tree in self.trees]))
        num_nodes_per_tree = [len(arr[-1][i]) for i in range(self.n_trees)]
        tree_id = np.array(
            [[i] * n for i, n in enumerate(num_nodes_per_tree)]
        ).flatten()
        tree_data, centers, dists = [np.concatenate(a, axis=0) for a in arr[2:]]

        parent_lists = []
        offset = 0
        for tree in self.trees:
            node_data = tree.get_arrays()[2]
            edge_idx = np.unique(np.array(list(zip(*node_data))[:1]))
            tmp = np.zeros(len(edge_idx), dtype=int)
            parents = []
            for j, node in enumerate(node_data):
                start, end = node[:2]
                parents.append(tmp[start == edge_idx][0])
                tmp[(start <= edge_idx) & (edge_idx < end)] = j
            parent_lists.append(np.array(parents) + offset)
            offset += len(parents)
        parents = np.concatenate((parent_lists), axis=0)
        is_root = tree_data[:, 2] == 0
        self.metadata = np.concatenate(
            [tree_data, tree_id[:, None], parents[:, None], is_root[:, None]], axis=1
        )
        return self.metadata, centers, dists


if __name__ == "__main__":
    n, d = 10, 2
    X = np.random.rand(n, d)
    labels = np.random.rand(10, 5) > 0.7

    mt = MetricTreeCollection(tree_type=ClusterTree, n_trees=2)
    mt.fit_transform(X, labels)
    mt.get_node_data()

    exit()

    for tree in [KDTree, BallTree, QuadTree, ClusterTree]:
        mt = MetricTree(tree_type=tree, return_sparse=True)
        counts, edge_weights = mt.fit_transform(X, labels)
        mt = MetricTree(tree_type=tree, return_sparse=False)
        counts, edge_weights = mt.fit_transform(X, labels)
    exit()
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
