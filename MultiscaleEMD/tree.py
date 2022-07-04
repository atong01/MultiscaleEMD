from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import BallTree as BT
from sklearn.neighbors import KDTree as KDT

import itertools
import numpy as np


def collate_node_data(node_data):
    """
    each node knows the range of nodes that are in it stored as [start, end]
    indicies over some sorted index to find the parent.
    """
    # unique edge indices in the tree sorted
    edge_idx = np.unique(np.concatenate(list(zip(*node_data))[:1]))
    parent_index = np.zeros(len(edge_idx), dtype=int)
    level_index = np.zeros(len(edge_idx), dtype=int)
    parents = []
    levels = []
    for j, node in enumerate(node_data):
        start, end = node[:2]
        parents.append(parent_index[start == edge_idx][0])
        levels.append(level_index[start == edge_idx][0])
        parent_index[(start <= edge_idx) & (edge_idx < end)] = j
        level_index[(start <= edge_idx) & (edge_idx < end)] += 1
    return np.array(parents), np.array(levels)


class KDTree(KDT):
    def get_arrays(self):
        X, index, node_data, node_bounds = super().get_arrays()
        # Averages the two boundaries of the KD box
        centers = node_bounds.mean(axis=0)
        n = centers.shape[0]
        # Subtracts the child from the parent relying on the order of nodes in the tree
        dists = np.linalg.norm(
            centers[np.insert(np.arange(n - 1) // 2, 0, 0)] - centers[np.arange(n)],
            axis=1,
        )
        node_data = np.array(list(zip(*node_data))).astype(int).T
        node_data[:, 3] = node_data[:, 2]
        node_data[:, 2] = collate_node_data(node_data)[1]
        return X, index, node_data, centers, dists


class BallTree(BT):
    def get_arrays(self):
        X, index, node_data, node_bounds = super().get_arrays()
        centers = node_bounds[0]
        n = centers.shape[0]
        # Subtracts the child from the parent relying on the order of nodes in the tree
        dists = np.linalg.norm(
            centers[np.insert(np.arange(n - 1) // 2, 0, 0)] - centers[np.arange(n)],
            axis=1,
        )
        node_data = np.array(list(zip(*node_data))).astype(int).T
        node_data[:, 3] = node_data[:, 2]
        node_data[:, 2] = collate_node_data(node_data)[1]
        return X, index, node_data, centers, dists


class QuadTree(object):
    """
    This quadtree could be sped up, but is an easy implementation
    """

    def __init__(self, X, n_levels=25, noise=1.0, normalize=False, *args, **kwargs):
        assert np.all(np.min(X, axis=0) >= 0)
        assert np.all(np.max(X, axis=0) <= 1)
        assert n_levels >= 1
        # Normalize to [0,1]
        # self.scaler = MinMaxScaler()
        # self.X = self.scaler.fit_transform(X)
        self.X = X
        self.kwargs = kwargs
        self.noise = noise
        # self.X = self.X + np.random.randn(*self.X.shape) * noise
        self.dims = X.shape[1]
        self.n_clusters = 2 ** self.dims
        self.n_levels = n_levels
        center = np.random.rand(self.dims) * noise
        self.tree, self.indices, self.centers, self.dists = self._cluster(
            center, np.arange(X.shape[0]), n_levels=self.n_levels - 1, start=0
        )
        self.indices = np.array(self.indices)
        self.tree = np.array([(0, self.X.shape[0], n_levels, 0), *self.tree])
        self.tree[:, 2] = n_levels - self.tree[:, 2]
        self.dists = np.array([0, *self.dists])
        self.centers = np.array([center, *self.centers])
        self.centers = np.array(self.centers)

    def _cluster(self, center, index, n_levels, start):
        """
        Parameters
        ----------

        bounds:
            [2 x D] matrix giving min / max of bounding box for this cluster

        """
        if n_levels == 0 or len(index) == 0:
            return None
        labels = np.ones_like(index) * -1
        dim_masks = np.array([self.X[index, d] > center[d] for d in range(self.dims)])

        bin_masks = np.array(list(itertools.product([False, True], repeat=self.dims)))
        label_masks = np.all(bin_masks[..., None] == dim_masks[None, ...], axis=1)
        for i, mask in enumerate(label_masks):
            labels[mask] = i
        assert np.all(labels > -1)
        shift = 2 ** -(self.n_levels - n_levels + 2)
        shifts = np.array(list(itertools.product([-shift, shift], repeat=self.dims)))
        cluster_centers = shifts + center
        sorted_index = []
        children = []
        ccenters = []
        cdists = []
        is_leaf = [0] * self.n_clusters
        unique, ucounts = np.unique(labels, return_counts=True)
        counts = np.zeros(self.n_clusters, dtype=np.int32)
        for u, c in zip(unique, ucounts):
            counts[u] = c
        cstart = 0
        for i, count, ccenter in zip(unique, counts, cluster_centers):
            ret = self._cluster(
                ccenter, index[labels == i], n_levels - 1, start + cstart
            )
            if ret is None:
                sorted_index.extend(index[labels == i])
                is_leaf[i] = 1
                continue
            sorted_index.extend(ret[1])
            children.extend(ret[0])
            ccenters.extend(ret[2])
            cdists.extend(ret[3])
            cstart += count

        to_return = list(
            zip(
                *[
                    np.array([0, *np.cumsum(counts)]) + start,
                    np.cumsum(counts) + start,
                    [n_levels] * self.n_clusters,
                    is_leaf,
                ]
            )
        )
        dists = np.linalg.norm(cluster_centers - center[None, :], axis=1)
        return (
            [*to_return, *children],
            sorted_index,
            [*cluster_centers, *ccenters],
            [*dists, *cdists],
        )

    def get_arrays(self):
        return self.X, self.indices, self.tree, self.centers, self.dists


class ClusterMethod(object):
    def __init__(self, n_clusters, random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        # labels_ must be available after calling fit,
        self.labels_ = None
        self.cluster_centers_ = None
        self.is_fit = False

    def fit(self, data):
        self.is_fit = True

    def predict(self):
        if not self.is_fit:
            raise ValueError("Cannot predict before fit")
        return self.labels_, self.cluster_centers_


class RandomSplit(ClusterMethod):
    """
    Pick a random dimension and split on it
    """

    def __init__(self, n_clusters, dims_per_split, random_state=None):
        super().__init__(n_clusters=n_clusters, random_state=random_state)
        self.dims_per_split = dims_per_split

    def fit(self, data):
        super().fit(data)
        variance = np.var(data, axis=0)
        normed_var = variance / np.sum(variance)
        # chose dimension according to variance
        split_dims = self.rng.choice(
            data.shape[1], size=self.dims_per_split, replace=False, p=normed_var
        )
        # Location of split determined randomly between min and max of dimension
        sub_data = data[:, split_dims]
        min_data = np.min(sub_data, axis=0)
        max_data = np.max(sub_data, axis=0)
        split_point = self.rng.uniform(size=2) * (max_data - min_data) + min_data

        labels = np.ones(sub_data.shape[0], dtype=int) * -1
        dim_masks = np.array(
            [sub_data[:, d] > split_point[d] for d in range(self.dims_per_split)]
        )
        bin_masks = np.array(
            list(itertools.product([False, True], repeat=self.dims_per_split))
        )
        label_masks = np.all(bin_masks[..., None] == dim_masks[None, ...], axis=1)
        centers = []
        for i, mask in enumerate(label_masks):
            labels[mask] = i
            if sum(mask) == 0:
                centers.append(np.full(data.shape[1], np.nan))
            else:
                centers.append(np.mean(data[mask], axis=0))

        assert np.all(labels > -1)
        self.labels_ = labels
        print(labels)
        self.cluster_centers_ = np.array(centers)
        self.is_fit = True


class ClusterTree(object):
    def __init__(
        self,
        X,
        leaf_size=40,
        n_clusters=4,
        n_levels=4,
        cluster_method="kmeans",
        random_state=None,
        *args,
        **kwargs
    ):
        self.X = X
        self.leaf_size = min(leaf_size, X.shape[0])
        self.n_clusters = n_clusters
        self.n_levels = n_levels
        self.cluster_method = cluster_method
        if self.cluster_method == "random-kd":
            self.dims_per_split = np.floor(np.log2(self.n_clusters))
            if self.dims_per_split != np.ceil(np.log2(self.n_clusters)):
                raise ValueError(
                    "n_clusters must be power of two when using random-kd method"
                )
        if self.n_levels < 2:
            raise ValueError("n_levels must be at least 2 for a non-trivial tree")
        self.random_state = random_state
        center = self.X.mean(axis=0)
        self.tree, self.indices, self.centers, self.dists = self._cluster(
            center, np.arange(X.shape[0]), n_levels=self.n_levels - 1, start=0
        )
        self.indices = np.array(self.indices)
        self.tree = np.array(
            [(0, self.X.shape[0], n_levels, n_levels == 1), *self.tree]
        )
        self.tree[:, 2] = n_levels - self.tree[:, 2]
        self.centers = np.array([center, *self.centers])
        self.dists = np.array([0, *self.dists])
        self.prune()

    def prune(self):
        empty_mask = (self.tree[:, 0] != self.tree[:, 1])
        self.tree = self.tree[empty_mask]
        self.centers = self.centers[empty_mask]
        self.dists = self.dists[empty_mask]


    def parse_cluster_method(self):
        if self.cluster_method == "kmeans":
            cl = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                init="random",
                max_iter=10,
                n_init=1,
                random_state=self.random_state,
            )
            return cl
        elif self.cluster_method == "random-kd":
            cl = RandomSplit(
                n_clusters=self.n_clusters,
                dims_per_split=2,
                random_state=self.random_state,
            )
            return cl

    def _cluster(self, center, index, n_levels, start):
        """
        Returns a list of tuples corresponding to each subnode of the tree
        (center, level, start, end, is_leaf), sorted_index
        center is the cluster center
        level is the level of the node counting the root as the zeroth level
        sorted_index is athe list of
        """
        if n_levels == 0 or len(index) < self.n_clusters or len(index) < self.leaf_size:
            return None
        cl = self.parse_cluster_method()
        cl.fit(self.X[index])
        sorted_index = []
        children = []
        ccenters = []
        cdists = []
        is_leaf = [0] * self.n_clusters
        unique, ucounts = np.unique(cl.labels_, return_counts=True)
        counts = np.zeros(self.n_clusters, dtype=np.int32)
        for u, c in zip(unique, ucounts):
            counts[u] = c
        cstart = 0
        for i, count in zip(unique, counts):
            sub_mask = cl.labels_ == i
            ret = self._cluster(
                cl.cluster_centers_[i], index[sub_mask], n_levels - 1, start + cstart,
            )
            if ret is None:
                # then the subcluster should be a leaf and is not subclustered
                sorted_index.extend(index[cl.labels_ == i])
                is_leaf[i] = 1
                continue
            sorted_index.extend(ret[1])
            children.extend(ret[0])
            ccenters.extend(ret[2])
            cdists.extend(ret[3])
            cstart += count
        to_return = list(
            zip(
                *[
                    np.array([0, *np.cumsum(counts)]) + start,
                    np.cumsum(counts) + start,
                    [n_levels] * self.n_clusters,
                    is_leaf,
                ]
            )
        )
        dists = np.linalg.norm(cl.cluster_centers_ - center[None, :], axis=1)
        return (
            [*to_return, *children],
            sorted_index,
            [*cl.cluster_centers_, *ccenters],
            [*dists, *cdists],
        )

    def get_arrays(self):
        return self.X, self.indices, self.tree, self.centers, self.dists


if __name__ == "__main__":
    from pprint import pprint

    n = 10
    x = np.arange(2 * n).reshape(n, 2) / (2 * n)
    tree_obj = ClusterTree(x, leaf_size=40, cluster_method="random-kd")
    arr = tree_obj.get_arrays()
    pprint(arr[2])
    pprint(tree_obj.centers)
    """

    tree_obj = KDTree(x, leaf_size=2)
    arr = tree_obj.get_arrays()
    pprint(np.array(arr[2]))
    print(arr[3])

    tree_obj = BallTree(x)
    arr = tree_obj.get_arrays()
    pprint(arr[2])

    tree_obj = ClusterTree(x, leaf_size=40)
    arr = tree_obj.get_arrays()
    pprint(arr[2])

    tree_obj = QuadTree(x, n_levels=4)
    arr = tree_obj.get_arrays()
    pprint(arr[2])
    # print(type(arr[2]))
    """
