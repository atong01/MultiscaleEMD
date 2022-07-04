from MultiscaleEMD.metric_tree import MetricTree
from MultiscaleEMD.metric_tree import MetricTreeCollection
from MultiscaleEMD.tree import BallTree
from MultiscaleEMD.tree import ClusterTree
from MultiscaleEMD.tree import KDTree
from MultiscaleEMD.tree import QuadTree
from numpy.testing import assert_array_equal

import numpy as np
import pytest
import scipy.sparse

n, d, ll, n_trees = 10, 2, 5, 2

trees = [
    ["kd", KDTree, {"leaf_size": 40}],
    ["ball", BallTree, {"leaf_size": 40}],
    ["kd", KDTree, {"leaf_size": 2}],
    ["ball", BallTree, {"leaf_size": 2}],
    ["quad", QuadTree, {}],
    ["cluster", ClusterTree, {}],
    ["cluster", ClusterTree, {"cluster_method": "random-kd"}],
]


@pytest.mark.parametrize("tree_type,tree,args", trees)
def test_default(tree_type, tree, args):
    X = np.random.rand(n, d)
    tree_obj = tree(X, **args)
    arr = tree_obj.get_arrays()
    assert len(arr) == 5
    x, indices, tree, centers, dists = arr
    assert_array_equal(x, X)

    assert len(indices) == len(x) == n
    assert isinstance(indices, np.ndarray)
    np.testing.assert_array_equal(np.sort(indices), np.arange(len(x)))

    # Check tree consistency
    assert len(tree) == len(centers) == len(dists)
    n_nodes = len(tree)
    assert isinstance(tree, np.ndarray)
    root_node = tree[0]
    assert root_node[0] == 0
    assert root_node[1] == len(x)
    assert root_node[2] == 0
    assert root_node[3] == 0 if n_nodes > 1 else root_node[3] == 1
    assert np.all(np.isfinite(tree))

    assert isinstance(centers, np.ndarray)
    assert len(centers.shape) == 2
    assert centers.shape[0] == n_nodes
    assert centers.shape[1] == d
    assert centers.dtype == float
    assert np.all(np.isfinite(centers))

    assert isinstance(dists, np.ndarray)
    assert len(dists.shape) == 1
    assert dists.shape[0] == n_nodes
    assert dists.dtype == float
    assert np.all(np.isfinite(dists))


def test_quadtree_out_of_bounds():
    x = -np.random.rand(10, 2)
    with pytest.raises(Exception):
        QuadTree(x)
    x = np.random.rand(10, 2) + 1
    with pytest.raises(Exception):
        QuadTree(x)


def test_metric_tree_sparse():
    tree_type, tree, args = trees[0]
    X = np.random.rand(n, d)
    labels = np.random.rand(10, 5) > 0.7
    mt_dense = MetricTree(tree_type=tree, return_sparse=False, **args)
    counts_dense, edge_weights_dense = mt_dense.fit_transform(X, labels)
    embeddings_dense = mt_dense.embed()

    mt_sparse = MetricTree(tree_type=tree, return_sparse=True, **args)
    counts_sparse, edge_weights_sparse = mt_sparse.fit_transform(X, labels)
    embeddings_sparse = mt_sparse.embed()

    np.testing.assert_array_equal(edge_weights_dense, edge_weights_sparse)
    assert isinstance(counts_dense, np.ndarray)
    assert isinstance(counts_sparse, scipy.sparse.csr_matrix)
    np.testing.assert_array_equal(counts_dense, counts_sparse.toarray())
    np.testing.assert_array_equal(embeddings_dense, embeddings_sparse.toarray())


labels_list = [
    np.random.rand(n, ll) > 0.7,
    np.random.rand(n, ll),
    scipy.sparse.csr_matrix(np.random.rand(n, ll) > 0.7),
    scipy.sparse.csr_matrix((np.random.rand(n, ll) > 0.7).astype(float)),
    scipy.sparse.coo_matrix((np.random.rand(n, ll) > 0.7).astype(float)),
    scipy.sparse.coo_matrix(np.random.rand(n, ll).astype(float)),
    scipy.sparse.csc_matrix(np.random.rand(n, ll).astype(float)),
]


# TODO test continuously valued lavels to extend to non-pointcloud data
@pytest.mark.parametrize("tree_type,tree,args", trees)
@pytest.mark.parametrize("labels", labels_list)
def test_metric_tree(tree_type, tree, args, labels):
    X = np.random.rand(n, d)
    mt = MetricTree(tree_type=tree, return_sparse=False, **args)
    embeddings = mt.fit_embed(X, labels)
    counts = mt.get_counts()
    weights = mt.get_weights()
    tree_data = mt.tree.get_arrays()[2]
    n_nodes = len(tree_data)
    assert counts.shape == (ll, n_nodes)
    assert embeddings.shape == (ll, n_nodes)
    np.testing.assert_array_equal(embeddings, counts * weights)
    labels = labels if isinstance(labels, np.ndarray) else labels.toarray()
    np.testing.assert_allclose(counts[:, 0], labels.sum(axis=0))


@pytest.mark.parametrize("tree_type,tree,args", trees)
@pytest.mark.parametrize("labels", labels_list)
def test_metric_collection(tree_type, tree, args, labels):
    X = np.random.rand(n, d)
    mt = MetricTreeCollection(n_trees=n_trees, tree_type=tree, **args)
    embeddings = mt.fit_embed(X, labels)
    counts = mt.get_counts()
    weights = mt.get_weights()
    n_nodes = counts.shape[1]
    assert counts.shape == (ll, n_nodes)
    assert embeddings.shape == (ll, n_nodes)
    np.testing.assert_array_equal(embeddings, counts * weights)
    labels = labels if isinstance(labels, np.ndarray) else labels.toarray()
    np.testing.assert_allclose(counts[:, 0], labels.sum(axis=0))


def test_get_node_data():
    tree_type, tree, args = trees[0]
    labels = labels_list[0]
    X = np.random.rand(n, d)
    mt = MetricTreeCollection(n_trees=n_trees, tree_type=tree, **args)
    embeddings = mt.fit_embed(X, labels)
    counts = mt.get_counts()
    node_data, centers, dists = mt.get_node_data()
    arrs = [node_data, centers, dists, counts.T, embeddings.T]
    lens = np.array([len(a) for a in arrs])
    np.testing.assert_array_equal(lens, np.ones_like(lens) * lens[0])
