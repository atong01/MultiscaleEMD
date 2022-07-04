from MultiscaleEMD.tree import BallTree
from MultiscaleEMD.tree import ClusterTree
from MultiscaleEMD.tree import KDTree
from MultiscaleEMD.tree import QuadTree
from numpy.testing import assert_array_equal

import numpy as np
import pytest

trees = [
    pytest.param("kd", KDTree, {"leaf_size": 40}, marks=pytest.mark.xfail),
    ["kd", KDTree, {"leaf_size": 2}],
    pytest.param("ball", BallTree, {"leaf_size": 40}, marks=pytest.mark.xfail),
    ["ball", BallTree, {"leaf_size": 2}],
    ["quad", QuadTree, {}],
    ["kmeans", ClusterTree, {}],
    ["kmeans", ClusterTree, {"cluster_method": "random-kd"}],
]


@pytest.mark.parametrize("name,tree,args", trees)
def test_default(name, tree, args):
    n, d = 10, 2
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
    assert root_node[3] == 0

    assert isinstance(centers, np.ndarray)
    assert len(centers.shape) == 2
    assert centers.shape[0] == n_nodes
    assert centers.shape[1] == d
    assert centers.dtype == float

    assert isinstance(dists, np.ndarray)
    assert len(dists.shape) == 1
    assert dists.shape[0] == n_nodes
    assert dists.dtype == float


def test_quadtree_out_of_bounds():
    x = -np.random.rand(10, 2)
    with pytest.raises(Exception):
        QuadTree(x)
    x = np.random.rand(10, 2) + 1
    with pytest.raises(Exception):
        QuadTree(x)
