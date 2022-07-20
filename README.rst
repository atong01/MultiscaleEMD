Implementation of Multiscale EMD Methods
===============================

Multiscale Earth Mover's Distances embeds the Wasserstein distance between two distributions into `L^1`. For each distribution we build an embedding where the `L^1` distance between embeddings equivalent to the Earth Mover's Distance between distributions. This creates a geometry between distributions which can be exploited to find EMD-nearest-neighbors in sub-linear time.

We offer two main types of MultiscaleEMDs at the moment:

* DiffusionEMD which embeds the Wasserstein distance between two distributions on a graph approximately into `L^1` in log-linear time.
* TreeEMD / Trellis which embeds the Wasserstein distance between distributions over a tree into `L^1`.

Installation
------------

MultiscaleEMD is available in `pypi`. Install by running the following::

    pip install MultiscaleEMD

Quick Start
-----------

MultiscaleEMD is written following the `sklearn` estimator framework.

For DiffusionEMD: We provide two functions that operate quite differently. First the Chebyshev approxiamtion of the operator in `DiffusionCheb`, which we recommend when the number of distributions is small compared to the number of points. Second, the Interpolative Decomposition method that computes dyadic powers of $P^{2^k}$ directly in `DiffusionTree`. These two classes are used in the same way, first supplying parameters, fitting to a graph and array of distributions::

    import numpy as np
    from DiffusionEMD import DiffusionCheb

    # Setup an adjacency matrix and a set of distributions to embed
    adj = np.ones((10, 10))
    distributions = np.random.randn(10, 5)
    dc = DiffusionCheb()

    # Embeddings where the L1 distance approximates the Earth Mover's Distance
    embeddings = dc.fit_transform(adj, distributions)
    # Shape: (5, 60)

For Tree Earth Mover's Distances and Trellis: we provide a number of ways to embed pointcloud data in `R^d` into a hierarchical tree. These are implemented as options in the `MetricTree` class.

Requirements can be found in `requirements.txt`

Examples
--------

Examples are in the `notebooks` directory.

Take a look at the examples provided there to get a sense of how the parameters
behave on simple examples that are easy to visualize.

Paper
-----

This code implements the algorithms described in this paper:

ArXiv Link: http://arxiv.org/abs/2102.12833::

    @InProceedings{pmlr-v139-tong21a,
      title =       {Diffusion Earth Mover’s Distance and Distribution Embeddings},
      author =      {Tong, Alexander and Huguet, Guillaume and Natik, Amine and Macdonald, Kincaid and Kuchroo, Manik and Coifman, Ronald and Wolf, Guy and Krishnaswamy, Smita},
      booktitle =   {Proceedings of the 38th International Conference on Machine Learning},
      pages = 	    {10336--10346},
      year = 	    {2021},
      editor = 	    {Meila, Marina and Zhang, Tong},
      volume = 	    {139},
      series = 	    {Proceedings of Machine Learning Research},
      month = 	    {18--24 Jul},
      publisher =   {PMLR},
      pdf = 	    {http://proceedings.mlr.press/v139/tong21a/tong21a.pdf},
      url = 	    {http://proceedings.mlr.press/v139/tong21a.html},
    }

And this paper:

ArXiv Link: https://arxiv.org/abs/2107.12334::

    @inproceedings{tong_embedding_2022,
      author={Tong, Alexander and Huguet, Guillaume and Shung, Dennis and Natik, Amine and Kuchroo, Manik and Lajoie, Guillaume and Wolf, Guy and Krishnaswamy, Smita},
      booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      title={Embedding Signals on Graphs with Unbalanced Diffusion Earth Mover&#x2019;s Distance},
      year={2022},
      volume={},
      number={},
      pages={5647-5651},
      doi={10.1109/ICASSP43922.2022.9746556}
    }

As well as other algorithms under development.
