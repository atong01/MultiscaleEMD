Implementation of Multiscale EMD Methods
========================================

Multiscale Earth Mover's Distances embeds the Wasserstein distance between two distributions into $L^1$. For each distribution we build an embedding where the $L^1$ distance between embeddings equivalent to the Earth Mover's Distance between distributions. This creates a geometry between distributions which can be exploited to find EMD-nearest-neighbors in sub-linear time.

We offer two main types of MultiscaleEMDs at the moment:

* `Diffusion EMD <http://arxiv.org/abs/2102.12833>`_ which embeds the Wasserstein distance between two distributions on a graph approximately into $L^1$ in log-linear time.
* TreeEMD / `Trellis <https://www.biorxiv.org/content/10.1101/2022.10.19.512668v1>`_ which embeds the Wasserstein distance between distributions over a tree exactly into $L^1$. TreeEMD / Trellis also provides utilities for building a tree over data in represented in $\\mathbb{R}^d$ using divisive hierarchical clustering. Where TreeEMD computes the Wasserstein distance, Trellis extends this to the Kantorovich-Rubenstein distance between treatment distribution changes.

These EMDs can also easily be extended to Kantorovich-Rubenstein (KR) norms between signals over the graph which do not sum to 1. As in the Trellis paper, subtracting a "control" vectors may prove useful in removing confounders under certain assumptions on the data generating process. This allows for more general treatment of data with multiple controls matched to different batches of data. For an example of this see the :code:`notebooks/Trellis-Embedding-Comparison.ipynb` notebook comparing "Trellis" to "Paired-Trellis", which subtracts out the control density vectors. 

Installation
------------

MultiscaleEMD is available in `pypi`. Install by running the following

.. code-block:: bash

    pip install MultiscaleEMD

Quick Start
-----------

MultiscaleEMD is written following the `sklearn` estimator framework.

For DiffusionEMD: We provide two functions that operate quite differently. First the Chebyshev approximation of the operator in `DiffusionCheb`, which we recommend when the number of distributions is small compared to the number of points. Second, the Interpolative Decomposition method that computes dyadic powers of $P^{2^k}$ directly in `DiffusionTree`. These two classes are used in the same way, first supplying parameters, fitting to a graph and array of distributions

.. code-block:: python

    import numpy as np
    from MultiscaleEMD import DiffusionCheb

    # Setup an adjacency matrix and a set of distributions to embed
    adj = np.ones((10, 10))
    distributions = np.random.randn(10, 5)
    dc = DiffusionCheb()

    # Embeddings where the L1 distance approximates the Earth Mover's Distance
    embedding = dc.fit_transform(adj, distributions)
    # Shape: (5, 60)

For Tree Earth Mover's Distances and Trellis: we provide a number of ways to embed pointcloud data in $\\mathbb{R}^d$ into a hierarchical tree. These are implemented as options in the `MetricTree` class.

.. code-block:: python

    from MultiscaleEMD.metric_tree import MetricTreeCollection
    
    mt = MetricTreeCollection(n_trees=10, tree_type="cluster", n_levels=4, n_clusters=4)
    embedding = mt.fit_embed(data, distributions)
    

Requirements can be found in :code:`requirements.txt`

Examples
--------

Examples are in the :code:`notebooks` directory.

Take a look at the examples provided there to get a sense of how the parameters
behave on simple examples that are easy to visualize.

Papers
------

This code implements the algorithms described in the following papers:

1. `Diffusion EMD <http://arxiv.org/abs/2102.12833>`_ (ICML 2021)
2. `Unbalanced Diffusion EMD <https://arxiv.org/abs/2107.12334>`_ (ICASSP 2022)
3. `Trellis <https://www.biorxiv.org/content/10.1101/2022.10.19.512668v1>`_ (Preprint 2022)

For bibtex see below:

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

BioRXiv Link: https://www.biorxiv.org/content/10.1101/2022.10.19.512668v1::

    @article {Ramos Zapatero2022.10.19.512668,
        author = {Ramos Zapatero, Maria and Tong, Alexander and Sufi, Jahangir and Vlckova, Petra and Cardoso Rodriguez, Ferran and Nattress, Callum and Qin, Xiao and Hochhauser, Daniel and Krishnaswamy, Smita and Tape, Christopher J},
        title = {Cancer-Associated Fibroblasts Regulate Patient-Derived Organoid Drug Responses},
        elocation-id = {2022.10.19.512668},
        year = {2022},
        doi = {10.1101/2022.10.19.512668},
        publisher = {Cold Spring Harbor Laboratory},
        URL = {https://www.biorxiv.org/content/early/2022/10/21/2022.10.19.512668},
        eprint = {https://www.biorxiv.org/content/early/2022/10/21/2022.10.19.512668.full.pdf},
        journal = {bioRxiv}
    }

As well as other algorithms under development.
