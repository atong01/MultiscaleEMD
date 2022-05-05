from MultiscaleEMD import DiffusionCheb
from sklearn.neighbors import NearestNeighbors

import graphtools


def diffusion_emd(data, labels):
    dc = DiffusionCheb()
    graph = graphtools.Graph(data, use_pygsp=True)
    embeddings = dc.fit_transform(graph.W, labels)
    # Calculate all pairwise? Or nearest neighbors?
    neigh = NearestNeighbors(n_neighbors=10, algorithm="auto", metric="manhattan")
    neigh.fit(embeddings)
    adj = neigh.kneighbors_graph()
    return adj


if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler

    import dataset

    ds = dataset.SklearnDataset(
        name="s_curve", n_distributions=100, n_points_per_distribution=20
    )
    labels = ds.labels
    labels /= labels.sum(0)

    X = ds.X
    X_std = StandardScaler().fit_transform(X)
    print(diffusion_emd(X_std, labels))
