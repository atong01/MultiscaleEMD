from methods import diffusion_emd
from methods import evaluate
from methods import pairwise_emd
from methods import pairwise_mean_diff
from methods import pairwise_sinkhorn
from methods import phemd
from methods import tree_emd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import dataset
import itertools
import pandas as pd


def run_ablation(
    seeds=5,
    dataset_name="swiss_roll",
    n_neighbors=10,
    n_clusters=4,
    n_levels=4,
    n_trees=1,
    clustering_methods="kmeans",
):
    methods = {
        "Exact": pairwise_emd,
        "TreeEMD": tree_emd,
    }
    if isinstance(n_neighbors, int):
        n_neighbors = [n_neighbors]
    if isinstance(n_clusters, int):
        n_clusters = [n_clusters]
    if isinstance(n_levels, int):
        n_levels = [n_levels]
    if isinstance(n_trees, int):
        n_trees = [n_trees]
    if isinstance(clustering_methods, str):
        clustering_methods = [clustering_methods]
    ks = [1, 5, 10, 25, 50]
    # dataset_name = "s_curve"
    n_distributions_list = [100]  # , 50, 100]
    n_points_per_distribution = 20
    version = "0.0.3"

    iterator = itertools.product(
        range(seeds),
        n_distributions_list,
        n_neighbors,
        n_clusters,
        n_levels,
        n_trees,
        clustering_methods,
    )
    results2 = []
    exact_results = {}
    for args in tqdm(list(iterator)):
        seed, n_distributions, neighbors, clusters, levels, trees, cmethod = args

        ds = dataset.SklearnDataset(
            name=dataset_name,
            n_distributions=n_distributions,
            n_points_per_distribution=n_points_per_distribution,
            random_state=42 + seed,
        )
        labels = ds.labels
        labels /= labels.sum(0)
        X = ds.X
        X_std = StandardScaler().fit_transform(X)
        results = tree_emd(
            X_std,
            labels,
            n_neighbors=neighbors,
            n_clusters=clusters,
            n_levels=levels,
            n_trees=trees,
            cluster_method=cmethod,
        )
        if n_distributions not in exact_results:
            exact_results[n_distributions] = pairwise_emd(
                X_std, labels, n_neighbors=neighbors
            )

        results2.append(
            (
                "TreeEMD",
                *args,
                *evaluate(results[1], exact_results[n_distributions][1], ks=ks),
                *results[-2:],
            )
        )
    df = pd.DataFrame(
        results2,
        columns=[
            "Method",
            "Seed",
            "# distributions",
            "# Neighbors",
            "# Clusters",
            "# levels",
            "# trees",
            "Clustering Method",
            "SpearmanR",
            *[f"P@{k}" for k in ks],
            "10-NN time (s)",
            "All-pairs time(s)",
        ],
    )
    df.to_pickle(
        f"results_{dataset_name}_{n_points_per_distribution}_{seeds}_{n_clusters}_{n_levels}_{n_trees}_{clustering_methods}_{version}.pkl"
    )
    return df


def run_sklearn_test(seeds=5):
    methods = {
        "DiffusionEMD": diffusion_emd,
        "PhEMD": phemd,
        "Exact": pairwise_emd,
        "Sinkhorn": pairwise_sinkhorn,
        "Mean": pairwise_mean_diff,
        "TreeEMD": tree_emd,
    }
    n_neighbors = 10
    ks = [1, 5, 10, 25]
    dataset_name = "swiss_roll"
    # dataset_name = "s_curve"
    n_distributions_list = [25, 75, 150, 200]  # , 50, 100]
    n_points_per_distribution = 20
    version = "0.0.1"
    results2 = []
    for seed in range(seeds):
        for n_distributions in n_distributions_list:
            ds = dataset.SklearnDataset(
                name=dataset_name,
                n_distributions=n_distributions,
                n_points_per_distribution=n_points_per_distribution,
                random_state=42 + seed,
            )
            labels = ds.labels
            labels /= labels.sum(0)
            X = ds.X
            X_std = StandardScaler().fit_transform(X)

            results = {}
            for name, fn in methods.items():
                results.update({name: fn(X_std, labels, n_neighbors=n_neighbors)})
                print(f"{name} with M={n_distributions} took {results[name][-1]:0.2f}s")

            for name, res in results.items():
                results2.append(
                    (
                        name,
                        seed,
                        n_distributions,
                        *evaluate(res[1], results["Exact"][1], ks=ks),
                        *res[-2:],
                    )
                )
    df = pd.DataFrame(
        results2,
        columns=[
            "Method",
            "Seed",
            "# distributions",
            "SpearmanR",
            *[f"P@{k}" for k in ks],
            "10-NN time (s)",
            "All-pairs time(s)",
        ],
    )
    df.to_pickle(
        f"results_{dataset_name}_{n_points_per_distribution}_{seeds}_{version}.pkl"
    )
    return df


if __name__ == "__main__":
    # run_sklearn_test()

    run_ablation(
        seeds=10,
        dataset_name="swiss_roll",
        n_neighbors=10,
        n_clusters=[2, 3, 4, 5, 6, 7, 8],
        n_levels=[2, 3, 4, 5, 6, 7, 8],
        n_trees=[1, 4, 8, 16, 32, 64],
        clustering_methods=["kmeans"],  # , "random-kd"],
    )
