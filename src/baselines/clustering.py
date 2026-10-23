"""
Shared clustering step for all baselines (image-only, text-only, multimodal).

UMAP and HDBSCAN settings live here so every approach is clustered
with exactly the same parameters.
"""

import hdbscan
import umap


RANDOM_STATE = 42

UMAP_PARAMS = {
    "n_neighbors": 10,
    "n_components": 5,
    "min_dist": 0.0,
    "metric": "cosine",
    "random_state": RANDOM_STATE,
}

HDBSCAN_PARAMS = {
    "min_cluster_size": 20,
    "min_samples": 1,
    "metric": "euclidean",
    "cluster_selection_method": "leaf",
}


def reduce_dimensions(embeddings):
    return umap.UMAP(**UMAP_PARAMS).fit_transform(embeddings)


def cluster_embeddings(reduced_embeddings):
    return hdbscan.HDBSCAN(**HDBSCAN_PARAMS).fit_predict(reduced_embeddings)