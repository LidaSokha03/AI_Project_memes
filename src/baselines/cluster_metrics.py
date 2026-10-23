"""
Shared evaluation for all baselines and later approaches.

Every approach is evaluated with the same code and its results are written
in the same metrics.csv format, so runs can be compared directly.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from clustering import UMAP_PARAMS, HDBSCAN_PARAMS


def evaluate(embeddings, reduced_embeddings, labels):
    """
    Internal (label-free) metrics.

    embeddings must be L2-normalized (so a dot product equals cosine similarity).
    Outliers (label -1) are excluded from silhouette and similarity metrics,
    so always read them together with outlier_ratio.
    """
    n_samples = len(labels)
    mask = labels != -1
    cluster_ids = np.unique(labels[mask])
    n_clusters = len(cluster_ids)
    n_outliers = int(np.sum(~mask))
    cluster_sizes = [int(np.sum(labels == c)) for c in cluster_ids]

    metrics = {
        "n_samples": int(n_samples),
        "n_clusters": int(n_clusters),
        "n_outliers": n_outliers,
        "outlier_ratio": n_outliers / n_samples,
        "largest_cluster_ratio": (max(cluster_sizes) / n_samples) if cluster_sizes else 0.0,
        "median_cluster_size": float(np.median(cluster_sizes)) if cluster_sizes else 0.0,
        "silhouette_original_cosine": None,
        "silhouette_umap_euclidean": None,
        "intra_cluster_similarity": None,
        "inter_cluster_similarity": None,
    }

    if n_clusters > 1 and mask.sum() > n_clusters:
        metrics["silhouette_original_cosine"] = float(
            silhouette_score(embeddings[mask], labels[mask], metric="cosine"))
        metrics["silhouette_umap_euclidean"] = float(
            silhouette_score(reduced_embeddings[mask], labels[mask], metric="euclidean"))

    if n_clusters > 0:
        centroids = []
        intra_values = []
        for cluster_id in cluster_ids:
            members = embeddings[labels == cluster_id]
            centroid = members.mean(axis=0)
            centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
            centroids.append(centroid)
            intra_values.extend((members @ centroid).tolist())
        # Mean similarity of memes to their own cluster centroid (higher = tighter).
        metrics["intra_cluster_similarity"] = float(np.mean(intra_values))

        if n_clusters > 1:
            centroids = np.vstack(centroids)
            similarity = centroids @ centroids.T
            off_diagonal = similarity[~np.eye(n_clusters, dtype=bool)]
            # Mean similarity between different cluster centroids (lower = more distinct).
            metrics["inter_cluster_similarity"] = float(off_diagonal.mean())

    return metrics


def save_metrics(output_dir, approach, metrics, extra_config=None):
    """Write a one-row metrics.csv: approach, settings and metrics together."""
    config = {
        "approach": approach,
        **{f"umap_{k}": v for k, v in UMAP_PARAMS.items()},
        **{f"hdbscan_{k}": v for k, v in HDBSCAN_PARAMS.items()},
        **(extra_config or {}),
    }
    pd.DataFrame([{**config, **metrics}]).to_csv(output_dir / "metrics.csv", index=False)


def print_summary(title, metrics):
    print(f"\n{title}")
    print(f"Memes: {metrics['n_samples']}")
    print(f"Clusters: {metrics['n_clusters']}")
    print(f"Outliers: {metrics['n_outliers']} ({metrics['outlier_ratio']:.1%})")
    print(f"Silhouette (cosine, original): {metrics['silhouette_original_cosine']}")
    print(f"Silhouette (euclidean, UMAP): {metrics['silhouette_umap_euclidean']}")
    print(f"Intra-cluster similarity: {metrics['intra_cluster_similarity']}")
    print(f"Inter-cluster similarity: {metrics['inter_cluster_similarity']}")