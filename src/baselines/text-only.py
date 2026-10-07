from pathlib import Path
import json
import re

import hdbscan
import numpy as np
import pandas as pd
import umap

from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path.cwd()

DATA_PATH = (PROJECT_ROOT / "data" / "processed" / "metadata4k_with_ocr.csv")
OUTPUT_DIR = (PROJECT_ROOT / "outputs" / "text_only")


MODEL_NAME = "intfloat/multilingual-e5-base"
RANDOM_STATE = 42

UMAP_N_NEIGHBORS = 10
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.0

HDBSCAN_MIN_CLUSTER_SIZE = 20
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_SELECTION_METHOD = "leaf"


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"t\.me/\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\b\S+\.(?:com|net|org|ru|ua|io)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text,)
    return text.strip()


def has_meaningful_text(text):
    if not text:
        return False
    letters = re.findall(r"[A-Za-zА-Яа-яІіЇїЄєҐґ]", text)
    return len(letters) >= 4


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["ocr_text"] = (df["ocr_raw_y"].apply(clean_text))
    df["has_text"] = (df["ocr_text"] .apply(has_meaningful_text))
    total = len(df)
    with_text = int(df["has_text"].sum())

    print(f"Total memes: {total}")
    print(f"With usable OCR text: {with_text}")

    text_df = (df[df["has_text"]].copy().reset_index(drop=True))

    return text_df


def create_embeddings(text_df):
    model = SentenceTransformer(MODEL_NAME)
    texts = ["passage: " + text for text in text_df["ocr_text"].tolist()]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True,)
    embeddings = embeddings.astype(np.float32)

    return embeddings


def reduce_dimensions(embeddings):
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=RANDOM_STATE)

    reduced_embeddings = (reducer.fit_transform(embeddings))
    return reduced_embeddings


def cluster_embeddings(reduced_embeddings):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=(HDBSCAN_MIN_CLUSTER_SIZE),
        min_samples=(HDBSCAN_MIN_SAMPLES),
        metric="euclidean", cluster_selection_method=(HDBSCAN_SELECTION_METHOD))

    labels = clusterer.fit_predict(reduced_embeddings)
    return labels


def evaluate(embeddings, reduced_embeddings, labels):
    n_samples = len(labels)
    cluster_ids = np.unique(labels[labels != -1])
    n_clusters = len(cluster_ids)
    n_outliers = int(np.sum(labels == -1))
    outlier_ratio = (n_outliers / n_samples)

    cluster_sizes = [int(np.sum(labels == cluster_id))
        for cluster_id in cluster_ids]

    if cluster_sizes:
        largest_cluster_size = max(cluster_sizes)
        largest_cluster_ratio = (largest_cluster_size / n_samples)
        median_cluster_size = float(np.median(cluster_sizes))
    else:
        largest_cluster_size = 0
        largest_cluster_ratio = 0.0
        median_cluster_size = 0.0

    mask = labels != -1

    silhouette_original = None
    silhouette_umap = None

    if (mask.sum() > 1 and n_clusters > 1):
        silhouette_original = (silhouette_score( embeddings[mask], labels[mask], metric="cosine"))
        silhouette_umap = (silhouette_score(reduced_embeddings[mask], labels[mask], metric="euclidean"))

    metrics = {
        "n_clusters": int(n_clusters),
        "n_outliers": int(n_outliers),
        "silhouette_original_cosine": (
            float(silhouette_original)
            if silhouette_original is not None
            else None),
        "silhouette_umap_euclidean": (float(silhouette_umap) if silhouette_umap is not None else None)}

    return metrics


def save_cluster_examples(text_df, embeddings, output_dir, n=10):
    rows = []
    labels = (text_df["cluster_id"].to_numpy())
    rng = np.random.default_rng(RANDOM_STATE)
    for cluster_id in sorted(text_df["cluster_id"].unique()):
        indices = np.where(labels == cluster_id)[0]
        cluster_size = len(indices)
        if cluster_id == -1:
            selected_indices = rng.choice(indices, size=min(n, cluster_size), replace=False)
            similarities = {int(index): None for index in selected_indices}
        else:
            cluster_embeddings = (embeddings[indices])
            centroid = (cluster_embeddings.mean(axis=0))
            centroid_norm = np.linalg.norm(centroid)

            if centroid_norm > 0:
                centroid = (centroid / centroid_norm)

            cosine_similarities = (cluster_embeddings @ centroid)

            order = np.argsort(cosine_similarities)[::-1]

            selected_local = order[:min(n, cluster_size)]

            selected_indices = (indices[selected_local])

            similarities = {int(index): float(similarity)
                for index, similarity in zip(selected_indices, cosine_similarities[ selected_local])}

        for index in selected_indices:
            row = text_df.iloc[index]

            rows.append({
                    "cluster_id": int(cluster_id),
                    "cluster_size": int(cluster_size),
                    "meme_id": row["meme_id"],
                    "ocr_text": row["ocr_text"],
                    "ocr_confidence": row.get("ocr_confidence", None),
                    "representative_similarity":similarities[int(index)]})

    examples_df = pd.DataFrame(rows)
    examples_df.to_csv(output_dir / "cluster_examples.csv", index=False)


def save_results(text_df, embeddings, reduced_embeddings, metrics,):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    text_df.to_csv(OUTPUT_DIR / "clusters.csv", index=False)
    np.save(OUTPUT_DIR / "text_embeddings.npy", embeddings)
    np.save(OUTPUT_DIR / "umap_embeddings.npy", reduced_embeddings)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True,)
    text_df = load_data()
    embeddings = create_embeddings(text_df)
    reduced_embeddings = (reduce_dimensions(embeddings))
    labels = cluster_embeddings(reduced_embeddings)
    text_df["cluster_id"] = labels
    metrics = evaluate(embeddings, reduced_embeddings, labels)

    save_results(text_df, embeddings, reduced_embeddings, metrics)
    save_cluster_examples(text_df, embeddings, OUTPUT_DIR, n=10)


    print("\nTEXT-ONLY BASELINE")
    print(f"Clusters: " f"{metrics['n_clusters']}")
    print(f"Outliers: "f"{metrics['n_outliers']}")

if __name__ == "__main__":
    main()