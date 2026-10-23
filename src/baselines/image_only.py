from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

from clustering import reduce_dimensions, cluster_embeddings
from cluster_metrics import evaluate, save_metrics, print_summary


PROJECT_ROOT = Path.cwd()
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "03_metadata_ocr.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "image_only"

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["image_path_abs"] = df["image_path"].apply(lambda p: Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p)
    df = df[df["image_path_abs"].apply(lambda p: p.is_file())].copy().reset_index(drop=True)
    print(f"Existing images: {len(df)}")
    return df


def create_embeddings(image_df):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    embeddings = []
    valid_indices = []

    for start in tqdm(range(0, len(image_df), BATCH_SIZE), desc="Creating CLIP embeddings"):
        images = []
        indices = []
        for index, row in image_df.iloc[start:start + BATCH_SIZE].iterrows():
            try:
                with Image.open(row["image_path_abs"]) as img:
                    images.append(img.convert("RGB"))
                indices.append(index)
            except Exception:
                continue
        if not images:
            continue
        inputs = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            features = model.get_image_features(pixel_values=inputs)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            features = features / features.norm(dim=-1, keepdim=True)
        embeddings.append(features.cpu().numpy())
        valid_indices.extend(indices)

    if not embeddings:
        raise RuntimeError("No image embeddings were created")
    return np.vstack(embeddings).astype(np.float32), image_df.loc[valid_indices].copy().reset_index(drop=True)


def save_cluster_examples(image_df, embeddings, output_dir, n=10):
    examples_dir = output_dir / "cluster_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    labels = image_df["cluster_id"].to_numpy()

    for cluster_id in sorted(np.unique(labels)):
        if cluster_id == -1:
            continue
        indices = np.where(labels == cluster_id)[0]
        cluster_embeddings = embeddings[indices]
        centroid = cluster_embeddings.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-12)
        selected_indices = indices[np.argsort(cluster_embeddings @ centroid)[::-1][:n]]
        cols = 5
        rows = math.ceil(len(selected_indices) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        for ax, index in zip(np.atleast_1d(axes).ravel(), selected_indices):
            row = image_df.iloc[index]
            with Image.open(row["image_path_abs"]) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_title(f"ID: {row['meme_id']}", fontsize=10)
        fig.suptitle(f"Cluster {cluster_id} — {len(indices)} memes", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(examples_dir / f"cluster_{cluster_id}_size_{len(indices)}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_results(image_df, embeddings, reduced_embeddings, metrics):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_df.drop(columns=["image_path_abs"]).to_csv(OUTPUT_DIR / "clusters.csv", index=False)
    np.save(OUTPUT_DIR / "image_embeddings.npy", embeddings)
    np.save(OUTPUT_DIR / "umap_embeddings.npy", reduced_embeddings)
    save_metrics(OUTPUT_DIR, "image_only", metrics, {"model": MODEL_NAME})


def main():
    image_df = load_data()
    embeddings, image_df = create_embeddings(image_df)
    reduced_embeddings = reduce_dimensions(embeddings)
    labels = cluster_embeddings(reduced_embeddings)
    image_df["cluster_id"] = labels
    metrics = evaluate(embeddings, reduced_embeddings, labels)
    save_results(image_df, embeddings, reduced_embeddings, metrics)
    save_cluster_examples(image_df, embeddings, OUTPUT_DIR, n=10)

    print_summary("IMAGE-ONLY BASELINE", metrics)


if __name__ == "__main__":
    main()