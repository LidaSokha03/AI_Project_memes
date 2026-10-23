from pathlib import Path
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer

from clustering import reduce_dimensions, cluster_embeddings, RANDOM_STATE
from cluster_metrics import evaluate, save_metrics, print_summary


PROJECT_ROOT = Path.cwd()
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "03_metadata_ocr.csv"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
E5_MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE_IMAGE = 32
BATCH_SIZE_TEXT = 64
CLIP_MAX_TOKENS = 77  # token limit of the CLIP text encoder

# --- Experiment configuration --------------------------------------------
# Which encoder to use for the OCR text:
#   "clip" - CLIP text encoder (shared space with the images, 512 dims;
#            as in TM-FID, but English-centric and truncates text to 77 tokens)
#   "e5"   - multilingual-e5-base (768 dims; better for Ukrainian)
TEXT_ENCODER = "clip"

# How to combine image and text embeddings:
#   "concat" - weighted concatenation (works with any text encoder)
#   "mean"   - weighted normalized average (only with TEXT_ENCODER="clip",
#              because both vectors must have the same dimensionality)
FUSION = "mean"

IMAGE_WEIGHT = 1.0
TEXT_WEIGHT = 1.0

# False: use all memes (memes without text rely on the image part only);
# True:  use only memes with usable OCR text (same subset as in text-only).
ONLY_MEMES_WITH_TEXT = False
# --------------------------------------------------------------------------

if TEXT_ENCODER not in {"clip", "e5"}:
    raise ValueError("TEXT_ENCODER must be 'clip' or 'e5'")
if FUSION not in {"concat", "mean"}:
    raise ValueError("FUSION must be 'concat' or 'mean'")
if FUSION == "mean" and TEXT_ENCODER != "clip":
    raise ValueError("FUSION='mean' requires TEXT_ENCODER='clip' (vectors must have equal dimensionality)")

# A separate output folder per configuration makes ablation runs easy to compare.
OUTPUT_DIR = PROJECT_ROOT / "outputs" / f"multimodal"


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
    df["image_path_abs"] = df["image_path"].apply(lambda p: Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p)
    df = df[df["image_path_abs"].apply(lambda p: p.is_file())].copy().reset_index(drop=True)
    df["ocr_text"] = df["ocr_raw"].apply(clean_text)
    df["has_text"] = df["ocr_text"].apply(has_meaningful_text)

    print(f"Existing images: {len(df)}")
    print(f"With usable OCR text: {int(df['has_text'].sum())}")

    if ONLY_MEMES_WITH_TEXT:
        df = df[df["has_text"]].copy().reset_index(drop=True)
        print(f"Using only memes with text: {len(df)}")
    return df


def l2_normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def to_tensor(features):
    if not isinstance(features, torch.Tensor):
        features = features.pooler_output
    return features


def create_image_embeddings(df, model, processor, device):
    embeddings = []
    valid_indices = []

    for start in tqdm(range(0, len(df), BATCH_SIZE_IMAGE), desc="Creating CLIP image embeddings"):
        images = []
        indices = []
        for index, row in df.iloc[start:start + BATCH_SIZE_IMAGE].iterrows():
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
            features = to_tensor(model.get_image_features(pixel_values=inputs))
            features = features / features.norm(dim=-1, keepdim=True)
        embeddings.append(features.cpu().numpy())
        valid_indices.extend(indices)

    if not embeddings:
        raise RuntimeError("No image embeddings were created")
    return np.vstack(embeddings).astype(np.float32), df.loc[valid_indices].copy().reset_index(drop=True)


def encode_texts_clip(texts, model, processor, device):
    out = []
    for start in tqdm(range(0, len(texts), BATCH_SIZE_TEXT), desc="Creating CLIP text embeddings"):
        batch = texts[start:start + BATCH_SIZE_TEXT]
        inputs = processor(text=batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=CLIP_MAX_TOKENS).to(device)
        with torch.no_grad():
            features = to_tensor(model.get_text_features(**inputs))
            features = features / features.norm(dim=-1, keepdim=True)
        out.append(features.cpu().numpy())
    return np.vstack(out)


def encode_texts_e5(texts):
    model = SentenceTransformer(E5_MODEL_NAME)
    return model.encode(["passage: " + t for t in texts], batch_size=BATCH_SIZE_TEXT,
                        show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)


def create_text_embeddings(df, clip_model, clip_processor, device):
    """Text embeddings for memes with text; a zero vector for the rest."""
    has_text = df["has_text"].to_numpy()
    texts = df.loc[has_text, "ocr_text"].tolist()

    if TEXT_ENCODER == "clip":
        dim = clip_model.config.projection_dim
        encoded = encode_texts_clip(texts, clip_model, clip_processor, device) if texts else None
    else:
        dim = SentenceTransformer(E5_MODEL_NAME).get_sentence_embedding_dimension()
        encoded = encode_texts_e5(texts) if texts else None

    text_embeddings = np.zeros((len(df), dim), dtype=np.float32)
    if encoded is not None:
        text_embeddings[has_text] = encoded.astype(np.float32)
    return text_embeddings


def fuse_embeddings(image_embeddings, text_embeddings, has_text):
    if FUSION == "concat":
        # For memes without text, the text part is all zeros.
        fused = np.hstack([IMAGE_WEIGHT * image_embeddings, TEXT_WEIGHT * text_embeddings])
    else:
        # Normalized weighted average; memes without text use the image embedding only.
        fused = IMAGE_WEIGHT * image_embeddings + TEXT_WEIGHT * text_embeddings * has_text[:, None]
    return l2_normalize(fused).astype(np.float32)


def save_cluster_examples(df, embeddings, output_dir, n=10):
    """Save a CSV with representative memes and a PNG grid per cluster."""
    examples_dir = output_dir / "cluster_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    labels = df["cluster_id"].to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    for cluster_id in sorted(np.unique(labels)):
        indices = np.where(labels == cluster_id)[0]
        cluster_size = len(indices)

        if cluster_id == -1:
            selected_indices = rng.choice(indices, size=min(n, cluster_size), replace=False)
            similarities = {int(index): None for index in selected_indices}
        else:
            members = embeddings[indices]
            centroid = members.mean(axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-12)
            cosine_similarities = members @ centroid
            order = np.argsort(cosine_similarities)[::-1][:min(n, cluster_size)]
            selected_indices = indices[order]
            similarities = {int(index): float(similarity)
                            for index, similarity in zip(selected_indices, cosine_similarities[order])}

        for index in selected_indices:
            row = df.iloc[index]
            rows.append({
                "cluster_id": int(cluster_id),
                "cluster_size": int(cluster_size),
                "meme_id": row["meme_id"],
                "has_text": bool(row["has_text"]),
                "ocr_text": row["ocr_text"],
                "representative_similarity": similarities[int(index)]})

        if cluster_id == -1:
            continue

        cols = 5
        n_rows = math.ceil(len(selected_indices) / cols)
        fig, axes = plt.subplots(n_rows, cols, figsize=(4 * cols, 4 * n_rows))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, index in zip(axes, selected_indices):
            row = df.iloc[index]
            with Image.open(row["image_path_abs"]) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_title(f"ID: {row['meme_id']}", fontsize=10)
        fig.suptitle(f"Cluster {cluster_id} — {cluster_size} memes", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(examples_dir / f"cluster_{cluster_id}_size_{cluster_size}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    pd.DataFrame(rows).to_csv(output_dir / "cluster_examples.csv", index=False)


def save_results(df, image_embeddings, text_embeddings, fused_embeddings, reduced_embeddings, metrics):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # clusters.csv is saved in the same row order as the embeddings.
    df.drop(columns=["image_path_abs"]).to_csv(OUTPUT_DIR / "clusters.csv", index=False)
    np.save(OUTPUT_DIR / "image_embeddings.npy", image_embeddings)
    np.save(OUTPUT_DIR / "text_embeddings.npy", text_embeddings)
    np.save(OUTPUT_DIR / "multimodal_embeddings.npy", fused_embeddings)
    np.save(OUTPUT_DIR / "umap_embeddings.npy", reduced_embeddings)
    save_metrics(OUTPUT_DIR, "multimodal", metrics, {
        "image_model": CLIP_MODEL_NAME,
        "text_encoder": TEXT_ENCODER,
        "fusion": FUSION,
        "image_weight": IMAGE_WEIGHT,
        "text_weight": TEXT_WEIGHT,
        "only_memes_with_text": ONLY_MEMES_WITH_TEXT})


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Config: text_encoder={TEXT_ENCODER}, fusion={FUSION}, "
          f"weights=(image {IMAGE_WEIGHT}, text {TEXT_WEIGHT}), "
          f"only_with_text={ONLY_MEMES_WITH_TEXT}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    df = load_data()
    image_embeddings, df = create_image_embeddings(df, clip_model, clip_processor, device)
    text_embeddings = create_text_embeddings(df, clip_model, clip_processor, device)
    fused_embeddings = fuse_embeddings(
        image_embeddings, text_embeddings, df["has_text"].to_numpy().astype(np.float32))

    reduced_embeddings = reduce_dimensions(fused_embeddings)
    labels = cluster_embeddings(reduced_embeddings)
    df["cluster_id"] = labels
    metrics = evaluate(fused_embeddings, reduced_embeddings, labels)

    save_results(df, image_embeddings, text_embeddings, fused_embeddings, reduced_embeddings, metrics)
    save_cluster_examples(df, fused_embeddings, OUTPUT_DIR, n=10)

    print_summary("MULTIMODAL BASELINE (image + OCR text)", metrics)


if __name__ == "__main__":
    main()