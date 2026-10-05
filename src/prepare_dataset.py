from pathlib import Path
import hashlib
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path.cwd()
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
OUTPUT_PATH = INTERIM_DIR / "metadata_all.csv"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_image(path: Path) -> dict:
    row = {
        "dataset": path.parts[path.parts.index("raw") + 1],
        "image_path": str(path.relative_to(PROJECT_ROOT)),
        "filename": path.name,
        "is_valid": True,
        "width": None,
        "height": None,
        "format": None,
        "sha256": None,
        "error": None,}

    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            row["width"] = image.width
            row["height"] = image.height
            row["format"] = image.format

        row["sha256"] = sha256_file(path)
    except Exception as exc:
        row["is_valid"] = False
        row["error"] = str(exc)

    return row


def main() -> None:
    image_paths = sorted(
        path
        for path in RAW_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)

    if not image_paths:
        print(f"No images found in {RAW_DIR}")
        return

    rows = [inspect_image(path) for path in image_paths]
    metadata = pd.DataFrame(rows)

    metadata.insert(0, "meme_id", [f"meme_{i:06d}" for i in range(1, len(metadata) + 1)])

    metadata["is_exact_duplicate"] = metadata["sha256"].duplicated(keep=False)

    for column in ["source", "published_at", "telegram_caption", "ocr_raw", "ocr_clean"]:
        metadata[column] = None

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(metadata)} rows to {OUTPUT_PATH}")
    print(f"Valid images: {int(metadata['is_valid'].sum())}")
    print(f"Invalid images: {int((~metadata['is_valid']).sum())}")
    print(f"Rows belonging to exact-duplicate groups: {int(metadata['is_exact_duplicate'].sum())}")


if __name__ == "__main__":
    main()
