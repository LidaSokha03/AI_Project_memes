from pathlib import Path
import pandas as pd
from paddleocr import PaddleOCR


PROJECT_ROOT = Path.cwd()
METADATA_PATH = PROJECT_ROOT / "data" / "interim" / "working_subset_4000.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "metadata4k_with_ocr.csv"
BATCH_SIZE = 50

# PaddleOCR for ukrainian and russian text recognition
ocr = PaddleOCR(
    lang="uk",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)


def save_checkpoint(metadata, results):
    result_df = pd.DataFrame(results)
    updated = metadata.merge(result_df, on="meme_id", how="left")
    updated.to_csv(OUTPUT_PATH, index=False)
    # print(f"\nCheckpoint saved: {len(result_df)} memes processed\n")


def extract_text(image_path):
    try:
        result = ocr.predict(str(image_path))
        all_texts = []
        all_scores = []
        for res in result:
            texts = res["rec_texts"]
            scores = res["rec_scores"]
            for text, score in zip(texts, scores):
                if text.strip():
                    all_texts.append(text.strip())
                    all_scores.append(float(score))
        ocr_text = " ".join(all_texts)
        if all_scores:
            avg_confidence = sum(all_scores) / len(all_scores)
        else:
            avg_confidence = 0.0

        return ocr_text, avg_confidence

    except Exception as error:
        print(f"OCR error for {image_path}: {error}")
        return "", 0.0


def main():
    metadata = pd.read_csv(METADATA_PATH)
    results = []
    if OUTPUT_PATH.exists():
        previous = pd.read_csv(OUTPUT_PATH)
        previous_processed = previous[previous["ocr_raw"].notna()][["meme_id", "ocr_raw", "ocr_confidence"]]
        results = previous_processed.to_dict("records")
        processed_ids = set(previous_processed["meme_id"])
        print(f"Resuming: {len(processed_ids)} memes already processed.")

    else:
        processed_ids = set()

    total = len(metadata)

    for _, row in metadata.iterrows():
        meme_id = row["meme_id"]
        if meme_id in processed_ids:
            continue

        image_path = PROJECT_ROOT / row["image_path"]
        text, confidence = extract_text(image_path)
        results.append({"meme_id": meme_id, "ocr_raw": text, "ocr_confidence": confidence})
        processed_ids.add(meme_id)
        # print(
        #     f"[{len(processed_ids)}/{total}] "
        #     f"{meme_id}: {text}")

        if len(processed_ids) % BATCH_SIZE == 0:
            save_checkpoint(metadata, results)

    save_checkpoint(metadata,results)

    # print("OCR completed!")
    # print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
