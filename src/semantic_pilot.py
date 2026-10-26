"""
Pilot for VLM semantic extraction (MemeTT-style).

For every meme in a small fixed sample the VLM receives the image and the OCR
text and returns structured JSON: description, key message and a list of
viewpoints (target, aspect, opinion, sentiment).

The same sample (fixed seed) is used for every model, so models can be compared
fairly. Results are appended to a JSONL file after each meme, so a stopped run
can be resumed.

Usage (from the project root):
    python src/semantic_pilot.py --model qwen2.5vl:7b --n 20
    python src/semantic_pilot.py --model llava:7b --n 20
"""

from pathlib import Path
import argparse
import base64
import io
import json
import re
import time

import pandas as pd
import requests

from PIL import Image
from tqdm.auto import tqdm


PROJECT_ROOT = Path.cwd()
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "03_metadata_ocr.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "semantic_pilot"

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5vl:7b"
DEFAULT_N = 20
RANDOM_STATE = 42

MAX_IMAGE_SIDE = 1024   # large images are downscaled before sending
MAX_OCR_CHARS = 600
MAX_ATTEMPTS = 2        # one retry if the answer is not valid JSON
REQUEST_TIMEOUT = 300   # seconds per request (override with --timeout)
OUTPUT_LANGUAGE = "Ukrainian"

SENTIMENTS = {"positive", "neutral", "negative"}

# Plain placeholders instead of str.format, because the prompt contains JSON braces.
PROMPT_TEMPLATE = """You are analysing an internet meme from Ukrainian Telegram channels.

You receive the meme image and the text that OCR extracted from it. The OCR text may contain errors or be empty. If the OCR text and the image disagree, trust the image.

OCR text:
<<OCR>>

Return ONLY one JSON object with exactly these keys:
- "description": 1-2 sentences about what is visibly shown (people, objects, layout, visible text). Describe, do not interpret.
- "key_message": one sentence with the main point the meme makes.
- "viewpoints": a list of 1-3 objects. Each object has:
  - "target": the specific person, group, institution or thing the meme talks about. Use a concrete name, not a broad category like "politics" or "society".
  - "aspect": the specific feature of the target that is being judged.
  - "opinion": the stance expressed about that aspect, as a short phrase.
  - "sentiment": exactly one of "positive", "neutral", "negative" (the attitude towards the target).

Rules:
- Write all values in <<LANGUAGE>>, except "sentiment", which must stay in English.
- Describe only what is visible. If some text is unreadable, say so instead of guessing.
- If the meme expresses no clear opinion, return an empty "viewpoints" list.
- Output nothing except the JSON object.

Example of the expected format (illustrative, unrelated to this meme):
{"description": "Двоє людей стоять біля зупинки, на табло написано, що автобус запізнюється на годину.", "key_message": "Громадський транспорт ходить нерегулярно.", "viewpoints": [{"target": "громадський транспорт", "aspect": "розклад", "opinion": "непередбачуваний", "sentiment": "negative"}]}
"""


def build_prompt(ocr_text):
    ocr_text = "" if pd.isna(ocr_text) else " ".join(str(ocr_text).split())
    ocr_text = ocr_text[:MAX_OCR_CHARS] or "(no text detected)"
    return (PROMPT_TEMPLATE
            .replace("<<OCR>>", ocr_text)
            .replace("<<LANGUAGE>>", OUTPUT_LANGUAGE))


def encode_image(path, max_side):
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def call_model(model, prompt, image_b64, timeout):
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()["message"]["content"]


def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def validate(data):
    """Return (normalized_data, error). error is None when the structure is valid."""
    if not isinstance(data, dict):
        return None, "answer is not a JSON object"

    for key in ("description", "key_message"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            return None, f"missing or empty '{key}'"

    viewpoints = data.get("viewpoints")
    if not isinstance(viewpoints, list):
        return None, "'viewpoints' is not a list"

    cleaned = []
    for viewpoint in viewpoints:
        if not isinstance(viewpoint, dict):
            return None, "a viewpoint is not an object"
        item = {}
        for key in ("target", "aspect", "opinion"):
            value = viewpoint.get(key)
            if not isinstance(value, str) or not value.strip():
                return None, f"viewpoint is missing '{key}'"
            item[key] = value.strip()
        sentiment = str(viewpoint.get("sentiment", "")).strip().lower()
        if sentiment not in SENTIMENTS:
            return None, f"invalid sentiment: {sentiment!r}"
        item["sentiment"] = sentiment
        cleaned.append(item)

    return {
        "description": data["description"].strip(),
        "key_message": data["key_message"].strip(),
        "viewpoints": cleaned,
    }, None


def extract_one(model, row, max_side, timeout):
    start = time.time()
    record = {"meme_id": row["meme_id"], "model": model, "ok": False,
              "attempts": 0, "seconds": None, "parsed": None,
              "error": None, "raw_response": None}
    try:
        image_b64 = encode_image(row["image_path_abs"], max_side)
    except Exception as error:
        record["error"] = f"cannot read image: {error}"
        return record

    prompt = build_prompt(row.get("ocr_raw"))
    for attempt in range(1, MAX_ATTEMPTS + 1):
        record["attempts"] = attempt
        try:
            raw = call_model(model, prompt, image_b64, timeout)
        except requests.exceptions.ConnectionError:
            raise SystemExit(
                "Cannot connect to Ollama at " + OLLAMA_URL + ". "
                "Make sure Ollama is running and the model is downloaded "
                f"(ollama pull {model}).")
        except Exception as error:
            # A timeout or server error will not fix itself on a retry, so stop here.
            record["error"] = f"request failed: {error}"
            break

        record["raw_response"] = raw
        parsed, error = validate(parse_json(raw))
        if error is None:
            record.update(ok=True, parsed=parsed, error=None)
            break
        record["error"] = error

    record["seconds"] = round(time.time() - start, 2)
    return record


def select_memes(n):
    df = pd.read_csv(DATA_PATH)
    df["image_path_abs"] = df["image_path"].apply(
        lambda p: Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p)
    df = df[df["image_path_abs"].apply(lambda p: p.is_file())].copy().reset_index(drop=True)
    return df.sample(n=min(n, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)


def load_records(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def summarize(records, model):
    total = len(records)
    ok = [r for r in records if r["ok"]]
    print(f"\nSEMANTIC EXTRACTION PILOT: {model}")
    print(f"Memes: {total}")
    print(f"Valid JSON: {len(ok)} ({len(ok) / max(total, 1):.0%})")
    print(f"Needed a retry: {sum(1 for r in records if r['attempts'] > 1)}")

    times = [r["seconds"] for r in records if r["seconds"] is not None]
    if times:
        print(f"Mean time per meme: {sum(times) / len(times):.1f} s")

    if ok:
        counts = [len(r["parsed"]["viewpoints"]) for r in ok]
        print(f"Mean viewpoints per meme: {sum(counts) / len(counts):.2f}")
        print(f"Memes with no viewpoints: {sum(1 for c in counts if c == 0)}")
        sentiments = pd.Series([v["sentiment"] for r in ok for v in r["parsed"]["viewpoints"]])
        if not sentiments.empty:
            print("Sentiment distribution:")
            print(sentiments.value_counts().to_string())

    errors = pd.Series([r["error"] for r in records if not r["ok"]])
    if not errors.empty:
        print("Failure reasons:")
        print(errors.value_counts().to_string())


def write_review_csv(records, sample, path):
    """A sheet for manual evaluation: fill in the empty columns by hand."""
    meta = sample.set_index("meme_id")
    rows = []
    for record in records:
        parsed = record["parsed"] or {}
        rows.append({
            "meme_id": record["meme_id"],
            "image_path": meta.loc[record["meme_id"], "image_path"],
            "ocr_raw": meta.loc[record["meme_id"], "ocr_raw"],
            "ok": record["ok"],
            "description": parsed.get("description"),
            "key_message": parsed.get("key_message"),
            "viewpoints": json.dumps(parsed.get("viewpoints", []), ensure_ascii=False),
            "description_ok": "",
            "key_message_ok": "",
            "viewpoints_ok": "",
            "notes": "",
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="number of memes in the pilot")
    parser.add_argument("--max-side", type=int, default=MAX_IMAGE_SIDE, help="downscale images so the longer side is at most this many pixels")
    parser.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT, help="seconds to wait for one answer")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    jsonl_path = OUTPUT_DIR / f"{safe_name}.jsonl"
    review_path = OUTPUT_DIR / f"{safe_name}_review.csv"

    sample = select_memes(args.n)
    sample_ids = set(sample["meme_id"])

    records = load_records(jsonl_path)
    ok_records = [r for r in records if r["ok"]]
    if len(ok_records) != len(records):
        # Failed memes are retried on the next run: keep only successful records in the file.
        with jsonl_path.open("w", encoding="utf-8") as file:
            for record in ok_records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        records = ok_records
    done_ids = {r["meme_id"] for r in records}
    if done_ids:
        print(f"Resuming: {len(done_ids & sample_ids)} memes already processed.")

    with jsonl_path.open("a", encoding="utf-8") as file:
        for _, row in tqdm(sample.iterrows(), total=len(sample), desc=f"Extracting with {args.model}"):
            if row["meme_id"] in done_ids:
                continue
            record = extract_one(args.model, row, args.max_side, args.timeout)
            tqdm.write(f"{record['meme_id']}: ok={record['ok']}, {record['seconds']} s, {record['error'] or ''}")
            records.append(record)
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()

    records = [r for r in records if r["meme_id"] in sample_ids]
    summarize(records, args.model)
    write_review_csv(records, sample, review_path)
    print(f"\nRaw results: {jsonl_path}")
    print(f"Sheet for manual review: {review_path}")


if __name__ == "__main__":
    main()