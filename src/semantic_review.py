"""
Build an HTML page for manual review of the VLM output.

Every meme is shown as a card: the image on the left, the OCR text and the
VLM answer (description, key message, viewpoints) on the right. Images are
embedded into the file, so the page can be opened anywhere and shared.

Cards follow the row order of the *_review.csv file, so a card number matches
a row of the CSV where the manual marks are filled in.

Usage (from the project root):
    python src/semantic_review.py --model qwen2.5vl:3b
"""

from pathlib import Path
import argparse
import base64
import html
import io
import json
import re

import pandas as pd

from PIL import Image


PROJECT_ROOT = Path.cwd()
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "03_metadata_ocr.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "semantic_pilot"

THUMB_SIDE = 700  # embedded images are downscaled to keep the file small

CSS = """
body { font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; margin: 0; padding: 24px; background: #f4f5f7; color: #1f2328; }
h1 { font-size: 22px; margin: 0 0 4px; }
.summary { color: #57606a; margin-bottom: 20px; }
.card { display: flex; gap: 20px; background: #fff; border: 1px solid #d0d7de; border-radius: 10px; padding: 16px; margin-bottom: 18px; }
.image { flex: 0 0 380px; }
.image img { max-width: 100%; border-radius: 6px; border: 1px solid #d0d7de; }
.content { flex: 1; min-width: 0; }
.head { font-weight: 600; margin-bottom: 10px; }
.head .meta { font-weight: 400; color: #57606a; margin-left: 8px; font-size: 13px; }
.label { font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: #57606a; margin-top: 10px; }
.text { margin: 2px 0 0; white-space: pre-wrap; overflow-wrap: anywhere; }
.ocr { background: #f6f8fa; border-radius: 6px; padding: 6px 8px; font-size: 14px; }
table { border-collapse: collapse; margin-top: 4px; width: 100%; font-size: 14px; }
th, td { border: 1px solid #d0d7de; padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: #f6f8fa; font-weight: 600; }
.badge { padding: 1px 8px; border-radius: 10px; font-size: 12px; font-weight: 600; }
.positive { background: #dafbe1; color: #116329; }
.neutral { background: #eaeef2; color: #424a53; }
.negative { background: #ffebe9; color: #a40e26; }
.failed { background: #fff1e5; border: 1px solid #f0b27a; border-radius: 6px; padding: 6px 8px; }
.empty { color: #8c959f; font-style: italic; }
@media (max-width: 800px) { .card { flex-direction: column; } .image { flex-basis: auto; } }
"""


def esc(value):
    return html.escape("" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value))


def image_data_uri(path):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((THUMB_SIDE, THUMB_SIDE))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception:
        return None


def load_records(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def render_viewpoints(viewpoints):
    if not viewpoints:
        return '<p class="text empty">(model returned no viewpoints)</p>'
    rows = "".join(
        f"<tr><td>{esc(v['target'])}</td><td>{esc(v['aspect'])}</td><td>{esc(v['opinion'])}</td>"
        f"<td><span class=\"badge {esc(v['sentiment'])}\">{esc(v['sentiment'])}</span></td></tr>"
        for v in viewpoints)
    return ("<table><tr><th>target</th><th>aspect</th><th>opinion</th><th>sentiment</th></tr>"
            f"{rows}</table>")


def render_card(number, record, meta_row):
    uri = image_data_uri(PROJECT_ROOT / meta_row["image_path"]) if meta_row is not None else None
    image_html = (f'<img src="{uri}" alt="{esc(record["meme_id"])}">' if uri
                  else '<p class="empty">(image not found)</p>')
    ocr = meta_row["ocr_raw"] if meta_row is not None else None
    ocr_html = (f'<p class="text ocr">{esc(ocr)}</p>' if esc(ocr).strip()
                else '<p class="text empty">(no OCR text)</p>')
    seconds = f"{record['seconds']} s" if record.get("seconds") is not None else ""

    if record["ok"]:
        parsed = record["parsed"]
        answer = (
            '<div class="label">Description</div>'
            f'<p class="text">{esc(parsed["description"])}</p>'
            '<div class="label">Key message</div>'
            f'<p class="text">{esc(parsed["key_message"])}</p>'
            '<div class="label">Viewpoints</div>'
            f'{render_viewpoints(parsed["viewpoints"])}')
    else:
        answer = f'<div class="failed">Failed: {esc(record.get("error"))}</div>'

    return (
        '<div class="card">'
        f'<div class="image">{image_html}</div>'
        '<div class="content">'
        f'<div class="head">#{number} · {esc(record["meme_id"])}<span class="meta">{esc(seconds)}</span></div>'
        '<div class="label">OCR text</div>'
        f'{ocr_html}'
        f'{answer}'
        '</div></div>')


def build_html(model, records, meta):
    ok = sum(1 for r in records if r["ok"])
    cards = "\n".join(
        render_card(i, record, meta.loc[record["meme_id"]] if record["meme_id"] in meta.index else None)
        for i, record in enumerate(records, start=1))
    return (
        '<!DOCTYPE html><html lang="uk"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>Semantic extraction review: {esc(model)}</title><style>{CSS}</style></head><body>'
        f'<h1>Semantic extraction review: {esc(model)}</h1>'
        f'<div class="summary">{len(records)} memes · valid JSON: {ok} ({ok / max(len(records), 1):.0%})</div>'
        f'{cards}</body></html>')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="the same model tag that was used for the pilot")
    args = parser.parse_args()

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    jsonl_path = OUTPUT_DIR / f"{safe_name}.jsonl"
    if not jsonl_path.exists():
        raise SystemExit(f"File not found: {jsonl_path}. Run semantic_pilot.py first.")

    records = load_records(jsonl_path)
    meta = pd.read_csv(DATA_PATH).drop_duplicates("meme_id").set_index("meme_id")

    output_path = OUTPUT_DIR / f"{safe_name}_review.html"
    output_path.write_text(build_html(args.model, records, meta), encoding="utf-8")
    print(f"Saved {len(records)} cards to: {output_path}")


if __name__ == "__main__":
    main()