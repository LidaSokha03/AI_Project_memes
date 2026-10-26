"""
Build an HTML page for manual review of the VLM output.

Every meme is shown as a card: the image on the left, the OCR text and the
VLM answer (description, key message, viewpoints) on the right. Under each
answer there are rating buttons and a note field, so the marks are set right
next to the meme instead of in a spreadsheet.

In the browser:
  - marks are autosaved in the browser (localStorage);
  - the bar at the top shows live counts for every criterion;
  - "Export CSV" downloads the marks in the same format as the *_review.csv
    written by semantic_pilot.py, so it can simply replace that file.

Images are embedded into the HTML, so the page can be opened anywhere.

Usage (from the project root):
    # for an experiment folder (the HTML is saved next to the .jsonl file)
    python src/semantic_review.py --jsonl experiments/semantic_pilot/qwen2.5vl-3b_prompt-v1/qwen2.5vl_3b.jsonl

    # for a fresh pilot run in outputs/semantic_pilot
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

RATING_FIELDS = [("d", "Опис"), ("k", "Key message"), ("v", "Viewpoints")]
RATING_BUTTONS = [("1", "✓ 1", "v1"), ("0.5", "~ 0.5", "v05"), ("0", "✗ 0", "v0")]

CSS = """
body { font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; margin: 0; padding: 0 24px 24px; background: #f4f5f7; color: #1f2328; }
h1 { font-size: 22px; margin: 20px 0 4px; }
.summary { color: #57606a; margin-bottom: 12px; }
.bar { position: sticky; top: 0; z-index: 10; display: flex; flex-wrap: wrap; align-items: center; gap: 8px 24px; background: #fff; border-bottom: 1px solid #d0d7de; margin: 0 -24px 16px; padding: 10px 24px; }
.bar .sums { color: #424a53; font-size: 14px; }
.bar .actions { margin-left: auto; display: flex; gap: 8px; }
.bar button { border: 1px solid #d0d7de; background: #f6f8fa; border-radius: 6px; padding: 6px 12px; cursor: pointer; font-size: 14px; }
.bar button.primary { background: #1f883d; border-color: #1f883d; color: #fff; font-weight: 600; }
details { background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 8px 12px; margin-bottom: 16px; font-size: 14px; }
details summary { cursor: pointer; font-weight: 600; }
.warn { color: #9a6700; font-size: 13px; margin-bottom: 8px; display: none; }
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
.rate { margin-top: 14px; padding-top: 10px; border-top: 1px dashed #d0d7de; }
.rrow { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; }
.rlabel { width: 110px; font-size: 14px; font-weight: 600; }
.opt { border: 1px solid #d0d7de; background: #fff; border-radius: 6px; padding: 4px 12px; cursor: pointer; font-size: 14px; }
.opt:hover { background: #f6f8fa; }
.opt.v1.active { background: #dafbe1; border-color: #1f883d; color: #116329; font-weight: 600; }
.opt.v05.active { background: #fff8c5; border-color: #bf8700; color: #7d4e00; font-weight: 600; }
.opt.v0.active { background: #ffebe9; border-color: #cf222e; color: #a40e26; font-weight: 600; }
.note { width: 100%; box-sizing: border-box; margin-top: 4px; padding: 6px 8px; border: 1px solid #d0d7de; border-radius: 6px; font: inherit; font-size: 14px; }
@media (max-width: 800px) { .card { flex-direction: column; } .image { flex-basis: auto; } }
"""

RUBRIC = """
<details>
<summary>Критерії оцінювання</summary>
<table>
<tr><th></th><th>✓ 1</th><th>~ 0.5</th><th>✗ 0</th></tr>
<tr><td>Опис</td><td>відповідає картинці, без вигадок</td><td>здебільшого правильний, помилки в деталях</td><td>вигадані об’єкти чи дії</td></tr>
<tr><td>Key message</td><td>передає суть мему</td><td>вловлює тему, але не суть (наприклад, лише буквальний переказ)</td><td>хибний або безглуздий</td></tr>
<tr><td>Viewpoints</td><td>target, aspect, opinion, sentiment осмислені</td><td>частина полів доречна</td><td>загальні або вигадані поля</td></tr>
</table>
<p>Повторне натискання на вибрану оцінку знімає її. Оцінки зберігаються в браузері автоматично; після завершення натисни «Експортувати CSV» і заміни ним файл <code>*_review.csv</code>.</p>
</details>
"""

SCRIPT = r"""
const ROWS = __DATA__;
const CSV_NAME = __CSV_NAME__;
const KEY = "semantic-review:" + document.title + ":" + ROWS.length + ":" + (ROWS.length ? ROWS[0].meme_id : "");
const FIELDS = ["d", "k", "v"];
const COLUMNS = ["meme_id", "image_path", "ocr_raw", "ok", "description", "key_message",
                 "viewpoints", "description_ok", "key_message_ok", "viewpoints_ok", "notes"];

let marks = {};
try {
  marks = JSON.parse(localStorage.getItem(KEY) || "{}");
} catch (e) {
  document.getElementById("warn").style.display = "block";
}

function persist() {
  try {
    localStorage.setItem(KEY, JSON.stringify(marks));
  } catch (e) {
    document.getElementById("warn").style.display = "block";
  }
  refresh();
}

function markOf(id) {
  if (!marks[id]) marks[id] = {};
  return marks[id];
}

function summarize() {
  const counts = {d: [0, 0, 0], k: [0, 0, 0], v: [0, 0, 0]};
  let done = 0;
  let total = 0;
  ROWS.forEach(function (row) {
    if (!row.ok) return;
    total++;
    const m = marks[row.meme_id] || {};
    let complete = true;
    FIELDS.forEach(function (f) {
      if (m[f] === 1) counts[f][0]++;
      else if (m[f] === 0.5) counts[f][1]++;
      else if (m[f] === 0) counts[f][2]++;
      else complete = false;
    });
    if (complete) done++;
  });
  return {counts: counts, done: done, total: total};
}

function refresh() {
  document.querySelectorAll(".opt").forEach(function (b) {
    const m = marks[b.dataset.id] || {};
    b.classList.toggle("active", m[b.dataset.field] === parseFloat(b.dataset.value));
  });
  const s = summarize();
  document.getElementById("progress").textContent = "Оцінено повністю: " + s.done + " з " + s.total;
  FIELDS.forEach(function (f) {
    const c = s.counts[f];
    document.getElementById("sum-" + f).textContent = "✓ " + c[0] + " · ~ " + c[1] + " · ✗ " + c[2];
  });
}

function cell(value) {
  const text = value === null || value === undefined ? "" : String(value);
  return /[",\n\r]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
}

function buildCsv() {
  const lines = [COLUMNS.join(",")];
  ROWS.forEach(function (r) {
    const m = marks[r.meme_id] || {};
    const values = [r.meme_id, r.image_path, r.ocr_raw, r.ok ? "True" : "False",
                    r.description, r.key_message, r.viewpoints, m.d, m.k, m.v, m.n];
    lines.push(values.map(cell).join(","));
  });
  return "\ufeff" + lines.join("\r\n") + "\r\n";
}

document.addEventListener("click", function (e) {
  const b = e.target.closest ? e.target.closest(".opt") : null;
  if (!b) return;
  const m = markOf(b.dataset.id);
  const v = parseFloat(b.dataset.value);
  if (m[b.dataset.field] === v) delete m[b.dataset.field];
  else m[b.dataset.field] = v;
  persist();
});

document.addEventListener("input", function (e) {
  if (!e.target.classList || !e.target.classList.contains("note")) return;
  markOf(e.target.dataset.id).n = e.target.value;
  persist();
});

document.getElementById("export").addEventListener("click", function () {
  const blob = new Blob([buildCsv()], {type: "text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = CSV_NAME;
  document.body.appendChild(a);
  a.click();
  a.remove();
});

document.getElementById("reset").addEventListener("click", function () {
  if (!confirm("Очистити всі оцінки й нотатки на цій сторінці?")) return;
  marks = {};
  document.querySelectorAll(".note").forEach(function (el) { el.value = ""; });
  persist();
});

document.querySelectorAll(".note").forEach(function (el) {
  el.value = (marks[el.dataset.id] || {}).n || "";
});
refresh();
"""


def esc(value):
    return html.escape("" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value))


def clean(value):
    return "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)


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


def render_rating(meme_id):
    rows = ""
    for field, label in RATING_FIELDS:
        buttons = "".join(
            f'<button type="button" class="opt {css}" data-id="{esc(meme_id)}" '
            f'data-field="{field}" data-value="{value}">{text}</button>'
            for value, text, css in RATING_BUTTONS)
        rows += f'<div class="rrow"><span class="rlabel">{label}</span>{buttons}</div>'
    return (f'<div class="rate">{rows}'
            f'<input class="note" type="text" data-id="{esc(meme_id)}" '
            'placeholder="Нотатка: чому така оцінка"></div>')


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
            f'{render_viewpoints(parsed["viewpoints"])}'
            f'{render_rating(record["meme_id"])}')
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


def export_rows(records, meta):
    """Data embedded into the page; the CSV export is built from it."""
    rows = []
    for record in records:
        parsed = record.get("parsed") or {}
        in_meta = record["meme_id"] in meta.index
        rows.append({
            "meme_id": record["meme_id"],
            "image_path": clean(meta.loc[record["meme_id"], "image_path"]) if in_meta else "",
            "ocr_raw": clean(meta.loc[record["meme_id"], "ocr_raw"]) if in_meta else "",
            "ok": bool(record["ok"]),
            "description": parsed.get("description") or "",
            "key_message": parsed.get("key_message") or "",
            "viewpoints": json.dumps(parsed.get("viewpoints", []), ensure_ascii=False),
        })
    return rows


def build_html(model, records, meta, csv_name):
    ok = sum(1 for r in records if r["ok"])
    cards = "\n".join(
        render_card(i, record, meta.loc[record["meme_id"]] if record["meme_id"] in meta.index else None)
        for i, record in enumerate(records, start=1))
    data = json.dumps(export_rows(records, meta), ensure_ascii=False).replace("</", "<\\/")
    script = (SCRIPT
              .replace("__DATA__", data)
              .replace("__CSV_NAME__", json.dumps(csv_name)))
    return (
        '<!DOCTYPE html><html lang="uk"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>Semantic extraction review: {esc(model)}</title><style>{CSS}</style></head><body>'
        '<div class="bar">'
        '<strong id="progress"></strong>'
        '<div class="sums">Опис: <span id="sum-d"></span> &nbsp;|&nbsp; '
        'Key message: <span id="sum-k"></span> &nbsp;|&nbsp; '
        'Viewpoints: <span id="sum-v"></span></div>'
        '<div class="actions">'
        '<button type="button" id="reset">Очистити</button>'
        '<button type="button" id="export" class="primary">Експортувати CSV</button>'
        '</div></div>'
        f'<h1>Semantic extraction review: {esc(model)}</h1>'
        f'<div class="summary">{len(records)} memes · valid JSON: {ok} ({ok / max(len(records), 1):.0%})</div>'
        '<div class="warn" id="warn">Автозбереження в браузері недоступне. '
        'Регулярно натискай «Експортувати CSV».</div>'
        f'{RUBRIC}'
        f'{cards}'
        f'<script>{script}</script></body></html>')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", help="path to a results .jsonl file; the HTML is saved next to it")
    parser.add_argument("--model", help="model tag; reads outputs/semantic_pilot/<model>.jsonl")
    args = parser.parse_args()

    if args.jsonl:
        jsonl_path = Path(args.jsonl)
    elif args.model:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
        jsonl_path = OUTPUT_DIR / f"{safe_name}.jsonl"
    else:
        parser.error("provide --jsonl or --model")

    if not jsonl_path.exists():
        raise SystemExit(f"File not found: {jsonl_path}")

    records = load_records(jsonl_path)
    model = (records[0].get("model") if records else None) or jsonl_path.stem
    meta = pd.read_csv(DATA_PATH).drop_duplicates("meme_id").set_index("meme_id")

    output_path = jsonl_path.with_name(f"{jsonl_path.stem}_review.html")
    csv_name = f"{jsonl_path.stem}_review.csv"
    output_path.write_text(build_html(model, records, meta, csv_name), encoding="utf-8")
    print(f"Saved {len(records)} cards to: {output_path}")


if __name__ == "__main__":
    main()