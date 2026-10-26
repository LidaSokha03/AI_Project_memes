"""
Summary of the VLM semantic extraction results.

Reads a results .jsonl file (for example the merged 04_semantic.jsonl) and
  - prints and saves summary statistics (how many memes are valid, viewpoints per
    meme, sentiment distribution, time per meme, most frequent targets);
  - saves a flat table with one row per viewpoint, ready for the next steps
    (embeddings of target / aspect / opinion and hierarchical clustering).

Both files are saved next to the .jsonl file.

Usage (from the project root):
    python src/semantic_report.py --jsonl data/processed/04_semantic.jsonl
"""

from collections import Counter
from pathlib import Path
import argparse
import json

import pandas as pd


TOP_N = 30


def load_records(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def flatten(records):
    rows = []
    for record in records:
        if not record["ok"]:
            continue
        parsed = record["parsed"]
        for index, viewpoint in enumerate(parsed["viewpoints"]):
            rows.append({
                "meme_id": record["meme_id"],
                "viewpoint_index": index,
                "target": viewpoint["target"],
                "aspect": viewpoint["aspect"],
                "opinion": viewpoint["opinion"],
                "sentiment": viewpoint["sentiment"],
                "key_message": parsed["key_message"],
            })
    return pd.DataFrame(rows, columns=["meme_id", "viewpoint_index", "target", "aspect",
                                       "opinion", "sentiment", "key_message"])


def summarize(records, table):
    lines = []
    total = len(records)
    ok = [r for r in records if r["ok"]]
    failed = [r for r in records if not r["ok"]]

    settings = sorted({(str(r.get("model")), str(r.get("prompt"))) for r in records})
    lines.append(f"Model / prompt: {', '.join(f'{m} / {p}' for m, p in settings)}")
    lines.append(f"Memes: {total}, valid: {len(ok)} ({len(ok) / max(total, 1):.1%}), failed: {len(failed)}")
    if failed:
        reasons = Counter(r.get("error") for r in failed).most_common(5)
        lines.append("Failure reasons: " + "; ".join(f"{reason} ({count})" for reason, count in reasons))

    times = pd.Series([r["seconds"] for r in records if r.get("seconds") is not None])
    if not times.empty:
        lines.append(f"Time per meme: median {times.median():.1f} s, mean {times.mean():.1f} s "
                     "(the first meme of each session includes model loading)")

    if ok:
        counts = pd.Series([len(r["parsed"]["viewpoints"]) for r in ok])
        lines.append(f"Viewpoints per meme: mean {counts.mean():.2f}; "
                     + ", ".join(f"{k}: {int((counts == k).sum())}" for k in sorted(counts.unique())))
        lines.append(f"Memes with no viewpoints: {int((counts == 0).sum())}")

    if not table.empty:
        sentiment = table["sentiment"].value_counts()
        lines.append("Sentiment (share of viewpoints): "
                     + ", ".join(f"{k} {v / len(table):.0%}" for k, v in sentiment.items()))

        targets = table["target"].str.strip().str.lower()
        counts = Counter(targets)
        unique_ratio = len(counts) / len(targets)
        top10_share = sum(c for _, c in counts.most_common(10)) / len(targets)
        lines.append(f"Viewpoints: {len(table)}, unique targets: {len(counts)} "
                     f"({unique_ratio:.0%} of viewpoints), top-10 targets cover {top10_share:.0%}")
        lines.append(f"Top {TOP_N} targets:")
        lines.extend(f"  {count:4d}  {target}" for target, count in counts.most_common(TOP_N))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="path to the results .jsonl file")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise SystemExit(f"File not found: {jsonl_path}")

    records = load_records(jsonl_path)
    table = flatten(records)
    text = summarize(records, table)

    table_path = jsonl_path.with_name(f"{jsonl_path.stem}_viewpoints.csv")
    summary_path = jsonl_path.with_name(f"{jsonl_path.stem}_summary.txt")
    table.to_csv(table_path, index=False)
    summary_path.write_text(text + "\n", encoding="utf-8")

    print(text)
    print(f"\nSaved: {table_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()