"""Show where a run makes mistakes, broken down by field and error type.

Usage:
    python scripts/inspect_errors.py                        # latest full run
    python scripts/inspect_errors.py --method llm_text      # latest run of this method
    python scripts/inspect_errors.py --run <run_folder_name>
    python scripts/inspect_errors.py --examples 10
"""

import argparse
import json
from collections import defaultdict

from idp.config import RESULTS_DIR
from idp.metrics import HEADER_FIELDS
from idp.normalize import norm_amount, norm_text


def find_run(run: str | None, method: str | None, min_docs: int):
    if run:
        return RESULTS_DIR / run
    candidates = []
    for path in sorted(RESULTS_DIR.glob("*/summary.json")):
        summary = json.loads(path.read_text())
        if summary["n_docs"] >= min_docs and (method is None or summary["method"] == method):
            candidates.append(path.parent)
    if not candidates:
        raise SystemExit("No matching runs found in results/")
    return candidates[-1]  # folder names start with a timestamp, so the last is the newest


def classify_items(pred_items: list[dict], gt_items: list[dict]) -> dict[str, list]:
    """Pair unmatched items to explain why they did not match."""
    key = lambda i: (norm_text(i.get("name")), norm_amount(i.get("price")))
    pred_left = list(pred_items)
    gt_left = []
    for gt in gt_items:
        match = next((p for p in pred_left if key(p) == key(gt)), None)
        if match is not None:
            pred_left.remove(match)
        else:
            gt_left.append(gt)

    errors = defaultdict(list)
    for gt in gt_left:
        same_name = next((p for p in pred_left if key(p)[0] == key(gt)[0]), None)
        if same_name is not None:
            pred_left.remove(same_name)
            errors["wrong price"].append((gt, same_name))
            continue
        same_price = next(
            (p for p in pred_left if key(p)[1] is not None and key(p)[1] == key(gt)[1]), None
        )
        if same_price is not None:
            pred_left.remove(same_price)
            errors["name differs"].append((gt, same_price))
            continue
        errors["missed"].append((gt, None))
    for pred in pred_left:
        errors["extra"].append((None, pred))
    return errors


def fmt_item(item: dict | None) -> str:
    if item is None:
        return "-"
    return f"{item.get('name')!r} @ {item.get('price')!r}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="results folder name")
    parser.add_argument("--method", help="pick the latest run of this method")
    parser.add_argument("--min-docs", type=int, default=50)
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()

    run_dir = find_run(args.run, args.method, args.min_docs)
    records = [json.loads(line) for line in open(run_dir / "predictions.jsonl")]
    print(f"Run: {run_dir.name}  ({len(records)} docs)")

    failed = [r for r in records if r.get("error")]
    if failed:
        print(f"\n{len(failed)} documents raised errors, e.g. {failed[0]['doc_id']}: {failed[0]['error']}")

    # Header fields
    for field in HEADER_FIELDS:
        buckets = defaultdict(list)
        for r in records:
            pred, gt = r["prediction"][field], r["ground_truth"][field]
            p, g = norm_amount(pred), norm_amount(gt)
            if p == g:
                continue
            if p is None:
                buckets["missed"].append((r["doc_id"], pred, gt))
            elif g is None:
                buckets["not in label"].append((r["doc_id"], pred, gt))
            else:
                buckets["wrong value"].append((r["doc_id"], pred, gt))
        total = sum(len(v) for v in buckets.values())
        print(f"\n== {field}: {total} errors ==")
        for kind, rows in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
            print(f"  {kind}: {len(rows)}")
            for doc_id, pred, gt in rows[: args.examples]:
                print(f"    {doc_id}: predicted {pred!r}, label {gt!r}")

    # Line items
    item_buckets = defaultdict(list)
    per_doc_errors = []
    for r in records:
        errors = classify_items(r["prediction"]["items"], r["ground_truth"]["items"])
        per_doc_errors.append((sum(len(v) for v in errors.values()), r["doc_id"]))
        for kind, pairs in errors.items():
            item_buckets[kind].extend((r["doc_id"], gt, pred) for gt, pred in pairs)

    total = sum(len(v) for v in item_buckets.values())
    print(f"\n== items: {total} errors ==")
    for kind, rows in sorted(item_buckets.items(), key=lambda kv: -len(kv[1])):
        print(f"  {kind}: {len(rows)}")
        for doc_id, gt, pred in rows[: args.examples]:
            print(f"    {doc_id}: label {fmt_item(gt)}  |  predicted {fmt_item(pred)}")

    worst = sorted(per_doc_errors, reverse=True)[:5]
    print("\nDocuments with most item errors:", ", ".join(f"{d} ({n})" for n, d in worst if n))


if __name__ == "__main__":
    main()
