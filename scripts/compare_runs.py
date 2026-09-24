"""Print a markdown table comparing runs in results/ (ready to paste into the README).

Usage:
    python scripts/compare_runs.py                 # hides runs with fewer than 50 docs
    python scripts/compare_runs.py --min-docs 1    # show everything, including smoke tests
"""

import argparse
import json

from idp.config import RESULTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-docs", type=int, default=50)
    parser.add_argument("--split", choices=["dev", "test"])
    args = parser.parse_args()

    summaries = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*/summary.json"))]
    summaries = [
        s for s in summaries
        if s["n_docs"] >= args.min_docs and (args.split is None or s["split"] == args.split)
    ]
    if not summaries:
        raise SystemExit("No matching runs found in results/")

    header = (
        "| Split | N | Method | Model | Prompt | Total F1 | Subtotal F1 | Tax F1 | Items F1 "
        "| Items F1 (fuzzy) | Header doc acc | Full doc acc | $ / 1k docs | p50 ms |"
    )
    print(header)
    print("|" + "---|" * (header.count("|") - 1))
    for s in summaries:
        f = s["fields"]
        # Runs made before prompt versions existed used v1
        prompt = s.get("prompt", "v1" if s["model"] else None) or "-"
        fuzzy = f"{f['items_fuzzy']['f1']:.3f}" if "items_fuzzy" in f else "-"
        print(
            f"| {s['split']} | {s['n_docs']} | {s['method']} | {s['model'] or '-'} | {prompt} "
            f"| {f['total']['f1']:.3f} | {f['subtotal']['f1']:.3f} | {f['tax']['f1']:.3f} "
            f"| {f['items']['f1']:.3f} | {fuzzy} | {s['header_doc_accuracy']:.3f} "
            f"| {s.get('full_doc_accuracy', 0):.3f} "
            f"| {s['cost_per_1k_docs_usd']} | {s['latency_ms_p50'] or '-'} |"
        )


if __name__ == "__main__":
    main()
