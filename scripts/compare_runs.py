"""Print a markdown table comparing all runs in results/ (ready to paste into the README)."""

import json

from idp.config import RESULTS_DIR


def main() -> None:
    summaries = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*/summary.json"))]
    if not summaries:
        raise SystemExit("No runs found in results/")

    header = "| Split | Method | Model | Total F1 | Subtotal F1 | Tax F1 | Items F1 | Header doc acc | $ / 1k docs | p50 ms |"
    print(header)
    print("|" + "---|" * (header.count("|") - 1))
    for s in summaries:
        f = s["fields"]
        print(
            f"| {s['split']} | {s['method']} | {s['model'] or '-'} "
            f"| {f['total']['f1']:.3f} | {f['subtotal']['f1']:.3f} | {f['tax']['f1']:.3f} "
            f"| {f['items']['f1']:.3f} | {s['header_doc_accuracy']:.3f} "
            f"| {s['cost_per_1k_docs_usd']} | {s['latency_ms_p50'] or '-'} |"
        )


if __name__ == "__main__":
    main()
