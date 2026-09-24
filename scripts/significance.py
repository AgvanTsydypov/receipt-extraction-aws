"""Paired comparison of two runs on the same documents, with a bootstrap confidence interval.

Runs are selected as method[:model[:prompt]]; the newest full run that matches is used.

Examples:
    python scripts/significance.py textract llm_hybrid:claude-haiku-4.5:v2
    python scripts/significance.py textract llm_image:claude-haiku-4.5:v2 --split test
"""

import argparse
import json
import random

from idp.config import RESULTS_DIR
from idp.metrics import HEADER_FIELDS


def find_run(selector: str, split: str, min_docs: int):
    parts = selector.split(":")
    method = parts[0]
    model = parts[1] if len(parts) > 1 else None
    prompt = parts[2] if len(parts) > 2 else None
    matches = []
    for path in sorted(RESULTS_DIR.glob("*/summary.json")):
        s = json.loads(path.read_text())
        if (
            s["split"] == split
            and s["n_docs"] >= min_docs
            and s["method"] == method
            and (model is None or s["model"] == model)
            and (prompt is None or s.get("prompt", "v1" if s["model"] else None) == prompt)
        ):
            matches.append(path.parent)
    if not matches:
        raise SystemExit(f"No {split} run matches '{selector}'")
    return matches[-1]


def per_doc_correctness(run_dir) -> dict[str, dict[str, bool]]:
    """doc_id -> {'header': all header fields correct, 'full': header and all items correct}."""
    result = {}
    for line in open(run_dir / "predictions.jsonl"):
        r = json.loads(line)
        perfect = lambda name: r["scores"][name]["fp"] == 0 and r["scores"][name]["fn"] == 0
        header = all(perfect(f) for f in HEADER_FIELDS)
        result[r["doc_id"]] = {"header": header, "full": header and perfect("items")}
    return result


def bootstrap_diff(a: list[bool], b: list[bool], n_boot: int, seed: int) -> tuple[float, float]:
    """95% CI for mean(b) - mean(a), resampling documents with replacement."""
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] - a[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a", help="baseline, e.g. textract")
    parser.add_argument("b", help="challenger, e.g. llm_hybrid:claude-haiku-4.5:v2")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--min-docs", type=int, default=50)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_a = find_run(args.a, args.split, args.min_docs)
    run_b = find_run(args.b, args.split, args.min_docs)
    corr_a, corr_b = per_doc_correctness(run_a), per_doc_correctness(run_b)
    docs = sorted(set(corr_a) & set(corr_b))

    print(f"A: {run_a.name}")
    print(f"B: {run_b.name}")
    print(f"Paired documents: {len(docs)}\n")

    for metric in ("header", "full"):
        a = [corr_a[d][metric] for d in docs]
        b = [corr_b[d][metric] for d in docs]
        both = sum(x and y for x, y in zip(a, b))
        only_a = sum(x and not y for x, y in zip(a, b))
        only_b = sum(y and not x for x, y in zip(a, b))
        neither = len(docs) - both - only_a - only_b
        diff = (sum(b) - sum(a)) / len(docs)
        low, high = bootstrap_diff(a, b, args.n_boot, args.seed)
        verdict = (
            "B is better" if low > 0 else "A is better" if high < 0 else "no significant difference"
        )
        print(f"{metric} doc accuracy: A={sum(a) / len(docs):.3f}  B={sum(b) / len(docs):.3f}")
        print(f"  both correct {both}, only A {only_a}, only B {only_b}, both wrong {neither}")
        print(f"  B - A = {diff:+.3f}, 95% CI [{low:+.3f}, {high:+.3f}] -> {verdict}\n")


if __name__ == "__main__":
    main()
