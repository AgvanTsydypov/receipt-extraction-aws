"""Recompute metrics for all saved runs from their predictions (free, no AWS calls).

Use after changing metrics.py, so old and new runs are compared with the same metrics.
"""

import json
from dataclasses import asdict

from idp.config import RESULTS_DIR
from idp.metrics import aggregate, score_document
from idp.schema import Receipt


def main() -> None:
    runs = sorted(p.parent for p in RESULTS_DIR.glob("*/summary.json"))
    for run_dir in runs:
        records = [json.loads(line) for line in open(run_dir / "predictions.jsonl")]
        scores = []
        for r in records:
            pred = Receipt.model_validate(r["prediction"])
            gt = Receipt.model_validate(r["ground_truth"])
            r["scores"] = score_document(pred, gt)
            scores.append(r["scores"])

        summary = json.loads((run_dir / "summary.json").read_text())
        summary.update(aggregate(scores))
        if "prompt" not in summary:
            summary["prompt"] = "v1" if summary["model"] else None
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        with open(run_dir / "predictions.jsonl", "w") as f:
            for r in records:
                r = {**r, "scores": {k: asdict(v) for k, v in r["scores"].items()}}
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Rescored {run_dir.name}")


if __name__ == "__main__":
    main()
