"""Helpers for locating and loading evaluation runs in results/."""

import json
from pathlib import Path

from idp.config import RESULTS_DIR


def find_run(selector: str, split: str, min_docs: int = 50) -> Path:
    """Newest full run matching 'method[:model[:prompt]]' on the given split."""
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


def load_records(run_dir: Path) -> dict[str, dict]:
    """doc_id -> record from predictions.jsonl."""
    records = {}
    with open(run_dir / "predictions.jsonl") as f:
        for line in f:
            r = json.loads(line)
            records[r["doc_id"]] = r
    return records
