"""Run one extraction method over a split and score it against ground truth.

Methods:
    textract   - Textract AnalyzeExpense fields only (baseline, no LLM)
    llm_text   - Textract OCR text -> Bedrock LLM with structured output
    llm_layout - Textract text rebuilt into visual rows -> Bedrock LLM
    llm_image  - receipt image -> multimodal Bedrock LLM (no Textract)
    llm_hybrid - receipt image + layout text -> multimodal Bedrock LLM

Examples:
    python scripts/run_eval.py --split dev --method textract
    python scripts/run_eval.py --split dev --method llm_text --model nova-2-lite
    python scripts/run_eval.py --split dev --method llm_image --model nova-2-lite --limit 10

    # Re-run only the documents that failed in an earlier run (settings are taken from it)
    python scripts/run_eval.py --retry <run_folder_name> --workers 1
"""

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime

from tqdm import tqdm

from idp.config import DATA_DIR, MODELS, RESULTS_DIR, TEXTRACT_EXPENSE_PRICE_PER_PAGE
from idp.extract import DEFAULT_PROMPT, PROMPTS, extract_with_llm
from idp.metrics import aggregate, score_document
from idp.ocr import analyze_expense, ocr_layout_text, ocr_text, textract_to_receipt
from idp.schema import Receipt

METHODS = ("textract", "llm_text", "llm_layout", "llm_image", "llm_hybrid")
USES_TEXTRACT = {"textract", "llm_text", "llm_layout", "llm_hybrid"}
USES_IMAGE = {"llm_image", "llm_hybrid"}


def list_docs(split: str, limit: int) -> list[str]:
    ids = sorted(p.stem for p in (DATA_DIR / "labels" / split).glob("*.json"))
    if not ids:
        raise SystemExit(f"No labels found for split '{split}'. Run scripts/prepare_cord.py first.")
    return ids[:limit] if limit else ids


def predict(
    split: str, doc_id: str, method: str, model: str | None, prompt: str
) -> tuple[Receipt, dict]:
    info = {"textract_fresh": False, "llm_cost_usd": 0.0, "latency_ms": None}
    if method in USES_TEXTRACT:
        analysis, info["textract_fresh"] = analyze_expense(split, doc_id)
    if method == "textract":
        return textract_to_receipt(analysis), info
    text = None
    if method == "llm_text":
        text = ocr_text(analysis)
    elif method in ("llm_layout", "llm_hybrid"):
        text = ocr_layout_text(analysis)
    image = None
    if method in USES_IMAGE:
        image = (DATA_DIR / "images" / split / f"{doc_id}.jpg").read_bytes()
    receipt, usage = extract_with_llm(model, ocr_text=text, image_bytes=image, prompt=prompt)
    info.update(usage)
    return receipt, info


def process(
    split: str, doc_id: str, method: str, model: str | None, prompt: str, fail_fast: bool
) -> dict:
    gt = Receipt.model_validate_json((DATA_DIR / "labels" / split / f"{doc_id}.json").read_text())
    error = None
    try:
        pred, info = predict(split, doc_id, method, model, prompt)
    except Exception as exc:  # noqa: BLE001 - one bad document must not stop the run
        if fail_fast:
            raise
        pred, info = Receipt(), {"textract_fresh": False, "llm_cost_usd": 0.0, "latency_ms": None}
        error = f"{type(exc).__name__}: {exc}"
    return {
        "doc_id": doc_id,
        "error": error,
        **info,
        "scores": score_document(pred, gt),
        "prediction": pred.model_dump(),
        "ground_truth": gt.model_dump(),
    }


def load_previous(run: str) -> tuple[dict, list[dict], list[str]]:
    """Return (settings, successful records, failed doc ids) of an earlier run."""
    run_dir = RESULTS_DIR / run
    old = json.loads((run_dir / "summary.json").read_text())
    kept, failed = [], []
    for line in open(run_dir / "predictions.jsonl"):
        r = json.loads(line)
        if r["error"]:
            failed.append(r["doc_id"])
            continue
        r["scores"] = score_document(
            Receipt.model_validate(r["prediction"]), Receipt.model_validate(r["ground_truth"])
        )
        # Reused documents cost nothing in this run
        r["llm_cost_usd_reused"] = r.pop("llm_cost_usd", 0.0)
        r["llm_cost_usd"] = 0.0
        r["textract_fresh"] = False
        kept.append(r)
    settings = {
        "split": old["split"],
        "method": old["method"],
        "model": old["model"],
        "prompt": old.get("prompt", "v1" if old["model"] else None),
    }
    return settings, kept, failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--model", choices=sorted(MODELS), help="required for llm_* methods")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, choices=sorted(PROMPTS))
    parser.add_argument("--limit", type=int, default=0, help="max documents (0 = all)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retry", help="run folder name: re-run only its failed documents")
    args = parser.parse_args()

    if args.retry:
        settings, kept, doc_ids = load_previous(args.retry)
        split, method, model, prompt = (
            settings["split"], settings["method"], settings["model"], settings["prompt"]
        )
        print(f"Retrying {len(doc_ids)} failed documents of {args.retry}")
    else:
        if not args.method:
            parser.error("--method is required (or use --retry)")
        if args.method != "textract" and not args.model:
            parser.error("--model is required for llm_* methods")
        split, method = args.split, args.method
        model = args.model if method != "textract" else None
        prompt = args.prompt if model else None
        kept, doc_ids = [], list_docs(split, args.limit)

    new_records = []
    todo = list(doc_ids)
    if todo and not args.retry:
        # First document runs without error handling, so auth or model-access
        # problems fail immediately with a full traceback instead of 100 errors.
        new_records.append(process(split, todo.pop(0), method, model, prompt, fail_fast=True))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, split, d, method, model, prompt, False) for d in todo]
        for future in tqdm(futures, total=len(futures), desc=method):
            new_records.append(future.result())

    records = sorted(kept + new_records, key=lambda r: r["doc_id"])
    # Keep variable names used below
    args.split, args.method = split, method

    summary = aggregate([r["scores"] for r in records])
    n = len(records)
    avg_llm_cost = sum(r["llm_cost_usd"] + r.get("llm_cost_usd_reused", 0.0) for r in records) / n
    textract_per_doc = TEXTRACT_EXPENSE_PRICE_PER_PAGE if args.method in USES_TEXTRACT else 0.0
    latencies = sorted(r["latency_ms"] for r in records if r["latency_ms"] is not None)

    run_name = (
        f"{datetime.now():%Y%m%d-%H%M%S}_{args.split}_{args.method}_{model or 'none'}"
        + (f"_{prompt}" if prompt else "")
    )
    summary.update(
        {
            "run": run_name,
            "split": args.split,
            "method": args.method,
            "model": model,
            "prompt": prompt,
            "n_docs": n,
            "n_errors": sum(1 for r in records if r["error"]),
            # Steady-state cost if every document were processed from scratch
            "cost_per_1k_docs_usd": round(1000 * (avg_llm_cost + textract_per_doc), 3),
            # What this particular run actually cost (cached Textract calls are free)
            "spent_this_run_usd": round(
                sum(r["llm_cost_usd"] for r in records)
                + TEXTRACT_EXPENSE_PRICE_PER_PAGE * sum(r["textract_fresh"] for r in records),
                4,
            ),
            "latency_ms_p50": statistics.median(latencies) if latencies else None,
            "latency_ms_p95": latencies[int(0.95 * (len(latencies) - 1))] if latencies else None,
        }
    )

    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "predictions.jsonl", "w") as f:
        for r in records:
            r = {**r, "scores": {k: asdict(v) for k, v in r["scores"].items()}}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nRun: {run_name}  ({n} docs, {summary['n_errors']} errors)")
    for name, m in summary["fields"].items():
        print(f"  {name:<9} P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    print(f"  header doc accuracy: {summary['header_doc_accuracy']:.3f}")
    print(f"  full doc accuracy:   {summary['full_doc_accuracy']:.3f}")
    print(f"  cost per 1k docs:    ${summary['cost_per_1k_docs_usd']}")
    print(f"  spent this run:      ${summary['spent_this_run_usd']}")
    print(f"  latency p50/p95 ms:  {summary['latency_ms_p50']} / {summary['latency_ms_p95']}")
    print(f"Saved to {run_dir}")
    if summary["n_errors"]:
        print(
            f"\n!!! WARNING: {summary['n_errors']} documents failed and were scored as empty."
            f"\n!!! Metrics of this run are too low. Fix the cause, then run:"
            f"\n!!!   python scripts/run_eval.py --retry {run_name} --workers 1"
        )


if __name__ == "__main__":
    main()
