"""Run one extraction method over a split and score it against ground truth.

Methods:
    textract   - Textract AnalyzeExpense fields only (baseline, no LLM)
    llm_text   - Textract OCR text -> Bedrock LLM with structured output
    llm_image  - receipt image -> multimodal Bedrock LLM (no Textract)

Examples:
    python scripts/run_eval.py --split dev --method textract
    python scripts/run_eval.py --split dev --method llm_text --model nova-lite
    python scripts/run_eval.py --split dev --method llm_image --model nova-lite --limit 10
"""

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime

from tqdm import tqdm

from idp.config import DATA_DIR, MODELS, RESULTS_DIR, TEXTRACT_EXPENSE_PRICE_PER_PAGE
from idp.extract import extract_with_llm
from idp.metrics import aggregate, score_document
from idp.ocr import analyze_expense, ocr_text, textract_to_receipt
from idp.schema import Receipt

METHODS = ("textract", "llm_text", "llm_image")
USES_TEXTRACT = {"textract", "llm_text"}


def list_docs(split: str, limit: int) -> list[str]:
    ids = sorted(p.stem for p in (DATA_DIR / "labels" / split).glob("*.json"))
    if not ids:
        raise SystemExit(f"No labels found for split '{split}'. Run scripts/prepare_cord.py first.")
    return ids[:limit] if limit else ids


def predict(split: str, doc_id: str, method: str, model: str | None) -> tuple[Receipt, dict]:
    info = {"textract_fresh": False, "llm_cost_usd": 0.0, "latency_ms": None}
    if method in USES_TEXTRACT:
        analysis, info["textract_fresh"] = analyze_expense(split, doc_id)
    if method == "textract":
        return textract_to_receipt(analysis), info
    if method == "llm_text":
        receipt, usage = extract_with_llm(model, ocr_text=ocr_text(analysis))
    else:
        image = (DATA_DIR / "images" / split / f"{doc_id}.jpg").read_bytes()
        receipt, usage = extract_with_llm(model, image_bytes=image)
    info.update(usage)
    return receipt, info


def process(split: str, doc_id: str, method: str, model: str | None, fail_fast: bool) -> dict:
    gt = Receipt.model_validate_json((DATA_DIR / "labels" / split / f"{doc_id}.json").read_text())
    error = None
    try:
        pred, info = predict(split, doc_id, method, model)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--model", choices=sorted(MODELS), help="required for llm_* methods")
    parser.add_argument("--limit", type=int, default=0, help="max documents (0 = all)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.method != "textract" and not args.model:
        parser.error("--model is required for llm_text and llm_image")
    model = args.model if args.method != "textract" else None

    doc_ids = list_docs(args.split, args.limit)

    # First document runs without error handling, so auth or model-access
    # problems fail immediately with a full traceback instead of 100 errors.
    records = [process(args.split, doc_ids[0], args.method, model, fail_fast=True)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(process, args.split, d, args.method, model, False) for d in doc_ids[1:]
        ]
        for future in tqdm(futures, total=len(futures), desc=args.method):
            records.append(future.result())

    summary = aggregate([r["scores"] for r in records])
    n = len(records)
    avg_llm_cost = sum(r["llm_cost_usd"] for r in records) / n
    textract_per_doc = TEXTRACT_EXPENSE_PRICE_PER_PAGE if args.method in USES_TEXTRACT else 0.0
    latencies = sorted(r["latency_ms"] for r in records if r["latency_ms"] is not None)

    run_name = f"{datetime.now():%Y%m%d-%H%M%S}_{args.split}_{args.method}_{model or 'none'}"
    summary.update(
        {
            "run": run_name,
            "split": args.split,
            "method": args.method,
            "model": model,
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


if __name__ == "__main__":
    main()
