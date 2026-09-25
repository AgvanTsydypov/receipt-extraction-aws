"""Features for the confidence model.

The model decides whether a document can be accepted automatically. Its inputs are
signals that do not need ground truth: agreement between two independent extractions
(e.g. Claude hybrid vs Textract), arithmetic consistency of the receipt and Textract's
own confidence scores.
"""

import json
import statistics
from types import SimpleNamespace

from idp.config import DATA_DIR
from idp.metrics import HEADER_FIELDS, score_items_fuzzy
from idp.normalize import norm_amount

TEXTRACT_TYPES = {"subtotal": "SUBTOTAL", "tax": "TAX", "total": "TOTAL"}


def _num(value) -> int | None:
    digits = norm_amount(value)
    return int(digits) if digits else None


def _as_items(items: list[dict]) -> list:
    return [SimpleNamespace(name=i.get("name"), price=i.get("price")) for i in items]


def textract_signals(split: str, doc_id: str) -> dict[str, float]:
    """Confidence scores from the locally cached Textract response of a dataset document."""
    path = DATA_DIR / "ocr" / split / f"{doc_id}.json"
    return textract_signals_from_result(json.loads(path.read_text()) if path.exists() else {})


def textract_signals_from_result(result: dict) -> dict[str, float]:
    """Confidence scores from a Textract AnalyzeExpense response."""
    docs = result.get("ExpenseDocuments") or [{}]
    doc = docs[0]

    signals = {}
    for field, textract_type in TEXTRACT_TYPES.items():
        confs = [
            (f.get("ValueDetection") or {}).get("Confidence", 0.0)
            for f in doc.get("SummaryFields") or []
            if (f.get("Type") or {}).get("Text") == textract_type
        ]
        signals[f"tx_conf_{field}"] = max(confs, default=0.0) / 100

    words = [
        b.get("Confidence", 0.0) / 100
        for b in doc.get("Blocks") or []
        if b.get("BlockType") == "WORD"
    ]
    signals["ocr_word_conf_mean"] = statistics.fmean(words) if words else 0.0
    signals["ocr_word_conf_min"] = min(words, default=0.0)
    signals["ocr_low_conf_share"] = (
        sum(w < 0.8 for w in words) / len(words) if words else 1.0
    )
    return signals


def build_features(primary: dict, secondary: dict, signals: dict[str, float]) -> dict[str, float]:
    """primary/secondary are prediction dicts from two different extraction methods."""
    f: dict[str, float] = {}

    # Agreement between the two methods on header fields
    for field in HEADER_FIELDS:
        a, b = norm_amount(primary.get(field)), norm_amount(secondary.get(field))
        f[f"agree_{field}"] = float(a == b)
        f[f"has_{field}"] = float(a is not None)
        f[f"one_missing_{field}"] = float((a is None) != (b is None))
    f["n_agree_header"] = sum(f[f"agree_{x}"] for x in HEADER_FIELDS)

    # Agreement on line items
    items_a, items_b = primary.get("items") or [], secondary.get("items") or []
    f["items_agreement_f1"] = score_items_fuzzy(_as_items(items_a), _as_items(items_b)).f1
    f["n_items"] = float(len(items_a))
    f["n_items_diff"] = float(abs(len(items_a) - len(items_b)))
    f["items_missing_price"] = float(sum(1 for i in items_a if _num(i.get("price")) is None))

    # Arithmetic consistency of the primary prediction
    total, subtotal, tax = (_num(primary.get(x)) for x in ("total", "subtotal", "tax"))
    item_sum = sum(p for p in (_num(i.get("price")) for i in items_a) if p is not None)
    f["items_sum_eq_subtotal"] = float(subtotal is not None and item_sum == subtotal)
    f["items_sum_eq_total"] = float(total is not None and item_sum == total)
    f["subtotal_plus_tax_eq_total"] = float(
        None not in (subtotal, tax, total) and subtotal + tax == total
    )
    f["total_ge_subtotal"] = float(
        total is not None and subtotal is not None and total >= subtotal
    )

    f.update(signals)
    return f


def is_correct(record: dict, target: str) -> bool:
    """Ground-truth label: were all header fields (or everything) extracted correctly?"""
    scores = record["scores"]
    perfect = lambda name: scores[name]["fp"] == 0 and scores[name]["fn"] == 0
    header = all(perfect(x) for x in HEADER_FIELDS)
    return header if target == "header" else header and perfect("items")
