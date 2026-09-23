"""Textract AnalyzeExpense with a two-level cache (local disk, then S3).

Each page is sent to Textract only once. All later experiments reuse the cached
JSON, which is the main cost saving in this project.
"""

import json

from botocore.exceptions import ClientError

from idp.aws import client
from idp.config import DATA_DIR, require_bucket
from idp.schema import LineItem, Receipt


def analyze_expense(split: str, doc_id: str) -> tuple[dict, bool]:
    """Return (AnalyzeExpense response, whether Textract was actually called)."""
    local_path = DATA_DIR / "ocr" / split / f"{doc_id}.json"
    if local_path.exists():
        return json.loads(local_path.read_text()), False

    bucket = require_bucket()
    s3 = client("s3")
    cache_key = f"ocr/{split}/{doc_id}.json"
    fresh = False
    try:
        result = json.loads(s3.get_object(Bucket=bucket, Key=cache_key)["Body"].read())
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("NoSuchKey", "404"):
            raise
        result = client("textract").analyze_expense(
            Document={"S3Object": {"Bucket": bucket, "Name": f"images/{split}/{doc_id}.jpg"}}
        )
        result.pop("ResponseMetadata", None)
        s3.put_object(
            Bucket=bucket,
            Key=cache_key,
            Body=json.dumps(result, default=str).encode(),
            ContentType="application/json",
        )
        fresh = True

    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(json.dumps(result, default=str))
    return result, fresh


def _first_document(result: dict) -> dict:
    docs = result.get("ExpenseDocuments") or []
    return docs[0] if docs else {}


def ocr_text(result: dict) -> str:
    """Plain text lines detected by Textract, in reading order."""
    blocks = _first_document(result).get("Blocks") or []
    return "\n".join(b["Text"] for b in blocks if b.get("BlockType") == "LINE" and b.get("Text"))


def _best_summary_value(doc: dict, field_type: str) -> str | None:
    best, best_conf = None, -1.0
    for field in doc.get("SummaryFields") or []:
        if (field.get("Type") or {}).get("Text") != field_type:
            continue
        value = field.get("ValueDetection") or {}
        text, conf = value.get("Text"), value.get("Confidence", 0.0)
        if text and conf > best_conf:
            best, best_conf = text, conf
    return best


def textract_to_receipt(result: dict) -> Receipt:
    """Baseline extractor: map Textract's own expense fields to our schema."""
    doc = _first_document(result)
    items = []
    for group in doc.get("LineItemGroups") or []:
        for line in group.get("LineItems") or []:
            fields = {}
            for f in line.get("LineItemExpenseFields") or []:
                key = (f.get("Type") or {}).get("Text")
                value = (f.get("ValueDetection") or {}).get("Text")
                if key and value and key not in fields:
                    fields[key] = value
            if fields.get("ITEM"):
                items.append(
                    LineItem(
                        name=fields["ITEM"],
                        quantity=fields.get("QUANTITY"),
                        price=fields.get("PRICE"),
                    )
                )
    return Receipt(
        subtotal=_best_summary_value(doc, "SUBTOTAL"),
        tax=_best_summary_value(doc, "TAX"),
        total=_best_summary_value(doc, "TOTAL"),
        items=items,
    )
