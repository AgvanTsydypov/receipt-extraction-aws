"""Send test receipts through the deployed AWS pipeline and compare results with labels.

Uploads images to s3://<bucket>/incoming/<run>/, waits until every document appears in
DynamoDB, then prints per-document routing decisions and a summary.

Usage:
    python scripts/pipeline_demo.py --n 10
    python scripts/pipeline_demo.py --n 30 --interval 2
"""

import argparse
import json
import time
from datetime import datetime

from idp.aws import client
from idp.config import DATA_DIR, require_bucket
from idp.metrics import HEADER_FIELDS
from idp.normalize import norm_amount

TABLE = "idp-documents"


def scan_prefix(prefix: str) -> list[dict]:
    """All DynamoDB items whose source_key starts with the prefix (small demo table)."""
    items = []
    paginator = client("dynamodb").get_paginator("scan")
    for page in paginator.paginate(
        TableName=TABLE,
        FilterExpression="begins_with(source_key, :p)",
        ExpressionAttributeValues={":p": {"S": prefix}},
    ):
        items.extend(page["Items"])
    return items


def header_correct(pred: dict, label: dict) -> bool:
    return all(norm_amount(pred.get(f)) == norm_amount(label.get(f)) for f in HEADER_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["dev", "test", "train"])
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--interval", type=float, default=3.0, help="seconds between uploads")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    bucket = require_bucket()
    s3 = client("s3")
    doc_ids = sorted(p.stem for p in (DATA_DIR / "images" / args.split).glob("*.jpg"))[: args.n]
    run = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = f"incoming/{run}/"

    for doc_id in doc_ids:
        s3.put_object(
            Bucket=bucket,
            Key=f"{prefix}{doc_id}.jpg",
            Body=(DATA_DIR / "images" / args.split / f"{doc_id}.jpg").read_bytes(),
            ContentType="image/jpeg",
        )
        time.sleep(args.interval)  # gentle pacing reduces Bedrock throttling
    print(f"Uploaded {len(doc_ids)} receipts to s3://{bucket}/{prefix}")

    started = time.time()
    items: dict[str, dict] = {}
    while len(items) < len(doc_ids) and time.time() - started < args.timeout:
        time.sleep(10)
        for item in scan_prefix(prefix):
            items[item["source_key"]["S"]] = item
        print(f"  processed {len(items)}/{len(doc_ids)}  ({time.time() - started:.0f}s)")

    print(f"\n| Document | Status | Confidence | Predicted total | Label total | Header correct |")
    print("|---|---|---|---|---|---|")
    stats = {"AUTO_APPROVED": [], "NEEDS_REVIEW": [], "FAILED": 0, "MISSING": 0}
    for doc_id in doc_ids:
        item = items.get(f"{prefix}{doc_id}.jpg")
        if item is None:
            stats["MISSING"] += 1
            print(f"| {doc_id} | not processed yet | - | - | - | - |")
            continue
        status = item["status"]["S"]
        if status == "FAILED":
            stats["FAILED"] += 1
            print(f"| {doc_id} | FAILED: {item['error']['S']} | - | - | - | - |")
            continue
        pred = json.loads(item["result"]["S"])
        label = json.loads((DATA_DIR / "labels" / args.split / f"{doc_id}.json").read_text())
        correct = header_correct(pred, label)
        stats[status].append(correct)
        print(
            f"| {doc_id} | {status} | {float(item['confidence']['N']):.3f} "
            f"| {pred.get('total')} | {label.get('total')} | {'yes' if correct else 'NO'} |"
        )

    approved, review = stats["AUTO_APPROVED"], stats["NEEDS_REVIEW"]
    print(f"\nAuto-approved: {len(approved)}/{len(doc_ids)}, correct among them: {sum(approved)}")
    print(f"Sent to review: {len(review)}, of which actually wrong: {len(review) - sum(review)}")
    print(f"Failed: {stats['FAILED']}, not finished: {stats['MISSING']}")


if __name__ == "__main__":
    main()
