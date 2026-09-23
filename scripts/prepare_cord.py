"""Download CORD v2 receipts, convert labels to our schema and upload images to S3.

Usage:
    python scripts/prepare_cord.py                 # dev (100) + test (100), uploads to S3
    python scripts/prepare_cord.py --limit 20      # quick smoke test
    python scripts/prepare_cord.py --no-upload     # local only
"""

import argparse
import io
import json

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from idp.aws import client
from idp.config import DATA_DIR, require_bucket
from idp.schema import LineItem, Receipt

# Our split name -> Hugging Face split name
SPLITS = {"dev": "validation", "test": "test"}
MAX_SIDE = 1600  # keeps images small for Bedrock and fast for Textract


def _text(value) -> str | None:
    if value is None or isinstance(value, dict):
        return None
    if isinstance(value, list):
        value = " ".join(str(v) for v in value if v is not None and not isinstance(v, dict))
    value = str(value).strip()
    return value or None


def _as_dict(value) -> dict:
    if isinstance(value, list):
        value = value[0] if value else {}
    return value if isinstance(value, dict) else {}


def cord_to_receipt(gt_parse: dict) -> Receipt:
    menu = gt_parse.get("menu") or []
    if isinstance(menu, dict):
        menu = [menu]
    items = []
    for entry in menu:
        if not isinstance(entry, dict):
            continue
        name = _text(entry.get("nm"))
        if name:
            items.append(
                LineItem(name=name, quantity=_text(entry.get("cnt")), price=_text(entry.get("price")))
            )
    sub_total = _as_dict(gt_parse.get("sub_total"))
    total = _as_dict(gt_parse.get("total"))
    return Receipt(
        subtotal=_text(sub_total.get("subtotal_price")),
        tax=_text(sub_total.get("tax_price")),
        total=_text(total.get("total_price")),
        items=items,
    )


def _to_jpeg(image: Image.Image) -> bytes:
    image = image.convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="max documents per split (0 = all)")
    parser.add_argument("--no-upload", action="store_true", help="skip uploading images to S3")
    args = parser.parse_args()

    bucket = None if args.no_upload else require_bucket()
    s3 = None if args.no_upload else client("s3")

    for alias, hf_split in SPLITS.items():
        dataset = load_dataset("naver-clova-ix/cord-v2", split=hf_split)
        image_dir = DATA_DIR / "images" / alias
        label_dir = DATA_DIR / "labels" / alias
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        total = min(args.limit, len(dataset)) if args.limit else len(dataset)
        for idx in tqdm(range(total), desc=f"{alias}"):
            example = dataset[idx]
            doc_id = f"{alias}_{idx:04d}"
            gt_parse = json.loads(example["ground_truth"])["gt_parse"]

            receipt = cord_to_receipt(gt_parse)
            (label_dir / f"{doc_id}.json").write_text(receipt.model_dump_json(indent=2))

            jpeg = _to_jpeg(example["image"])
            (image_dir / f"{doc_id}.jpg").write_bytes(jpeg)
            if s3 is not None:
                s3.put_object(
                    Bucket=bucket,
                    Key=f"images/{alias}/{doc_id}.jpg",
                    Body=jpeg,
                    ContentType="image/jpeg",
                )

    print(f"Done. Local data in {DATA_DIR}")


if __name__ == "__main__":
    main()
