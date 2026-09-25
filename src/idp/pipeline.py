"""AWS Lambda handlers for the production pipeline.

One container image, three handlers, orchestrated by Step Functions:
    textract_handler -> extract_handler -> score_handler -> (Step Functions writes to DynamoDB)

Each handler receives the state dict from the previous step, adds its own keys and returns it.
Large intermediate results are stored in S3 under pipeline/<doc_id>/ to stay far below the
256 KB Step Functions payload limit.
"""

import io
import json
import math
import os

from botocore.exceptions import ClientError

from idp.aws import client
from idp.extract import RETRYABLE_CODES, extract_with_llm
from idp.features import build_features, textract_signals_from_result
from idp.ocr import ocr_layout_text, textract_to_receipt

MODEL_NAME = os.environ.get("IDP_MODEL", "claude-haiku-4.5")
PROMPT = os.environ.get("IDP_PROMPT", "v2")
CONFIDENCE_MODEL_KEY = os.environ.get("IDP_CONFIDENCE_MODEL_KEY", "models/confidence.joblib")
CONFIDENCE_MODEL_NAME = os.environ.get("IDP_CONFIDENCE_MODEL_NAME", "gradient_boosting")
MAX_IMAGE_SIDE = 1600

TEXTRACT_RETRYABLE = {
    "ThrottlingException",
    "ProvisionedThroughputExceededException",
    "LimitExceededException",
    "InternalServerError",
}


class RetryableError(Exception):
    """Transient failure. Step Functions retries the step on this error type."""


def _raise_retryable(exc: ClientError, codes: set[str]) -> None:
    if exc.response["Error"]["Code"] in codes:
        raise RetryableError(str(exc)) from exc


def _get_bytes(bucket: str, key: str) -> bytes:
    return client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()


def _put_json(bucket: str, key: str, data: dict) -> None:
    client("s3").put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, default=str, ensure_ascii=False).encode(),
        ContentType="application/json",
    )


def _prepare_image(raw: bytes) -> bytes:
    """Normalize any uploaded image to a JPEG that fits Bedrock limits."""
    from PIL import Image  # imported lazily: only the extract step needs Pillow

    image = Image.open(io.BytesIO(raw)).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def textract_handler(event: dict, context=None) -> dict:
    bucket, key, doc_id = event["bucket"], event["key"], event["doc_id"]
    try:
        result = client("textract").analyze_expense(
            Document={"S3Object": {"Bucket": bucket, "Name": key}}
        )
    except ClientError as exc:
        _raise_retryable(exc, TEXTRACT_RETRYABLE)
        raise
    result.pop("ResponseMetadata", None)
    textract_key = f"pipeline/{doc_id}/textract.json"
    _put_json(bucket, textract_key, result)
    return {**event, "textract_key": textract_key}


def extract_handler(event: dict, context=None) -> dict:
    bucket, doc_id = event["bucket"], event["doc_id"]
    textract = json.loads(_get_bytes(bucket, event["textract_key"]))
    image = _prepare_image(_get_bytes(bucket, event["key"]))
    try:
        receipt, usage = extract_with_llm(
            MODEL_NAME, ocr_text=ocr_layout_text(textract), image_bytes=image, prompt=PROMPT
        )
    except ClientError as exc:
        _raise_retryable(exc, RETRYABLE_CODES)
        raise
    extraction_key = f"pipeline/{doc_id}/extraction.json"
    _put_json(
        bucket,
        extraction_key,
        {
            "prediction": receipt.model_dump(),
            "textract_prediction": textract_to_receipt(textract).model_dump(),
            "model": MODEL_NAME,
            "prompt": PROMPT,
            "usage": usage,
        },
    )
    return {**event, "extraction_key": extraction_key}


_bundle = None  # confidence model, cached between warm invocations


def _confidence_model() -> dict:
    global _bundle
    if _bundle is None:
        import joblib

        path = "/tmp/confidence.joblib"
        client("s3").download_file(os.environ["IDP_BUCKET"], CONFIDENCE_MODEL_KEY, path)
        _bundle = joblib.load(path)
    return _bundle


def score_handler(event: dict, context=None) -> dict:
    bucket = event["bucket"]
    extraction = json.loads(_get_bytes(bucket, event["extraction_key"]))
    textract = json.loads(_get_bytes(bucket, event["textract_key"]))

    features = build_features(
        extraction["prediction"],
        extraction["textract_prediction"],
        textract_signals_from_result(textract),
    )
    bundle = _confidence_model()
    entry = bundle["models"][CONFIDENCE_MODEL_NAME]
    row = [[features[name] for name in bundle["features"]]]
    confidence = float(entry["model"].predict_proba(row)[0, 1])
    threshold = float(entry["threshold"])
    approved = math.isfinite(threshold) and confidence >= threshold

    return {
        **event,
        "status": "AUTO_APPROVED" if approved else "NEEDS_REVIEW",
        "confidence": round(confidence, 4),
        # inf means "target precision unreachable": never auto-approve. JSON has no inf.
        "threshold": round(threshold, 4) if math.isfinite(threshold) else 1.0,
        "result_json": json.dumps(extraction["prediction"], ensure_ascii=False),
    }
