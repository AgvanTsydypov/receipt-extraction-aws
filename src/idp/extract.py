"""LLM extraction through the Bedrock Converse API with forced tool use."""

import json
import os
import random
import time

from botocore.exceptions import ClientError

from idp.aws import client
from idp.config import MODELS
from idp.schema import RECEIPT_TOOL_SCHEMA, Receipt

TOOL_NAME = "record_receipt"

# Our own retry loop for throttling. boto3's built-in retries share a retry budget per
# client; on long throttled runs the budget drains and every later request fails at once.
RETRYABLE_CODES = {"ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException"}
# Lambda sets this low: there Step Functions retries the whole step instead
MAX_ATTEMPTS = int(os.environ.get("IDP_BEDROCK_MAX_ATTEMPTS", "8"))
BASE_DELAY_S = 2.0
MAX_DELAY_S = 60.0


def _converse_with_backoff(**kwargs) -> dict:
    """Call Bedrock Converse, retrying throttling with exponential backoff and jitter."""
    delay = BASE_DELAY_S
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return client("bedrock-runtime").converse(**kwargs)
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in RETRYABLE_CODES or attempt == MAX_ATTEMPTS:
                raise
            time.sleep(delay * (0.5 + random.random()))
            delay = min(delay * 2, MAX_DELAY_S)
    raise RuntimeError("unreachable")

PROMPTS = {
    "v1": """You are a precise data extraction system for retail receipts.
Rules:
- Copy amounts exactly as printed, keeping separators. Do not calculate or convert anything.
- The price of a line item is the line total printed on that line, not the unit price.
- Omit a field when it is not printed on the receipt. Never guess.
- Copy item names exactly as printed. Do not translate them.
- Subtotal, tax, service, total, payment and change lines are not items.""",
    # v2 encodes the annotation guidelines found during error analysis on the dev split
    "v2": """You are a precise data extraction system for receipts from shops and restaurants.
Follow these annotation guidelines exactly.

Line items
- One item per purchased product line. Copy the name as printed, without translating it.
- Do not include quantity markers in the name, such as "1x", "2 X", "1Prs" or a leading count. Put the count in "quantity".
- Options printed on the same line as the product (for example "50%") are part of the name.
- Lines printed under a product that describe it (toppings, sugar or ice level, notes like "Less Ice", lines starting with "-" or "+") are not separate items and not part of the name. Put them in "modifiers" of that product.
- A standalone product line is an item even if its price is 0, for example a free plastic bag.
- Discount, voucher, service charge, tax, subtotal, total, payment and change lines are not items.
- "price" is the line total printed for that product, not the unit price.

Amounts
- Copy each amount as printed, keeping separators, but without currency symbols or labels. For example, for "PB1-TAX  Tax 6,364" return "6,364".
- "tax" is the amount on the line labeled tax, PB1, PPN or VAT. Service charge is not tax. If the tax line prints 0, return "0".
- "subtotal" is the amount before tax and service charge, if printed.
- "total" is the final amount to pay.
- Omit a field when it is not printed on the receipt. Never calculate or guess values.""",
}
DEFAULT_PROMPT = "v2"


def _clean_tool_input(raw: dict) -> dict:
    """Defensive cleanup for common model quirks before validation."""
    items = raw.get("items") or []
    if isinstance(items, str):  # some models return the array as a JSON string
        try:
            items = json.loads(items)
        except json.JSONDecodeError:
            items = []
    cleaned = []
    for item in items:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        modifiers = item.get("modifiers") or []
        if isinstance(modifiers, str):
            modifiers = [modifiers]
        item["modifiers"] = [str(m) for m in modifiers if m is not None]
        cleaned.append(item)
    raw["items"] = cleaned
    return raw


def extract_with_llm(
    model_name: str,
    *,
    ocr_text: str | None = None,
    image_bytes: bytes | None = None,
    image_format: str = "jpeg",
    prompt: str = DEFAULT_PROMPT,
) -> tuple[Receipt, dict]:
    """Extract a Receipt from OCR text and/or a JPEG image. Returns (receipt, usage info)."""
    spec = MODELS[model_name]
    if image_bytes is not None and not spec.supports_images:
        raise ValueError(f"Model '{model_name}' does not accept images")

    content = []
    if image_bytes is not None:
        content.append({"image": {"format": image_format, "source": {"bytes": image_bytes}}})
    if ocr_text is not None:
        content.append({"text": f"OCR text of the receipt:\n\n{ocr_text}"})
    content.append({"text": f"Extract the receipt data by calling the {TOOL_NAME} tool."})

    response = _converse_with_backoff(
        modelId=spec.model_id,
        system=[{"text": PROMPTS[prompt]}],
        messages=[{"role": "user", "content": content}],
        toolConfig={
            "tools": [
                {
                    "toolSpec": {
                        "name": TOOL_NAME,
                        "description": "Record structured data extracted from a receipt.",
                        "inputSchema": {"json": RECEIPT_TOOL_SCHEMA},
                    }
                }
            ],
            "toolChoice": {"any": {}},  # force a tool call instead of free text
        },
        inferenceConfig={"maxTokens": 2000, "temperature": 0},
    )

    tool_input = next(
        (
            block["toolUse"]["input"]
            for block in response["output"]["message"]["content"]
            if "toolUse" in block
        ),
        None,
    )
    receipt = Receipt.model_validate(_clean_tool_input(tool_input)) if tool_input else Receipt()

    in_tok = response["usage"]["inputTokens"]
    out_tok = response["usage"]["outputTokens"]
    usage = {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "llm_cost_usd": (in_tok * spec.input_price_per_m + out_tok * spec.output_price_per_m) / 1e6,
        "latency_ms": response.get("metrics", {}).get("latencyMs"),
        "tool_called": tool_input is not None,
    }
    return receipt, usage
