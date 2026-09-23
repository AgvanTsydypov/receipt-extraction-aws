"""LLM extraction through the Bedrock Converse API with forced tool use."""

import json

from idp.aws import client
from idp.config import MODELS
from idp.schema import RECEIPT_TOOL_SCHEMA, Receipt

TOOL_NAME = "record_receipt"

SYSTEM_PROMPT = """You are a precise data extraction system for retail receipts.
Rules:
- Copy amounts exactly as printed, keeping separators. Do not calculate or convert anything.
- The price of a line item is the line total printed on that line, not the unit price.
- Omit a field when it is not printed on the receipt. Never guess.
- Copy item names exactly as printed. Do not translate them.
- Subtotal, tax, service, total, payment and change lines are not items."""


def _clean_tool_input(raw: dict) -> dict:
    """Defensive cleanup for common model quirks before validation."""
    items = raw.get("items") or []
    if isinstance(items, str):  # some models return the array as a JSON string
        try:
            items = json.loads(items)
        except json.JSONDecodeError:
            items = []
    raw["items"] = [i for i in items if isinstance(i, dict) and i.get("name")]
    return raw


def extract_with_llm(
    model_name: str, *, ocr_text: str | None = None, image_bytes: bytes | None = None
) -> tuple[Receipt, dict]:
    """Extract a Receipt from OCR text and/or a JPEG image. Returns (receipt, usage info)."""
    spec = MODELS[model_name]
    if image_bytes is not None and not spec.supports_images:
        raise ValueError(f"Model '{model_name}' does not accept images")

    content = []
    if image_bytes is not None:
        content.append({"image": {"format": "jpeg", "source": {"bytes": image_bytes}}})
    if ocr_text is not None:
        content.append({"text": f"OCR text of the receipt:\n\n{ocr_text}"})
    content.append({"text": f"Extract the receipt data by calling the {TOOL_NAME} tool."})

    response = client("bedrock-runtime").converse(
        modelId=spec.model_id,
        system=[{"text": SYSTEM_PROMPT}],
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
