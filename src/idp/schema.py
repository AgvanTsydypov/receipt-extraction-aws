"""Target data model for receipt extraction."""

from pydantic import BaseModel, ConfigDict, Field


class LineItem(BaseModel):
    # LLMs sometimes return numbers instead of strings; accept both
    model_config = ConfigDict(coerce_numbers_to_str=True)

    name: str
    quantity: str | None = None
    price: str | None = None  # line total as printed
    # Toppings, options and notes printed under the item. Not scored, but giving the
    # model a place for them stops it from emitting them as separate items.
    modifiers: list[str] = Field(default_factory=list)


class Receipt(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)

    subtotal: str | None = None
    tax: str | None = None
    total: str | None = None
    items: list[LineItem] = Field(default_factory=list)


# Hand-written JSON schema for the Bedrock tool. It is kept flat on purpose
# (no $ref, no type unions) so that every model family accepts it.
RECEIPT_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "description": "Purchased line items in the order they appear on the receipt.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Item name exactly as printed."},
                    "quantity": {"type": "string", "description": "Quantity as printed, if any."},
                    "price": {"type": "string", "description": "Line total as printed."},
                    "modifiers": {
                        "type": "array",
                        "description": "Toppings, options or notes printed under this item.",
                        "items": {"type": "string"},
                    },
                },
                "required": ["name"],
            },
        },
        "subtotal": {"type": "string", "description": "Subtotal as printed, if present."},
        "tax": {"type": "string", "description": "Tax amount as printed, if present."},
        "total": {"type": "string", "description": "Final amount to pay, as printed."},
    },
    "required": ["items"],
}
