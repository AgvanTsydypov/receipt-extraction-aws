"""Central configuration: paths, AWS settings and model price table."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


def require_bucket() -> str:
    """Return the project bucket name or fail with a helpful message."""
    bucket = os.environ.get("IDP_BUCKET", "")
    if not bucket:
        raise RuntimeError(
            "IDP_BUCKET is not set. Run:\n"
            '  export IDP_BUCKET="$(terraform -chdir=infra output -raw bucket_name)"'
        )
    return bucket


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    input_price_per_m: float  # USD per 1M input tokens
    output_price_per_m: float  # USD per 1M output tokens
    supports_images: bool


# Approximate on-demand prices in us-east-1. Verify on the Bedrock pricing page
# before quoting numbers in the README. Model IDs use US cross-region inference profiles.
MODELS: dict[str, ModelSpec] = {
    "nova-micro": ModelSpec("us.amazon.nova-micro-v1:0", 0.035, 0.14, supports_images=False),
    "nova-lite": ModelSpec("us.amazon.nova-lite-v1:0", 0.06, 0.24, supports_images=True),
    "nova-pro": ModelSpec("us.amazon.nova-pro-v1:0", 0.80, 3.20, supports_images=True),
    "claude-haiku-4.5": ModelSpec(
        "us.anthropic.claude-haiku-4-5-20251001-v1:0", 1.00, 5.00, supports_images=True
    ),
}

# Approximate Textract AnalyzeExpense price per page (USD)
TEXTRACT_EXPENSE_PRICE_PER_PAGE = 0.01
