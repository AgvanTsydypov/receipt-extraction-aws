# Receipt data extraction on AWS (Textract + Bedrock)

Extracts structured data (line items, subtotal, tax, total) from receipt images and measures
field-level accuracy, cost and latency of three approaches:

| Method | How it works |
|---|---|
| `textract` | Textract AnalyzeExpense fields only (baseline, no LLM) |
| `llm_text` | Textract OCR text, then a Bedrock LLM with forced tool use for structured output |
| `llm_layout` | Textract text rebuilt into visual rows by line geometry, then a Bedrock LLM |
| `llm_image` | Receipt image sent directly to a multimodal Bedrock LLM |
| `llm_hybrid` | Receipt image plus layout text sent together to a multimodal Bedrock LLM |

Dataset: [CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) receipts
(100 dev, 100 test).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest

# Infrastructure (S3 bucket)
terraform -chdir=infra init
terraform -chdir=infra apply
export IDP_BUCKET="$(terraform -chdir=infra output -raw bucket_name)"
export AWS_REGION=eu-west-2
```

## Run

```bash
# 1. Download dataset and upload images to S3
python scripts/prepare_cord.py

# 2. Evaluate
python scripts/run_eval.py --split dev --method textract
python scripts/run_eval.py --split dev --method llm_text --model nova-2-lite
python scripts/run_eval.py --split dev --method llm_image --model nova-2-lite

# 3. Compare runs and inspect errors
python scripts/rescore.py        # recompute metrics for old runs after metric changes
python scripts/compare_runs.py
python scripts/inspect_errors.py --method llm_text
python scripts/significance.py textract llm_hybrid:claude-haiku-4.5:v2
```

Textract responses are cached locally and in S3, so each page is billed only once.

## Results

_Coming soon._

## Cleanup

```bash
terraform -chdir=infra destroy
```
