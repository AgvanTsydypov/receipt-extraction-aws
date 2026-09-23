# Receipt data extraction on AWS (Textract + Bedrock)

Extracts structured data (line items, subtotal, tax, total) from receipt images and measures
field-level accuracy, cost and latency of three approaches:

| Method | How it works |
|---|---|
| `textract` | Textract AnalyzeExpense fields only (baseline, no LLM) |
| `llm_text` | Textract OCR text, then a Bedrock LLM with forced tool use for structured output |
| `llm_image` | Receipt image sent directly to a multimodal Bedrock LLM |

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
export AWS_REGION=us-east-1
```

## Run

```bash
# 1. Download dataset and upload images to S3
python scripts/prepare_cord.py

# 2. Evaluate
python scripts/run_eval.py --split dev --method textract
python scripts/run_eval.py --split dev --method llm_text --model nova-lite
python scripts/run_eval.py --split dev --method llm_image --model nova-lite

# 3. Compare all runs
python scripts/compare_runs.py
```

Textract responses are cached locally and in S3, so each page is billed only once.

## Results

_Coming soon._

## Cleanup

```bash
terraform -chdir=infra destroy
```
