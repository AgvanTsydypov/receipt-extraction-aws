#!/usr/bin/env bash
# Build the Lambda image for arm64, push it to ECR and record the tag for Terraform.
set -euo pipefail
cd "$(dirname "$0")/.."

REGION="${AWS_REGION:-eu-west-2}"
REPO_URL="$(terraform -chdir=infra output -raw ecr_repository_url)"
REGISTRY="${REPO_URL%%/*}"
TAG="$(date +%Y%m%d-%H%M%S)"

# Pin the same versions that were used to train and pickle the confidence model
SKLEARN_VERSION="$(python -c 'import sklearn; print(sklearn.__version__)')"
NUMPY_VERSION="$(python -c 'import numpy; print(numpy.__version__)')"
echo "scikit-learn ${SKLEARN_VERSION}, numpy ${NUMPY_VERSION}"

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

# --provenance=false: Lambda rejects multi-manifest images that buildx creates by default
docker build \
  --platform linux/arm64 \
  --provenance=false \
  --build-arg SKLEARN_VERSION="$SKLEARN_VERSION" \
  --build-arg NUMPY_VERSION="$NUMPY_VERSION" \
  -t "${REPO_URL}:${TAG}" .

docker push "${REPO_URL}:${TAG}"

# Terraform picks up *.auto.tfvars automatically (the file is gitignored)
echo "image_tag = \"${TAG}\"" > infra/image.auto.tfvars
echo "Pushed ${REPO_URL}:${TAG}"
