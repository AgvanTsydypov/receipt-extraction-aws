# Lambda container image with the idp package. All three pipeline Lambdas use this image,
# each with a different handler (set in Terraform via image_config.command).
FROM public.ecr.aws/lambda/python:3.12

# The confidence model is pickled locally, so the image must use the same versions
ARG SKLEARN_VERSION
ARG NUMPY_VERSION

RUN pip install --no-cache-dir --target "${LAMBDA_TASK_ROOT}" \
        "scikit-learn==${SKLEARN_VERSION}" \
        "numpy==${NUMPY_VERSION}" \
        "boto3>=1.35" \
        "pydantic>=2.7" \
        "pillow>=10.0" \
        "joblib>=1.3"

COPY src/idp ${LAMBDA_TASK_ROOT}/idp

CMD ["idp.pipeline.textract_handler"]
