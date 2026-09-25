# Production pipeline:
# S3 incoming/ -> EventBridge -> Step Functions -> Lambda (Textract) -> Lambda (LLM)
#   -> Lambda (confidence model) -> DynamoDB (AUTO_APPROVED / NEEDS_REVIEW / FAILED)

locals {
  functions = {
    textract = { handler = "idp.pipeline.textract_handler", memory = 512, timeout = 60 }
    extract  = { handler = "idp.pipeline.extract_handler", memory = 1024, timeout = 120 }
    score    = { handler = "idp.pipeline.score_handler", memory = 1024, timeout = 60 }
  }

  lambda_env = {
    IDP_BUCKET                = aws_s3_bucket.data.bucket
    IDP_MODEL                 = var.llm_model
    IDP_PROMPT                = var.prompt_version
    IDP_CONFIDENCE_MODEL_KEY  = "models/confidence.joblib"
    IDP_CONFIDENCE_MODEL_NAME = "gradient_boosting"
    # Short in-Lambda retries: Step Functions retries the whole step with backoff instead
    IDP_BEDROCK_MAX_ATTEMPTS = "2"
  }
}

# ---------- Container registry ----------

resource "aws_ecr_repository" "pipeline" {
  name         = "idp-pipeline"
  force_delete = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Keep only the last 5 images to avoid paying for old ones
resource "aws_ecr_lifecycle_policy" "pipeline" {
  repository = aws_ecr_repository.pipeline.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 5 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}

# ---------- Results table ----------

resource "aws_dynamodb_table" "documents" {
  name         = "idp-documents"
  billing_mode = "PAY_PER_REQUEST" # pay per request, no idle cost
  hash_key     = "doc_id"

  attribute {
    name = "doc_id"
    type = "S"
  }
}

# ---------- Lambda ----------

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "idp-pipeline-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "lambda" {
  statement {
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.data.arn}/*"]
  }
  statement {
    actions   = ["textract:AnalyzeExpense"]
    resources = ["*"]
  }
  # Global inference profiles route requests to models in several regions
  statement {
    actions   = ["bedrock:InvokeModel"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "lambda" {
  name   = "idp-pipeline-lambda"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.lambda.json
}

resource "aws_cloudwatch_log_group" "lambda" {
  for_each          = local.functions
  name              = "/aws/lambda/idp-${each.key}"
  retention_in_days = 14
}

resource "aws_lambda_function" "step" {
  for_each = local.functions

  function_name = "idp-${each.key}"
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.pipeline.repository_url}:${var.image_tag}"
  architectures = ["arm64"] # matches Apple Silicon builds, and Graviton is cheaper
  memory_size   = each.value.memory
  timeout       = each.value.timeout

  image_config {
    command = [each.value.handler]
  }

  environment {
    variables = local.lambda_env
  }

  depends_on = [aws_cloudwatch_log_group.lambda, aws_iam_role_policy_attachment.lambda_logs]

  lifecycle {
    precondition {
      condition     = var.image_tag != ""
      error_message = "image_tag is empty. Run scripts/build_and_push.sh first."
    }
  }
}

# ---------- Step Functions ----------

locals {
  retry_transient = [{
    ErrorEquals = [
      "RetryableError",
      "Lambda.TooManyRequestsException",
      "Lambda.ServiceException",
      "Lambda.AWSLambdaException",
      "Lambda.SdkClientException",
    ]
    IntervalSeconds = 5
    BackoffRate     = 2
    MaxAttempts     = 6
    MaxDelaySeconds = 120
    JitterStrategy  = "FULL"
  }]

  catch_all = [{
    ErrorEquals = ["States.ALL"]
    ResultPath  = "$.error" # keep the original input next to the error
    Next        = "MarkFailed"
  }]

  lambda_state = {
    for name, fn in aws_lambda_function.step : name => {
      Type       = "Task"
      Resource   = "arn:aws:states:::lambda:invoke"
      Parameters = { FunctionName = fn.arn, "Payload.$" = "$" }
      OutputPath = "$.Payload"
      Retry      = local.retry_transient
      Catch      = local.catch_all
    }
  }
}

resource "aws_sfn_state_machine" "pipeline" {
  name     = "idp-pipeline"
  role_arn = aws_iam_role.sfn.arn

  definition = jsonencode({
    Comment = "Receipt extraction with confidence-based routing"
    StartAt = "Prepare"
    States = {
      Prepare = {
        Type = "Pass"
        Parameters = {
          "bucket.$" = "$.detail.bucket.name"
          "key.$"    = "$.detail.object.key"
          "doc_id.$" = "$$.Execution.Name"
        }
        Next = "Textract"
      }
      Textract = merge(local.lambda_state["textract"], { Next = "Extract" })
      Extract  = merge(local.lambda_state["extract"], { Next = "Score" })
      Score    = merge(local.lambda_state["score"], { Next = "Save" })
      Save = {
        Type     = "Task"
        Resource = "arn:aws:states:::dynamodb:putItem"
        Parameters = {
          TableName = aws_dynamodb_table.documents.name
          Item = {
            doc_id       = { "S.$" = "$.doc_id" }
            status       = { "S.$" = "$.status" }
            confidence   = { "N.$" = "States.Format('{}', $.confidence)" }
            threshold    = { "N.$" = "States.Format('{}', $.threshold)" }
            result       = { "S.$" = "$.result_json" }
            source_key   = { "S.$" = "$.key" }
            execution    = { "S.$" = "$$.Execution.Id" }
            processed_at = { "S.$" = "$$.State.EnteredTime" }
          }
        }
        End = true
      }
      MarkFailed = {
        Type     = "Task"
        Resource = "arn:aws:states:::dynamodb:putItem"
        Parameters = {
          TableName = aws_dynamodb_table.documents.name
          Item = {
            doc_id       = { "S.$" = "$.doc_id" }
            status       = { S = "FAILED" }
            source_key   = { "S.$" = "$.key" }
            error        = { "S.$" = "$.error.Error" }
            cause        = { "S.$" = "$.error.Cause" }
            execution    = { "S.$" = "$$.Execution.Id" }
            processed_at = { "S.$" = "$$.State.EnteredTime" }
          }
        }
        Next = "Failed"
      }
      Failed = {
        Type  = "Fail"
        Error = "PipelineFailed"
        Cause = "See the FAILED item in DynamoDB"
      }
    }
  })
}

data "aws_iam_policy_document" "sfn_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sfn" {
  name               = "idp-pipeline-sfn"
  assume_role_policy = data.aws_iam_policy_document.sfn_assume.json
}

data "aws_iam_policy_document" "sfn" {
  statement {
    actions = ["lambda:InvokeFunction"]
    resources = flatten([
      for fn in aws_lambda_function.step : [fn.arn, "${fn.arn}:*"]
    ])
  }
  statement {
    actions   = ["dynamodb:PutItem"]
    resources = [aws_dynamodb_table.documents.arn]
  }
}

resource "aws_iam_role_policy" "sfn" {
  name   = "idp-pipeline-sfn"
  role   = aws_iam_role.sfn.id
  policy = data.aws_iam_policy_document.sfn.json
}

# ---------- Trigger: new object in incoming/ ----------

resource "aws_s3_bucket_notification" "eventbridge" {
  bucket      = aws_s3_bucket.data.id
  eventbridge = true
}

resource "aws_cloudwatch_event_rule" "incoming" {
  name = "idp-incoming-document"
  event_pattern = jsonencode({
    source        = ["aws.s3"]
    "detail-type" = ["Object Created"]
    detail = {
      bucket = { name = [aws_s3_bucket.data.bucket] }
      object = { key = [{ prefix = "incoming/" }] }
    }
  })
}

data "aws_iam_policy_document" "events_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "events" {
  name               = "idp-pipeline-events"
  assume_role_policy = data.aws_iam_policy_document.events_assume.json
}

resource "aws_iam_role_policy" "events" {
  name = "idp-pipeline-events"
  role = aws_iam_role.events.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "states:StartExecution"
      Resource = aws_sfn_state_machine.pipeline.arn
    }]
  })
}

resource "aws_cloudwatch_event_target" "pipeline" {
  rule     = aws_cloudwatch_event_rule.incoming.name
  arn      = aws_sfn_state_machine.pipeline.arn
  role_arn = aws_iam_role.events.arn
}
