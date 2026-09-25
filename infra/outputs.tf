output "bucket_name" {
  description = "Name of the project S3 bucket"
  value       = aws_s3_bucket.data.bucket
}

output "region" {
  description = "Region where resources were created"
  value       = var.region
}

output "ecr_repository_url" {
  description = "ECR repository for the Lambda image"
  value       = aws_ecr_repository.pipeline.repository_url
}

output "documents_table" {
  description = "DynamoDB table with pipeline results"
  value       = aws_dynamodb_table.documents.name
}

output "state_machine_arn" {
  description = "Step Functions state machine of the pipeline"
  value       = aws_sfn_state_machine.pipeline.arn
}
