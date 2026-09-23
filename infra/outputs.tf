output "bucket_name" {
  description = "Name of the project S3 bucket"
  value       = aws_s3_bucket.data.bucket
}

output "region" {
  description = "Region where resources were created"
  value       = var.region
}
