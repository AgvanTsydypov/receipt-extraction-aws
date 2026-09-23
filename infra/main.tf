terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "aws" {
  region = var.region

  # Every resource gets this tag, so project costs can be filtered in Cost Explorer
  default_tags {
    tags = {
      Project = "idp-portfolio"
    }
  }
}

# Bucket names are global across all AWS accounts, so add a random suffix
resource "random_id" "suffix" {
  byte_length = 4
}

# Stores dataset images, cached Textract responses and later pipeline outputs
resource "aws_s3_bucket" "data" {
  bucket = "idp-data-${random_id.suffix.hex}"

  # Allows `terraform destroy` to delete the bucket even if it still has objects.
  # Fine for a portfolio project: all data here can be regenerated.
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
