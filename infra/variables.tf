variable "region" {
  description = "AWS region for all resources"
  type        = string
  default     = "eu-west-2"
}

variable "image_tag" {
  description = "Lambda image tag in ECR. Written to image.auto.tfvars by scripts/build_and_push.sh"
  type        = string
  default     = ""
}

variable "llm_model" {
  description = "Model key from idp.config.MODELS used by the extract step"
  type        = string
  default     = "claude-haiku-4.5"
}

variable "prompt_version" {
  description = "Prompt version from idp.extract.PROMPTS"
  type        = string
  default     = "v2"
}
