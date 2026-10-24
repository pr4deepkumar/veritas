variable "aws_region" {
  description = "AWS region to deploy resources into"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment name (e.g. dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "app_name" {
  description = "Application identifier prefix"
  type        = string
  default     = "veritas"
}

variable "s3_bucket_name" {
  description = "Unique name for the S3 document storage bucket"
  type        = string
  default     = "veritas-document-storage"
}

variable "anthropic_api_key" {
  description = "Anthropic Claude API key for reasoning & verification"
  type        = string
  sensitive   = true
  default     = ""
}

variable "pinecone_api_key" {
  description = "Pinecone API key for vector database management"
  type        = string
  sensitive   = true
  default     = ""
}

variable "pinecone_index_name" {
  description = "Pinecone vector index name"
  type        = string
  default     = "veritas-research"
}

variable "pinecone_cloud" {
  description = "Cloud provider for Pinecone serverless index"
  type        = string
  default     = "aws"
}

variable "pinecone_region" {
  description = "Region for Pinecone serverless index"
  type        = string
  default     = "us-east-1"
}

variable "api_container_port" {
  description = "Port exposed by the unified VERITAS FastAPI container"
  type        = number
  default     = 8000
}

variable "container_cpu" {
  description = "CPU units reserved for each Fargate task (1024 = 1 vCPU)"
  type        = number
  default     = 512
}

variable "container_memory" {
  description = "Memory reserved for each Fargate task (in MiB)"
  type        = number
  default     = 1024
}

variable "image_tag" {
  description = "Image tag the ECS task runs. scripts/deploy_aws.sh pushes :latest and the git SHA."
  type        = string
  default     = "latest"
}

variable "cpu_architecture" {
  description = "Fargate CPU architecture. Must match the image platform (deploy_aws.sh builds linux/amd64)."
  type        = string
  default     = "X86_64"

  validation {
    condition     = contains(["X86_64", "ARM64"], var.cpu_architecture)
    error_message = "cpu_architecture must be X86_64 or ARM64."
  }
}

variable "log_retention_days" {
  description = "Days to keep the API's CloudWatch logs"
  type        = number
  default     = 30
}

variable "s3_force_destroy" {
  description = "Let `terraform destroy` delete the document bucket even when it still contains files"
  type        = bool
  default     = false
}
