terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    pinecone = {
      source  = "pinecone-io/pinecone"
      version = "~> 0.8"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "VERITAS"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

provider "pinecone" {
  # Pinecone API key configured via environment variable PINECONE_API_KEY or variable
  api_key = var.pinecone_api_key
}
