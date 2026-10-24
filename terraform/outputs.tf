output "aws_region" {
  description = "Deployed AWS region"
  value       = var.aws_region
}

output "s3_bucket_name" {
  description = "Name of the created document storage S3 bucket"
  value       = aws_s3_bucket.documents.bucket
}

output "ecr_api_repository_url" {
  description = "ECR Repository URL for the unified VERITAS container image"
  value       = aws_ecr_repository.api.repository_url
}

output "ecs_cluster_name" {
  description = "Name of the ECS Fargate cluster"
  value       = aws_ecs_cluster.veritas.name
}

output "ecs_service_name" {
  description = "Name of the ECS service running the VERITAS container"
  value       = aws_ecs_service.api.name
}

output "log_group_name" {
  description = "CloudWatch log group for the API container"
  value       = aws_cloudwatch_log_group.api.name
}

output "anthropic_secret_arn" {
  description = "ARN of AWS Secrets Manager secret for Anthropic API key"
  value       = aws_secretsmanager_secret.anthropic_key.arn
}

output "pinecone_secret_arn" {
  description = "ARN of AWS Secrets Manager secret for Pinecone API key"
  value       = aws_secretsmanager_secret.pinecone_key.arn
}
