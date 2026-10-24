# Each key comes from its tfvars variable on the first apply, or a placeholder when that is
# empty. ignore_changes means a value set later in the AWS console is not overwritten by the
# next `terraform apply`; after changing a secret, redeploy the ECS service so tasks read it.

resource "aws_secretsmanager_secret" "anthropic_key" {
  name        = "${var.app_name}-anthropic-api-key-${var.environment}"
  description = "Anthropic API Key for VERITAS logic & reasoning verification"
}

resource "aws_secretsmanager_secret_version" "anthropic_key_val" {
  secret_id     = aws_secretsmanager_secret.anthropic_key.id
  secret_string = var.anthropic_api_key != "" ? var.anthropic_api_key : "PLACEHOLDER_SET_IN_AWS"

  lifecycle {
    ignore_changes = [secret_string]
  }
}

resource "aws_secretsmanager_secret" "pinecone_key" {
  name        = "${var.app_name}-pinecone-api-key-${var.environment}"
  description = "Pinecone API Key for VERITAS vector index authentication"
}

resource "aws_secretsmanager_secret_version" "pinecone_key_val" {
  secret_id     = aws_secretsmanager_secret.pinecone_key.id
  secret_string = var.pinecone_api_key != "" ? var.pinecone_api_key : "PLACEHOLDER_SET_IN_AWS"

  lifecycle {
    ignore_changes = [secret_string]
  }
}
