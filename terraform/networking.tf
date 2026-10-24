# Use Default VPC and Subnets for standard deployment
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Security Group for VERITAS Services
resource "aws_security_group" "ecs_sg" {
  name        = "${var.app_name}-ecs-sg-${var.environment}"
  description = "Security Group allowing inbound traffic for the VERITAS API"
  vpc_id      = data.aws_vpc.default.id

  # FastAPI Ingress (serves both API and static web UI)
  ingress {
    from_port   = var.api_container_port
    to_port     = var.api_container_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "VERITAS unified application ingress"
  }

  # Outbound Egress (allow all traffic to Pinecone, Anthropic, S3, internet)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
