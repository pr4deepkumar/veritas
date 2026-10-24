resource "aws_ecs_cluster" "veritas" {
  name = "${var.app_name}-cluster-${var.environment}"
}

# Created here rather than with the awslogs "create-group" option, which needs
# logs:CreateLogGroup — a permission AmazonECSTaskExecutionRolePolicy does not grant.
resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${var.app_name}-api"
  retention_in_days = var.log_retention_days
}

# ECS Task Definition for the unified FastAPI container (API + static UI)
resource "aws_ecs_task_definition" "api" {
  family                   = "${var.app_name}-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.container_cpu
  memory                   = var.container_memory
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = var.cpu_architecture
  }

  container_definitions = jsonencode([
    {
      name      = "api"
      image     = "${aws_ecr_repository.api.repository_url}:${var.image_tag}"
      essential = true
      command   = ["uvicorn", "veritas.api.main:app", "--host", "0.0.0.0", "--port", tostring(var.api_container_port)]
      portMappings = [
        {
          containerPort = var.api_container_port
          hostPort      = var.api_container_port
        }
      ]
      environment = [
        { name = "LLM_PROVIDER", value = "claude" },
        { name = "VECTOR_STORE", value = "pinecone" },
        { name = "PINECONE_INDEX", value = var.pinecone_index_name },
        { name = "PINECONE_CLOUD", value = var.pinecone_cloud },
        { name = "PINECONE_REGION", value = var.pinecone_region },
        { name = "S3_BUCKET", value = aws_s3_bucket.documents.bucket },
        { name = "AWS_REGION", value = var.aws_region }
      ]
      secrets = [
        { name = "ANTHROPIC_API_KEY", valueFrom = aws_secretsmanager_secret.anthropic_key.arn },
        { name = "PINECONE_API_KEY", valueFrom = aws_secretsmanager_secret.pinecone_key.arn }
      ]
      healthCheck = {
        command     = ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${var.api_container_port}/health', timeout=4)"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.api.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "api"
        }
      }
    }
  ])
}

# ECS Service for the unified VERITAS application
resource "aws_ecs_service" "api" {
  name            = "${var.app_name}-api-service"
  cluster         = aws_ecs_cluster.veritas.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  # Roll back automatically when a new deployment's tasks keep failing.
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.ecs_sg.id]
    assign_public_ip = true
  }
}
