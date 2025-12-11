#infra/terraform/app/rag-api-gateway/main.tf
terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "rag-milvus-tfstate"
    key            = "app/rag-api-gateway.tfstate"
    region         = "us-east-1"
    dynamodb_table = "rag-milvus-tfstate-lock"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region
}

# 读取 dev 层的 remote state
data "terraform_remote_state" "dev" {
  backend = "s3"
  config = {
    bucket = "rag-milvus-tfstate"
    key    = "envs/dev/terraform.tfstate"
    region = "us-east-1"
  }
}

locals {
  dev_outputs = data.terraform_remote_state.dev.outputs

  # 把 dev 里的 map(string) → list(object{name,value})
  environment = [
    for k, v in local.dev_outputs.rag_api_gateway_environment_variables :
    {
      name  = k
      value = v
    }
  ]
}

resource "aws_ecs_task_definition" "rag_api_gateway" {
  family                   = "rag-api-gateway"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]

  execution_role_arn = local.dev_outputs.rag_api_gateway_task_execution_role_arn
  task_role_arn      = local.dev_outputs.rag_api_gateway_task_role_arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode([
    {
      name      = "rag-api-gateway"
      image     = "${local.dev_outputs.ecr_repository_urls["rag-api-gateway"]}:${var.image_tag}"
      essential = true

      portMappings = [
        {
          containerPort = var.container_port
          protocol      = "tcp"
        }
      ]

      environment = local.environment

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/rag-api-gateway"
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}

output "task_definition_arn" {
  description = "New ECS task definition ARN"
  value       = aws_ecs_task_definition.rag_api_gateway.arn
}

# 方便 GitHub Actions 读 cluster / service 名称
output "ecs_cluster_name" {
  value = local.dev_outputs.ecs_cluster_name
}

output "rag_api_gateway_service_name" {
  value = local.dev_outputs.rag_api_gateway_service_name
}
