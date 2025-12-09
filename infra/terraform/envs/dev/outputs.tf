output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = module.vpc.public_subnet_ids
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = module.vpc.private_subnet_ids
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = module.ecs_cluster.cluster_name
}

output "ecr_repository_urls" {
  description = "Map of ECR repository URLs"
  value       = module.ecr.repository_urls
}

output "rag_api_gateway_alb_dns_name" {
  value = module.rag_api_gateway_service.alb_dns_name
}

output "rag_api_gateway_task_role_arn" {
  description = "Task IAM role ARN for rag-api-gateway"
  value       = module.rag_api_gateway_service.task_role_arn
}

output "rag_api_gateway_task_execution_role_arn" {
  description = "Task execution IAM role ARN for rag-api-gateway"
  value       = module.rag_api_gateway_service.task_execution_role_arn
}

output "rag_api_gateway_service_name" {
  description = "ECS service name for rag-api-gateway"
  value       = module.rag_api_gateway_service.service_name
}

output "github_actions_role_arn" {
  description = "IAM Role ARN for GitHub Actions OIDC"
  value       = aws_iam_role.github_actions.arn
}

output "rag_api_gateway_environment_variables" {
  description = "Env vars for rag-api-gateway ECS service"
  value       = module.rag_api_gateway_service.environment_variables
  sensitive = true
}
