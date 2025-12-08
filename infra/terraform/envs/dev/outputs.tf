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

output "ecs_cluster_arn" {
  description = "ECS cluster ARN"
  value       = module.ecs_cluster.cluster_arn
}

output "ecs_service_sg_id" {
  description = "Security group for ECS service"
  value       = module.sg.ecs_service_sg_id
}

output "alb_sg_id" {
  description = "Security group for ALB"
  value       = module.sg.alb_sg_id
}

output "github_actions_role_arn" {
  value = module.github_oidc_role.github_actions_role_arn
}
