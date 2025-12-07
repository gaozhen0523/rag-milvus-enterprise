output "service_name" {
  value = module.rag_api_gateway_service.service_name
}

output "service_arn" {
  value = module.rag_api_gateway_service.service_arn
}

output "task_definition_arn" {
  value = module.rag_api_gateway_service.task_definition_arn
}

output "alb_dns_name" {
  value = module.rag_api_gateway_service.alb_dns_name
}
