module "rag_api_gateway_service" {
  source = "../../modules/ecs_service"

  service_name = "rag-api-gateway"

  cluster_arn = module.ecs_cluster.cluster_arn
  vpc_id      = module.vpc.vpc_id

  private_subnet_ids = module.vpc.private_subnet_ids
  public_subnet_ids  = module.vpc.public_subnet_ids

  security_group_ids    = [module.sg.ecs_service_sg_id]
  alb_security_group_id = module.sg.alb_sg_id

  container_image = "${module.ecr.repository_urls["rag-api-gateway"]}:latest"
  container_port  = 8000

  task_cpu    = 512
  task_memory = 1024

  desired_count    = 1
  health_check_path = "/health"

  environment_variables = {
    ENVIRONMENT = var.environment
    # 这里后续可以加 MILVUS / REDIS 等配置
  }

  tags = local.tags
}
