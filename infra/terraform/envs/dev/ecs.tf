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
  # --- Milvus Local Mode ---
  MILVUS_HOST         = "host.docker.internal"
  MILVUS_PORT         = "19530"
  MILVUS_IS_ZILLIZ    = "True"

  # --- Zilliz Cloud Mode ---
  MILVUS_ZILLIZ_HOST     = "https://in03-d0c7c2a9b8dcff3.serverless.aws-eu-central-1.cloud.zilliz.com"
  MILVUS_ZILLIZ_API_KEY  = "data.aws_ssm_parameter.milvus_api_key.value"

  # --- Embedding Settings ---
  EMBEDDING_MODEL     = "dummy"
  EMBEDDING_DIM       = "768"
  EMBEDDING_METRIC    = "IP"
  TEST_DOC            = "sample.txt"

  # --- Redis (TEMP PLACEHOLDER) ---
  REDIS_HOST          = "host.docker.internal"
  REDIS_PORT          = "6379"
  REDIS_DB            = "0"

  # --- Environment Name ---
  ENVIRONMENT         = var.environment
  }

  tags = local.tags
}

data "aws_ssm_parameter" "milvus_api_key" {
  name = "/dev/milvus/api_key"
  with_decryption = true
}
