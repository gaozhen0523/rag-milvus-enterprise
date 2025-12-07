locals {
  service_name = "rag-api-gateway"
}

# 读取 envs/dev 的 state，复用那里的 VPC / Subnets / SG / Cluster / ECR 配置
data "terraform_remote_state" "dev" {
  backend = "local"
  config = {
    path = "../envs/dev/terraform.tfstate"
  }
}

# 从 dev 的 outputs 读取基础设施信息
locals {
  vpc_id             = data.terraform_remote_state.dev.outputs.vpc_id
  public_subnet_ids  = data.terraform_remote_state.dev.outputs.public_subnet_ids
  private_subnet_ids = data.terraform_remote_state.dev.outputs.private_subnet_ids
  ecs_cluster_arn    = data.terraform_remote_state.dev.outputs.ecs_cluster_arn
  ecs_service_sg_id  = data.terraform_remote_state.dev.outputs.ecs_service_sg_id
  alb_sg_id          = data.terraform_remote_state.dev.outputs.alb_sg_id
  ecr_repo_urls      = data.terraform_remote_state.dev.outputs.ecr_repository_urls
}

module "rag_api_gateway_service" {
  source = "../modules/ecs_service"

  service_name = local.service_name

  cluster_arn = local.ecs_cluster_arn
  vpc_id      = local.vpc_id

  private_subnet_ids = local.private_subnet_ids
  public_subnet_ids  = local.public_subnet_ids

  security_group_ids    = [local.ecs_service_sg_id]
  alb_security_group_id = local.alb_sg_id

  # 使用 dev 层输出的 ECR 地址 + 传入的 image_tag
  container_image = "${local.ecr_repo_urls["rag-api-gateway"]}:${var.image_tag}"
  container_port  = 8000

  task_cpu    = 512
  task_memory = 1024

  desired_count     = 1
  health_check_path = "/health"

  environment_variables = {
    # --- Milvus Local Mode ---
    MILVUS_HOST      = "host.docker.internal"
    MILVUS_PORT      = "19530"
    MILVUS_IS_ZILLIZ = "True"

    # --- Zilliz Cloud Mode ---
    MILVUS_ZILLIZ_HOST    = "https://in03-d0c7c2a9b8dcff3.serverless.aws-eu-central-1.cloud.zilliz.com"
    MILVUS_ZILLIZ_API_KEY = data.aws_ssm_parameter.milvus_api_key.value

    # --- Embedding Settings ---
    EMBEDDING_MODEL  = "dummy"
    EMBEDDING_DIM    = "768"
    EMBEDDING_METRIC = "IP"
    TEST_DOC         = "sample.txt"

    # --- Redis (TEMP PLACEHOLDER) ---
    REDIS_HOST = "host.docker.internal"
    REDIS_PORT = "6379"
    REDIS_DB   = "0"

    # --- Environment Name ---
    ENVIRONMENT = var.environment
  }

  tags = {
    Project     = "rag-milvus-enterprise"
    Environment = var.environment
  }
}

data "aws_ssm_parameter" "milvus_api_key" {
  name            = "/dev/milvus/api_key"
  with_decryption = true
}
