variable "aws_region" {
  description = "AWS region to deploy to"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name for tagging"
  type        = string
  default     = "rag-milvus-enterprise"
}

variable "environment" {
  description = "Environment name (e.g. dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "vpc_cidr" {
  description = "CIDR for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = [
    "10.0.1.0/24",
    "10.0.2.0/24"
  ]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = [
    "10.0.101.0/24",
    "10.0.102.0/24"
  ]
}

variable "ecr_repository_names" {
  description = "ECR repositories for the RAG services"
  type        = list(string)
  default = [
    "rag-api-gateway"
  ]
}

variable "root_domain" {
  description = "Root domain name managed in Route53"
  type        = string
  default     = "zhencloud.com"
}

variable "rag_api_subdomain" {
  description = "Subdomain for rag-api-gateway"
  type        = string
  default     = "rag"
}
