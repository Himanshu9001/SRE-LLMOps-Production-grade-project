locals {
  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

module "vpc" {
  source = "../../modules/vpc"

  project_name         = var.project_name
  environment          = var.environment
  vpc_cidr             = var.vpc_cidr
  availability_zones   = var.availability_zones
  private_subnet_cidrs = var.private_subnet_cidrs
  public_subnet_cidrs  = var.public_subnet_cidrs
  tags                 = local.tags
}

module "eks" {
  source = "../../modules/eks"

  project_name         = var.project_name
  environment          = var.environment
  cluster_version      = var.eks_cluster_version
  vpc_id               = module.vpc.vpc_id
  private_subnet_ids   = module.vpc.private_subnet_ids
  eks_cluster_role_arn = module.iam.eks_cluster_role_arn
  tags                 = local.tags
}

module "iam" {
  source = "../../modules/iam"

  project_name          = var.project_name
  environment           = var.environment
  eks_cluster_name      = module.eks.cluster_name
  eks_oidc_provider_arn = module.eks.oidc_provider_arn
  eks_oidc_provider_url = module.eks.oidc_provider_url
  s3_bucket_name        = var.s3_bucket_name
  tags                  = local.tags
}

module "node_groups" {
  source = "../../modules/node-groups"

  project_name       = var.project_name
  environment        = var.environment
  cluster_name       = module.eks.cluster_name
  eks_node_role_arn  = module.iam.eks_node_role_arn
  private_subnet_ids = module.vpc.private_subnet_ids
  gpu_subnet_ids     = [module.vpc.private_subnet_ids[0]]
  tags               = local.tags
}

# FSx for Lustre — uncomment only when running training jobs
# Costs ~$168/day minimum — destroy after each training run
#
# module "fsx" {
#   source = "../../modules/fsx"
#
#   project_name     = var.project_name
#   environment      = var.environment
#   vpc_id           = module.vpc.vpc_id
#   subnet_id        = module.vpc.private_subnet_ids[0]
#   vpc_cidr         = var.vpc_cidr
#   s3_bucket_name   = var.s3_bucket_name
#   s3_import_prefix = "base-models/"
#   storage_capacity = var.fsx_storage_capacity
#   tags             = local.tags
# }
