# ------------------------------------------------------------
# EKS Cluster — control plane only
# Node groups are in a separate module for independent scaling
# Private API endpoint: control plane not exposed to internet
# Public endpoint: enabled for kubectl from local machine
# ------------------------------------------------------------

locals {
  name = "${var.project_name}-${var.environment}"
}

resource "aws_eks_cluster" "main" {
  name     = local.name
  version  = var.cluster_version
  role_arn = var.eks_cluster_role_arn

  vpc_config {
    subnet_ids              = var.private_subnet_ids
    endpoint_private_access = true   # nodes communicate with API server privately
    endpoint_public_access  = true   # kubectl access from local machine
    public_access_cidrs     = ["0.0.0.0/0"]  # restrict to your IP in production
  }

  # Enable envelope encryption for Kubernetes secrets using KMS
  encryption_config {
    provider {
      key_arn = aws_kms_key.eks.arn
    }
    resources = ["secrets"]
  }

  # Enable control plane logging — critical for debugging auth/API issues
  enabled_cluster_log_types = [
    "api",
    "audit",
    "authenticator",
    "controllerManager",
    "scheduler"
  ]

  tags = merge(var.tags, { Name = local.name })

  depends_on = [aws_cloudwatch_log_group.eks]
}

# CloudWatch log group for EKS control plane logs
resource "aws_cloudwatch_log_group" "eks" {
  name              = "/aws/eks/${local.name}/cluster"
  retention_in_days = 30   # 30 days — enough for audit trail without excessive cost
  tags              = var.tags
}

# KMS key for encrypting Kubernetes secrets at rest
resource "aws_kms_key" "eks" {
  description             = "${local.name} EKS secrets encryption"
  deletion_window_in_days = 7
  enable_key_rotation     = true   # rotate annually — security best practice
  tags                    = merge(var.tags, { Name = "${local.name}-eks-kms" })
}

resource "aws_kms_alias" "eks" {
  name          = "alias/${local.name}-eks"
  target_key_id = aws_kms_key.eks.key_id
}

# OIDC Provider — required for IRSA
# EKS uses OIDC to federate Kubernetes service account tokens to AWS IAM roles
data "tls_certificate" "eks" {
  url = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "eks" {
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.eks.certificates[0].sha1_fingerprint]
  url             = aws_eks_cluster.main.identity[0].oidc[0].issuer

  tags = merge(var.tags, { Name = "${local.name}-oidc" })
}
