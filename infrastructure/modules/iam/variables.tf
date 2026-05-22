variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "eks_cluster_name" {
  description = "EKS cluster name — used for IRSA trust policies"
  type        = string
}

variable "eks_oidc_provider_arn" {
  description = "EKS OIDC provider ARN — required for IRSA"
  type        = string
}

variable "eks_oidc_provider_url" {
  description = "EKS OIDC provider URL (without https://) — used in trust policy conditions"
  type        = string
}

variable "s3_bucket_name" {
  description = "S3 bucket for dataset and model artifacts"
  type        = string
}

variable "tags" {
  type    = map(string)
  default = {}
}
