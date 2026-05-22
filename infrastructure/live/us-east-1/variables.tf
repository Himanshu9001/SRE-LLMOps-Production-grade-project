variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project_name" {
  type    = string
  default = "sre-llmops"
}

variable "environment" {
  type    = string
  default = "production"
}

variable "s3_bucket_name" {
  type    = string
  default = "sre-llmops-artifacts"
}

variable "vpc_cidr" {
  type    = string
  default = "10.0.0.0/16"
}

variable "availability_zones" {
  type    = list(string)
  default = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "private_subnet_cidrs" {
  type    = list(string)
  default = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  type    = list(string)
  default = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "eks_cluster_version" {
  type    = string
  default = "1.32"
}

variable "fsx_storage_capacity" {
  description = "FSx storage in GB — 1200 minimum for SCRATCH_2"
  type        = number
  default     = 1200
}
