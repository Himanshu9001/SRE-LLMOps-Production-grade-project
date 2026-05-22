variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "cluster_name" {
  type = string
}

variable "eks_node_role_arn" {
  type = string
}

variable "private_subnet_ids" {
  type = list(string)
}

variable "gpu_subnet_ids" {
  description = "Subset of private subnets in AZs where G5 is available"
  type        = list(string)
}

variable "tags" {
  type    = map(string)
  default = {}
}
