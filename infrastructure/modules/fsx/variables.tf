variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "subnet_id" {
  description = "Single subnet for FSx — FSx Lustre is single-AZ"
  type        = string
}

variable "vpc_cidr" {
  type = string
}

variable "s3_bucket_name" {
  type = string
}

variable "s3_import_prefix" {
  description = "S3 prefix to auto-import into FSx"
  type        = string
  default     = ""
}

variable "storage_capacity" {
  description = "FSx storage in GB — minimum 1200GB for SCRATCH_2"
  type        = number
  default     = 1200
}

variable "tags" {
  type    = map(string)
  default = {}
}
