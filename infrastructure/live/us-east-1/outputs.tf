output "vpc_id" {
  value = module.vpc.vpc_id
}

output "cluster_name" {
  value = module.eks.cluster_name
}

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "mlflow_irsa_arn" {
  value = module.iam.mlflow_irsa_arn
}

output "training_irsa_arn" {
  value = module.iam.training_irsa_arn
}

output "inference_irsa_arn" {
  value = module.iam.inference_irsa_arn
}

output "rds_endpoint" {
  value = module.rds.rds_endpoint
}
