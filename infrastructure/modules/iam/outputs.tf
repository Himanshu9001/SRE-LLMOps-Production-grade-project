output "eks_cluster_role_arn" {
  value = aws_iam_role.eks_cluster.arn
}

output "eks_node_role_arn" {
  value = aws_iam_role.eks_node.arn
}

output "mlflow_irsa_arn" {
  description = "Annotate MLflow service account with this ARN"
  value       = aws_iam_role.mlflow.arn
}

output "training_irsa_arn" {
  description = "Annotate training job service account with this ARN"
  value       = aws_iam_role.training.arn
}

output "inference_irsa_arn" {
  description = "Annotate vLLM service account with this ARN"
  value       = aws_iam_role.inference.arn
}
