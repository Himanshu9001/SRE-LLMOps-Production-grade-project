output "rds_endpoint" {
  description = "RDS endpoint — used in MLflow DATABASE_URI env var"
  value       = aws_db_instance.mlflow.endpoint
}

output "rds_db_name" {
  value = aws_db_instance.mlflow.db_name
}
