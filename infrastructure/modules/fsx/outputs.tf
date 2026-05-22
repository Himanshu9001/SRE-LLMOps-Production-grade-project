output "fsx_id" {
  value = aws_fsx_lustre_file_system.main.id
}

output "fsx_dns_name" {
  description = "FSx DNS name — used in Kubernetes PersistentVolume spec"
  value       = aws_fsx_lustre_file_system.main.dns_name
}

output "fsx_mount_name" {
  description = "FSx mount name — required for Lustre client mount command"
  value       = aws_fsx_lustre_file_system.main.mount_name
}

output "fsx_security_group_id" {
  value = aws_security_group.fsx.id
}
