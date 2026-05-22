locals {
  name = "${var.project_name}-${var.environment}"
}

resource "aws_eks_node_group" "system" {
  cluster_name    = var.cluster_name
  node_group_name = "${local.name}-system"
  node_role_arn   = var.eks_node_role_arn
  subnet_ids      = var.private_subnet_ids
  instance_types  = ["m5.xlarge"]

  scaling_config {
    desired_size = 2
    min_size     = 2
    max_size     = 4
  }

  update_config {
    max_unavailable = 1
  }

  taint {
    key    = "dedicated"
    value  = "system"
    effect = "NO_SCHEDULE"
  }

  labels = {
    role      = "system"
    node-type = "cpu"
  }

  tags = merge(var.tags, { Name = "${local.name}-system-ng" })
}

resource "aws_eks_node_group" "cpu" {
  cluster_name    = var.cluster_name
  node_group_name = "${local.name}-cpu"
  node_role_arn   = var.eks_node_role_arn
  subnet_ids      = var.private_subnet_ids
  instance_types  = ["m5.2xlarge"]

  scaling_config {
    desired_size = 2
    min_size     = 1
    max_size     = 6
  }

  update_config {
    max_unavailable = 1
  }

  labels = {
    role      = "cpu-workload"
    node-type = "cpu"
  }

  tags = merge(var.tags, { Name = "${local.name}-cpu-ng" })
}

resource "aws_launch_template" "gpu" {
  name_prefix = "${local.name}-gpu-lt-"
  description = "GPU nodes — 200GB root volume, IMDSv2"

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = 200
      volume_type           = "gp3"
      iops                  = 3000
      throughput            = 125
      delete_on_termination = true
      encrypted             = true
    }
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }

  tag_specifications {
    resource_type = "instance"
    tags          = merge(var.tags, { Name = "${local.name}-gpu-node" })
  }

  tags = var.tags

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_eks_node_group" "gpu" {
  cluster_name    = var.cluster_name
  node_group_name = "${local.name}-gpu"
  node_role_arn   = var.eks_node_role_arn
  subnet_ids      = var.private_subnet_ids
  instance_types  = ["g5.2xlarge"]
  capacity_type   = "ON_DEMAND"

  scaling_config {
    desired_size = 1
    min_size     = 0
    max_size     = 2
  }

  update_config {
    max_unavailable = 1
  }

  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  labels = {
    role             = "gpu-workload"
    node-type        = "gpu"
    "nvidia.com/gpu" = "true"
  }

  launch_template {
    id      = aws_launch_template.gpu.id
    version = "$Latest"
  }

  depends_on = [aws_launch_template.gpu]

  tags = merge(var.tags, { Name = "${local.name}-gpu-ng" })
}
