# ------------------------------------------------------------
# Node Groups — three tiers for different workloads
# System:   m5.xlarge   — kube-system, monitoring, ArgoCD
# CPU:      m5.2xlarge  — MLflow, pipeline orchestration
# GPU:      g5.2xlarge  — training (QLoRA) and inference (vLLM)
#
# GPU nodes use SPOT pricing — up to 70% cheaper than on-demand
# Spot interruption handled by Karpenter in later phases
# ------------------------------------------------------------

locals {
  name = "${var.project_name}-${var.environment}"
}

# --- System Node Group — always-on, manages cluster infrastructure ---
resource "aws_eks_node_group" "system" {
  cluster_name    = var.cluster_name
  node_group_name = "${local.name}-system"
  node_role_arn   = var.eks_node_role_arn
  subnet_ids      = var.private_subnet_ids

  instance_types = ["m5.xlarge"]   # 4 vCPU, 16GB RAM — sufficient for system pods

  scaling_config {
    desired_size = 2
    min_size     = 2   # always keep 2 for HA — system pods need redundancy
    max_size     = 4
  }

  update_config {
    max_unavailable = 1   # rolling update — never take both nodes down simultaneously
  }

  # Taint system nodes so only system pods schedule here
  taint {
    key    = "dedicated"
    value  = "system"
    effect = "NO_SCHEDULE"
  }

  labels = {
    role        = "system"
    node-type   = "cpu"
  }

  tags = merge(var.tags, { Name = "${local.name}-system-ng" })
}

# --- CPU Node Group — MLflow, training pipelines, data preprocessing ---
resource "aws_eks_node_group" "cpu" {
  cluster_name    = var.cluster_name
  node_group_name = "${local.name}-cpu"
  node_role_arn   = var.eks_node_role_arn
  subnet_ids      = var.private_subnet_ids

  instance_types = ["m5.2xlarge"]   # 8 vCPU, 32GB RAM — MLflow + pipeline jobs

  scaling_config {
    desired_size = 2
    min_size     = 1
    max_size     = 6   # scale up for parallel preprocessing jobs
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

# --- GPU Node Group — QLoRA training and vLLM inference ---
resource "aws_eks_node_group" "gpu" {
  cluster_name    = var.cluster_name
  node_group_name = "${local.name}-gpu"
  node_role_arn   = var.eks_node_role_arn

  # G5 only available in specific AZs — use gpu_subnet_ids not all private subnets
  subnet_ids     = var.gpu_subnet_ids
  instance_types = ["g5.2xlarge"]   # 1x A10G GPU 24GB VRAM, 8 vCPU, 32GB RAM

  # SPOT pricing — G5 spot is ~70% cheaper than on-demand (~$0.34/hr vs ~$1.21/hr)
  # Acceptable for training (can checkpoint + resume) and inference (stateless)
  capacity_type = "SPOT"

  scaling_config {
    desired_size = 0   # scale to 0 when not training — major cost saving
    min_size     = 0   # 0 min: no GPU cost when idle
    max_size     = 2   # max 2 for distributed training in P5
  }

  update_config {
    max_unavailable = 1
  }

  # GPU taint — only pods that explicitly tolerate this will schedule on GPU nodes
  # Prevents CPU workloads from accidentally consuming expensive GPU capacity
  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  labels = {
    role             = "gpu-workload"
    node-type        = "gpu"
    "nvidia.com/gpu" = "true"
    accelerator      = "nvidia-a10g"
  }

  # Launch template for GPU-specific configuration
  launch_template {
    id      = aws_launch_template.gpu.id
    version = aws_launch_template.gpu.latest_version
  }

  tags = merge(var.tags, { Name = "${local.name}-gpu-ng" })
}

# --- Launch Template for GPU nodes ---
# Configures larger root volume (model weights need space) and GPU optimized AMI settings
resource "aws_launch_template" "gpu" {
  name_prefix   = "${local.name}-gpu-lt-"
  description   = "Launch template for GPU nodes — larger root volume, GPU optimized settings"

  # 200GB root volume — Llama 3 8B is 16GB, need space for model + Docker layers
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
    http_tokens                 = "required"   # IMDSv2 — prevents SSRF attacks on metadata endpoint
    http_put_response_hop_limit = 2            # 2 hops needed for containers to access IMDS
  }

  tag_specifications {
    resource_type = "instance"
    tags          = merge(var.tags, { Name = "${local.name}-gpu-node" })
  }

  tags = var.tags
}
