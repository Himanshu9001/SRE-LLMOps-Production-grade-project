# ------------------------------------------------------------
# VPC — foundation for all networking
# Single VPC, 3 public + 3 private subnets across 3 AZs
# Private subnets: EKS nodes, FSx for Lustre
# Public subnets:  NAT Gateway, ALB ingress
# ------------------------------------------------------------

locals {
  name = "${var.project_name}-${var.environment}"
}

# --- VPC ---
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true   # required for EKS node registration
  enable_dns_support   = true   # required for FSx DNS resolution

  tags = merge(var.tags, {
    Name = "${local.name}-vpc"
    # EKS requires these tags on VPC for cluster auto-discovery
    "kubernetes.io/cluster/${local.name}" = "shared"
  })
}

# --- Internet Gateway — public subnet egress ---
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = merge(var.tags, { Name = "${local.name}-igw" })
}

# --- Public Subnets — one per AZ ---
resource "aws_subnet" "public" {
  count                   = length(var.availability_zones)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true  # instances get public IPs for NAT GW

  tags = merge(var.tags, {
    Name = "${local.name}-public-${var.availability_zones[count.index]}"
    # Required tag: tells EKS this subnet can host internet-facing ALBs
    "kubernetes.io/role/elb" = "1"
    "kubernetes.io/cluster/${local.name}" = "shared"
  })
}

# --- Private Subnets — one per AZ ---
resource "aws_subnet" "private" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = var.availability_zones[count.index]

  tags = merge(var.tags, {
    Name = "${local.name}-private-${var.availability_zones[count.index]}"
    # Required tag: tells EKS this subnet hosts internal load balancers
    "kubernetes.io/role/internal-elb" = "1"
    "kubernetes.io/cluster/${local.name}" = "shared"
  })
}

# --- Elastic IPs for NAT Gateways — one per AZ ---
# One NAT GW per AZ: if one AZ goes down, other AZs still have outbound internet
# Cost trade-off: 3 NAT GWs vs 1 — use 1 for dev, 3 for production
resource "aws_eip" "nat" {
  count  = length(var.availability_zones)
  domain = "vpc"
  tags   = merge(var.tags, { Name = "${local.name}-nat-eip-${count.index}" })

  depends_on = [aws_internet_gateway.main]
}

# --- NAT Gateways — one per public subnet ---
resource "aws_nat_gateway" "main" {
  count         = length(var.availability_zones)
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = merge(var.tags, {
    Name = "${local.name}-nat-${var.availability_zones[count.index]}"
  })

  depends_on = [aws_internet_gateway.main]
}

# --- Public Route Table — routes internet traffic through IGW ---
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = merge(var.tags, { Name = "${local.name}-public-rt" })
}

# Associate all public subnets with public route table
resource "aws_route_table_association" "public" {
  count          = length(var.availability_zones)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# --- Private Route Tables — one per AZ, routes outbound through AZ-local NAT GW ---
resource "aws_route_table" "private" {
  count  = length(var.availability_zones)
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[count.index].id
  }

  tags = merge(var.tags, {
    Name = "${local.name}-private-rt-${var.availability_zones[count.index]}"
  })
}

# Associate each private subnet with its AZ-local route table
resource "aws_route_table_association" "private" {
  count          = length(var.availability_zones)
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# --- VPC Endpoints — keep AWS API traffic off the internet ---
# S3 Gateway endpoint: EKS nodes pull ECR images via S3 — free, no NAT GW charges
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${data.aws_region.current.name}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = aws_route_table.private[*].id

  tags = merge(var.tags, { Name = "${local.name}-s3-endpoint" })
}

# ECR API endpoint: allows EKS nodes to authenticate with ECR privately
resource "aws_vpc_endpoint" "ecr_api" {
  vpc_id              = aws_vpc.main.id
  service_name        = "com.amazonaws.${data.aws_region.current.name}.ecr.api"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = aws_subnet.private[*].id
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  private_dns_enabled = true

  tags = merge(var.tags, { Name = "${local.name}-ecr-api-endpoint" })
}

# ECR DKR endpoint: allows EKS nodes to pull images from ECR privately
resource "aws_vpc_endpoint" "ecr_dkr" {
  vpc_id              = aws_vpc.main.id
  service_name        = "com.amazonaws.${data.aws_region.current.name}.ecr.dkr"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = aws_subnet.private[*].id
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  private_dns_enabled = true

  tags = merge(var.tags, { Name = "${local.name}-ecr-dkr-endpoint" })
}

# --- Security Group for VPC Endpoints ---
resource "aws_security_group" "vpc_endpoints" {
  name        = "${local.name}-vpc-endpoints-sg"
  description = "Allow HTTPS from VPC CIDR to VPC endpoints"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "HTTPS from VPC"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, { Name = "${local.name}-vpc-endpoints-sg" })
}

data "aws_region" "current" {}
