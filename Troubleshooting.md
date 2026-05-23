# Troubleshooting Log — SRE LLMOps Production Platform

Real issues encountered during the build, root causes, and fixes applied.
Every entry here represents an actual failure — not hypothetical scenarios.

---

## Table of Contents

1. [Infrastructure Issues](#1-infrastructure-issues)
2. [Docker & Container Issues](#2-docker--container-issues)
3. [Kubernetes Issues](#3-kubernetes-issues)
4. [AWS Quota & Capacity Issues](#4-aws-quota--capacity-issues)
5. [Python & Dependency Issues](#5-python--dependency-issues)
6. [Data Pipeline Issues](#6-data-pipeline-issues)
7. [MLflow Issues](#7-mlflow-issues)
8. [Terraform Issues](#8-terraform-issues)
9. [Git & DVC Issues](#9-git--dvc-issues)
10. [Network & Upload Issues](#10-network--upload-issues)

---

## 1. Infrastructure Issues

---

### 1.1 EKS Node Group — Unsupported AMI for Kubernetes 1.29

**Phase:** P2  
**Symptom:**
```
Error: creating EKS Node Group: InvalidParameterException:
Requested AMI for version 1.29 is not supported
```

**Root Cause:**  
EKS 1.29 AMIs deprecated and removed from us-east-1. AWS removes old Kubernetes version AMIs periodically. New accounts requesting old versions hit this error.

**Fix Applied:**  
Updated `eks_cluster_version` from `1.29` → `1.32` in `variables.tf` and `terraform.tfvars`. Since the cluster had no workloads, destroyed and recreated at correct version instead of sequential upgrade path.

**Prevention:**
```bash
# Always check latest supported EKS versions first
aws eks describe-cluster-versions --region us-east-1 \
  --query 'clusterVersions[*].clusterVersion' --output table
```

**Key Learning:**  
EKS in-place upgrades must follow sequential minor version path. Cannot skip 1.29→1.32 on a running cluster. For empty clusters: destroy + recreate is faster than 3 sequential upgrades.

---

### 1.2 RDS PostgreSQL 15.4 — Version Not Found

**Phase:** P3  
**Symptom:**
```
Error: creating RDS DB Instance: InvalidParameterCombination:
Cannot find version 15.4 for postgres
```

**Root Cause:**  
PostgreSQL 15.4 is no longer available as a minor version in us-east-1. AWS only maintains the latest patch for each major version (15.x becomes 15.latest).

**Fix Applied:**  
Checked available versions and updated to `17.4`:
```bash
aws rds describe-db-engine-versions \
  --engine postgres \
  --query 'DBEngineVersions[*].EngineVersion' \
  --output table \
  --region us-east-1
```
Updated `engine_version = "17.4"` in `infrastructure/modules/rds/main.tf`.

**General Rule Going Forward:**  
Never hardcode minor version numbers for managed AWS services. Use latest stable major version and let AWS manage patches.

---

### 1.3 GPU Node Group Stuck in CREATING for 28+ Minutes

**Phase:** P2/P4  
**Symptom:**
```
module.node_groups.aws_eks_node_group.gpu: Still creating... [28m00s elapsed]
```

**Root Cause:**  
`VcpuLimitExceeded` — new AWS account had GPU vCPU quota of 0 for G-family instances. The node group was stuck in CREATING state silently retrying failed instance launches.

**Diagnosis:**
```bash
aws autoscaling describe-scaling-activities \
  --region us-east-1 \
  --query 'Activities[*].{Status:StatusCode,Message:StatusMessage}' \
  --output table
# Output: VcpuLimitExceeded — quota of 0 for G and VT instances
```

**Fix Applied:**  
Submitted quota increase request:
```bash
aws service-quotas request-service-quota-increase \
  --service-code ec2 \
  --quota-code L-DB2E81BA \
  --desired-value 8 \
  --region us-east-1
```
Also submitted via AWS Support console with detailed use case description for faster review.

**Workaround While Waiting:**  
Ran smoke test with `opt-125m` on CPU nodes to validate entire training pipeline end-to-end without GPU.

**Key Learning:**  
New AWS accounts have GPU quota = 0. Always check service quotas before designing GPU-dependent architecture. Check with:
```bash
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-DB2E81BA \
  --region us-east-1 \
  --query 'Quota.Value'
```

---

### 1.4 G5 Spot Capacity Unavailable — UnfulfillableCapacity

**Phase:** P4  
**Symptom:**
```
Could not launch Spot Instances. UnfulfillableCapacity —
Unable to fulfill capacity due to your request configuration.
```

**Root Cause:**  
G5.2xlarge spot capacity exhausted in us-east-1a (single AZ configured). Peak hours + high demand = no spot available.

**Fix Applied:**  
Expanded GPU node group to all 3 AZs and added 4 fallback instance types:
```hcl
subnet_ids     = var.private_subnet_ids  # all 3 AZs
instance_types = [
  "g5.2xlarge",
  "g5.4xlarge",
  "g4dn.2xlarge",
  "g4dn.4xlarge",
]
capacity_type = "SPOT"
```

**Additional Fix:**  
Switched to `ON_DEMAND` for actual training runs since quota was already 0 — spot vs on-demand didn't matter until quota was approved.

**Key Learning:**  
Always configure GPU node groups with multiple AZs and multiple instance type fallbacks. EKS picks whichever has available capacity. Never single-AZ for spot GPU workloads.

---

## 2. Docker & Container Issues

---

### 2.1 Cross-Platform Build Failure — ARM64 Image on AMD64 Nodes

**Phase:** P4  
**Symptom:**  
Training pod scheduled on G5 (AMD64) but image built on M-series Mac (ARM64). Pod failed immediately with exec format error.

**Root Cause:**  
`docker build` on Apple Silicon produces ARM64 image by default. EKS G5 nodes are x86_64/AMD64. Architecture mismatch causes immediate pod failure.

**Fix Applied:**  
Switched to `docker buildx` with explicit platform targeting:
```bash
docker buildx create --use --name multiarch
docker buildx build \
  --platform linux/amd64 \
  -f docker/Dockerfile.training \
  -t $ECR_REPO:p4-v1 \
  --push \
  .
```

**Trade-off:**  
Cross-platform build via QEMU emulation: first build 1178 seconds (20 min). Subsequent builds with layer cache: 55 seconds. The `COPY src/ ./src/` layer is always rebuilt (source changes frequently) but PyTorch/pip layers are cached.

**Key Learning:**  
Always use `--platform linux/amd64` when building on Mac for AWS deployment. In CI/CD (GitHub Actions on ubuntu-latest), this is automatic — no buildx needed.

---

### 2.2 Dependency Conflict — tqdm Version Pinned Too Low

**Phase:** P4  
**Symptom:**
```
ERROR: Cannot install tqdm==4.66.2 because datasets==2.21.0
requires tqdm>=4.66.3
ResolutionImpossible
```

**Root Cause:**  
`tqdm==4.66.2` pinned in `requirements-training.txt` but `datasets==2.21.0` requires `>=4.66.3`. One patch version difference caused full build failure.

**Fix Applied:**  
Removed `tqdm` pin entirely — let pip resolve it as a transitive dependency:
```diff
- tqdm==4.66.2
# removed — transitive dep resolved by datasets==2.21.0
```

**General Rule:**  
Only pin direct dependencies. Never pin transitive dependencies unless you have a specific known-good reason. Use `pip-compile` to generate a full locked requirements file for production.

---

### 2.3 pathspec Import Error Breaking DVC

**Phase:** P1  
**Symptom:**
```
ERROR: unexpected error - cannot import name '_DIR_MARK'
from 'pathspec.patterns.gitwildmatch'
```

**Root Cause:**  
`pathspec==1.1.1` (installed by another package) broke DVC's internal import. DVC requires `pathspec==0.12.1` specifically — the `_DIR_MARK` symbol was removed in 1.1.x.

**Fix Applied:**
```bash
pip install "pathspec==0.12.1" --force-reinstall
```

Added to `requirements.txt` with pin comment:
```
pathspec==0.12.1  # pinned — 1.1.1 breaks DVC _DIR_MARK import
```

**Key Learning:**  
This is the one case where pinning a transitive dependency IS correct — when a library has a known incompatibility with a specific version of another library. Document why with a comment.

---

## 3. Kubernetes Issues

---

### 3.1 Pod Pending — Insufficient Ephemeral Storage

**Phase:** P1, P4  
**Symptom:**
```
0/4 nodes are available: 2 Insufficient ephemeral-storage,
2 node(s) had untolerated taint {dedicated: system}
```

**Root Cause:**  
Default EKS node root volume is 20GB. After system pods and Docker layers, only ~8-10GB free. Requesting `ephemeral-storage: 20Gi` exceeded available space.

**Fix Applied:**  
Reduced ephemeral storage request to 5Gi and switched to emptyDir without size limit for /tmp:
```yaml
resources:
  requests:
    ephemeral-storage: "5Gi"
  limits:
    ephemeral-storage: "10Gi"
volumes:
  - name: tmp-storage
    emptyDir: {}   # no sizeLimit — uses node disk space
```

**For GPU nodes:**  
GPU launch template configured with 200GB root volume to handle model weights (16GB Llama 3) + Docker layers + training artifacts.

---

### 3.2 MLflow CrashLoopBackOff — OOMKilled at Startup

**Phase:** P3  
**Symptom:**
```
Last State: OOMKilled, Exit Code: 137
Memory limit: 1Gi
```

**Root Cause:**  
`ghcr.io/mlflow/mlflow:latest` spawns multiple Uvicorn workers by default. Each worker loads the full MLflow application into memory. 4 workers × ~300MB = ~1.2GB peak usage at startup — exceeded 1Gi limit.

**Fix Applied:**  
Added `--workers=1` flag and increased memory limit to 2Gi:
```yaml
command:
  - mlflow
  - server
  - --workers=1      # prevent multi-worker OOM
  - --host=0.0.0.0
  - --port=5000
resources:
  limits:
    memory: "2Gi"    # increased from 1Gi
```

---

### 3.3 MLflow CrashLoopBackOff — Liveness Probe Killing Pod During Startup

**Phase:** P3  
**Symptom:**  
MLflow logs showed successful startup followed by SIGTERM:
```
Uvicorn running on http://0.0.0.0:5000
Received SIGTERM, exiting.
```
Pod restarted repeatedly despite no application error.

**Root Cause:**  
`livenessProbe` with `initialDelaySeconds: 30` fired before Uvicorn finished forking workers (~40-50 seconds). Kubernetes declared the pod unhealthy and killed it.

**Fix Applied:**  
Replaced `livenessProbe` + `readinessProbe` `initialDelaySeconds` with a `startupProbe` that blocks both until MLflow is fully ready:
```yaml
startupProbe:
  httpGet:
    path: /health
    port: 5000
  initialDelaySeconds: 15
  periodSeconds: 5
  failureThreshold: 24   # 24 × 5s = 120s max startup window
livenessProbe:
  httpGet:
    path: /health
    port: 5000
  periodSeconds: 15      # no initialDelaySeconds needed
  failureThreshold: 3
```

**Key Learning:**  
`startupProbe` was introduced specifically for this pattern — slow-starting applications that need a large initial window without disabling liveness checks entirely. Never use large `initialDelaySeconds` on liveness — use startupProbe instead.

---

### 3.4 MLflow 403 — DNS Rebinding Attack Detected

**Phase:** P3  
**Symptom:**
```
MlflowException: API request to /api/2.0/mlflow/experiments/create
failed with error code 403.
Response: 'Invalid Host header - possible DNS rebinding attack detected'
```

**Root Cause:**  
MLflow's security middleware blocks requests from non-localhost hosts by default. Even though the pod is binding to `0.0.0.0:5000`, the application-level security middleware rejects requests with non-localhost Host headers. Inter-pod communication from training namespace used the Kubernetes DNS name as Host header, which triggered the check.

**Fix Applied:**  
Added `--allowed-hosts=*` to MLflow server command:
```yaml
command:
  - mlflow
  - server
  - --allowed-hosts=*
  - --host=0.0.0.0
```

**Security Note:**  
`--allowed-hosts=*` disables DNS rebinding protection entirely. In production exposed via ALB, set to the specific domain: `--allowed-hosts=mlflow.internal.company.com`. For internal cluster-only access, `*` is acceptable.

---

### 3.5 Kubernetes Job Immutability — Cannot Update Running Job

**Phase:** P1, P3, P4 (multiple times)  
**Symptom:**
```
The Job "download-llama3-8b" is invalid: spec.template:
Invalid value: core.PodTemplateSpec{...}
```

**Root Cause:**  
Kubernetes Job specs are immutable after creation. Running `kubectl apply` with a changed Job spec fails. This happened multiple times when fixing job configurations mid-run.

**Fix Applied:**  
Delete then reapply:
```bash
kubectl delete job <job-name> -n training --force --grace-period=0
kubectl apply -f <job-manifest>.yaml
```

**One-liner alternative:**
```bash
kubectl replace --force -f <job-manifest>.yaml
```

**Key Learning:**  
Jobs are immutable by design — changing a running job's spec would invalidate already-running pods. Use `replace --force` or delete + apply. In CI/CD, use unique job names (include run number) to avoid this entirely.

---

### 3.6 GitHub Issues Scraper — 0 Records from All Repos

**Phase:** P1  
**Symptom:**
```
0 quality issues from kubernetes/kubernetes
0 quality issues from prometheus/prometheus
... (all 7 repos returned 0)
```

**Root Cause:**  
`labels: ["bug", "incident"]` filter in scraper config. Most open source repos don't use these exact label names. The API returned empty results before the `min_comments` filter even ran.

**Fix Applied:**  
Removed label filter entirely and relied on `min_comments >= 3` + `state=closed` as quality signal:
```yaml
github:
  filters:
    state: closed
    labels: []         # removed — repos don't use 'bug'/'incident'
    min_comments: 3
```

**Additional Fix:**  
Replaced monorepos (`kubernetes/kubernetes` with 100k+ issues) with focused smaller repos (`argoproj/argo-cd`, `fluent/fluent-bit`, `grafana/loki`) that complete in 3-5 minutes instead of getting stuck indefinitely.

---

## 4. AWS Quota & Capacity Issues

---

### 4.1 GPU Quota Request Rejected

**Phase:** P4  
**Symptom:**
```
We are unable to approve your service quota increase request.
Service quotas are put in place to help you gradually ramp up activity.
```

**Root Cause:**  
New AWS accounts are automatically flagged for GPU quota requests. The automated system rejected the initial request for 8 vCPUs (1 G5 instance) without human review.

**Fix Applied:**  
Resubmitted via AWS Support console (not CLI) with detailed business justification:
- Infrastructure already deployed (EKS, RDS, S3, ECR)
- Specific model and training workload described
- Expected usage hours and cost acknowledged
- Requested higher value (32 vCPUs) to signal production intent

**Workaround:**  
Ran complete pipeline smoke test on CPU with `facebook/opt-125m` (125M parameter model). Validated: S3 download → tokenization → QLoRA → MLflow logging → adapter upload. All code paths confirmed working without GPU.

**Cost Alternative Considered:**  
RunPod/Lambda Labs for ~$0.50/hr A100 with no quota approval. Kaggle for free T4/P100 GPU.

---

### 4.2 Terraform State Lock — PreconditionFailed

**Phase:** P2, P4  
**Symptom:**
```
Error acquiring the state lock
api error PreconditionFailed: At least one of the pre-conditions
you specified did not hold
Lock Info:
  ID: 3e5ed703-c6bc-73df-18a5-333579373487
  Operation: OperationTypeApply
```

**Root Cause:**  
Previous `terraform apply` was interrupted (Ctrl+C). The S3 lock file (`terraform.tfstate.tflock`) was left behind. Subsequent apply couldn't acquire lock.

**Fix Applied:**
```bash
terraform force-unlock 3e5ed703-c6bc-73df-18a5-333579373487
# Enter 'yes' when prompted
```

**When Force-Unlock is Safe:**  
Only when the previous apply process is confirmed dead (not just slow). Check: no running Terraform processes, no CI/CD jobs running. If uncertain — wait 10 minutes and try again.

**Key Learning:**  
The new `use_lockfile = true` S3 backend (replaced deprecated `dynamodb_table`) uses S3 object locking. The lock file is `terraform.tfstate.tflock` in the same S3 bucket/prefix.

---

## 5. Python & Dependency Issues

---

### 5.1 `Optional` NameError in Synthetic Generator

**Phase:** P1  
**Symptom:**
```
NameError: name 'Optional' is not defined
  File "src/scrapers/synthetic_generator.py", line 406
  def generate_sample(self, category: str) -> Optional[dict]:
```

**Root Cause:**  
`from typing import Optional` was placed at the BOTTOM of the file after the class definition that used it. Python executes top-to-bottom — the class was parsed before `Optional` was imported.

**Fix Applied:**
```bash
# Move import to top of file
sed -i '' 's/^from typing import Optional$//' src/scrapers/synthetic_generator.py
sed -i '' '1s/^/from typing import Optional\n/' src/scrapers/synthetic_generator.py
```

---

### 5.2 HuggingFace CLI Deprecated — `huggingface-cli` Not Found

**Phase:** P3  
**Symptom:**
```
Warning: `huggingface-cli` is deprecated and no longer works.
Use `hf` instead.
```

**Root Cause:**  
`huggingface_hub` package deprecated its `huggingface-cli` command in newer versions. The new CLI is `hf`. Training Docker image had newer `huggingface_hub` version than expected.

**Fix Applied:**  
Updated all scripts to use `hf auth login` instead of `huggingface-cli login`. For programmatic use, passed token directly to `hf_hub_download()`:
```python
hf_hub_download(
    repo_id=model_id,
    filename=filename,
    token=token,         # pass token directly, no CLI login needed
    local_dir=local_dir,
)
```

**Key Learning:**  
Never rely on CLI tools inside Docker containers when Python SDK alternatives exist. SDK calls are more stable across versions.

---

### 5.3 GatedRepoError 401 — Revoked HuggingFace Token

**Phase:** P3  
**Symptom:**
```
huggingface_hub.errors.GatedRepoError: 401 Client Error.
Cannot access gated repo for url https://huggingface.co/meta-llama/...
```

**Root Cause:**  
HuggingFace token was accidentally shared in chat conversation and needed to be revoked for security. The old token stored in Kubernetes secret became invalid. Subsequent download attempts all failed with 401.

**Fix Applied:**  
1. Revoked compromised token at `huggingface.co/settings/tokens`
2. Generated new token with `read` scope only
3. Updated Kubernetes secret:
```bash
kubectl delete secret huggingface-secret -n training
kubectl create secret generic huggingface-secret \
  --namespace=training \
  --from-literal=token='<new_token>'
```

**Security Lesson:**  
Never share tokens, API keys, or credentials in any chat interface, email, or document. Treat them like passwords. Rotate immediately if exposed — assume compromised.

---

### 5.4 zsh History Expansion — `!` in Double-Quoted Strings

**Phase:** Multiple  
**Symptom:**
```
dquote>
dquote>
(shell hangs waiting for closing quote)
```
Or:
```
zsh: command not found: #
```

**Root Cause 1:** `!` inside double-quoted strings triggers zsh history expansion. `"MLflow2024SecurePass!"` causes the shell to try to expand `!` as a history event.

**Root Cause 2:** Pasting multi-line commands with `#` comment lines — zsh treats `# comment` as a command when pasted interactively (unlike bash which ignores them).

**Fix Applied:**
```bash
# Use single quotes for strings containing !
kubectl create secret generic mlflow-db-secret \
  --from-literal=password='MLflow2024SecurePass!'  # single quotes

# For multi-line commands with comments — run as a script file
# or remove comments before pasting into terminal
```

---

## 6. Data Pipeline Issues

---

### 6.1 GitHub Scraper Stuck on kubernetes/kubernetes

**Phase:** P1  
**Symptom:**  
GitHub scraper stuck on `kubernetes/kubernetes` for 20+ minutes with no output. Process had to be killed.

**Root Cause:**  
`kubernetes/kubernetes` has 100,000+ closed issues. GitHub API paginates through them sorted by comments. Even with `max_issues_per_repo=100`, the iterator had to scan thousands of issues to find 100 with sufficient comments. At 0.3s delay per issue = 30,000 × 0.3s = 9,000 seconds theoretical maximum.

**Fix Applied:**  
Replaced large monorepos with focused smaller repos that complete in 3-5 minutes each:
```yaml
repositories:
  - argoproj/argo-cd          # ~8k issues, focused ArgoCD bugs
  - fluent/fluent-bit          # ~3k issues, log pipeline problems
  - grafana/loki               # ~4k issues, logging stack
  - VictoriaMetrics/VictoriaMetrics
  - cert-manager/cert-manager
  - external-secrets/external-secrets
  - aws/karpenter
```

**Key Learning:**  
Add `break` when `issue.comments < min_comments` only works if issues are sorted by `comments desc`. This optimization makes the early-exit effective — once you hit an issue with fewer than min_comments, all subsequent issues also have fewer (sorted order).

---

### 6.2 Dataset S3 Upload Missing pytorch_model.bin

**Phase:** P3  
**Symptom:**  
`aws s3 ls` showed 8 files but `pytorch_model.bin` (250MB) missing despite upload script printing it.

**Root Cause:**  
Upload interrupted by `Ctrl+C` mid-multipart upload. S3 multipart upload stores parts until completed or aborted. Interrupted upload leaves orphaned parts consuming storage but no complete object.

**Fix Applied:**  
Used AWS CLI instead of boto3 for large file uploads — CLI has built-in retry and resume:
```bash
aws s3 cp /tmp/opt-125m/pytorch_model.bin \
  s3://sre-llmops-artifacts/smoke-test/opt-125m/pytorch_model.bin \
  --sse AES256 \
  --region us-east-1
```

**Cleanup:**  
Added S3 lifecycle rule to abort incomplete multipart uploads after 7 days:
```json
{
  "Rules": [{
    "ID": "abort-incomplete-multipart",
    "Status": "Enabled",
    "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7}
  }]
}
```

---

### 6.3 DVC Commit Failed — Output File Does Not Exist

**Phase:** P1  
**Symptom:**
```
ERROR: failed to commit - output
'data/processed/github_issues_formatted.jsonl' does not exist
```

**Root Cause:**  
`dvc.yaml` listed `github_issues_formatted.jsonl` as an output of the `format` stage. But since GitHub scraping was skipped (no token), the file was never created. DVC couldn't commit a stage with missing outputs.

**Fix Applied:**
```bash
touch data/processed/github_issues_formatted.jsonl
dvc commit --force
```
Created an empty placeholder file. DVC warned it was empty but committed successfully. File was replaced with real content after GitHub scraping completed.

---

## 7. MLflow Issues

---

### 7.1 MLflow Experiment Not Found — Creates New One Instead

**Phase:** P3  
**Symptom:**  
MLflow training logs showed:
```
Experiment with name 'sre-smoke-test' does not exist. Creating a new experiment.
```
Expected to find existing experiment but created a duplicate.

**Root Cause:**  
This is actually correct MLflow behavior — not an error. When `mlflow.set_experiment()` is called with a non-existent experiment name, MLflow creates it automatically. The message is INFO level, not an error.

**Resolution:**  
No fix needed. Confirmed experiment was created correctly with ID=2. MLflow UI showed the experiment with the correct run.

---

### 7.2 MLflow S3 Artifacts — AccessDenied on PutObject

**Phase:** P3/P4  
**Symptom:**
```
botocore.exceptions.ClientError: AccessDenied when calling
PutObject on s3://sre-llmops-artifacts/adapters/...
```

**Root Cause:**  
IAM IRSA policy for `training-irsa` role only allowed `s3:PutObject` on `adapters/*`, `checkpoints/*`, and `distilled/*` — NOT on `base-models/*`. The model download job tried to write to `base-models/` and was denied.

**Fix Applied:**  
Updated IAM policy in `infrastructure/modules/iam/main.tf` to include `base-models/*` in write permissions:
```hcl
{
  Effect = "Allow"
  Action = ["s3:PutObject", "s3:DeleteObject"]
  Resource = [
    "arn:aws:s3:::${var.s3_bucket_name}/base-models/*",
    "arn:aws:s3:::${var.s3_bucket_name}/adapters/*",
    "arn:aws:s3:::${var.s3_bucket_name}/checkpoints/*",
    "arn:aws:s3:::${var.s3_bucket_name}/distilled/*",
    "arn:aws:s3:::${var.s3_bucket_name}/quantized/*"
  ]
}
```
Applied with `terraform apply`.

**Key Learning:**  
IAM policy changes propagate eventually — there is a short window after `terraform apply` where old policy may still be cached (~30-60 seconds). If job fails immediately after apply, wait 60 seconds and retry.

---

## 8. Terraform Issues

---

### 8.1 HCL Syntax Error — Semicolons Not Valid

**Phase:** P2  
**Symptom:**
```
Error: Invalid character
The ";" character is not valid. Use newlines to separate arguments.
```

**Root Cause:**  
Used compact single-line variable syntax with semicolons:
```hcl
variable "cluster_version" { type = string; default = "1.29" }
```
HCL uses newlines as statement terminators — semicolons are invalid.

**Fix Applied:**  
Rewrote all variable blocks with proper newline separation:
```hcl
variable "cluster_version" {
  type    = string
  default = "1.32"
}
```

---

### 8.2 Launch Template Version Drift — Provider Bug

**Phase:** P2  
**Symptom:**
```
Error: Provider produced inconsistent final plan
.launch_template[0].version: was cty.StringVal("1"),
but now cty.StringVal("2")
```

**Root Cause:**  
Known AWS Terraform provider bug. When a launch template is updated during apply, its version number increments. The plan was computed with version "1" but by apply time it was "2" — provider rejected the inconsistency.

**Fix Applied:**  
Used `"$Latest"` instead of `latest_version` attribute:
```hcl
launch_template {
  id      = aws_launch_template.gpu.id
  version = "$Latest"   # AWS resolves this at runtime
}
```
Added `lifecycle { create_before_destroy = true }` to launch template to avoid version conflicts.

---

### 8.3 Terraform Tainted Resource After Interrupted Apply

**Phase:** P2  
**Symptom:**
```
# module.node_groups.aws_eks_node_group.gpu: (tainted)
resource "aws_eks_node_group" "gpu" {
```

**Root Cause:**  
Ctrl+C during `terraform apply` interrupted mid-resource-creation. Terraform marked the partially-created resource as "tainted" — flagged for forced replacement on next apply.

**Fix Applied:**  
In this case the resource was already deleted in AWS (failed creation), so running `terraform apply` was correct — it destroyed the tainted resource and recreated it cleanly.

**If resource exists in AWS but is tainted:**
```bash
# Remove taint without destroying resource
terraform untaint module.node_groups.aws_eks_node_group.gpu

# Or import existing resource into state
terraform import module.node_groups.aws_eks_node_group.gpu \
  cluster-name:nodegroup-name
```

---

## 9. Git & DVC Issues

---

### 9.1 Git Push Rejected — 648MB Terraform Provider Binary

**Phase:** P2  
**Symptom:**
```
remote: error: File infrastructure/live/us-east-1/.terraform/providers/
registry.terraform.io/hashicorp/aws/5.100.0/darwin_arm64/
terraform-provider-aws_v5.100.0_x5 is 648.39 MB;
this exceeds GitHub's file size limit of 100.00 MB
```

**Root Cause:**  
`.terraform/` directory containing downloaded provider binaries was accidentally committed. The AWS provider binary alone is 648MB.

**Fix Applied:**
```bash
# Remove from Git tracking (keep local)
git rm -r --cached infrastructure/live/us-east-1/.terraform/

# Add to .gitignore
echo "**/.terraform/" >> .gitignore
echo "*.tfstate" >> .gitignore
echo "*.tfstate.backup" >> .gitignore

# Amend commit to exclude it
git commit --amend --no-edit
git push origin main --force
```

**What Should Be Committed:**
- `*.tf` files — Terraform configuration
- `.terraform.lock.hcl` — provider version lock file (reproducibility)

**What Should NOT Be Committed:**
- `.terraform/` — downloaded providers and modules (regenerated by `terraform init`)
- `*.tfstate` — state files (stored in S3 remote backend)
- `terraform.tfvars` — if it contains secrets

---

### 9.2 DVC Pipeline Stages Overlap With dvc add

**Phase:** P1  
**Symptom:**
```
ERROR: cannot update 'sre_ops_train_v1.jsonl': overlaps with
an output of stage: 'validate' in 'dvc.yaml'.
Run the pipeline or use 'dvc commit' to force update it.
```

**Root Cause:**  
Tried to run `dvc add` on files that were already tracked as outputs of `dvc.yaml` pipeline stages. DVC prevents double-tracking — a file can't be both a pipeline output and a standalone tracked file.

**Fix Applied:**
```bash
dvc commit --force
# Instead of dvc add — commits current state of pipeline outputs
```

**Key Learning:**  
If a file is an output in `dvc.yaml`, use `dvc commit` to update its cache. Use `dvc add` only for files that aren't part of any pipeline stage.

---

## 10. Network & Upload Issues

---

### 10.1 boto3 S3 Upload Timeout — ConnectionClosedError

**Phase:** P3  
**Symptom:**
```
botocore.exceptions.ConnectionClosedError: Connection was closed
before we received a valid response from endpoint URL:
"https://sre-llmops-artifacts.s3.amazonaws.com/...
?uploadId=QvWFEA0n...&partNumber=5"
```

**Root Cause:**  
Unstable home internet connection dropped during multipart upload of 250MB `pytorch_model.bin`. boto3's default S3 Transfer Manager doesn't retry failed parts — the entire upload fails.

**Fix Applied:**  
Switched from boto3 `upload_file()` to AWS CLI `s3 cp` which has built-in multipart retry:
```bash
aws s3 cp /tmp/opt-125m/pytorch_model.bin \
  s3://sre-llmops-artifacts/smoke-test/opt-125m/pytorch_model.bin \
  --sse AES256 \
  --region us-east-1
```

**Production Fix (for programmatic uploads):**
```python
from boto3.s3.transfer import TransferConfig

config = TransferConfig(
    multipart_threshold=8 * 1024 * 1024,   # 8MB
    max_concurrency=10,
    multipart_chunksize=8 * 1024 * 1024,
    retries={'max_attempts': 5},
)
s3.upload_file(local_path, bucket, key, Config=config)
```

---

### 10.2 HuggingFace Download — Slow on Unauthenticated Requests

**Phase:** P3  
**Symptom:**
```
Warning: You are sending unauthenticated requests to the HF Hub.
Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Downloading (incomplete total...): 0.00B [00:00, ?B/s]
```

**Root Cause:**  
HuggingFace applies lower rate limits and bandwidth to unauthenticated requests. The `0.00B` shown is because the download progress bar shows total progress across all files — individual files were downloading but the total was unknown.

**Fix Applied:**  
Always set `HF_TOKEN` environment variable before any HuggingFace operations:
```bash
export HF_TOKEN=$(kubectl get secret huggingface-secret \
  -n training \
  -o jsonpath='{.data.token}' | base64 -d)
```

**Note:**  
The `0.00B` total is a known HuggingFace Hub display issue when total file sizes can't be determined upfront. Actual download was proceeding normally.

---

## Summary — Issues by Phase

| Phase | Issues Encountered | Most Impactful |
|-------|--------------------|----------------|
| P1 | GitHub scraper stuck, label filter returning 0, DVC commit conflict, tqdm version conflict | Label filter bug — wasted 2 hours debugging |
| P2 | EKS 1.29 AMI deprecated, HCL semicolons, launch template drift, .terraform committed to Git | Git push rejected at 648MB — required history rewrite |
| P3 | RDS 15.4 not found, MLflow OOMKilled, liveness probe killing pod, 403 DNS rebinding, token expired | MLflow probe loop — subtle, took multiple restart cycles to diagnose |
| P4 | GPU quota 0, spot UnfulfillableCapacity, cross-platform build, ephemeral storage insufficient | GPU quota rejection — blocked real training, required CPU smoke test workaround |
| P4 (data) | pytorch_model.bin upload interrupted, boto3 timeout, HuggingFace token in plaintext | Token exposure — required immediate revocation and rotation |

---

## Quick Reference — Common Fixes

```bash
# Terraform state lock stuck
terraform force-unlock <lock-id>

# Kubernetes Job immutable
kubectl delete job <name> -n <namespace> --force --grace-period=0
kubectl apply -f <manifest>.yaml

# Pod stuck in Pending — check why
kubectl describe pod <name> -n <namespace> | grep -A10 "Events:"

# Check GPU quota
aws service-quotas get-service-quota \
  --service-code ec2 --quota-code L-DB2E81BA --region us-east-1

# Check spot capacity failure
aws autoscaling describe-scaling-activities --region us-east-1 \
  --query 'Activities[*].{Status:StatusCode,Message:StatusMessage}' \
  --output table

# DVC pipeline outputs — use commit not add
dvc commit --force

# Large file accidentally committed
git rm -r --cached <path>
git commit --amend --no-edit
git push origin main --force

# pathspec DVC fix
pip install "pathspec==0.12.1" --force-reinstall

# Check EKS node group status
aws eks describe-nodegroup \
  --cluster-name sre-llmops-production \
  --nodegroup-name sre-llmops-production-gpu \
  --region us-east-1 \
  --query 'nodegroup.{Status:status,Health:health}'
```