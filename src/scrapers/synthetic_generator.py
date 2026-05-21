from typing import Optional
import random
import json
from pathlib import Path

import jsonlines
from rich.progress import track

from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Incident templates — patterns derived from real SRE production incidents.
# Each template has: instruction variants, context variants, resolution steps.
# The generator samples and combines these to produce diverse training pairs.
# ---------------------------------------------------------------------------

INCIDENT_TEMPLATES = {
    "kubernetes_crashloop": {
        "instructions": [
            "A pod is in CrashLoopBackOff state. Diagnose and fix the issue.",
            "Kubernetes pod keeps restarting with CrashLoopBackOff. What are the steps to debug?",
            "My deployment pods are CrashLoopBackOff. Walk me through the troubleshooting process.",
        ],
        "contexts": [
            "Pod: {pod_name} in namespace {namespace}. Last exit code: {exit_code}. Restarts: {restarts}.",
            "kubectl get pods shows {pod_name} as CrashLoopBackOff with {restarts} restarts in {namespace}.",
            "Alert fired: KubePodCrashLooping for {pod_name} in {namespace}. Exit code {exit_code}.",
        ],
        "resolutions": [
            """1. Check pod logs for root cause:
   kubectl logs {pod_name} -n {namespace} --previous

2. Describe the pod for events and resource issues:
   kubectl describe pod {pod_name} -n {namespace}

3. Common causes by exit code:
   - Exit 1: Application error — check app logs for stack trace
   - Exit 137: OOMKilled — increase memory limits or fix memory leak
   - Exit 139: Segfault — binary or dependency issue
   - Exit 143: SIGTERM not handled — fix graceful shutdown

4. If config/secret related:
   kubectl get events -n {namespace} --sort-by='.lastTimestamp'

5. If image issue:
   kubectl set image deployment/{deployment_name} {container}={new_image} -n {namespace}

6. Verify fix:
   kubectl rollout status deployment/{deployment_name} -n {namespace}""",
        ],
        "variables": {
            "pod_name": ["api-server-7d9f8b-xkp2m", "worker-deployment-5c8d9-mnb4k", "fluent-bit-ds-z9x2p"],
            "namespace": ["production", "monitoring", "default", "kube-system"],
            "exit_code": ["1", "137", "139", "143"],
            "restarts": ["5", "12", "23", "47"],
            "deployment_name": ["api-server", "worker-deployment", "backend-service"],
            "container": ["api", "worker", "backend"],
            "new_image": ["myapp:v2.1.1", "myapp:v1.9.3-hotfix", "myapp:stable"],
        }
    },

    "pod_oom_killed": {
        "instructions": [
            "A pod was OOMKilled. How do I diagnose and prevent this?",
            "Pod terminated with reason OOMKilled. What are the steps to investigate and fix?",
            "Kubernetes node shows memory pressure, pods getting OOMKilled. How to resolve?",
        ],
        "contexts": [
            "Pod {pod_name} in {namespace} was OOMKilled. Current memory limit: {mem_limit}. Node memory usage: {node_mem}%.",
            "kubectl describe pod {pod_name} shows Last State: OOMKilled. Memory limit set to {mem_limit}.",
            "Alert: KubeContainerOOMKilled for {pod_name}. Container {container} exceeded memory limit {mem_limit}.",
        ],
        "resolutions": [
            """1. Confirm OOMKill:
   kubectl describe pod {pod_name} -n {namespace} | grep -A5 "Last State"

2. Check actual memory usage before the kill:
   kubectl top pod {pod_name} -n {namespace}
   kubectl top node

3. Check if limit is too low — compare request vs actual usage:
   kubectl get pod {pod_name} -n {namespace} -o jsonpath='{{.spec.containers[*].resources}}'

4. Increase memory limit in deployment manifest:
   resources:
     requests:
       memory: "256Mi"
     limits:
       memory: "{new_mem_limit}"

5. If memory leak suspected — check heap dump or profiler output in app logs

6. Apply and verify:
   kubectl apply -f deployment.yaml
   kubectl rollout status deployment/{deployment_name} -n {namespace}

7. Set up VPA (Vertical Pod Autoscaler) for automatic right-sizing in future.""",
        ],
        "variables": {
            "pod_name": ["ml-inference-7f9d-xp2k", "data-processor-5b8c-mn3j", "api-gateway-3d7f-kl9m"],
            "namespace": ["production", "ml-serving", "data-pipeline"],
            "mem_limit": ["512Mi", "1Gi", "2Gi"],
            "node_mem": ["87", "92", "95"],
            "container": ["inference", "processor", "gateway"],
            "new_mem_limit": ["1Gi", "2Gi", "4Gi"],
            "deployment_name": ["ml-inference", "data-processor", "api-gateway"],
        }
    },

    "prometheus_alert_firing": {
        "instructions": [
            "A Prometheus alert is firing. How do I investigate and resolve it?",
            "PagerDuty alert triggered from Prometheus. Walk me through the investigation.",
            "Prometheus AlertManager fired a critical alert. What is the SRE response process?",
        ],
        "contexts": [
            "Alert: {alert_name} firing for {duration}. Severity: {severity}. Labels: namespace={namespace}, service={service}.",
            "AlertManager notification: {alert_name} - {alert_description}. Started {duration} ago.",
            "Grafana alert triggered: {alert_name} in {namespace}. Threshold breached: {metric} > {threshold}.",
        ],
        "resolutions": [
            """1. Acknowledge alert and start incident channel:
   Acknowledge in PagerDuty/AlertManager to stop escalation

2. Check alert definition to understand what fired:
   kubectl get prometheusrule -n {namespace}
   # Or check Prometheus UI: /alerts

3. Query the firing metric in Prometheus:
   {metric}{{namespace="{namespace}", service="{service}"}}

4. Check recent deployments that may have caused regression:
   kubectl rollout history deployment/{service} -n {namespace}

5. Check pod health and logs:
   kubectl get pods -n {namespace} -l app={service}
   kubectl logs -l app={service} -n {namespace} --tail=100

6. If high error rate — check downstream dependencies:
   kubectl get endpoints -n {namespace}

7. Escalation path if not resolved in {slo_window}:
   Page on-call lead + open war room

8. After resolution — write postmortem with timeline and action items.""",
        ],
        "variables": {
            "alert_name": ["HighErrorRate", "PodCrashLooping", "NodeMemoryPressure", "TargetDown", "HighLatency"],
            "duration": ["5m", "15m", "30m", "1h"],
            "severity": ["critical", "warning", "page"],
            "namespace": ["production", "monitoring", "api"],
            "service": ["payment-service", "auth-api", "data-pipeline", "ml-inference"],
            "alert_description": ["Error rate > 5%", "Pod restarted 5 times", "Memory usage > 90%"],
            "metric": ["http_request_errors_total", "container_memory_usage_bytes", "up"],
            "threshold": ["0.05", "0.9", "0"],
            "slo_window": ["15 minutes", "30 minutes", "1 hour"],
        }
    },

    "terraform_state_lock": {
        "instructions": [
            "Terraform state is locked. How do I safely resolve this?",
            "terraform apply is failing with state lock error. What are the steps to fix?",
            "Terraform backend shows state locked by another process. How to investigate and unlock?",
        ],
        "contexts": [
            "Error: Error acquiring the state lock. Lock ID: {lock_id}. Info: {lock_info}.",
            "terraform plan fails: state file locked by {locker} since {locked_since}.",
            "CI pipeline stuck: Terraform state lock on {workspace} workspace. Lock ID: {lock_id}.",
        ],
        "resolutions": [
            """1. Identify who holds the lock:
   terraform force-unlock --help
   # Check DynamoDB table (if S3 backend):
   aws dynamodb get-item --table-name terraform-state-lock \\
     --key '{{"LockID": {{"S": "{state_path}"}}}}'

2. Verify the locking process is actually dead (not just slow):
   # Check CI job status — never unlock if job is still running
   # Check AWS console for active ECS/CodeBuild jobs

3. If lock is stale (process confirmed dead):
   terraform force-unlock {lock_id}
   # This is IRREVERSIBLE — confirm the holding process is dead first

4. Re-run terraform plan to verify state is clean:
   terraform plan -out=tfplan

5. If state is corrupted after force-unlock:
   # Pull last known good state from S3 versioning
   aws s3api list-object-versions --bucket {state_bucket} --prefix {state_key}
   aws s3api get-object --bucket {state_bucket} \\
     --key {state_key} --version-id {version_id} terraform.tfstate.backup

6. Prevention: set lock timeout in CI pipeline:
   terraform apply -lock-timeout=5m""",
        ],
        "variables": {
            "lock_id": ["a1b2c3d4-e5f6-7890-abcd-ef1234567890", "f9e8d7c6-b5a4-3210-fedc-ba9876543210"],
            "lock_info": ["CI pipeline job #1234", "local apply by engineer", "scheduled automation"],
            "locker": ["github-actions", "jenkins-agent", "local terraform"],
            "locked_since": ["2024-01-15 10:23:45", "2024-01-15 09:45:00"],
            "workspace": ["production", "staging", "dev"],
            "state_path": ["eks-cluster/terraform.tfstate", "networking/terraform.tfstate"],
            "state_bucket": ["company-terraform-state", "infra-tf-state-prod"],
            "state_key": ["eks/terraform.tfstate", "vpc/terraform.tfstate"],
            "version_id": ["KD8s9fJKLm3nOpQrStUvWx", "Ab1Cd2Ef3Gh4Ij5Kl6Mn7O"],
        }
    },

    "argocd_sync_failed": {
        "instructions": [
            "ArgoCD application sync has failed. How do I diagnose and fix it?",
            "ArgoCD shows OutOfSync status and sync operation failed. Walk me through resolution.",
            "GitOps deployment via ArgoCD is stuck in failed sync state. How to resolve?",
        ],
        "contexts": [
            "ArgoCD app {app_name} in {namespace} shows status: OutOfSync, Health: {health}. Last sync: {last_sync}.",
            "argocd app sync {app_name} failed with error: {error_msg}.",
            "GitOps alert: ArgoCD application {app_name} sync failed. Revision: {git_rev}.",
        ],
        "resolutions": [
            """1. Check sync status and error details:
   argocd app get {app_name}
   argocd app sync {app_name} --dry-run   # see what would change

2. Check ArgoCD server logs for root cause:
   kubectl logs -n argocd -l app.kubernetes.io/name=argocd-application-controller --tail=50

3. Common failure causes:
   a) Resource conflict — another controller managing same resource:
      kubectl describe {resource_type} {resource_name} -n {namespace}
   b) RBAC — ArgoCD service account lacks permission:
      kubectl auth can-i create deployment --as=system:serviceaccount:argocd:argocd-application-controller
   c) Invalid manifest — check git diff for syntax errors:
      argocd app diff {app_name}

4. If manifest is valid but sync fails — hard refresh:
   argocd app get {app_name} --hard-refresh

5. If resource is stuck in terminating state:
   kubectl patch {resource_type} {resource_name} -n {namespace} \\
     -p '{{"metadata":{{"finalizers":[]}}}}' --type=merge

6. Force sync if needed (use with caution):
   argocd app sync {app_name} --force

7. Verify after fix:
   argocd app wait {app_name} --health --timeout 120""",
        ],
        "variables": {
            "app_name": ["payment-service", "ml-inference", "api-gateway", "monitoring-stack"],
            "namespace": ["production", "staging", "monitoring"],
            "health": ["Degraded", "Progressing", "Missing"],
            "last_sync": ["10m ago", "1h ago", "3h ago"],
            "error_msg": ["failed to sync", "resource conflict", "permission denied", "invalid manifest"],
            "git_rev": ["a1b2c3d", "f9e8d7c", "b5a4321"],
            "resource_type": ["deployment", "service", "configmap", "ingress"],
            "resource_name": ["payment-service", "ml-inference-svc", "app-config"],
        }
    },

    "fluent_bit_config_error": {
        "instructions": [
            "Fluent Bit DaemonSet pods are in CrashLoopBackOff. How do I debug the config?",
            "Log pipeline is broken — Fluent Bit pods crashing on startup. Walk me through diagnosis.",
            "Fluent Bit not shipping logs after ConfigMap change. How to identify and fix the config error?",
        ],
        "contexts": [
            "fluent-bit DaemonSet pods in {namespace} crashing. ConfigMap last updated {last_update}. Exit code: 1.",
            "kubectl logs fluent-bit-{pod_suffix} -n {namespace} shows config parse error.",
            "After ConfigMap update for {pipeline_name} pipeline, Fluent Bit pods CrashLoopBackOff.",
        ],
        "resolutions": [
            """1. Check pod logs for specific config error:
   kubectl logs -n {namespace} -l app=fluent-bit --previous | head -50

2. Common Fluent Bit config errors:
   a) Spaces-only blank lines between config blocks — Fluent Bit parser fails silently:
      # BAD: blank lines with whitespace characters
      [INPUT]
         Name tail
      {space}{space}
      [OUTPUT]
      # FIX: use truly empty lines (no spaces/tabs)

   b) Indentation with tabs instead of spaces
   c) Missing required fields (Path, Name, Match)
   d) Invalid regex in parser

3. Validate config before applying:
   # Dry-run config validation
   kubectl create configmap fluent-bit-config \\
     --from-file=fluent-bit.conf --dry-run=client -o yaml

4. Check ConfigMap content for hidden characters:
   kubectl get configmap fluent-bit-config -n {namespace} -o yaml | cat -A | grep '\\$'
   # $ at end of line = clean, ^I = tab (bad), ^M = Windows CRLF (bad)

5. Rollback ConfigMap to last working version:
   kubectl rollout history configmap/fluent-bit-config -n {namespace}
   # Or restore from git

6. After fix, restart DaemonSet:
   kubectl rollout restart daemonset/fluent-bit -n {namespace}
   kubectl rollout status daemonset/fluent-bit -n {namespace}

7. Verify logs flowing to destination:
   kubectl logs -n {namespace} -l app=fluent-bit --tail=20""",
        ],
        "variables": {
            "namespace": ["logging", "monitoring", "kube-system"],
            "last_update": ["30m ago", "2h ago", "yesterday"],
            "pod_suffix": ["xkp2m", "mn3j4", "z9x2p"],
            "pipeline_name": ["application-logs", "host-logs", "audit-logs"],
        }
    },

    "node_notready": {
        "instructions": [
            "A Kubernetes node is in NotReady state. How do I diagnose and recover it?",
            "kubectl get nodes shows a node as NotReady. Walk me through the investigation.",
            "Node pressure is causing NotReady status on EKS worker node. How to resolve?",
        ],
        "contexts": [
            "Node {node_name} is NotReady for {duration}. Condition: {condition}. Instance type: {instance_type}.",
            "kubectl describe node {node_name} shows {condition}: True. Pods evicted: {evicted_count}.",
            "EKS cluster alert: Node {node_name} NotReady. Region: {region}. Node group: {node_group}.",
        ],
        "resolutions": [
            """1. Check node conditions:
   kubectl describe node {node_name} | grep -A 20 Conditions

2. Check kubelet status on the node (SSH or SSM):
   # Via AWS SSM Session Manager (no SSH needed):
   aws ssm start-session --target {instance_id}
   sudo systemctl status kubelet
   sudo journalctl -u kubelet -n 100 --no-pager

3. Common NotReady causes:
   a) DiskPressure — disk full:
      df -h
      du -sh /var/log/containers/* | sort -rh | head
      # Flush old logs: journalctl --vacuum-time=1d
   b) MemoryPressure — node OOM:
      free -h
      # Eviction threshold hit — drain and replace node
   c) NetworkPlugin not ready:
      kubectl get pods -n kube-system | grep aws-node
      # Restart CNI: kubectl delete pod -n kube-system -l k8s-app=aws-node
   d) Kubelet cert expired:
      sudo ls -la /var/lib/kubelet/pki/

4. Cordon node to stop new scheduling:
   kubectl cordon {node_name}

5. Drain workloads safely:
   kubectl drain {node_name} --ignore-daemonsets --delete-emptydir-data

6. If unrecoverable — terminate and let ASG replace:
   aws ec2 terminate-instances --instance-ids {instance_id}

7. Verify new node joins:
   kubectl get nodes -w""",
        ],
        "variables": {
            "node_name": ["ip-10-0-1-45.ap-south-1.compute.internal", "ip-10-0-2-78.ap-south-1.compute.internal"],
            "duration": ["5m", "15m", "1h"],
            "condition": ["DiskPressure", "MemoryPressure", "NetworkUnavailable"],
            "instance_type": ["m5.xlarge", "m5.2xlarge", "c5.2xlarge"],
            "evicted_count": ["3", "7", "12"],
            "region": ["ap-south-1", "us-east-1"],
            "node_group": ["general-workers", "gpu-nodes", "spot-workers"],
            "instance_id": ["i-0a1b2c3d4e5f67890", "i-0f9e8d7c6b5a43210"],
        }
    },
}


class SyntheticIncidentGenerator:
    """
    Generates synthetic SRE incident training pairs from templates.
    Templates are derived from real production incident patterns.
    Variable substitution creates diverse surface forms from each template.
    Output: Alpaca-format JSONL ready for formatter stage.
    """

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = config["num_samples"]
        self.categories = config["categories"]

    def _fill_template(self, template: str, variables: dict[str, list]) -> str:
        """
        Fill a template string with randomly sampled variable values.
        Uses Python str.format_map — missing keys left as-is (safe fallback).
        """
        sampled = {k: random.choice(v) for k, v in variables.items()}
        try:
            return template.format_map(sampled)
        except (KeyError, IndexError):
            return template  # Return unfilled on error — validator will catch bad quality

    def generate_sample(self, category: str) -> Optional[dict]:
        """
        Generate one training sample for a given category.
        Randomly picks instruction, context, and resolution templates then fills variables.
        Returns Alpaca-format dict: instruction, input, output.
        """
        if category not in INCIDENT_TEMPLATES:
            logger.debug(f"No template for category: {category} — skipping")
            return None

        template = INCIDENT_TEMPLATES[category]
        variables = template.get("variables", {})

        instruction = random.choice(template["instructions"])
        context = self._fill_template(random.choice(template["contexts"]), variables)
        resolution = self._fill_template(random.choice(template["resolutions"]), variables)

        return {
            "instruction": instruction,
            "input": context,
            "output": resolution,
            "source": "synthetic",
            "category": category,
            "metadata": {
                "template_version": "v1",
                "generated": True,
            }
        }

    def generate_all(self) -> Path:
        """
        Generate num_samples total, distributed across configured categories.
        Writes directly to Alpaca JSONL — synthetic data is already formatted.
        Returns path to output file.
        """
        output_file = self.output_dir / "synthetic_incidents.jsonl"

        # Available categories = intersection of configured + templated
        available = [c for c in self.categories if c in INCIDENT_TEMPLATES]
        samples_per_category = max(1, self.num_samples // len(available))

        total = 0

        with jsonlines.open(output_file, mode="w") as writer:
            for category in track(available, description="Generating synthetic incidents..."):
                generated = 0
                attempts = 0
                max_attempts = samples_per_category * 3

                while generated < samples_per_category and attempts < max_attempts:
                    sample = self.generate_sample(category)
                    attempts += 1

                    if sample and len(sample["output"]) >= 50:
                        writer.write(sample)
                        generated += 1
                        total += 1

        logger.info(f"Synthetic generation complete: {total} samples → {output_file}")
        return output_file


# Allow Optional import at module level

