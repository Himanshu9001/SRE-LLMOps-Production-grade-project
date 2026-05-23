"""
P12a — Cost Analyzer
Tracks and attributes costs across the full ML lifecycle.

Cost breakdown for SRE LLMOps platform:
┌─────────────────────────────────────────────────────────┐
│ INFRASTRUCTURE (always-on)                              │
│   EKS control plane:     $0.10/hr  = $72/month         │
│   3x NAT Gateways:       $0.135/hr = $97/month         │
│   RDS t3.micro:          $0.017/hr = $12/month         │
│   VPC endpoints (3):     $0.03/hr  = $21/month         │
│   S3 storage (50GB):     $1.15/month                   │
│   Total infrastructure:            ~$203/month          │
│                                                         │
│ COMPUTE (on-demand)                                     │
│   Training (GPU):        $1.21/hr × hours               │
│   Inference (GPU):       $1.21/hr × hours               │
│   CPU nodes (2×m5.xlarge): $0.192/hr when active       │
│                                                         │
│ OPTIMIZATION LEVERS                                     │
│   Spot training:         70% savings on GPU             │
│   Scale-to-zero:         0 GPU cost when idle           │
│   Model quantization:    Same GPU serves 4x more        │
│   Prompt caching:        Avoid recompute on repeat req  │
│   Request batching:      Amortize GPU overhead          │
└─────────────────────────────────────────────────────────┘
"""

import json
import time
import boto3
import datetime
from dataclasses import dataclass, field
from typing import Optional
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# AWS Cost Explorer integration
# ---------------------------------------------------------------------------

@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a time period."""
    period_start:    str
    period_end:      str
    total_usd:       float
    by_service:      dict = field(default_factory=dict)
    by_resource:     dict = field(default_factory=dict)
    by_tag:          dict = field(default_factory=dict)


class AWSCostAnalyzer:
    """
    Fetches and analyzes AWS costs using Cost Explorer API.
    Tags resources by project/component for granular cost attribution.
    """

    def __init__(self, region: str = "us-east-1"):
        self.ce = boto3.client("ce", region_name="us-east-1")
        # Cost Explorer is always us-east-1 regardless of resource region

    def get_cost_by_service(
        self,
        start_date: str,
        end_date: str,
        granularity: str = "DAILY",
    ) -> dict:
        """
        Get costs grouped by AWS service.
        start_date/end_date: YYYY-MM-DD format.
        """
        response = self.ce.get_cost_and_usage(
            TimePeriod={"Start": start_date, "End": end_date},
            Granularity=granularity,
            Metrics=["UnblendedCost"],
            GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
            Filter={
                "Tags": {
                    "Key":    "Project",
                    "Values": ["sre-llmops"],
                }
            }
        )

        costs = {}
        for result in response["ResultsByTime"]:
            for group in result["Groups"]:
                service = group["Keys"][0]
                amount  = float(group["Metrics"]["UnblendedCost"]["Amount"])
                costs[service] = costs.get(service, 0) + amount

        return costs

    def get_gpu_cost(self, start_date: str, end_date: str) -> float:
        """Get total EC2 GPU instance cost for period."""
        response = self.ce.get_cost_and_usage(
            TimePeriod={"Start": start_date, "End": end_date},
            Granularity="DAILY",
            Metrics=["UnblendedCost"],
            Filter={
                "And": [
                    {"Tags": {"Key": "Project", "Values": ["sre-llmops"]}},
                    {"Dimensions": {
                        "Key":    "INSTANCE_TYPE",
                        "Values": ["g5.2xlarge", "g5.4xlarge", "g4dn.2xlarge"],
                    }},
                ]
            }
        )

        total = sum(
            float(r["Total"]["UnblendedCost"]["Amount"])
            for r in response["ResultsByTime"]
        )
        return total

    def get_monthly_forecast(self) -> dict:
        """Forecast current month's total cost."""
        today     = datetime.date.today()
        month_end = datetime.date(today.year, today.month + 1, 1) \
                    if today.month < 12 \
                    else datetime.date(today.year + 1, 1, 1)

        try:
            response = self.ce.get_cost_forecast(
                TimePeriod={
                    "Start": today.isoformat(),
                    "End":   month_end.isoformat(),
                },
                Metric="UNBLENDED_COST",
                Granularity="MONTHLY",
            )

            return {
                "forecast_usd":    float(response["Total"]["Amount"]),
                "lower_bound_usd": float(response["ForecastResultsByTime"][0]
                                         ["PredictionIntervalLowerBound"]),
                "upper_bound_usd": float(response["ForecastResultsByTime"][0]
                                         ["PredictionIntervalUpperBound"]),
            }
        except Exception as e:
            logger.error(f"Cost forecast failed: {e}")
            return {}

    def get_savings_opportunities(self) -> list[dict]:
        """
        Identify cost savings opportunities using AWS Compute Optimizer
        and Cost Explorer recommendations.
        """
        opportunities = []

        # Check for idle GPU nodes
        try:
            ec2 = boto3.client("ec2", region_name="us-east-1")
            instances = ec2.describe_instances(
                Filters=[
                    {"Name": "tag:Project",   "Values": ["sre-llmops"]},
                    {"Name": "instance-state-name", "Values": ["running"]},
                    {"Name": "instance-type", "Values": ["g5.2xlarge"]},
                ]
            )

            for reservation in instances["Reservations"]:
                for instance in reservation["Instances"]:
                    # GPU node running but no training jobs?
                    # This is wasted money
                    opportunities.append({
                        "type":        "idle_gpu",
                        "instance_id": instance["InstanceId"],
                        "hourly_cost": 1.21,
                        "recommendation": "Scale GPU node group to 0 when not training",
                        "annual_savings": 1.21 * 24 * 365,
                    })

        except Exception as e:
            logger.warning(f"Could not check EC2 instances: {e}")

        # Check Savings Plans coverage
        opportunities.append({
            "type":        "savings_plan",
            "description": "1-year Compute Savings Plan for EKS control plane",
            "current_cost": 72,     # $72/month for EKS
            "savings_pct":  30,
            "savings_usd":  21.6,   # $21.6/month saved
            "recommendation": "Purchase 1-year Compute Savings Plan for predictable EKS cost",
        })

        return opportunities


# ---------------------------------------------------------------------------
# Per-Request Cost Tracker
# ---------------------------------------------------------------------------

class RequestCostTracker:
    """
    Tracks cost for every inference request.
    Enables per-team cost attribution and chargeback.

    Cost model:
      GPU cost = (prompt_tokens + completion_tokens) × gpu_cost_per_token
      gpu_cost_per_token = instance_hourly_rate / (3600 × tokens_per_sec)

    For G5.2xlarge spot ($0.34/hr), Llama 3 8B AWQ at 500 tok/s:
      cost_per_token = 0.34 / (3600 × 500) = $0.000000189
      cost per request (100 tokens): $0.0000189 ≈ $0.00002

    Comparison:
      vLLM:       $0.00002  per request
      GPT-3.5:    $0.0002   per request  (10x more expensive)
      GPT-4:      $0.003    per request  (150x more expensive)
    """

    # Instance cost models
    GPU_COSTS = {
        "g5.2xlarge_spot":     {"hourly": 0.34,  "tokens_per_sec": 500},
        "g5.2xlarge_ondemand": {"hourly": 1.21,  "tokens_per_sec": 500},
        "g4dn.2xlarge_spot":   {"hourly": 0.19,  "tokens_per_sec": 300},
        "g4dn.2xlarge_ondemand": {"hourly": 0.75, "tokens_per_sec": 300},
    }

    OPENAI_COSTS = {
        "gpt-3.5-turbo":       {"input": 0.0000005,  "output": 0.0000015},
        "gpt-4":               {"input": 0.00001,     "output": 0.00003},
        "gpt-4-turbo":         {"input": 0.00001,     "output": 0.00003},
    }

    def __init__(
        self,
        instance_type: str = "g5.2xlarge_spot",
        s3_bucket: str = "sre-llmops-artifacts",
    ):
        self.instance_type = instance_type
        self.s3_bucket     = s3_bucket
        self._records      = []

    def _cost_per_token(self) -> float:
        """Calculate cost per token for current instance."""
        config = self.GPU_COSTS.get(self.instance_type, {
            "hourly": 1.21, "tokens_per_sec": 500
        })
        return config["hourly"] / (3600 * config["tokens_per_sec"])

    def record_request(
        self,
        team_id:         str,
        user_id:         str,
        prompt_tokens:   int,
        completion_tokens: int,
        model:           str = "vllm",
        latency_ms:      float = 0.0,
    ) -> float:
        """
        Record a request and return its cost in USD.
        Logs to in-memory buffer, flushes to S3 hourly.
        """
        if model == "vllm":
            cost = (prompt_tokens + completion_tokens) * self._cost_per_token()
        elif model in self.OPENAI_COSTS:
            costs = self.OPENAI_COSTS[model]
            cost  = (prompt_tokens   * costs["input"] +
                     completion_tokens * costs["output"])
        else:
            cost = 0.0

        record = {
            "timestamp":        time.time(),
            "team_id":          team_id,
            "user_id":          user_id,
            "model":            model,
            "instance_type":    self.instance_type,
            "prompt_tokens":    prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":     prompt_tokens + completion_tokens,
            "cost_usd":         cost,
            "latency_ms":       latency_ms,
        }

        self._records.append(record)

        # Flush every 1000 records
        if len(self._records) >= 1000:
            self._flush_to_s3()

        return cost

    def get_team_costs(self, team_id: str, hours: int = 24) -> dict:
        """Get cost summary for a team over the last N hours."""
        cutoff   = time.time() - (hours * 3600)
        records  = [
            r for r in self._records
            if r["team_id"] == team_id and r["timestamp"] > cutoff
        ]

        if not records:
            return {"team_id": team_id, "period_hours": hours,
                    "total_usd": 0, "requests": 0, "tokens": 0}

        return {
            "team_id":        team_id,
            "period_hours":   hours,
            "total_usd":      sum(r["cost_usd"] for r in records),
            "requests":       len(records),
            "tokens":         sum(r["total_tokens"] for r in records),
            "avg_cost_usd":   sum(r["cost_usd"] for r in records) / len(records),
            "avg_tokens":     sum(r["total_tokens"] for r in records) / len(records),
            "vs_openai_gpt35": sum(r["cost_usd"] for r in records) /
                               max(0.001, sum(
                                   r["total_tokens"] * 0.000002
                                   for r in records
                               )),  # ratio: our cost / GPT-3.5 cost
        }

    def _flush_to_s3(self):
        """Flush cost records to S3 for long-term storage."""
        if not self._records:
            return

        s3  = boto3.client("s3", region_name="us-east-1")
        key = f"cost-records/{datetime.date.today()}/{int(time.time())}.jsonl"

        body = "\n".join(json.dumps(r) for r in self._records)
        s3.put_object(
            Bucket=self.s3_bucket,
            Key=key,
            Body=body.encode(),
        )

        logger.info(f"Flushed {len(self._records)} cost records to s3://{self.s3_bucket}/{key}")
        self._records = []
