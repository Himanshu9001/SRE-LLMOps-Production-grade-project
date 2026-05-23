"""
P11c — AI-Specific Metrics Collection
Custom metrics beyond what vLLM exposes natively.

Tracks:
  - Model accuracy over time (online evaluation)
  - Response quality scores
  - Token distribution analysis
  - Request pattern analysis
  - Cost per request
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    CollectorRegistry, push_to_gateway,
)
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# AI-Specific Prometheus Metrics
# ---------------------------------------------------------------------------

# Online evaluation metrics
MODEL_ACCURACY = Gauge(
    "sre_model_sre_accuracy_current",
    "Current SRE benchmark accuracy (rolling 1h window)",
    ["model_name", "category"]
)

MODEL_ACCURACY_BASELINE = Gauge(
    "sre_model_sre_accuracy_baseline",
    "Baseline SRE benchmark accuracy at deployment",
    ["model_name", "category"]
)

# Request quality
RESPONSE_KEYWORD_HIT_RATE = Gauge(
    "sre_response_keyword_hit_rate",
    "Rate of responses containing expected SRE keywords",
    ["model_name"]
)

# Token economics
COST_PER_REQUEST = Histogram(
    "sre_cost_per_request_usd",
    "Estimated cost per inference request in USD",
    ["model_name"],
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
)

TOKENS_PER_DOLLAR = Gauge(
    "sre_tokens_per_dollar",
    "Tokens generated per dollar spent",
    ["model_name"]
)

# Request patterns
REQUEST_CATEGORY = Counter(
    "sre_requests_by_category_total",
    "Requests grouped by detected SRE category",
    ["model_name", "category"]
)

PROMPT_LENGTH_DISTRIBUTION = Histogram(
    "sre_prompt_length_tokens",
    "Distribution of prompt lengths in tokens",
    ["model_name"],
    buckets=[50, 100, 200, 500, 1000, 2000, 4000]
)

RESPONSE_LENGTH_DISTRIBUTION = Histogram(
    "sre_response_length_tokens",
    "Distribution of response lengths in tokens",
    ["model_name"],
    buckets=[10, 50, 100, 200, 500, 1000]
)

# Cache metrics
PROMPT_CACHE_HIT_RATE = Gauge(
    "sre_prompt_cache_hit_rate",
    "Prompt cache hit rate (0-1)",
    ["model_name"]
)

PROMPT_CACHE_SIZE = Gauge(
    "sre_prompt_cache_size_entries",
    "Number of entries in prompt cache",
    ["model_name"]
)

# Model drift
RESPONSE_PERPLEXITY = Histogram(
    "sre_response_perplexity",
    "Response perplexity (lower = more confident)",
    ["model_name"],
    buckets=[1, 2, 5, 10, 20, 50, 100]
)


# ---------------------------------------------------------------------------
# SRE Category Classifier
# ---------------------------------------------------------------------------

SRE_CATEGORIES = {
    "kubernetes": [
        "kubectl", "pod", "deployment", "namespace", "kubelet",
        "crashloopbackoff", "oomkilled", "daemonset", "node",
    ],
    "prometheus": [
        "promql", "alert", "firing", "scrape", "metric",
        "rate(", "histogram_quantile", "alertmanager",
    ],
    "terraform": [
        "terraform", "tfstate", "plan", "apply", "module",
        "resource", "provider", "backend",
    ],
    "argocd": [
        "argocd", "gitops", "sync", "application", "helm",
        "kustomize", "revision",
    ],
    "observability": [
        "grafana", "loki", "jaeger", "trace", "span",
        "dashboard", "alert", "slo", "sli",
    ],
}


def classify_sre_request(prompt: str) -> str:
    """
    Classify an SRE request into a category based on keywords.
    Used for request pattern analysis and per-category metrics.
    """
    prompt_lower = prompt.lower()
    scores = {}

    for category, keywords in SRE_CATEGORIES.items():
        score = sum(1 for kw in keywords if kw in prompt_lower)
        scores[category] = score

    best_category = max(scores, key=scores.get)
    return best_category if scores[best_category] > 0 else "general"


# ---------------------------------------------------------------------------
# Cost Calculator
# ---------------------------------------------------------------------------

# GPU cost per token (approximate)
# G5.2xlarge spot: $0.34/hr
# Throughput: ~500 tok/sec for 8B AWQ
# Cost per token: $0.34 / 3600 / 500 = $0.000000189
GPU_COST_PER_TOKEN = {
    "g5.2xlarge_spot":     0.000000189,
    "g5.2xlarge_ondemand": 0.000000672,
    "g4dn.2xlarge_spot":   0.000000139,
}

# OpenAI comparison (for fallback cost tracking)
OPENAI_COST_PER_TOKEN = {
    "gpt-3.5-turbo": 0.000002,
    "gpt-4":         0.00003,
}


def calculate_request_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model_type: str = "vllm",
    instance_type: str = "g5.2xlarge_spot",
) -> float:
    """
    Calculate estimated cost for a single inference request.

    For vLLM: cost = tokens * gpu_cost_per_token
    For OpenAI: cost = tokens * openai_price_per_token

    Used to:
    - Track per-team cost allocation
    - Compare vLLM vs API costs
    - Identify expensive request patterns
    """
    total_tokens = prompt_tokens + completion_tokens

    if model_type == "vllm":
        cost = total_tokens * GPU_COST_PER_TOKEN.get(instance_type, 0.0000002)
    elif model_type in OPENAI_COST_PER_TOKEN:
        cost = total_tokens * OPENAI_COST_PER_TOKEN[model_type]
    else:
        cost = 0.0

    return cost


# ---------------------------------------------------------------------------
# Online Evaluator (background thread)
# ---------------------------------------------------------------------------

@dataclass
class RequestRecord:
    """Record of a single inference request for online evaluation."""
    prompt:         str
    response:       str
    prompt_tokens:  int
    response_tokens: int
    latency_ms:     float
    model_name:     str
    timestamp:      float = field(default_factory=time.time)
    category:       str   = "general"
    cost_usd:       float = 0.0


class OnlineEvaluator:
    """
    Continuously evaluates model quality on recent requests.
    Runs in background thread — doesn't block inference.

    Evaluation:
    - Keyword hit rate: % of SRE responses with expected keywords
    - Response length distribution: detects verbosity changes
    - Category distribution: detects prompt pattern shifts
    - Cost tracking: per-team spending analysis

    Window: last 1000 requests (rolling)
    Update frequency: every 60 seconds
    """

    def __init__(
        self,
        model_name: str,
        window_size: int = 1000,
        eval_interval: float = 60.0,
    ):
        self.model_name    = model_name
        self.window        = deque(maxlen=window_size)
        self.eval_interval = eval_interval
        self._thread       = None
        self._running      = False

        # SRE evaluation keywords per category
        self.eval_keywords = {
            "kubernetes":    ["kubectl", "pod", "namespace", "deploy"],
            "prometheus":    ["rate(", "metric", "query", "alert"],
            "terraform":     ["terraform", "resource", "state"],
            "argocd":        ["argocd", "sync", "app"],
            "observability": ["grafana", "dashboard", "alert", "trace"],
        }

    def record_request(self, record: RequestRecord):
        """Add request to evaluation window."""
        record.category = classify_sre_request(record.prompt)
        self.window.append(record)

        # Update per-request metrics immediately
        PROMPT_LENGTH_DISTRIBUTION.labels(
            model_name=self.model_name
        ).observe(record.prompt_tokens)

        RESPONSE_LENGTH_DISTRIBUTION.labels(
            model_name=self.model_name
        ).observe(record.response_tokens)

        COST_PER_REQUEST.labels(
            model_name=self.model_name
        ).observe(record.cost_usd)

        REQUEST_CATEGORY.labels(
            model_name=self.model_name,
            category=record.category,
        ).inc()

    def _evaluate_window(self):
        """
        Compute rolling metrics over current window.
        Called every eval_interval seconds.
        """
        if not self.window:
            return

        records = list(self.window)

        # Keyword hit rate
        keyword_hits = 0
        for record in records:
            cat_keywords = self.eval_keywords.get(record.category, [])
            if any(kw in record.response.lower() for kw in cat_keywords):
                keyword_hits += 1

        hit_rate = keyword_hits / len(records)
        RESPONSE_KEYWORD_HIT_RATE.labels(
            model_name=self.model_name
        ).set(hit_rate)

        # Category-level accuracy
        by_category = {}
        for record in records:
            cat = record.category
            if cat not in by_category:
                by_category[cat] = {"total": 0, "hits": 0}
            by_category[cat]["total"] += 1
            cat_keywords = self.eval_keywords.get(cat, [])
            if any(kw in record.response.lower() for kw in cat_keywords):
                by_category[cat]["hits"] += 1

        for cat, counts in by_category.items():
            accuracy = counts["hits"] / max(1, counts["total"])
            MODEL_ACCURACY.labels(
                model_name=self.model_name,
                category=cat,
            ).set(accuracy)

        # Cost metrics
        total_cost   = sum(r.cost_usd for r in records)
        total_tokens = sum(r.response_tokens for r in records)
        if total_cost > 0:
            TOKENS_PER_DOLLAR.labels(
                model_name=self.model_name
            ).set(total_tokens / total_cost)

        logger.debug(
            f"Online eval: {len(records)} requests, "
            f"keyword_hit_rate={hit_rate:.3f}"
        )

    def start(self):
        """Start background evaluation thread."""
        self._running = True
        self._thread  = threading.Thread(
            target=self._run, daemon=True
        )
        self._thread.start()
        logger.info(f"OnlineEvaluator started for {self.model_name}")

    def stop(self):
        """Stop background thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        """Background evaluation loop."""
        while self._running:
            try:
                self._evaluate_window()
            except Exception as e:
                logger.error(f"Online evaluation error: {e}")
            time.sleep(self.eval_interval)
