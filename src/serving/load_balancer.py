"""
P10d — Load Balancing for Inference
Multiple vLLM replicas behind a load balancer.
Autoscaling based on GPU utilization and queue depth.

Load balancing strategies:
  Round-robin:         simple, ignores server state
  Least-connections:   route to replica with fewest active requests
  KV-cache-aware:      route to replica most likely to have KV cache hit
                       (for multi-turn conversations — route to same replica)

For SRE inference (stateless requests):
  Least-connections is optimal — balances GPU utilization
  KV-cache-aware unnecessary (single-turn Q&A)

Autoscaling triggers:
  Scale up:   GPU utilization > 80% OR queue depth > 10 requests
  Scale down: GPU utilization < 20% AND queue depth = 0 for 5 minutes
  Min replicas: 1 (always available)
  Max replicas: based on GPU quota

HPA configuration:
  Custom metrics from Prometheus → KEDA → HPA
  Metric: vllm_concurrent_requests / vllm_queue_depth
"""

import asyncio
import time
import httpx
from dataclasses import dataclass, field
from typing import Optional
from src.utils.logger import logger


@dataclass
class ReplicaStats:
    """Runtime statistics for a single vLLM replica."""
    endpoint:          str
    concurrent_reqs:   int   = 0
    queue_depth:       int   = 0
    gpu_utilization:   float = 0.0
    kv_cache_usage:    float = 0.0
    is_healthy:        bool  = True
    last_health_check: float = field(default_factory=time.time)
    total_requests:    int   = 0
    total_errors:      int   = 0


class LeastConnectionsBalancer:
    """
    Routes requests to vLLM replica with fewest concurrent requests.
    Updates replica stats via periodic Prometheus scraping.
    """

    def __init__(self, endpoints: list[str], health_check_interval: float = 10.0):
        self.replicas  = {ep: ReplicaStats(endpoint=ep) for ep in endpoints}
        self.interval  = health_check_interval
        self._lock     = asyncio.Lock()

        logger.info(f"LoadBalancer initialized with {len(endpoints)} replicas")

    async def start_health_checks(self):
        """Background task: periodically update replica stats."""
        while True:
            await self._update_all_stats()
            await asyncio.sleep(self.interval)

    async def _update_all_stats(self):
        """Update stats for all replicas in parallel."""
        tasks = [self._update_stats(ep) for ep in self.replicas]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _update_stats(self, endpoint: str):
        """
        Fetch metrics from a single replica's /metrics endpoint.
        Parses Prometheus text format for key metrics.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Health check
                health = await client.get(f"{endpoint}/health")
                is_healthy = health.status_code == 200

                # Metrics
                metrics_resp = await client.get(f"{endpoint}/metrics")
                metrics_text = metrics_resp.text

            stats = self.replicas[endpoint]
            stats.is_healthy        = is_healthy
            stats.last_health_check = time.time()

            # Parse Prometheus metrics
            for line in metrics_text.split("\n"):
                if line.startswith("#"):
                    continue
                if "vllm_concurrent_requests" in line and "{" in line:
                    try:
                        stats.concurrent_reqs = int(float(line.split()[-1]))
                    except (ValueError, IndexError):
                        pass
                elif "vllm_kv_cache_usage_ratio" in line and "{" in line:
                    try:
                        stats.kv_cache_usage = float(line.split()[-1])
                    except (ValueError, IndexError):
                        pass

        except Exception as e:
            async with self._lock:
                self.replicas[endpoint].is_healthy = False
            logger.warning(f"Health check failed for {endpoint}: {e}")

    async def get_best_replica(self) -> Optional[str]:
        """
        Select replica with fewest concurrent requests.
        Filters out unhealthy replicas.
        Raises if no healthy replicas available.
        """
        async with self._lock:
            healthy = {
                ep: stats
                for ep, stats in self.replicas.items()
                if stats.is_healthy
            }

        if not healthy:
            raise RuntimeError("No healthy replicas available")

        # Sort by concurrent requests (least first)
        best = min(healthy.items(), key=lambda x: x[1].concurrent_reqs)
        return best[0]

    async def forward_request(
        self,
        payload: dict,
        endpoint_override: str = None,
    ) -> dict:
        """Forward request to best replica, track stats."""
        endpoint = endpoint_override or await self.get_best_replica()

        async with self._lock:
            self.replicas[endpoint].concurrent_reqs += 1
            self.replicas[endpoint].total_requests  += 1

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{endpoint}/v1/chat/completions",
                    json=payload,
                )
                return response.json()

        except Exception as e:
            async with self._lock:
                self.replicas[endpoint].total_errors += 1
            raise

        finally:
            async with self._lock:
                self.replicas[endpoint].concurrent_reqs = max(
                    0, self.replicas[endpoint].concurrent_reqs - 1
                )

    def get_stats_summary(self) -> dict:
        """Return summary of all replica stats."""
        return {
            ep: {
                "healthy":        stats.is_healthy,
                "concurrent":     stats.concurrent_reqs,
                "kv_cache":       stats.kv_cache_usage,
                "total_requests": stats.total_requests,
                "error_rate":     (
                    stats.total_errors / max(1, stats.total_requests)
                ),
            }
            for ep, stats in self.replicas.items()
        }
