"""
P12c — Inference Cost Optimization
Techniques to maximize tokens per dollar during serving.

Cost levers for inference:
  1. Quantization (P8):       fp16→AWQ = 4x more requests per GPU
  2. Dynamic batching:        amortize GPU overhead across requests
  3. Request caching:         avoid recomputing identical prompts
  4. Scale-to-zero:           $0 cost when no traffic
  5. Spot instances:          70% cheaper than on-demand
  6. Model right-sizing:      8B vs 70B for appropriate tasks
  7. Continuous batching:     vLLM default, 10x vs naive batching
  8. KV cache optimization:   tune block_size and utilization
"""

import time
import hashlib
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Dynamic Batching Controller
# ---------------------------------------------------------------------------

@dataclass
class BatchConfig:
    """Configuration for dynamic batching."""
    max_batch_size:   int   = 32       # max requests per batch
    max_wait_ms:      float = 50.0     # max time to wait for batch to fill
    min_batch_size:   int   = 1        # don't wait if batch this size
    adaptive:         bool  = True     # adjust based on queue depth


class DynamicBatcher:
    """
    Batches incoming requests for more efficient GPU utilization.

    Problem: vLLM continuous batching handles this internally.
    This class is for pre-vLLM request aggregation at the gateway level.

    Use case: when AI Gateway receives burst of requests,
    can aggregate them before forwarding to vLLM for better throughput.

    Batching economics:
      Single request:  GPU processes 1 sequence → low utilization
      Batch of 32:     GPU processes 32 sequences → high utilization
      Throughput gain: ~10-20x for memory-bandwidth-bound decode
      Latency cost:    up to max_wait_ms added to TTFT

    Adaptive batching:
      Low queue depth:  smaller batches, lower latency
      High queue depth: larger batches, higher throughput
      Auto-adjusts to maintain target TTFT SLO
    """

    def __init__(self, config: BatchConfig = None):
        self.config  = config or BatchConfig()
        self._queue  = asyncio.Queue()
        self._stats  = {"batches": 0, "requests": 0, "avg_batch_size": 0}

    async def add_request(self, request: dict) -> asyncio.Future:
        """
        Add request to batching queue.
        Returns future that resolves when request is processed.
        """
        future = asyncio.get_event_loop().create_future()
        await self._queue.put((request, future))
        return future

    async def _collect_batch(self) -> list[tuple]:
        """
        Collect requests into a batch.
        Waits up to max_wait_ms for batch to fill.
        Returns immediately if min_batch_size reached.
        """
        batch      = []
        deadline   = time.perf_counter() + (self.config.max_wait_ms / 1000)

        # Get first request (blocking)
        try:
            item = await asyncio.wait_for(
                self._queue.get(),
                timeout=self.config.max_wait_ms / 1000
            )
            batch.append(item)
        except asyncio.TimeoutError:
            return batch

        # Collect more until batch full or timeout
        while len(batch) < self.config.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break

            # Check if adaptive batching suggests smaller batch
            if (self.config.adaptive and
                len(batch) >= self.config.min_batch_size and
                self._queue.empty()):
                break  # don't wait if queue empty and min size reached

            try:
                item = await asyncio.wait_for(
                    self._queue.get(), timeout=remaining
                )
                batch.append(item)
            except asyncio.TimeoutError:
                break

        return batch

    async def process_loop(self, process_fn):
        """
        Main processing loop.
        Continuously collects and processes batches.
        process_fn: async callable that takes list of requests
        """
        while True:
            batch = await self._collect_batch()
            if not batch:
                await asyncio.sleep(0.001)
                continue

            requests = [item[0] for item in batch]
            futures  = [item[1] for item in batch]

            self._stats["batches"]  += 1
            self._stats["requests"] += len(batch)
            self._stats["avg_batch_size"] = (
                self._stats["requests"] / self._stats["batches"]
            )

            logger.debug(f"Processing batch of {len(batch)} requests")

            try:
                results = await process_fn(requests)
                for future, result in zip(futures, results):
                    if not future.done():
                        future.set_result(result)
            except Exception as e:
                for future in futures:
                    if not future.done():
                        future.set_exception(e)


# ---------------------------------------------------------------------------
# KV Cache Optimizer
# ---------------------------------------------------------------------------

class KVCacheOptimizer:
    """
    Optimizes vLLM KV cache configuration for specific workloads.

    KV cache sizing:
      Each token in KV cache consumes:
        2 (K, V) × num_layers × num_heads × head_dim × dtype_bytes
        = 2 × 32 × 32 × 128 × 2 (bf16) = 524,288 bytes per token
        ≈ 0.5 MB per token for Llama 3 8B

      For A10G (24GB), with 4GB model weights:
        Available for KV: ~18GB
        Max tokens: 18GB / 0.5MB = ~36,000 tokens
        Max concurrent sequences at 1024 tokens each: ~35

      PagedAttention stores in pages (block_size tokens per page):
        block_size=16: 36,000/16 = 2,250 pages
        Fragmentation: ~5% (PagedAttention eliminates most waste)

    Optimization:
      Larger block_size → less fragmentation, larger minimum allocation
      Smaller block_size → more granular, wastes more memory at end
      Sweet spot: 16 for A10G
    """

    @staticmethod
    def calculate_kv_cache_capacity(
        gpu_memory_gb: float,
        model_size_gb: float,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype_bytes: int = 2,   # bf16
        overhead_gb: float = 1.0,
    ) -> dict:
        """
        Calculate KV cache capacity for given GPU and model.

        Returns:
          max_tokens: maximum tokens that fit in KV cache
          max_seqs_at_1k: max concurrent sequences at 1024 tokens
          max_seqs_at_2k: max concurrent sequences at 2048 tokens
          recommended_block_size: optimal block size
          recommended_max_num_seqs: optimal max concurrent sequences
        """
        available_gb = gpu_memory_gb - model_size_gb - overhead_gb
        available_bytes = available_gb * 1024**3

        # Bytes per token in KV cache
        bytes_per_token = (
            2 *          # K and V
            num_layers * # one K, one V per layer
            num_heads *  # multi-head attention
            head_dim *   # head dimension
            dtype_bytes  # bytes per value (2 for bf16)
        )

        max_tokens = int(available_bytes / bytes_per_token)

        return {
            "available_gb":          round(available_gb, 2),
            "bytes_per_token":       bytes_per_token,
            "max_tokens":            max_tokens,
            "max_seqs_at_512":       max_tokens // 512,
            "max_seqs_at_1024":      max_tokens // 1024,
            "max_seqs_at_2048":      max_tokens // 2048,
            "recommended_block_size": 16,
            "recommended_max_num_seqs": max_tokens // 1024,
        }

    @staticmethod
    def llama3_8b_on_a10g() -> dict:
        """Pre-computed KV cache capacity for Llama 3 8B on A10G."""
        # Llama 3 8B: 32 layers, 32 heads, 128 head_dim, GQA with 8 KV heads
        return KVCacheOptimizer.calculate_kv_cache_capacity(
            gpu_memory_gb=24.0,
            model_size_gb=4.5,     # AWQ 4-bit
            num_layers=32,
            num_heads=8,            # GQA: 8 KV heads (not 32)
            head_dim=128,
            dtype_bytes=2,
            overhead_gb=1.0,
        )


# ---------------------------------------------------------------------------
# Scale-to-Zero Controller
# ---------------------------------------------------------------------------

class ScaleToZeroController:
    """
    Controls GPU node scaling based on traffic patterns.
    Scale to 0 when idle → $0 GPU cost.
    Scale up when requests arrive → 3-5 min cold start.

    Cold start mitigation:
      Option A: Keep 1 CPU pod with model metadata cached
                → Users get "warming up" message, ~3min wait
      Option B: Pre-warm on schedule (business hours only)
                → Always-on 9am-6pm, scale-to-zero nights/weekends
      Option C: Speculative scale-up based on traffic patterns
                → Scale up before traffic spike (predictive)

    For SRE use case: Option B is best
      SRE queries happen during incidents → business hours mostly
      Nights/weekends: scale to zero, cold start acceptable
    """

    BUSINESS_HOURS_START = 9   # 9 AM
    BUSINESS_HOURS_END   = 18  # 6 PM

    def __init__(
        self,
        cluster_name:  str,
        nodegroup_name: str,
        region:        str = "us-east-1",
    ):
        self.cluster_name   = cluster_name
        self.nodegroup_name = nodegroup_name
        self.eks            = boto3.client("eks", region_name=region)

    def should_scale_up(self, hour_of_day: int, queue_depth: int) -> bool:
        """
        Determine if GPU nodes should be scaled up.
        Business hours OR non-zero queue depth triggers scale-up.
        """
        is_business_hours = (
            self.BUSINESS_HOURS_START <= hour_of_day < self.BUSINESS_HOURS_END
        )
        return is_business_hours or queue_depth > 0

    def scale_gpu_nodes(self, desired: int):
        """Scale GPU node group to desired size."""
        try:
            self.eks.update_nodegroup_config(
                clusterName=self.cluster_name,
                nodegroupName=self.nodegroup_name,
                scalingConfig={
                    "minSize":     0,
                    "maxSize":     4,
                    "desiredSize": desired,
                }
            )
            logger.info(
                f"GPU nodes scaled to {desired}: "
                f"{self.cluster_name}/{self.nodegroup_name}"
            )
        except Exception as e:
            logger.error(f"Scale failed: {e}")

    def run_schedule(self):
        """
        Run scale-up/down on business hours schedule.
        Called by a CronJob every hour.
        """
        import datetime
        hour = datetime.datetime.utcnow().hour
        # Convert UTC to IST (UTC+5:30) — adjust for your timezone
        ist_hour = (hour + 5) % 24

        if self.should_scale_up(ist_hour, queue_depth=0):
            self.scale_gpu_nodes(1)
            logger.info(f"Business hours ({ist_hour}:00 IST) — scaling up GPU")
        else:
            self.scale_gpu_nodes(0)
            logger.info(f"Off hours ({ist_hour}:00 IST) — scaling down GPU")


# Add boto3 import
import boto3
