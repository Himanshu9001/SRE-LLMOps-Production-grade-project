"""
P10c — Prefill-Decode Disaggregation
One of the most advanced vLLM optimization topics.

Understanding the problem:
  LLM inference has two distinct phases:
  
  PREFILL: Process the entire input prompt in parallel
    - Compute: HIGH (all prompt tokens processed simultaneously)  
    - Memory bandwidth: MEDIUM
    - Duration: proportional to prompt length
    - GPU utilization: ~100% (compute-bound)
    
  DECODE: Generate one output token at a time
    - Compute: LOW (only 1 token generated per step)
    - Memory bandwidth: HIGH (read entire KV cache per step)
    - Duration: proportional to output length
    - GPU utilization: ~10-30% (memory-bandwidth-bound)

The conflict:
  When prefill and decode run on the SAME GPU:
  - Long prefill (e.g. 2000 token prompt) blocks all decode steps
  - Decode requests experience "prefill tax" — high latency spikes
  - Inter-Token Latency (ITL) becomes unpredictable
  - Users see stuttering in streaming responses

Disaggregation solution:
  PREFILL GPU: dedicated to processing new requests
    - Optimized for compute throughput
    - Large batch sizes for prefill
    - Can be paused mid-batch for decode (chunked prefill)
    
  DECODE GPU: dedicated to generating tokens
    - Optimized for memory bandwidth
    - Consistent ITL regardless of incoming requests
    - KV cache transferred from prefill GPU after processing

Implementation in vLLM:
  vLLM supports this via --enable-chunked-prefill
  Full disaggregation requires separate vLLM instances
  KV cache transfer via RDMA/NVLink between GPU pods

This module implements:
  1. Chunked prefill (single-GPU approximation)
  2. Two-pool architecture design (separate prefill/decode pods)
  3. Request routing logic
  4. KV cache transfer coordination
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional
from src.utils.logger import logger


@dataclass
class InferenceRequest:
    """Single inference request with routing metadata."""
    request_id:    str
    prompt:        str
    max_tokens:    int
    temperature:   float
    prompt_tokens: int   # estimated prompt length for routing
    created_at:    float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    @property
    def is_long_prefill(self, threshold: int = 512) -> bool:
        """Long prefill requests benefit most from disaggregation."""
        return self.prompt_tokens > threshold


@dataclass
class PoolConfig:
    """Configuration for a vLLM instance pool."""
    pool_type:     str     # "prefill" or "decode"
    endpoints:     list[str]
    max_queue:     int
    timeout_secs:  float


class PrefillDecodeRouter:
    """
    Routes requests between prefill and decode pools.

    Architecture:
      Client → Router → Prefill Pool (new requests)
                     ↓
                     KV cache transfer
                     ↓
              Decode Pool (token generation)

    For single-GPU approximation (chunked prefill):
      vLLM handles internally with --enable-chunked-prefill
      max_num_prefill_seqs=1 limits concurrent prefills
      This prevents long prefills from blocking decode

    For full disaggregation (two separate GPU pools):
      This router distributes based on request phase
    """

    def __init__(
        self,
        prefill_endpoints: list[str],
        decode_endpoints: list[str],
        long_prefill_threshold: int = 512,
    ):
        self.prefill_endpoints = prefill_endpoints
        self.decode_endpoints  = decode_endpoints
        self.threshold         = long_prefill_threshold

        # Round-robin counters
        self._prefill_idx = 0
        self._decode_idx  = 0

        logger.info(
            f"PrefillDecodeRouter initialized: "
            f"{len(prefill_endpoints)} prefill, "
            f"{len(decode_endpoints)} decode endpoints"
        )

    def _next_prefill_endpoint(self) -> str:
        """Round-robin selection of prefill endpoint."""
        endpoint = self.prefill_endpoints[self._prefill_idx % len(self.prefill_endpoints)]
        self._prefill_idx += 1
        return endpoint

    def _next_decode_endpoint(self) -> str:
        """Round-robin selection of decode endpoint."""
        endpoint = self.decode_endpoints[self._decode_idx % len(self.decode_endpoints)]
        self._decode_idx += 1
        return endpoint

    def route(self, request: InferenceRequest) -> str:
        """
        Route request to appropriate pool.

        Routing logic:
        - Short prompts (< threshold tokens): send to decode pool
          These have minimal prefill overhead — no need for dedicated prefill
        - Long prompts (>= threshold tokens): send to prefill pool
          Heavy prefill work, then KV cache transferred to decode pool

        In practice with vLLM's chunked prefill:
          All requests go to same pool but prefill is chunked
          This approximates disaggregation without separate GPU pools
        """
        if request.prompt_tokens >= self.threshold:
            endpoint = self._next_prefill_endpoint()
            logger.debug(
                f"Request {request.request_id}: "
                f"long prefill ({request.prompt_tokens} tokens) → {endpoint}"
            )
        else:
            endpoint = self._next_decode_endpoint()
            logger.debug(
                f"Request {request.request_id}: "
                f"short prompt ({request.prompt_tokens} tokens) → {endpoint}"
            )

        return endpoint

    async def forward_request(
        self,
        request: InferenceRequest,
        stream: bool = True,
    ):
        """
        Forward request to appropriate vLLM endpoint.
        Uses httpx for async HTTP client.
        """
        import httpx

        endpoint = self.route(request)
        url      = f"{endpoint}/v1/completions"

        payload = {
            "model":       "sre-llmops",
            "prompt":      request.prompt,
            "max_tokens":  request.max_tokens,
            "temperature": request.temperature,
            "stream":      stream,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            if stream:
                async with client.stream("POST", url, json=payload) as response:
                    async for chunk in response.aiter_text():
                        yield chunk
            else:
                response = await client.post(url, json=payload)
                yield response.json()


class ChunkedPrefillConfig:
    """
    Configuration for vLLM chunked prefill.
    Single-GPU approximation of prefill-decode disaggregation.

    How chunked prefill works:
    1. Long prompt arrives (e.g. 2000 tokens)
    2. vLLM splits prefill into chunks (e.g. 512 tokens each)
    3. Between chunks, decode steps run for existing sequences
    4. Prefill completes in 4 chunks, decode never stalls > 1 chunk
    5. Result: max ITL spike = time to process 1 chunk (not full prefill)

    Trade-off:
    - Prefill throughput slightly lower (interrupted by decode)
    - Decode ITL much more stable (predictable latency)
    - Use when: streaming responses where ITL consistency matters
    """

    @staticmethod
    def optimal_chunk_size(
        gpu_memory_gb: float = 24.0,
        target_max_itl_ms: float = 100.0,
        model_size_b: float = 8.0,
    ) -> int:
        """
        Calculate optimal chunk size for target ITL.

        A10G can process ~1000 tokens/sec in prefill mode.
        For 100ms max ITL: chunk_size = 1000 * 0.1 = 100 tokens.
        Add buffer: 80 tokens per chunk.

        For larger models or slower GPUs, reduce chunk size.
        """
        # Rough estimate: tokens per second for prefill
        tokens_per_sec = (gpu_memory_gb / model_size_b) * 500

        # Max tokens processable within target ITL
        max_chunk = int(tokens_per_sec * (target_max_itl_ms / 1000))

        # Round down to nearest 64 (alignment)
        chunk_size = max(64, (max_chunk // 64) * 64)

        logger.info(
            f"Optimal chunk size: {chunk_size} tokens "
            f"(target ITL: {target_max_itl_ms}ms, "
            f"GPU: {gpu_memory_gb}GB, "
            f"model: {model_size_b}B)"
        )

        return chunk_size

    @staticmethod
    def get_vllm_args(chunk_size: int = 512) -> dict:
        """Get vLLM args for chunked prefill."""
        return {
            "enable_chunked_prefill":    True,
            "max_num_prefill_seqs":      1,
            "max_num_batched_tokens":    chunk_size,
        }
