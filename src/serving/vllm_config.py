"""
P10a — vLLM Configuration
Production serving configuration for Llama 3 8B on A10G GPU.

vLLM key innovations:
1. PagedAttention: manages KV cache like OS virtual memory
   - Standard attention: KV cache pre-allocated per request (wasteful)
   - PagedAttention: KV cache allocated in fixed-size pages on demand
   - Result: near-zero KV cache waste, 2-4x more concurrent requests

2. Continuous batching:
   - Standard: batch fixed at request start, GPU waits for all to finish
   - Continuous: new requests join mid-flight as slots free up
   - Result: ~10x throughput improvement over naive batching

3. Tensor parallelism:
   - Split model across multiple GPUs (for large models)
   - Each GPU holds a shard of each layer
   - All-reduce after each layer — requires fast interconnect

Memory math for Llama 3 8B AWQ on A10G (24GB):
  Model weights (AWQ 4-bit):   ~4.5GB
  KV cache (PagedAttention):   ~16GB  (variable, scales with load)
  Overhead:                    ~1GB
  Available for KV cache:      ~18.5GB
  Max concurrent sequences:    ~100-200 (depends on seq length)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VLLMConfig:
    """
    vLLM engine configuration.
    Maps directly to vllm.AsyncLLMEngine parameters.
    """

    # Model
    model:              str   = "sre-llmops-artifacts/quantized/llama3-8b-awq-4bit"
    tokenizer:          str   = None   # defaults to model path
    quantization:       str   = "awq"  # awq | gptq | None

    # GPU
    tensor_parallel_size:  int   = 1       # GPUs per replica
    gpu_memory_utilization: float = 0.90   # fraction of GPU memory for KV cache
                                           # 0.90 = leave 10% for model weights overhead
                                           # Too high → OOM; Too low → fewer concurrent requests

    # KV Cache (PagedAttention)
    block_size:         int   = 16     # tokens per KV cache page
                                       # 16 is optimal for A10G — matches GPU memory alignment
                                       # Larger = less fragmentation, larger minimum allocation
    max_model_len:      int   = 4096   # maximum sequence length (prompt + completion)
                                       # Llama 3 supports 8192 but 4096 fits more concurrent requests
    max_num_seqs:       int   = 256    # max concurrent sequences
    max_num_batched_tokens: int = 8192 # max tokens across all sequences in one batch

    # Serving
    host:               str   = "0.0.0.0"
    port:               int   = 8000
    uvicorn_log_level:  str   = "info"

    # OpenAI-compatible API
    served_model_name:  str   = "sre-llmops"

    # Continuous batching
    max_paddings:       int   = 256    # max padding tokens per batch
    scheduler_delay_factor: float = 0.0  # 0 = no delay, process immediately

    # Speculative decoding (P13)
    speculative_model:     Optional[str] = None
    num_speculative_tokens: int = 5        # draft tokens per step

    # Prefill-decode disaggregation (P10c)
    enable_chunked_prefill: bool = True    # split long prefills across multiple steps
    max_num_prefill_seqs:   int  = 1       # concurrent prefill sequences
                                           # 1 = full disaggregation (prefill blocks decode)

    def to_engine_args(self) -> dict:
        """Convert to vLLM EngineArgs dict."""
        args = {
            "model":                    self.model,
            "quantization":             self.quantization,
            "tensor_parallel_size":     self.tensor_parallel_size,
            "gpu_memory_utilization":   self.gpu_memory_utilization,
            "block_size":               self.block_size,
            "max_model_len":            self.max_model_len,
            "max_num_seqs":             self.max_num_seqs,
            "max_num_batched_tokens":   self.max_num_batched_tokens,
            "served_model_name":        self.served_model_name,
            "enable_chunked_prefill":   self.enable_chunked_prefill,
        }

        if self.speculative_model:
            args["speculative_model"]      = self.speculative_model
            args["num_speculative_tokens"] = self.num_speculative_tokens

        return args


# Production configs for different scenarios
CONFIGS = {
    # Single GPU, AWQ quantized — default production config
    "production": VLLMConfig(
        model="s3://sre-llmops-artifacts/quantized/llama3-8b-awq-4bit",
        quantization="awq",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        max_num_seqs=256,
        enable_chunked_prefill=True,
    ),

    # High throughput — maximize batch size
    "high_throughput": VLLMConfig(
        model="s3://sre-llmops-artifacts/quantized/llama3-8b-awq-4bit",
        quantization="awq",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_model_len=2048,        # shorter sequences = more concurrent
        max_num_seqs=512,
        max_num_batched_tokens=16384,
        enable_chunked_prefill=True,
    ),

    # Low latency — minimize TTFT
    "low_latency": VLLMConfig(
        model="s3://sre-llmops-artifacts/quantized/llama3-8b-awq-4bit",
        quantization="awq",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        max_num_seqs=32,           # fewer concurrent = lower latency per request
        enable_chunked_prefill=False,  # disable chunked prefill for lowest TTFT
    ),

    # Speculative decoding config (P13)
    "speculative": VLLMConfig(
        model="s3://sre-llmops-artifacts/quantized/llama3-8b-awq-4bit",
        quantization="awq",
        speculative_model="s3://sre-llmops-artifacts/quantized/llama3-8b-awq-4bit",
        num_speculative_tokens=5,
        enable_chunked_prefill=False,  # incompatible with speculative decoding
    ),
}
