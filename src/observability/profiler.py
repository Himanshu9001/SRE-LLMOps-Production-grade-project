"""
P11e — Performance Profiling + GPU Bottleneck Troubleshooting
Identifies whether inference is compute-bound or memory-bandwidth-bound.

Key insight for LLM inference:
  PREFILL phase: compute-bound (matrix multiplications)
    - GPU utilization: HIGH (90-100%)
    - Memory bandwidth: MEDIUM
    - Bottleneck: FLOPs (can't go faster without faster GPU)
    - Solution: batch multiple prefills together

  DECODE phase: memory-bandwidth-bound (KV cache reads)
    - GPU utilization: LOW (10-30%)
    - Memory bandwidth: HIGH (100%)
    - Bottleneck: bytes/sec (reading KV cache for each token)
    - Solution: larger batch sizes (more seqs share BW overhead)

Arithmetic intensity:
  FLOPs / Bytes = how many operations per byte of memory accessed
  High intensity (>200) = compute-bound → faster GPU helps
  Low intensity (<10)   = memory-bound → faster HBM helps
  LLM decode: ~1-5 FLOPs/byte → ALWAYS memory-bandwidth-bound

Roofline analysis:
  A10G peak FP16: 125 TFLOPS
  A10G peak HBM:  600 GB/s
  Ridge point: 125e12 / 600e9 = ~208 FLOPs/byte
  LLM decode at 2 FLOPs/byte → 99% memory-limited
"""

import time
import json
import torch
from pathlib import Path
from typing import Optional
from src.utils.logger import logger


class GPUProfiler:
    """
    Profile GPU operations to identify bottlenecks.
    Uses PyTorch Profiler + NVIDIA DCGM metrics.
    """

    def __init__(self, output_dir: str = "/tmp/profiler"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def profile_inference_step(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label: str = "inference",
    ) -> dict:
        """
        Profile a single inference step.
        Returns timing breakdown: tokenize, attention, FFN, logits.
        """
        from torch.profiler import profile, record_function, ProfilerActivity

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=True,
        ) as prof:
            with torch.no_grad():
                with record_function("attention"):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

        # Export Chrome trace
        trace_path = self.output_dir / f"{label}_trace.json"
        prof.export_chrome_trace(str(trace_path))

        # Extract key stats
        key_avgs = prof.key_averages()

        stats = {
            "label": label,
            "trace_file": str(trace_path),
            "top_ops": [],
        }

        for item in key_avgs.table(sort_by="cuda_time_total", row_limit=10).split("\n"):
            if item.strip():
                stats["top_ops"].append(item)

        # Compute arithmetic intensity
        total_flops = sum(
            getattr(item, 'flops', 0) or 0
            for item in key_avgs
        )
        total_bytes = sum(
            (getattr(item, 'cpu_memory_usage', 0) or 0) +
            (getattr(item, 'cuda_memory_usage', 0) or 0)
            for item in key_avgs
        )

        if total_bytes > 0:
            intensity = total_flops / total_bytes
            stats["arithmetic_intensity"] = intensity
            stats["bottleneck"] = (
                "compute-bound" if intensity > 208
                else "memory-bandwidth-bound"
            )
        else:
            stats["arithmetic_intensity"] = 0
            stats["bottleneck"] = "unknown"

        logger.info(
            f"Profile [{label}]: "
            f"intensity={stats.get('arithmetic_intensity', 0):.1f} FLOPs/byte, "
            f"bottleneck={stats['bottleneck']}"
        )

        return stats

    def benchmark_kv_cache(
        self,
        model,
        tokenizer,
        seq_lengths: list[int] = [128, 256, 512, 1024],
    ) -> dict:
        """
        Benchmark inference at different sequence lengths.
        Shows how KV cache size affects decode throughput.

        Expected pattern:
          Short seqs: higher throughput (less KV cache to read)
          Long seqs:  lower throughput (more KV cache bandwidth)
          Shows memory-bandwidth bottleneck of decode phase
        """
        results = {}

        for seq_len in seq_lengths:
            prompt = "A " * seq_len
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=seq_len,
            ).to("cuda")

            # Warmup
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=5,
                              pad_token_id=tokenizer.eos_token_id)

            # Benchmark
            torch.cuda.synchronize()
            start = time.perf_counter()
            NUM_RUNS = 5

            for _ in range(NUM_RUNS):
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / NUM_RUNS

            output_tokens   = output.shape[1] - inputs["input_ids"].shape[1]
            throughput_tps  = output_tokens / elapsed

            results[seq_len] = {
                "seq_len":      seq_len,
                "throughput":   round(throughput_tps, 1),
                "latency_ms":   round(elapsed * 1000, 1),
                "output_tokens": output_tokens,
            }

            logger.info(
                f"KV cache benchmark: seq_len={seq_len} "
                f"→ {throughput_tps:.1f} tok/s, "
                f"{elapsed*1000:.0f}ms"
            )

        return results

    @staticmethod
    def troubleshoot_gpu_utilization(
        gpu_util: float,
        sm_occupancy: float,
        memory_bandwidth_util: float,
        kv_cache_usage: float,
    ) -> dict:
        """
        Diagnose GPU performance issues from DCGM metrics.

        Decision tree:
        - Low GPU util + Low SM occupancy: compute underutilization
          → Increase batch size, check for CPU bottleneck
        - High GPU util + Low throughput: compute-bound
          → Use larger batch size for prefill, chunked prefill
        - Low GPU util + High memory BW: memory-bound (normal for decode)
          → This is expected, increase batch size to improve BW utilization
        - High KV cache: OOM risk
          → Reduce max_num_seqs, reduce max_model_len
        """
        diagnosis = {
            "gpu_util":             gpu_util,
            "sm_occupancy":         sm_occupancy,
            "memory_bandwidth":     memory_bandwidth_util,
            "kv_cache_usage":       kv_cache_usage,
            "issues":               [],
            "recommendations":      [],
        }

        # Check for compute underutilization
        if gpu_util < 30 and sm_occupancy < 0.3:
            diagnosis["issues"].append("Compute underutilization")
            diagnosis["recommendations"].extend([
                "Increase batch size (max_num_seqs)",
                "Check if CPU preprocessing is bottleneck",
                "Verify requests are actually arriving (check queue depth)",
            ])

        # Memory bandwidth bound (normal for decode)
        if memory_bandwidth_util > 0.8 and gpu_util < 40:
            diagnosis["issues"].append("Memory bandwidth bound (normal for decode)")
            diagnosis["recommendations"].extend([
                "This is EXPECTED for LLM decode phase",
                "Increase batch size to amortize memory bandwidth cost",
                "Consider speculative decoding for latency-sensitive workloads",
            ])

        # KV cache pressure
        if kv_cache_usage > 0.85:
            diagnosis["issues"].append("KV cache near capacity")
            diagnosis["recommendations"].extend([
                "Reduce max_model_len or max_num_seqs",
                "Scale up additional vLLM replicas",
                "Enable KV cache offloading to CPU",
            ])

        # SM occupancy low despite high GPU util
        if gpu_util > 70 and sm_occupancy < 0.4:
            diagnosis["issues"].append("Poor SM occupancy — wasted compute")
            diagnosis["recommendations"].extend([
                "Increase batch size to improve SM utilization",
                "Check for small tensor operations (use larger batch)",
                "Profile with PyTorch Profiler to identify kernel inefficiency",
            ])

        if not diagnosis["issues"]:
            diagnosis["status"] = "healthy"
        else:
            diagnosis["status"] = "degraded"

        return diagnosis
