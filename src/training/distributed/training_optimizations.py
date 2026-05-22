"""
P5d — Training Optimization Techniques
FlashAttention-2, packed sequences, mixed precision strategies.
Applied on top of FSDP/DeepSpeed for maximum throughput.
"""

import torch
from transformers import AutoModelForCausalLM
from datasets import Dataset
from src.utils.logger import logger


def enable_flash_attention(model) -> None:
    """
    Enable FlashAttention-2 for memory-efficient attention computation.

    Standard attention: O(n²) memory — quadratic in sequence length.
    FlashAttention-2:  O(n) memory — uses tiling + recomputation.

    For seq_len=1024: ~4x memory reduction in attention layers.
    For seq_len=4096: ~16x memory reduction.

    Requires: CUDA GPU with compute capability >= 8.0 (A10G = 8.6 ✅)
    Install: pip install flash-attn --no-build-isolation

    How it works:
    1. Tiles Q, K, V matrices into blocks
    2. Computes attention score block by block
    3. Recomputes attention during backward instead of storing
    4. Uses fused CUDA kernels — single kernel for QKV + softmax + output
    """
    try:
        # Check if flash_attn is available
        import flash_attn
        logger.info(f"FlashAttention {flash_attn.__version__} available")

        # Enable via model config — transformers handles the rest
        model.config.attn_implementation = "flash_attention_2"
        logger.info("FlashAttention-2 enabled")

    except ImportError:
        logger.warning(
            "flash_attn not installed — using standard attention. "
            "Install with: pip install flash-attn --no-build-isolation"
        )
        # Fall back to SDPA (Scaled Dot Product Attention)
        # PyTorch 2.0+ built-in, ~2x faster than naive attention
        model.config.attn_implementation = "sdpa"
        logger.info("Using PyTorch SDPA (fallback)")


def pack_sequences(
    dataset: Dataset,
    tokenizer,
    max_length: int = 2048,
) -> Dataset:
    """
    Pack multiple short sequences into single training examples.

    Problem: SRE dataset has variable length sequences (50-800 tokens).
    Standard batching pads short sequences to max_length — wasteful.

    Packing: concatenate multiple sequences with EOS separator
    until max_length is reached. No padding needed.

    Example with max_length=512:
    Standard: [seq1: 100 tokens + 412 padding] — 80% waste
    Packed:   [seq1(100) + sep + seq2(200) + sep + seq3(150) + sep] — 0% waste

    Throughput improvement: 2-4x for short-sequence datasets.

    Loss masking: labels for separator tokens set to -100
    so model doesn't learn to predict EOS as content.
    """
    packed_examples = []
    current_input_ids = []
    current_labels    = []
    current_attn_mask = []

    eos_id = tokenizer.eos_token_id

    for example in dataset:
        ids    = example["input_ids"]
        labels = example["labels"]
        mask   = example["attention_mask"]

        # Add EOS separator between sequences
        ids_with_sep    = ids + [eos_id]
        labels_with_sep = labels + [-100]   # don't predict separator
        mask_with_sep   = mask + [1]

        # If adding this sequence would exceed max_length, flush current
        if len(current_input_ids) + len(ids_with_sep) > max_length:
            if current_input_ids:
                # Pad to max_length
                pad_len = max_length - len(current_input_ids)
                packed_examples.append({
                    "input_ids":      current_input_ids + [tokenizer.pad_token_id] * pad_len,
                    "labels":         current_labels    + [-100] * pad_len,
                    "attention_mask": current_attn_mask + [0] * pad_len,
                })
            current_input_ids = ids_with_sep
            current_labels    = labels_with_sep
            current_attn_mask = mask_with_sep
        else:
            current_input_ids.extend(ids_with_sep)
            current_labels.extend(labels_with_sep)
            current_attn_mask.extend(mask_with_sep)

    # Flush remaining
    if current_input_ids:
        pad_len = max_length - len(current_input_ids)
        packed_examples.append({
            "input_ids":      current_input_ids + [tokenizer.pad_token_id] * pad_len,
            "labels":         current_labels    + [-100] * pad_len,
            "attention_mask": current_attn_mask + [0] * pad_len,
        })

    original_count = len(dataset)
    packed_count   = len(packed_examples)
    reduction      = (1 - packed_count / original_count) * 100

    logger.info(
        f"Sequence packing: {original_count} → {packed_count} examples "
        f"({reduction:.1f}% reduction, ~{original_count/packed_count:.1f}x throughput gain)"
    )

    return Dataset.from_list(packed_examples)


def get_mixed_precision_config(gpu_compute_capability: float) -> dict:
    """
    Select optimal mixed precision strategy based on GPU architecture.

    bf16 vs fp16:
    - bf16: same exponent range as fp32, smaller mantissa
            No overflow/underflow with LLM's large activation values
            A10G (8.6), A100 (8.0), H100 (9.0) — native bf16 support
    - fp16: smaller exponent range — overflow risk with large activations
            Requires loss scaling to prevent underflow
            Better on older GPUs (V100, T4) that lack bf16 support

    Decision tree:
    compute >= 8.0 (Ampere+) → bf16, no loss scaling needed
    compute >= 7.0 (Volta/Turing) → fp16 with automatic loss scaling
    compute < 7.0 → fp32 only (no mixed precision benefit)
    """
    if gpu_compute_capability >= 8.0:
        return {
            "dtype": torch.bfloat16,
            "loss_scaling": False,
            "reason": "Ampere+ GPU — native bf16, no loss scaling needed"
        }
    elif gpu_compute_capability >= 7.0:
        return {
            "dtype": torch.float16,
            "loss_scaling": True,
            "reason": "Volta/Turing GPU — fp16 with automatic loss scaling"
        }
    else:
        return {
            "dtype": torch.float32,
            "loss_scaling": False,
            "reason": "Pre-Volta GPU — fp32 only"
        }


def profile_training_step(model, batch, profiler_output_dir: str = "/tmp/profiler"):
    """
    Profile a single training step with PyTorch Profiler.
    Identifies GPU bottlenecks: compute-bound vs memory-bound vs communication-bound.

    Output: Chrome trace JSON — open in chrome://tracing
    Key metrics:
    - GPU utilization (want >85%)
    - Memory bandwidth utilization
    - Kernel execution time
    - NCCL communication overhead (for distributed)
    """
    from torch.profiler import profile, record_function, ProfilerActivity

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("forward_pass"):
            outputs = model(**batch)
            loss    = outputs.loss

        with record_function("backward_pass"):
            loss.backward()

    # Export Chrome trace for visualization
    prof.export_chrome_trace(f"{profiler_output_dir}/trace.json")

    # Print top GPU kernels by CUDA time
    logger.info("\nTop 10 GPU operations by CUDA time:")
    logger.info(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=10
        )
    )

    return prof
