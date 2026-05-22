"""
P8a — AWQ (Activation-aware Weight Quantization)
Quantizes model weights to 4-bit using activation statistics.

Why AWQ over GPTQ:
  GPTQ: minimizes reconstruction error per layer independently
        Layer-by-layer second-order optimization (inverse Hessian)
        Fast quantization, decent quality

  AWQ:  analyzes activation magnitudes to find salient weights
        Protects 1% of weights that matter most (high activation)
        Scales protected weights before quantization → less error
        Better perplexity than GPTQ at same bit-width
        Faster inference than GPTQ (hardware-friendly layout)
        vLLM natively supports AWQ — preferred for serving

Memory reduction:
  fp16 Llama 3 8B:   16GB
  AWQ 4-bit:         ~4GB (4x reduction)
  AWQ 4-bit + group: ~4.5GB (slightly larger due to per-group scales)

Quality impact on SRE task:
  fp16 baseline:     perplexity X
  AWQ 4-bit:         perplexity X + 0.1-0.3 (minimal degradation)
  GPTQ 4-bit:        perplexity X + 0.3-0.5
  GGUF Q4_K_M:       perplexity X + 0.5-0.8 (CPU-optimized)
"""

import os
import boto3
import argparse
from pathlib import Path

from src.training.model_utils import download_model_from_s3
from src.utils.logger import logger


def quantize_awq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
    zero_point: bool = True,
    version: str = "GEMM",
) -> Path:
    """
    Quantize model to AWQ 4-bit.

    Parameters:
    - bits=4:        4-bit quantization (standard)
    - group_size=128: quantize in groups of 128 weights
                      Smaller groups = better quality, larger model
                      128 is optimal trade-off for LLMs
    - zero_point=True: asymmetric quantization (better quality than symmetric)
    - version="GEMM": optimized for GEMM operations (faster on A10G)
                      Alternative: "GEMV" for batch_size=1 inference

    AWQ algorithm:
    1. Run calibration forward passes with sample data
    2. Compute per-channel activation scales
    3. Identify salient weights (top 1% by activation magnitude)
    4. Scale up salient weights → quantize → scale down
    5. Store quantized weights + scales + zero points

    Calibration dataset: use training data subset (128 samples typical)
    More calibration samples → better quantization quality
    """
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "autoawq not installed. "
            "pip install autoawq"
        )

    logger.info(f"Starting AWQ {bits}-bit quantization")
    logger.info(f"Group size: {group_size}, Zero point: {zero_point}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model for AWQ quantization
    # AWQ loads in fp16 for calibration, then quantizes
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        safetensors=True,
        device_map="cuda",
    )

    # Quantization config
    quant_config = {
        "zero_point": zero_point,
        "q_group_size": group_size,
        "w_bit": bits,
        "version": version,
    }

    # Calibration data — use C4 dataset (standard) or custom SRE data
    # More domain-specific calibration → better quantization for that domain
    logger.info("Running AWQ calibration...")
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data="pileval",   # standard calibration dataset
        # For domain-specific: load SRE JSONL and pass as calib_data
    )

    # Save quantized model
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Log model size
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    logger.info(f"AWQ quantized model saved: {total_size/1024**3:.2f}GB → {output_dir}")

    return output_dir


def quantize_awq_with_adapter(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
) -> Path:
    """
    Merge LoRA adapter into base model BEFORE AWQ quantization.

    Workflow:
    1. Load base model (fp16)
    2. Load LoRA adapter (bf16)
    3. Merge adapter weights into base — creates full fine-tuned model
    4. AWQ quantize the merged model
    5. Save quantized model

    Why merge before quantize (not after):
    - Quantizing base then merging adapters degrades quality
    - Merged model has SRE domain knowledge baked in
    - AWQ calibration uses merged model's activations → correct scales
    - Final model: SRE-tuned + 4-bit compressed, ready for vLLM
    """
    from peft import PeftModel, AutoPeftModelForCausalLM
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    logger.info("Merging LoRA adapter into base model...")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Load base in fp16
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Load and merge LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()  # merge adapter → full model, remove LoRA structure

    # Save merged model temporarily
    merged_dir = Path(output_path).parent / "merged_temp"
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    logger.info(f"Merged model saved to {merged_dir}")

    # Now AWQ quantize the merged model
    return quantize_awq(
        model_path=str(merged_dir),
        output_path=output_path,
        bits=bits,
        group_size=group_size,
    )


def upload_quantized_model(
    local_path: Path,
    s3_bucket: str,
    s3_prefix: str,
    region: str = "us-east-1",
):
    """Upload quantized model to S3."""
    s3 = boto3.client("s3", region_name=region)

    files = list(local_path.rglob("*"))
    files = [f for f in files if f.is_file() and not f.name.startswith(".")]

    logger.info(f"Uploading {len(files)} files to s3://{s3_bucket}/{s3_prefix}/")

    for f in files:
        s3_key = f"{s3_prefix}/{f.relative_to(local_path)}"
        s3.upload_file(
            str(f), s3_bucket, s3_key,
            ExtraArgs={"ServerSideEncryption": "AES256"}
        )

    logger.info("Upload complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-s3-prefix",  default="base-models/llama3-8b")
    parser.add_argument("--adapter-s3-prefix",     default="adapters/llama3-8b-lora-kd-v1/final")
    parser.add_argument("--output-s3-prefix",      default="quantized/llama3-8b-awq-4bit")
    parser.add_argument("--s3-bucket",             default="sre-llmops-artifacts")
    parser.add_argument("--local-model-dir",       default="/tmp/base-model")
    parser.add_argument("--local-adapter-dir",     default="/tmp/adapter")
    parser.add_argument("--local-output-dir",      default="/tmp/awq-quantized")
    parser.add_argument("--bits",                  type=int, default=4)
    parser.add_argument("--group-size",            type=int, default=128)
    parser.add_argument("--merge-adapter",         action="store_true", default=True)
    args = parser.parse_args()

    # Download base model
    base_dir = download_model_from_s3(
        args.s3_bucket, args.base_model_s3_prefix, args.local_model_dir
    )

    if args.merge_adapter:
        # Download adapter
        adapter_dir = download_model_from_s3(
            args.s3_bucket, args.adapter_s3_prefix, args.local_adapter_dir
        )
        output_dir = quantize_awq_with_adapter(
            base_model_path=str(base_dir),
            adapter_path=str(adapter_dir),
            output_path=args.local_output_dir,
            bits=args.bits,
            group_size=args.group_size,
        )
    else:
        output_dir = quantize_awq(
            model_path=str(base_dir),
            output_path=args.local_output_dir,
            bits=args.bits,
            group_size=args.group_size,
        )

    upload_quantized_model(
        output_dir, args.s3_bucket, args.output_s3_prefix
    )


if __name__ == "__main__":
    main()
