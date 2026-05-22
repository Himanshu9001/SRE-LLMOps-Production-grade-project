"""
P8b — GPTQ (Generative Pre-trained Transformer Quantization)
Layer-by-layer quantization using second-order optimization.

GPTQ algorithm:
1. For each linear layer:
   a. Compute Hessian H = 2 * X^T * X (X = input activations)
   b. Quantize weights column by column
   c. Update remaining weights to compensate for quantization error
   d. Uses Cholesky decomposition for numerical stability

Why GPTQ for comparison:
- Widely supported (transformers, vLLM, llama.cpp)
- Slightly lower quality than AWQ but well-tested
- Good baseline for quality comparison
- AutoGPTQ library: simple API, production-ready
"""

import boto3
import argparse
from pathlib import Path

from src.training.model_utils import download_model_from_s3
from src.utils.logger import logger


def quantize_gptq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
    desc_act: bool = False,
    damp_percent: float = 0.01,
) -> Path:
    """
    Quantize model to GPTQ 4-bit.

    Parameters:
    - bits=4:          4-bit quantization
    - group_size=128:  per-group quantization scales
    - desc_act=False:  activation order (False = faster, True = better quality)
    - damp_percent:    dampening for Hessian inverse (numerical stability)
                       Higher = more stable, slightly lower quality

    desc_act=True (descending activation order):
      Sorts columns by activation magnitude before quantization
      Better quality but slower quantization and inference
      Use for maximum quality at cost of speed

    desc_act=False:
      Standard column order
      Faster quantization, compatible with more inference backends
      Recommended for production serving
    """
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("auto-gptq not installed. pip install auto-gptq")

    logger.info(f"Starting GPTQ {bits}-bit quantization")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # GPTQ quantization config
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        damp_percent=damp_percent,
    )

    # Load model for quantization
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config=quantize_config,
    )

    # Calibration examples — 128 samples standard for GPTQ
    # Using short SRE-style prompts for domain calibration
    calibration_examples = [
        tokenizer(
            "kubectl get pods -n production | grep CrashLoopBackOff",
            return_tensors="pt"
        ),
        tokenizer(
            "Error: terraform state lock. Lock ID: abc123. How to resolve?",
            return_tensors="pt"
        ),
        tokenizer(
            "Prometheus alert: HighErrorRate firing for payment-service",
            return_tensors="pt"
        ),
    ] * 43  # repeat to get ~128 calibration samples

    logger.info("Running GPTQ calibration and quantization...")
    model.quantize(calibration_examples)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_quantized(str(output_dir), use_safetensors=True)
    tokenizer.save_pretrained(str(output_dir))

    total_size = sum(
        f.stat().st_size for f in output_dir.rglob("*") if f.is_file()
    )
    logger.info(f"GPTQ quantized model: {total_size/1024**3:.2f}GB → {output_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-s3-prefix",  default="base-models/llama3-8b")
    parser.add_argument("--output-s3-prefix", default="quantized/llama3-8b-gptq-4bit")
    parser.add_argument("--s3-bucket",        default="sre-llmops-artifacts")
    parser.add_argument("--local-model-dir",  default="/tmp/base-model")
    parser.add_argument("--local-output-dir", default="/tmp/gptq-quantized")
    parser.add_argument("--bits",             type=int,   default=4)
    parser.add_argument("--group-size",       type=int,   default=128)
    parser.add_argument("--desc-act",         action="store_true", default=False)
    args = parser.parse_args()

    model_dir = download_model_from_s3(
        args.s3_bucket, args.model_s3_prefix, args.local_model_dir
    )

    output_dir = quantize_gptq(
        model_path=str(model_dir),
        output_path=args.local_output_dir,
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
    )

    s3 = boto3.client("s3", region_name="us-east-1")
    for f in output_dir.rglob("*"):
        if f.is_file() and not f.name.startswith("."):
            s3_key = f"{args.output_s3_prefix}/{f.relative_to(output_dir)}"
            s3.upload_file(str(f), args.s3_bucket, s3_key,
                          ExtraArgs={"ServerSideEncryption": "AES256"})

    logger.info(f"GPTQ model uploaded to s3://{args.s3_bucket}/{args.output_s3_prefix}/")


if __name__ == "__main__":
    main()
