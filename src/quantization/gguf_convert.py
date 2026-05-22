"""
P8c — GGUF Conversion (llama.cpp format)
Converts model to GGUF for CPU inference and edge deployment.

GGUF (GPT-Generated Unified Format):
  Successor to GGML — llama.cpp native format
  Designed for CPU inference with optional GPU offload
  Supports multiple quantization levels (Q2 to Q8)

Quantization levels:
  Q2_K:    2-bit — smallest, lowest quality, CPU only
  Q4_0:    4-bit basic — fast, moderate quality
  Q4_K_M:  4-bit K-quant medium — best 4-bit quality/speed balance
            "K" = uses k-quants algorithm (better than standard 4-bit)
            "M" = medium size variant
  Q5_K_M:  5-bit — higher quality, larger
  Q8_0:    8-bit — near fp16 quality, 2x compression

Recommended for SRE use case:
  Q4_K_M: best balance for laptop/edge deployment
  Fits in 8GB RAM — runs on MacBook Pro M2

Use cases for GGUF:
  - Local inference on developer laptops (no GPU needed)
  - Edge deployment (Raspberry Pi 4, Jetson Nano)
  - Offline incident response (no internet required)
  - Demo/POC without cloud infrastructure
  - llama.cpp server for CPU-only environments
"""

import os
import subprocess
import boto3
import argparse
from pathlib import Path

from src.training.model_utils import download_model_from_s3
from src.utils.logger import logger


QUANTIZATION_TYPES = {
    "Q4_K_M": "Best quality/size balance — recommended",
    "Q5_K_M": "Higher quality, 25% larger",
    "Q8_0":   "Near fp16 quality, 2x compression",
    "Q4_0":   "Basic 4-bit, fastest CPU inference",
    "Q2_K":   "Minimum size, lowest quality",
}


def install_llama_cpp():
    """
    Install llama.cpp Python bindings.
    Includes convert_hf_to_gguf.py script for HuggingFace model conversion.
    """
    try:
        import llama_cpp
        logger.info(f"llama-cpp-python already installed")
        return
    except ImportError:
        pass

    logger.info("Installing llama-cpp-python...")
    subprocess.run(
        ["pip", "install", "llama-cpp-python", "--no-cache-dir"],
        check=True
    )


def convert_to_gguf_f16(model_path: str, output_dir: str) -> Path:
    """
    Step 1: Convert HuggingFace model to GGUF fp16.
    This is the base GGUF before quantization.
    Uses llama.cpp's convert_hf_to_gguf.py script.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gguf_f16_path = output_dir / "model-f16.gguf"

    if gguf_f16_path.exists():
        logger.info(f"GGUF f16 already exists: {gguf_f16_path}")
        return gguf_f16_path

    # Find llama.cpp convert script
    try:
        import llama_cpp
        llama_cpp_dir = Path(llama_cpp.__file__).parent
        convert_script = llama_cpp_dir / "llama_cpp" / "llama-cpp-python" / "vendor" / "llama.cpp" / "convert_hf_to_gguf.py"
    except Exception:
        # Fall back to downloading convert script directly
        convert_script = Path("/tmp/convert_hf_to_gguf.py")
        if not convert_script.exists():
            subprocess.run([
                "wget", "-q",
                "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py",
                "-O", str(convert_script)
            ], check=True)

    logger.info(f"Converting {model_path} to GGUF f16...")
    result = subprocess.run([
        "python3", str(convert_script),
        model_path,
        "--outfile", str(gguf_f16_path),
        "--outtype", "f16",
    ], capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Conversion failed: {result.stderr}")
        raise RuntimeError(f"GGUF conversion failed: {result.stderr}")

    size_gb = gguf_f16_path.stat().st_size / 1024**3
    logger.info(f"GGUF f16 created: {size_gb:.2f}GB → {gguf_f16_path}")
    return gguf_f16_path


def quantize_gguf(
    f16_gguf_path: Path,
    output_dir: Path,
    quant_type: str = "Q4_K_M",
) -> Path:
    """
    Step 2: Quantize GGUF f16 to target quantization level.
    Uses llama.cpp's quantize binary.
    """
    output_path = output_dir / f"model-{quant_type}.gguf"

    if output_path.exists():
        logger.info(f"Quantized GGUF already exists: {output_path}")
        return output_path

    # Find llama-quantize binary
    quantize_bin = None
    for candidate in ["/usr/local/bin/llama-quantize", "/usr/bin/llama-quantize"]:
        if Path(candidate).exists():
            quantize_bin = candidate
            break

    if not quantize_bin:
        # Build from source
        logger.info("llama-quantize not found, building from source...")
        build_dir = Path("/tmp/llama-cpp-build")
        if not build_dir.exists():
            subprocess.run([
                "git", "clone", "--depth=1",
                "https://github.com/ggerganov/llama.cpp.git",
                str(build_dir)
            ], check=True)
            subprocess.run(
                ["make", "quantize", "-j4"],
                cwd=str(build_dir), check=True
            )
        quantize_bin = str(build_dir / "quantize")

    logger.info(f"Quantizing to {quant_type} ({QUANTIZATION_TYPES.get(quant_type, '')})")
    result = subprocess.run([
        quantize_bin,
        str(f16_gguf_path),
        str(output_path),
        quant_type,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Quantization failed: {result.stderr}")
        raise RuntimeError(f"GGUF quantization failed: {result.stderr}")

    size_gb = output_path.stat().st_size / 1024**3
    logger.info(f"GGUF {quant_type} created: {size_gb:.2f}GB → {output_path}")
    return output_path


def run_gguf_benchmark(gguf_path: Path, prompt: str = None) -> dict:
    """
    Quick benchmark of GGUF model — tokens/sec on CPU.
    Used to validate quantization quality and speed.
    """
    try:
        from llama_cpp import Llama

        if prompt is None:
            prompt = (
                "### Instruction:\nA pod is in CrashLoopBackOff. "
                "How do I diagnose it?\n\n### Input:\n"
                "Pod: api-server-7d9f8b in namespace production\n\n"
                "### Response:\n"
            )

        logger.info(f"Benchmarking {gguf_path.name}...")

        llm = Llama(
            model_path=str(gguf_path),
            n_ctx=512,
            n_threads=os.cpu_count(),
            verbose=False,
        )

        import time
        start = time.time()
        output = llm(prompt, max_tokens=100, echo=False)
        elapsed = time.time() - start

        tokens_generated = output["usage"]["completion_tokens"]
        tokens_per_sec   = tokens_generated / elapsed

        result = {
            "model":          gguf_path.name,
            "tokens_per_sec": round(tokens_per_sec, 1),
            "tokens_generated": tokens_generated,
            "elapsed_seconds":  round(elapsed, 2),
            "response_preview": output["choices"][0]["text"][:100],
        }

        logger.info(f"Benchmark: {tokens_per_sec:.1f} tokens/sec")
        return result

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-s3-prefix",  default="base-models/llama3-8b")
    parser.add_argument("--output-s3-prefix", default="quantized/llama3-8b-gguf")
    parser.add_argument("--s3-bucket",        default="sre-llmops-artifacts")
    parser.add_argument("--local-model-dir",  default="/tmp/base-model")
    parser.add_argument("--local-output-dir", default="/tmp/gguf-output")
    parser.add_argument("--quant-types",      nargs="+",
                        default=["Q4_K_M", "Q5_K_M", "Q8_0"])
    args = parser.parse_args()

    install_llama_cpp()

    # Download model
    model_dir = download_model_from_s3(
        args.s3_bucket, args.model_s3_prefix, args.local_model_dir
    )

    output_dir = Path(args.local_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert to GGUF f16
    f16_path = convert_to_gguf_f16(str(model_dir), str(output_dir))

    # Step 2: Quantize to multiple formats
    s3 = boto3.client("s3", region_name="us-east-1")

    for quant_type in args.quant_types:
        logger.info(f"\n=== Quantizing to {quant_type} ===")
        quantized_path = quantize_gguf(f16_path, output_dir, quant_type)

        # Benchmark
        benchmark = run_gguf_benchmark(quantized_path)
        if benchmark:
            logger.info(f"Speed: {benchmark['tokens_per_sec']} tokens/sec")

        # Upload to S3
        s3_key = f"{args.output_s3_prefix}/{quantized_path.name}"
        logger.info(f"Uploading to s3://{args.s3_bucket}/{s3_key}")
        s3.upload_file(
            str(quantized_path), args.s3_bucket, s3_key,
            ExtraArgs={"ServerSideEncryption": "AES256"}
        )

    logger.info(f"\nAll GGUF variants uploaded to s3://{args.s3_bucket}/{args.output_s3_prefix}/")


if __name__ == "__main__":
    main()
