"""
Model loading utilities for QLoRA fine-tuning.
Loads Llama 3 8B from S3 in 4-bit NF4 quantization.
Applies LoRA adapters on top of frozen quantized base.
"""

import os
import boto3
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from src.utils.logger import logger


def download_model_from_s3(
    s3_bucket: str,
    s3_prefix: str,
    local_dir: str,
    region: str = "us-east-1"
) -> Path:
    """
    Download model weights from S3 to local disk before loading.
    transformers cannot read directly from S3 — needs local path.
    Uses S3 transfer manager for parallel multipart downloads.
    """
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", region_name=region)

    # List all objects under the prefix
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)

    files = []
    for page in pages:
        for obj in page.get("Contents", []):
            files.append(obj["Key"])

    logger.info(f"Downloading {len(files)} files from s3://{s3_bucket}/{s3_prefix}")

    for s3_key in files:
        relative = s3_key[len(s3_prefix):].lstrip("/")
        local_file = local_path / relative
        local_file.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already downloaded — resume support
        if local_file.exists():
            logger.debug(f"  SKIP {relative} — already exists")
            continue

        logger.info(f"  Downloading {relative}...")
        s3.download_file(s3_bucket, s3_key, str(local_file))

    logger.info(f"Model downloaded to {local_path}")
    return local_path


def load_model_for_qlora(
    model_path: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> tuple:
    """
    Load Llama 3 8B in 4-bit NF4 quantization + LoRA adapters.

    QLoRA flow:
    1. Load base model weights in 4-bit NF4 (frozen, ~4GB VRAM)
    2. prepare_model_for_kbit_training: casts LayerNorm to fp32,
       enables gradient checkpointing
    3. Inject LoRA adapters in bf16 on q_proj + v_proj
    4. Only adapter weights (~0.5% of params) are trainable

    Why q_proj + v_proj only?
    These are the attention matrices most responsible for
    contextual understanding. Adding k_proj and o_proj gives
    diminishing returns vs memory cost for domain adaptation.
    """

    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Llama 3 has no pad token by default — use EOS as pad
    # Critical: without this, variable-length batch padding fails
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4-bit NF4 quantization config
    # NF4 (NormalFloat4): information-theoretically optimal for
    # normally distributed weights — better than int4 for LLMs
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NF4 vs int4: better perplexity
        bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bf16, store in 4bit
        bnb_4bit_use_double_quant=True,     # double quantization: quantize quant constants
                                             # saves ~0.4 bits/param additional
    )

    logger.info(f"Loading model in 4-bit NF4 from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",          # automatically places layers across GPU/CPU
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )

    # Prepare model for k-bit training:
    # - Upcasts LayerNorm layers to fp32 for training stability
    # - Enables gradient checkpointing to reduce VRAM usage
    # - Sets requires_grad=False on all base model params
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # LoRA configuration
    # r=16: rank of adapter matrices — higher = more capacity, more memory
    # alpha=32: scaling factor = alpha/r = 2.0 — controls adapter contribution
    # target_modules: inject adapters into attention projections only
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",                # don't train biases — not needed for domain adapt
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def load_model_cpu(
    model_path: str,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> tuple:
    """
    Load a small model in fp32 for CPU smoke testing.
    No quantization — used only to validate pipeline end-to-end.
    Not suitable for large models (>1B params) on CPU.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    logger.info(f"Loading model on CPU (smoke test) from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer
