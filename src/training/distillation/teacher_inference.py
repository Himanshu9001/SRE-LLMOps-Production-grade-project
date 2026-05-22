"""
P6 — Step 1: Teacher Inference — Generate Soft Labels
Run Llama 3 70B (teacher) over SRE training dataset.
Store logits (soft labels) on S3/FSx for offline distillation.

Why offline distillation:
  Teacher inference is expensive (70B params).
  Run teacher ONCE → store soft labels → train student REPEATEDLY.
  Student training doesn't need teacher at runtime — 10x cheaper.

Soft labels vs hard labels:
  Hard label: [0, 0, 1, 0, ...] — one-hot, only correct token
  Soft label: [0.001, 0.003, 0.89, 0.002, ...] — full probability distribution
  
  Soft labels carry inter-class similarity information.
  "kubectl" and "k8s" have similar distributions — student learns this.
  Hard labels treat all wrong tokens equally — loses this signal.

Temperature scaling:
  T=1: original distribution (peaked, low entropy)
  T=4: smoothed distribution (flatter, higher entropy)
  Higher T → softer labels → more information transferred
  Typical range: T=2 to T=6 for knowledge distillation
"""

import os
import json
import torch
import boto3
import argparse
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.nn.functional import softmax, log_softmax
from tqdm import tqdm

from src.training.data_utils import load_dataset_from_s3, format_alpaca_prompt
from src.training.model_utils import download_model_from_s3
from src.utils.logger import logger


class TeacherInference:
    """
    Runs teacher model inference to generate soft labels.
    Supports 4-bit quantization for 70B model on single A100 (80GB).
    For A10G (24GB): use 2x A10G with tensor parallelism or load in 8-bit.
    """

    def __init__(self, model_path: str, temperature: float = 4.0, device: str = "auto"):
        self.temperature = temperature
        self.device      = device

        logger.info(f"Loading teacher model from {model_path}")
        logger.info(f"Temperature: {temperature}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load teacher in 4-bit — reduces 70B from ~140GB to ~35GB
        # Fits on 2x A10G (48GB total) or 1x A100 (80GB)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        logger.info("Teacher model loaded")

    @torch.no_grad()
    def generate_soft_labels(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        top_k: int = 50,
    ) -> dict:
        """
        Generate soft labels for a single batch.

        Returns top-k token probabilities per position.
        Storing full vocabulary (128k tokens for Llama 3) is too expensive.
        Top-k=50 captures 95%+ of probability mass — good approximation.

        Output format per position:
        {
            "indices": [tok1, tok2, ..., tok50],   # top-k token IDs
            "probs":   [p1,   p2,   ..., p50],     # soft probabilities
        }
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # logits shape: [batch, seq_len, vocab_size]
        logits = outputs.logits

        # Temperature scaling — soften the distribution
        # Higher T → flatter distribution → more information per label
        scaled_logits = logits / self.temperature

        # Convert to probabilities
        probs = softmax(scaled_logits, dim=-1)

        # Keep only top-k per position — reduce storage 128k→50
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

        return {
            "top_k_indices": top_indices.cpu().numpy().tolist(),
            "top_k_probs":   top_probs.cpu().float().numpy().tolist(),
            "temperature":   self.temperature,
        }

    def run_inference_dataset(
        self,
        records: list[dict],
        output_path: str,
        batch_size: int = 4,
        max_length: int = 1024,
        top_k: int = 50,
    ) -> Path:
        """
        Run teacher inference over entire dataset.
        Writes soft labels to JSONL — one line per training example.
        Resume-safe: skips already-processed examples.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Resume: count already processed
        processed = 0
        if output_file.exists():
            with open(output_file) as f:
                processed = sum(1 for line in f if line.strip())
            logger.info(f"Resuming from {processed}/{len(records)}")

        logger.info(f"Generating soft labels for {len(records) - processed} records")

        with open(output_file, "a") as f_out:
            for i in tqdm(range(processed, len(records), batch_size),
                         desc="Teacher inference"):
                batch_records = records[i:i+batch_size]
                prompts = [format_alpaca_prompt(r) for r in batch_records]

                # Tokenize batch
                tokenized = self.tokenizer(
                    prompts,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_ids      = tokenized["input_ids"].cuda()
                attention_mask = tokenized["attention_mask"].cuda()

                soft_labels = self.generate_soft_labels(
                    input_ids, attention_mask, top_k=top_k
                )

                # Write one record per example
                for j, record in enumerate(batch_records):
                    output_record = {
                        "record_idx":    i + j,
                        "instruction":   record.get("instruction", ""),
                        "input":         record.get("input", ""),
                        "output":        record.get("output", ""),
                        "soft_labels": {
                            "top_k_indices": soft_labels["top_k_indices"][j],
                            "top_k_probs":   soft_labels["top_k_probs"][j],
                            "temperature":   self.temperature,
                            "top_k":         top_k,
                        }
                    }
                    f_out.write(json.dumps(output_record) + "\n")
                    f_out.flush()

        logger.info(f"Soft labels written to {output_file}")
        return output_file


def upload_soft_labels_to_s3(
    local_path: Path,
    s3_bucket: str,
    s3_key: str,
    region: str = "us-east-1",
):
    """Upload soft label file to S3 for student training."""
    s3 = boto3.client("s3", region_name=region)
    size_gb = local_path.stat().st_size / 1024**3
    logger.info(f"Uploading soft labels ({size_gb:.1f}GB) to s3://{s3_bucket}/{s3_key}")
    s3.upload_file(
        str(local_path), s3_bucket, s3_key,
        ExtraArgs={"ServerSideEncryption": "AES256"}
    )
    logger.info("Upload complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-s3-prefix",  default="base-models/llama3-70b")
    parser.add_argument("--dataset-s3-key",     default="datasets/v1/sre_ops_train_v1.jsonl")
    parser.add_argument("--output-s3-key",      default="soft-labels/llama3-70b-T4-top50.jsonl")
    parser.add_argument("--s3-bucket",          default="sre-llmops-artifacts")
    parser.add_argument("--local-output",       default="/tmp/soft-labels/soft_labels.jsonl")
    parser.add_argument("--local-model-dir",    default="/tmp/teacher-model")
    parser.add_argument("--temperature",        type=float, default=4.0)
    parser.add_argument("--top-k",              type=int,   default=50)
    parser.add_argument("--batch-size",         type=int,   default=4)
    parser.add_argument("--max-length",         type=int,   default=1024)
    args = parser.parse_args()

    # Download teacher model
    model_dir = download_model_from_s3(
        args.s3_bucket, args.teacher_s3_prefix, args.local_model_dir
    )

    # Load dataset
    records = load_dataset_from_s3(args.s3_bucket, args.dataset_s3_key)
    logger.info(f"Loaded {len(records)} records")

    # Run teacher inference
    teacher = TeacherInference(
        model_path=str(model_dir),
        temperature=args.temperature,
    )

    output_path = teacher.run_inference_dataset(
        records=records,
        output_path=args.local_output,
        batch_size=args.batch_size,
        max_length=args.max_length,
        top_k=args.top_k,
    )

    # Upload to S3
    upload_soft_labels_to_s3(
        output_path,
        args.s3_bucket,
        args.output_s3_key,
    )


if __name__ == "__main__":
    main()
