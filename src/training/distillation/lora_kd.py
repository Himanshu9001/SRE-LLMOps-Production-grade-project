"""
P7 — LoRA-KD: Combined LoRA Fine-tuning + Knowledge Distillation
Single training pass: student gets domain adaptation + teacher knowledge simultaneously.

Why combined vs sequential:
  Sequential: fine-tune with LoRA → then distill
              Two separate training runs, potential catastrophic forgetting
              Student forgets fine-tuning signal when distillation starts

  Combined:   single training loop with combined loss
              L_total = α * L_CE(hard_label) + (1-α) * L_KD(soft_label)
              Student learns BOTH simultaneously — better convergence

Memory efficiency:
  QLoRA student (4-bit base + bf16 adapters): ~5GB
  Teacher soft labels (pre-computed, stored in S3): 50-200GB
  Teacher NOT needed at training time — offline distillation
  Most memory-efficient path to teacher-quality output

Architecture:
  Llama 3 70B (Teacher) → offline soft labels → S3
                                                   ↓
  Llama 3 8B (Student, 4-bit) + LoRA adapters ← load from S3
                    ↓
  Combined loss: α*CE + (1-α)*KD
                    ↓
  Distilled + domain-adapted LoRA adapter → S3
"""

import os
import json
import torch
import torch.nn.functional as F
import boto3
import argparse
import mlflow
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel,
)

from src.training.distillation.student_distillation import (
    DistillationDataset,
    kl_divergence_loss,
)
from src.training.model_utils import download_model_from_s3
from src.utils.logger import logger


class LoRAKDTrainer:
    """
    Combined LoRA + Knowledge Distillation trainer.

    Key design decisions:
    1. Load base model in 4-bit NF4 (frozen)
    2. Apply LoRA adapters on q/v/k/o projections (trainable, bf16)
    3. Load pre-computed soft labels from S3 (no teacher at runtime)
    4. Training loop: forward → CE loss + KD loss → backward → update adapters only

    Compared to standard QLoRA (P4):
    - Same memory footprint
    - Same trainable parameter count
    - Additional KD loss term from teacher soft labels
    - Better generalization: student learns teacher's reasoning patterns

    Compared to pure distillation (P6):
    - Same KD loss computation
    - Hard label CE loss also included (α > 0)
    - LoRA structure means only adapter weights updated
    - More stable training than full fine-tuning with KD
    """

    def __init__(self, args):
        self.args = args

        # Download student base model from S3
        logger.info("=== Loading student model (QLoRA) ===")
        model_dir = download_model_from_s3(
            args.s3_bucket,
            args.student_s3_prefix,
            args.local_model_dir,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token     = self.tokenizer.eos_token
            self.tokenizer.pad_token_id  = self.tokenizer.eos_token_id

        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=True,
        )

        # Apply LoRA — only adapter weights are trainable
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def _download_soft_labels(self) -> str:
        """Download soft labels from S3 to local /tmp."""
        local_path = "/tmp/soft-labels/soft_labels.jsonl"
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        if Path(local_path).exists():
            logger.info(f"Soft labels already downloaded: {local_path}")
            return local_path

        logger.info(f"Downloading soft labels from s3://{self.args.s3_bucket}/{self.args.soft_labels_s3_key}")
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.download_file(self.args.s3_bucket, self.args.soft_labels_s3_key, local_path)
        logger.info(f"Soft labels downloaded: {Path(local_path).stat().st_size / 1024**2:.0f} MB")
        return local_path

    def _build_dataset(self, soft_labels_path: str) -> DistillationDataset:
        """Build distillation dataset from soft labels file."""
        return DistillationDataset(
            soft_labels_path=soft_labels_path,
            tokenizer=self.tokenizer,
            max_length=self.args.max_length,
        )

    def _build_optimizer_scheduler(self, num_training_steps: int):
        """Build AdamW optimizer + cosine LR scheduler with warmup."""
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.args.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )

        warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        return optimizer, scheduler

    def train(self):
        """
        Main LoRA-KD training loop.

        Each step:
        1. Forward pass → get student logits
        2. CE loss: cross-entropy against hard ground-truth labels
        3. KD loss: KL divergence against teacher soft labels
        4. Combined loss = α*CE + (1-α)*KD
        5. Backward → update only LoRA adapter weights
        """
        args = self.args

        # Download soft labels
        soft_labels_path = self._download_soft_labels()

        # Build dataset and dataloader
        dataset = self._build_dataset(soft_labels_path)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        total_steps = len(dataloader) * args.num_epochs
        optimizer, scheduler = self._build_optimizer_scheduler(total_steps)

        # MLflow setup
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.experiment_name)

        with mlflow.start_run(
            run_name=f"lora-kd-alpha{args.alpha}-T{args.temperature}-r{args.lora_r}"
        ) as run:
            mlflow.log_params({
                "method":          "lora_kd",
                "teacher":         "llama3-70b",
                "student":         "llama3-8b",
                "alpha":           args.alpha,
                "temperature":     args.temperature,
                "lora_r":          args.lora_r,
                "lora_alpha":      args.lora_alpha,
                "learning_rate":   args.learning_rate,
                "num_epochs":      args.num_epochs,
                "batch_size":      args.batch_size,
                "effective_batch": args.batch_size * args.grad_accum,
                "max_length":      args.max_length,
                "train_samples":   len(dataset),
                "soft_labels_key": args.soft_labels_s3_key,
            })

            global_step  = 0
            best_loss    = float("inf")
            self.model.train()

            for epoch in range(args.num_epochs):
                epoch_metrics = {
                    "ce_loss":    0.0,
                    "kd_loss":    0.0,
                    "total_loss": 0.0,
                }

                for step, batch in enumerate(dataloader):
                    # Move to GPU
                    input_ids          = batch["input_ids"].cuda()
                    attention_mask     = batch["attention_mask"].cuda()
                    labels             = batch["labels"].cuda()
                    soft_label_indices = batch["soft_label_indices"].cuda()
                    soft_label_probs   = batch["soft_label_probs"].cuda()

                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                    # CE loss — standard cross-entropy from HuggingFace
                    ce_loss = outputs.loss

                    # KD loss — KL divergence against teacher soft labels
                    kd_loss = kl_divergence_loss(
                        student_logits=outputs.logits,
                        soft_label_indices=soft_label_indices,
                        soft_label_probs=soft_label_probs,
                        temperature=args.temperature,
                        attention_mask=attention_mask,
                    )

                    # Combined LoRA-KD loss
                    total_loss = (
                        args.alpha * ce_loss +
                        (1 - args.alpha) * kd_loss
                    ) / args.grad_accum

                    total_loss.backward()

                    # Gradient accumulation
                    if (step + 1) % args.grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 1.0
                        )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1

                        # Log every 10 steps
                        if global_step % 10 == 0:
                            actual_total = total_loss.item() * args.grad_accum
                            lr = scheduler.get_last_lr()[0]

                            logger.info(
                                f"E{epoch+1} S{global_step} | "
                                f"CE={ce_loss.item():.4f} "
                                f"KD={kd_loss.item():.4f} "
                                f"Total={actual_total:.4f} "
                                f"LR={lr:.2e}"
                            )
                            mlflow.log_metrics({
                                "ce_loss":    ce_loss.item(),
                                "kd_loss":    kd_loss.item(),
                                "total_loss": actual_total,
                                "lr":         lr,
                            }, step=global_step)

                    epoch_metrics["ce_loss"]    += ce_loss.item()
                    epoch_metrics["kd_loss"]    += kd_loss.item()
                    epoch_metrics["total_loss"] += total_loss.item() * args.grad_accum

                # Epoch summary
                n = len(dataloader)
                avg = {k: v/n for k, v in epoch_metrics.items()}
                logger.info(
                    f"Epoch {epoch+1}/{args.num_epochs} — "
                    f"CE={avg['ce_loss']:.4f} "
                    f"KD={avg['kd_loss']:.4f} "
                    f"Total={avg['total_loss']:.4f}"
                )
                mlflow.log_metrics({
                    f"epoch_{k}": v for k, v in avg.items()
                }, step=epoch)

                # Save best checkpoint
                if avg["total_loss"] < best_loss:
                    best_loss = avg["total_loss"]
                    self._save_and_upload_adapter(
                        suffix="best",
                        run_id=run.info.run_id,
                    )

            # Final save
            self._save_and_upload_adapter(
                suffix="final",
                run_id=run.info.run_id,
            )
            mlflow.log_metric("best_loss", best_loss)
            logger.info(
                f"LoRA-KD training complete. "
                f"Best loss: {best_loss:.4f} "
                f"Run: {run.info.run_id}"
            )

    def _save_and_upload_adapter(self, suffix: str, run_id: str):
        """Save LoRA adapter locally and upload to S3."""
        adapter_dir = Path(f"{self.args.output_dir}/lora-kd-adapter-{suffix}")
        adapter_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(adapter_dir))
        self.tokenizer.save_pretrained(str(adapter_dir))

        # Upload to S3
        s3     = boto3.client("s3", region_name="us-east-1")
        prefix = f"{self.args.output_s3_prefix}/{suffix}"

        for f in adapter_dir.rglob("*"):
            if f.is_file() and not f.name.startswith("."):
                s3_key = f"{prefix}/{f.relative_to(adapter_dir)}"
                s3.upload_file(
                    str(f), self.args.s3_bucket, s3_key,
                    ExtraArgs={"ServerSideEncryption": "AES256"}
                )

        logger.info(f"Adapter saved: s3://{self.args.s3_bucket}/{prefix}/")
        mlflow.log_param(f"adapter_s3_{suffix}", f"s3://{self.args.s3_bucket}/{prefix}/")


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA-KD: Combined LoRA + Knowledge Distillation")
    parser.add_argument("--student-s3-prefix",   default="base-models/llama3-8b")
    parser.add_argument("--soft-labels-s3-key",  default="soft-labels/llama3-70b-T4-top50.jsonl")
    parser.add_argument("--output-s3-prefix",    default="adapters/llama3-8b-lora-kd-v1")
    parser.add_argument("--experiment-name",     default="sre-lora-kd-p7")
    parser.add_argument("--s3-bucket",           default="sre-llmops-artifacts")
    parser.add_argument("--mlflow-uri",          default="http://mlflow.mlflow.svc.cluster.local:5000")
    parser.add_argument("--local-model-dir",     default="/tmp/model")
    parser.add_argument("--output-dir",          default="/tmp/lora-kd-output")
    parser.add_argument("--alpha",               type=float, default=0.5)
    parser.add_argument("--temperature",         type=float, default=4.0)
    parser.add_argument("--lora-r",              type=int,   default=16)
    parser.add_argument("--lora-alpha",          type=int,   default=32)
    parser.add_argument("--num-epochs",          type=int,   default=3)
    parser.add_argument("--batch-size",          type=int,   default=4)
    parser.add_argument("--grad-accum",          type=int,   default=4)
    parser.add_argument("--learning-rate",       type=float, default=2e-4)
    parser.add_argument("--max-length",          type=int,   default=1024)
    parser.add_argument("--warmup-ratio",        type=float, default=0.03)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = LoRAKDTrainer(args)
    trainer.train()
