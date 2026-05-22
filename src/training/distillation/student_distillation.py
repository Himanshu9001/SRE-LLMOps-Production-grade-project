"""
P6 — Step 2: Student Training with Knowledge Distillation
Train Llama 3 8B student against stored teacher soft labels.

Loss function:
  L_total = α * L_CE(student, hard_label) + (1-α) * L_KD(student, teacher)

  L_CE:  standard cross-entropy against ground truth tokens
         Teaches the student what the correct answer IS
         
  L_KD:  KL divergence between student and teacher distributions
         KL(teacher || student) = Σ p_teacher * log(p_teacher / p_student)
         Teaches the student WHY the teacher chose that token
         Soft labels carry inter-token similarity — richer signal

  α (alpha):
    α=0.0: pure distillation — only learn from teacher
    α=1.0: pure CE — ignore teacher, standard fine-tuning
    α=0.5: balanced — recommended starting point
    
  Temperature during student training:
    Must match teacher's temperature T
    Student logits also scaled by T before KL divergence
    Without matching T, distributions are on different scales
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
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from src.utils.logger import logger


class DistillationDataset(Dataset):
    """
    Dataset that pairs tokenized inputs with teacher soft labels.
    Loads soft labels from S3 JSONL — one record per training example.

    Memory layout:
    - input_ids, attention_mask, labels: standard tokenized fields
    - soft_label_indices, soft_label_probs: top-k teacher distribution
    """

    def __init__(
        self,
        soft_labels_path: str,
        tokenizer,
        max_length: int = 1024,
        max_records: int = None,
    ):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.records    = []

        logger.info(f"Loading distillation dataset from {soft_labels_path}")

        with open(soft_labels_path) as f:
            for i, line in enumerate(f):
                if max_records and i >= max_records:
                    break
                if line.strip():
                    self.records.append(json.loads(line))

        logger.info(f"Loaded {len(self.records)} distillation records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        # Reconstruct Alpaca prompt
        prompt = (
            f"### Instruction:\n{record['instruction']}\n\n"
            f"### Input:\n{record['input']}\n\n"
            f"### Response:\n{record['output']}"
        )

        # Tokenize
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids      = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # Labels: mask padding positions with -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Soft labels: top-k indices and probabilities from teacher
        soft = record["soft_labels"]
        seq_len = input_ids.size(0)
        top_k   = len(soft["top_k_indices"][0])

        # Pad/truncate soft labels to match tokenized sequence length
        sl_indices = soft["top_k_indices"][:seq_len]
        sl_probs   = soft["top_k_probs"][:seq_len]

        # Pad if soft labels shorter than tokenized sequence
        while len(sl_indices) < seq_len:
            sl_indices.append([0] * top_k)
            sl_probs.append([1.0 / top_k] * top_k)

        return {
            "input_ids":          input_ids,
            "attention_mask":     attention_mask,
            "labels":             labels,
            "soft_label_indices": torch.tensor(sl_indices, dtype=torch.long),
            "soft_label_probs":   torch.tensor(sl_probs,   dtype=torch.float),
        }


def kl_divergence_loss(
    student_logits: torch.Tensor,
    soft_label_indices: torch.Tensor,
    soft_label_probs: torch.Tensor,
    temperature: float = 4.0,
    attention_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute KL divergence between student and teacher distributions.

    student_logits:     [batch, seq_len, vocab_size]
    soft_label_indices: [batch, seq_len, top_k]
    soft_label_probs:   [batch, seq_len, top_k]

    Key implementation details:
    1. Scale student logits by temperature (must match teacher's T)
    2. Convert student logits to log-probabilities
    3. Gather student log-probs at teacher's top-k positions
    4. Compute KL: sum(p_teacher * (log p_teacher - log p_student))
    5. Mask out padding positions
    6. Scale by T² — compensates for T in gradient magnitude
       (Hinton et al. 2015: multiply KD loss by T² to balance with CE)
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    top_k = soft_label_indices.shape[-1]

    # Temperature scale student logits
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # Gather student log-probs at teacher's top-k token positions
    # Shape: [batch, seq_len, top_k]
    student_log_probs_topk = student_log_probs.gather(
        dim=-1,
        index=soft_label_indices,
    )

    # Teacher probs are already scaled by T during generation
    # Clamp to avoid log(0)
    teacher_probs = soft_label_probs.clamp(min=1e-8)
    teacher_log_probs = teacher_probs.log()

    # KL divergence: sum(p * (log p - log q))
    # = sum(p * log p) - sum(p * log q)
    kl_loss = (teacher_probs * (teacher_log_probs - student_log_probs_topk)).sum(dim=-1)
    # Shape: [batch, seq_len]

    # Mask padding positions
    if attention_mask is not None:
        kl_loss = kl_loss * attention_mask.float()
        kl_loss = kl_loss.sum() / attention_mask.float().sum()
    else:
        kl_loss = kl_loss.mean()

    # Scale by T² (Hinton et al. 2015)
    return kl_loss * (temperature ** 2)


def train_with_distillation(args):
    """
    Full distillation training loop.
    Combines CE loss (hard labels) + KL loss (soft labels from teacher).
    """
    logger.info(f"Starting distillation training (alpha={args.alpha}, T={args.temperature})")

    # Load tokenizer and model in QLoRA
    tokenizer = AutoTokenizer.from_pretrained(args.student_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.student_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load distillation dataset
    train_dataset = DistillationDataset(
        soft_labels_path=args.soft_labels_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    total_steps  = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * 0.03)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # MLflow tracking
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=f"distillation-alpha{args.alpha}-T{args.temperature}") as run:
        mlflow.log_params({
            "teacher_model":   "llama3-70b",
            "student_model":   "llama3-8b",
            "alpha":           args.alpha,
            "temperature":     args.temperature,
            "lora_r":          args.lora_r,
            "learning_rate":   args.learning_rate,
            "num_epochs":      args.num_epochs,
            "distillation":    "offline-topk50",
        })

        global_step = 0
        model.train()

        for epoch in range(args.num_epochs):
            epoch_ce_loss  = 0.0
            epoch_kd_loss  = 0.0
            epoch_total    = 0.0

            for step, batch in enumerate(train_loader):
                # Move to GPU
                input_ids          = batch["input_ids"].cuda()
                attention_mask     = batch["attention_mask"].cuda()
                labels             = batch["labels"].cuda()
                soft_label_indices = batch["soft_label_indices"].cuda()
                soft_label_probs   = batch["soft_label_probs"].cuda()

                # Forward pass — get full logits for KD loss
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                # CE loss from HuggingFace (standard cross-entropy)
                ce_loss = outputs.loss

                # KD loss — KL divergence against teacher soft labels
                kd_loss = kl_divergence_loss(
                    student_logits=outputs.logits,
                    soft_label_indices=soft_label_indices,
                    soft_label_probs=soft_label_probs,
                    temperature=args.temperature,
                    attention_mask=attention_mask,
                )

                # Combined loss
                # α controls balance: 0=pure KD, 1=pure CE
                total_loss = args.alpha * ce_loss + (1 - args.alpha) * kd_loss
                total_loss = total_loss / args.grad_accum

                total_loss.backward()

                if (step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % 10 == 0:
                        logger.info(
                            f"Epoch {epoch+1} Step {global_step} "
                            f"CE: {ce_loss.item():.4f} "
                            f"KD: {kd_loss.item():.4f} "
                            f"Total: {total_loss.item()*args.grad_accum:.4f} "
                            f"LR: {scheduler.get_last_lr()[0]:.2e}"
                        )
                        mlflow.log_metrics({
                            "ce_loss":    ce_loss.item(),
                            "kd_loss":    kd_loss.item(),
                            "total_loss": total_loss.item() * args.grad_accum,
                            "lr":         scheduler.get_last_lr()[0],
                        }, step=global_step)

                epoch_ce_loss += ce_loss.item()
                epoch_kd_loss += kd_loss.item()
                epoch_total   += total_loss.item() * args.grad_accum

            # Epoch summary
            n = len(train_loader)
            logger.info(
                f"Epoch {epoch+1} — "
                f"CE: {epoch_ce_loss/n:.4f} "
                f"KD: {epoch_kd_loss/n:.4f} "
                f"Total: {epoch_total/n:.4f}"
            )
            mlflow.log_metrics({
                "epoch_ce_loss":    epoch_ce_loss / n,
                "epoch_kd_loss":    epoch_kd_loss / n,
                "epoch_total_loss": epoch_total / n,
            }, step=epoch)

        # Save adapter
        adapter_dir = f"{args.output_dir}/distilled-adapter"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        # Upload to S3
        s3 = boto3.client("s3", region_name="us-east-1")
        for f in Path(adapter_dir).rglob("*"):
            if f.is_file():
                s3_key = f"{args.output_s3_prefix}/{f.relative_to(adapter_dir)}"
                s3.upload_file(str(f), args.s3_bucket, s3_key,
                              ExtraArgs={"ServerSideEncryption": "AES256"})

        mlflow.log_param("adapter_s3", f"s3://{args.s3_bucket}/{args.output_s3_prefix}/")
        logger.info(f"Distillation complete. Run: {run.info.run_id}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-model-path", default="/tmp/model")
    parser.add_argument("--soft-labels-path",   default="/tmp/soft-labels/soft_labels.jsonl")
    parser.add_argument("--output-s3-prefix",   default="adapters/llama3-8b-distilled-v1")
    parser.add_argument("--experiment-name",    default="sre-distillation-p6")
    parser.add_argument("--s3-bucket",          default="sre-llmops-artifacts")
    parser.add_argument("--mlflow-uri",         default="http://mlflow.mlflow.svc.cluster.local:5000")
    parser.add_argument("--output-dir",         default="/tmp/distillation-output")
    parser.add_argument("--alpha",              type=float, default=0.5)
    parser.add_argument("--temperature",        type=float, default=4.0)
    parser.add_argument("--lora-r",             type=int,   default=16)
    parser.add_argument("--lora-alpha",         type=int,   default=32)
    parser.add_argument("--num-epochs",         type=int,   default=3)
    parser.add_argument("--batch-size",         type=int,   default=4)
    parser.add_argument("--grad-accum",         type=int,   default=4)
    parser.add_argument("--learning-rate",      type=float, default=2e-4)
    parser.add_argument("--max-length",         type=int,   default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_with_distillation(args)
