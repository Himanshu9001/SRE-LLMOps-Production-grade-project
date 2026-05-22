"""
P4 — QLoRA Fine-tuning Script
Supports both GPU (QLoRA 4-bit) and CPU (fp32 smoke test).
GPU: Llama 3 8B with 4-bit NF4 + LoRA
CPU: small models (opt-125m) for pipeline validation
"""

import os
import argparse
import boto3
import mlflow
import torch
from pathlib import Path
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from src.training.model_utils import download_model_from_s3, load_model_for_qlora, load_model_cpu
from src.training.data_utils import load_dataset_from_s3, tokenize_dataset
from src.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-s3-prefix",   default="base-models/llama3-8b")
    parser.add_argument("--dataset-s3-key",    default="datasets/v1/sre_ops_train_v1.jsonl")
    parser.add_argument("--val-s3-key",        default="datasets/v1/sre_ops_val_v1.jsonl")
    parser.add_argument("--output-s3-prefix",  default="adapters/llama3-8b-sre-v1")
    parser.add_argument("--experiment-name",   default="sre-qlora-p4")
    parser.add_argument("--s3-bucket",         default="sre-llmops-artifacts")
    parser.add_argument("--mlflow-uri",        default="http://mlflow.mlflow.svc.cluster.local:5000")
    parser.add_argument("--local-model-dir",   default="/tmp/model")
    parser.add_argument("--output-dir",        default="/tmp/training-output")
    parser.add_argument("--num-epochs",        type=int,   default=3)
    parser.add_argument("--batch-size",        type=int,   default=4)
    parser.add_argument("--grad-accum-steps",  type=int,   default=4)
    parser.add_argument("--learning-rate",     type=float, default=2e-4)
    parser.add_argument("--max-length",        type=int,   default=1024)
    parser.add_argument("--lora-r",            type=int,   default=16)
    parser.add_argument("--lora-alpha",        type=int,   default=32)
    parser.add_argument("--warmup-ratio",      type=float, default=0.03)
    return parser.parse_args()


def upload_adapter_to_s3(local_dir, s3_bucket, s3_prefix, region="us-east-1"):
    s3 = boto3.client("s3", region_name=region)
    uploaded = 0
    for file_path in Path(local_dir).rglob("*"):
        if file_path.is_file():
            s3_key = f"{s3_prefix}/{file_path.relative_to(local_dir)}"
            s3.upload_file(str(file_path), s3_bucket, s3_key,
                          ExtraArgs={"ServerSideEncryption": "AES256"})
            uploaded += 1
    logger.info(f"Uploaded {uploaded} files to s3://{s3_bucket}/{s3_prefix}/")


def main():
    args = parse_args()
    is_smoke_test = os.environ.get("SMOKE_TEST", "false").lower() == "true"
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f}GB VRAM)")
    else:
        if not is_smoke_test:
            raise RuntimeError("No GPU — set SMOKE_TEST=true for CPU training")
        logger.warning("No GPU detected — running CPU smoke test")

    # Step 1: Download model from S3
    logger.info("=== Step 1: Downloading model from S3 ===")
    model_local_dir = download_model_from_s3(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.model_s3_prefix,
        local_dir=args.local_model_dir,
    )

    # Step 2: Load model
    logger.info("=== Step 2: Loading model ===")
    if use_gpu:
        model, tokenizer = load_model_for_qlora(
            model_path=str(model_local_dir),
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )
    else:
        # CPU smoke test: load in fp32, apply LoRA without quantization
        model, tokenizer = load_model_cpu(
            model_path=str(model_local_dir),
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )

    # Step 3: Load datasets — use only 100 samples for smoke test
    logger.info("=== Step 3: Loading datasets ===")
    train_records = load_dataset_from_s3(args.s3_bucket, args.dataset_s3_key)
    val_records   = load_dataset_from_s3(args.s3_bucket, args.val_s3_key)

    if is_smoke_test:
        train_records = train_records[:100]
        val_records   = val_records[:20]
        logger.info("Smoke test: using 100 train / 20 val samples")

    train_dataset = tokenize_dataset(train_records, tokenizer, args.max_length)
    val_dataset   = tokenize_dataset(val_records,   tokenizer, args.max_length)

    # Step 4: MLflow
    logger.info("=== Step 4: Configuring MLflow ===")
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    # Step 5: Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=use_gpu,       # bf16 only on GPU
        fp16=False,
        optim="paged_adamw_32bit" if use_gpu else "adamw_torch",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,   # 0 for smoke test stability
        remove_unused_columns=False,
        no_cuda=not use_gpu,
    )

    # Step 6: Train
    logger.info("=== Step 6: Training ===")
    with mlflow.start_run(run_name=f"{'smoke-test' if is_smoke_test else 'qlora'}-r{args.lora_r}") as run:
        mlflow.log_params({
            "base_model":    args.model_s3_prefix.split("/")[-1],
            "lora_r":        args.lora_r,
            "lora_alpha":    args.lora_alpha,
            "learning_rate": args.learning_rate,
            "num_epochs":    args.num_epochs,
            "batch_size":    args.batch_size,
            "max_length":    args.max_length,
            "train_samples": len(train_dataset),
            "val_samples":   len(val_dataset),
            "device":        "gpu" if use_gpu else "cpu",
            "smoke_test":    is_smoke_test,
        })

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer, model=model, padding=True, pad_to_multiple_of=8,
            ),
        )

        train_result = trainer.train()

        mlflow.log_metrics({
            "train_loss":     train_result.training_loss,
            "train_runtime":  train_result.metrics["train_runtime"],
        })

        eval_results = trainer.evaluate()
        mlflow.log_metrics({
            "eval_loss":       eval_results["eval_loss"],
            "eval_perplexity": torch.exp(torch.tensor(eval_results["eval_loss"])).item(),
        })

        adapter_local_dir = f"{args.output_dir}/final-adapter"
        model.save_pretrained(adapter_local_dir)
        tokenizer.save_pretrained(adapter_local_dir)

        upload_adapter_to_s3(adapter_local_dir, args.s3_bucket, args.output_s3_prefix)
        mlflow.log_param("adapter_s3_path", f"s3://{args.s3_bucket}/{args.output_s3_prefix}/")

        logger.info(f"Done. Run ID: {run.info.run_id}")
        logger.info(f"Adapter: s3://{args.s3_bucket}/{args.output_s3_prefix}/")


if __name__ == "__main__":
    main()
