"""
P5c — DeepSpeed ZeRO Distributed Training
Integrates DeepSpeed ZeRO Stage 2/3 with HuggingFace Trainer.

Launch command (single node, 4 GPUs):
  deepspeed --num_gpus=4 src/training/distributed/deepspeed_train.py \
            --deepspeed_config=zero3

Launch command (multi-node, 2 nodes x 4 GPUs):
  deepspeed --num_nodes=2 --num_gpus=4 \
            --master_addr=$MASTER_ADDR \
            --hostfile=hostfile \
            src/training/distributed/deepspeed_train.py \
            --deepspeed_config=zero3

Key advantage over FSDP:
  DeepSpeed integrates directly with HuggingFace Trainer
  — no manual training loop needed.
  ZeRO-3 config passed to TrainingArguments.
"""

import os
import json
import argparse
import tempfile
import torch
import mlflow
import boto3
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

from src.training.data_utils import load_dataset_from_s3, tokenize_dataset
from src.training.model_utils import download_model_from_s3
from src.training.distributed.deepspeed_config import get_zero2_config, get_zero3_config
from src.utils.logger import logger


def load_model_for_deepspeed(model_path: str, use_qlora: bool = True):
    """
    Load model for DeepSpeed training.

    With QLoRA + DeepSpeed ZeRO-3:
    - Base model in 4-bit NF4 (frozen)
    - LoRA adapters in bf16 (trainable)
    - ZeRO-3 shards the LoRA adapter params across GPUs
    - Effective memory: ~4GB base + ~0.5GB adapters per GPU

    Without QLoRA (full fine-tuning with ZeRO-3):
    - All params in bf16
    - ZeRO-3 shards everything
    - Effective memory: 16GB / num_gpus per GPU
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
            # Map to specific GPU — DeepSpeed handles distribution
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            use_cache=False,
        )
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        model.print_trainable_parameters()

    return model, tokenizer


def train_deepspeed(args):
    """
    DeepSpeed training using HuggingFace Trainer.
    Trainer handles distributed setup, gradient sync, checkpointing.
    DeepSpeed ZeRO config passed via TrainingArguments.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main    = local_rank == 0

    # Generate DeepSpeed config
    total_batch  = args.batch_size * args.grad_accum * int(os.environ.get("WORLD_SIZE", 1))
    warmup_steps = int((len(range(args.num_epochs)) * 100) * args.warmup_ratio)

    if args.zero_stage == 2:
        ds_config = get_zero2_config(total_batch, args.batch_size, args.learning_rate, warmup_steps)
    else:
        ds_config = get_zero3_config(total_batch, args.batch_size, args.learning_rate, warmup_steps,
                                     cpu_offload=args.cpu_offload)

    # Write config to temp file — TrainingArguments requires file path
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(ds_config, f)
        ds_config_path = f.name

    if is_main:
        logger.info(f"DeepSpeed ZeRO Stage {args.zero_stage} config written to {ds_config_path}")

    # Download model (rank 0 only, others wait via barrier in download_model_from_s3)
    model_dir = download_model_from_s3(
        args.s3_bucket, args.model_s3_prefix, args.local_model_dir
    )

    # Load model
    model, tokenizer = load_model_for_deepspeed(str(model_dir), use_qlora=args.use_qlora)

    # Load datasets
    train_records = load_dataset_from_s3(args.s3_bucket, args.dataset_s3_key)
    val_records   = load_dataset_from_s3(args.s3_bucket, args.val_s3_key)
    train_dataset = tokenize_dataset(train_records, tokenizer, args.max_length)
    val_dataset   = tokenize_dataset(val_records,   tokenizer, args.max_length)

    # MLflow setup (rank 0 only)
    if is_main:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.experiment_name)

    # TrainingArguments with DeepSpeed
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=False,
        group_by_length=True,
        dataloader_num_workers=2,
        deepspeed=ds_config_path,  # ZeRO config injected here
        local_rank=local_rank,
    )

    with mlflow.start_run(run_name=f"deepspeed-zero{args.zero_stage}") as run:
        if is_main:
            mlflow.log_params({
                "base_model":   "llama3-8b",
                "zero_stage":   args.zero_stage,
                "use_qlora":    args.use_qlora,
                "cpu_offload":  args.cpu_offload,
                "world_size":   os.environ.get("WORLD_SIZE", 1),
                "lora_r":       16,
                "learning_rate": args.learning_rate,
                "num_epochs":   args.num_epochs,
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
        eval_results = trainer.evaluate()

        if is_main:
            mlflow.log_metrics({
                "train_loss":   train_result.training_loss,
                "eval_loss":    eval_results["eval_loss"],
                "train_runtime": train_result.metrics["train_runtime"],
                "samples_per_second": train_result.metrics["train_samples_per_second"],
            })

            # Save and upload adapter
            adapter_dir = f"{args.output_dir}/deepspeed-adapter"
            trainer.save_model(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)

            s3 = boto3.client("s3", region_name="us-east-1")
            for f in Path(adapter_dir).rglob("*"):
                if f.is_file():
                    s3_key = f"{args.output_s3_prefix}/{f.relative_to(adapter_dir)}"
                    s3.upload_file(str(f), args.s3_bucket, s3_key,
                                  ExtraArgs={"ServerSideEncryption": "AES256"})

            mlflow.log_param("adapter_s3", f"s3://{args.s3_bucket}/{args.output_s3_prefix}/")
            logger.info(f"DeepSpeed training complete. Run: {run.info.run_id}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-s3-prefix",   default="base-models/llama3-8b")
    parser.add_argument("--dataset-s3-key",    default="datasets/v1/sre_ops_train_v1.jsonl")
    parser.add_argument("--val-s3-key",        default="datasets/v1/sre_ops_val_v1.jsonl")
    parser.add_argument("--output-s3-prefix",  default="adapters/llama3-8b-deepspeed-v1")
    parser.add_argument("--experiment-name",   default="sre-deepspeed-p5")
    parser.add_argument("--s3-bucket",         default="sre-llmops-artifacts")
    parser.add_argument("--mlflow-uri",        default="http://mlflow.mlflow.svc.cluster.local:5000")
    parser.add_argument("--local-model-dir",   default="/tmp/model")
    parser.add_argument("--output-dir",        default="/tmp/training-output")
    parser.add_argument("--zero-stage",        type=int,   default=3, choices=[2, 3])
    parser.add_argument("--use-qlora",         action="store_true", default=True)
    parser.add_argument("--cpu-offload",       action="store_true", default=False)
    parser.add_argument("--num-epochs",        type=int,   default=3)
    parser.add_argument("--batch-size",        type=int,   default=4)
    parser.add_argument("--grad-accum",        type=int,   default=4)
    parser.add_argument("--learning-rate",     type=float, default=2e-4)
    parser.add_argument("--max-length",        type=int,   default=1024)
    parser.add_argument("--warmup-ratio",      type=float, default=0.03)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_deepspeed(args)
