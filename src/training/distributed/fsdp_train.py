"""
P5a — PyTorch FSDP Distributed Training
Fully Sharded Data Parallel across multiple GPUs/nodes.

Single node:  torchrun --nproc_per_node=4 -m src.training.distributed.fsdp_train
Multi node:   torchrun --nnodes=2 --nproc_per_node=4 
                       --master_addr=$MASTER_ADDR 
                       --master_port=$MASTER_PORT
                       -m src.training.distributed.fsdp_train

ZeRO Stage mapping:
  FSDP FULL_SHARD    = ZeRO Stage 3 (params + grads + optimizer states)
  FSDP SHARD_GRAD_OP = ZeRO Stage 2 (grads + optimizer states only)
  FSDP NO_SHARD      = DDP (no sharding, standard data parallel)
"""

import os
import argparse
import functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from peft import LoraConfig, get_peft_model, TaskType
import mlflow
import boto3
from pathlib import Path

from src.training.data_utils import load_dataset_from_s3, tokenize_dataset
from src.training.model_utils import download_model_from_s3
from src.utils.logger import logger


def setup_distributed():
    """
    Initialize distributed process group.
    NCCL backend: optimized for GPU-GPU communication via NVLink/InfiniBand.
    Called once per process — each GPU runs its own process.
    """
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    logger.info(
        f"Rank {dist.get_rank()}/{dist.get_world_size()} — "
        f"GPU {int(os.environ['LOCAL_RANK'])}"
    )


def cleanup_distributed():
    """Destroy process group — must be called at end of training."""
    dist.destroy_process_group()


def get_fsdp_model(model, sharding_strategy: str = "FULL_SHARD") -> FSDP:
    """
    Wrap model with FSDP for distributed training.

    Sharding strategies:
    - FULL_SHARD:    ZeRO-3 — shards params + grads + optimizer states
                     Best memory efficiency, highest communication overhead
                     Use when model barely fits across all GPUs
    - SHARD_GRAD_OP: ZeRO-2 — shards grads + optimizer states only
                     Params replicated, better for smaller models
                     Lower communication overhead than FULL_SHARD
    - NO_SHARD:      DDP equivalent — no sharding, all params replicated
                     Fastest training, highest memory usage

    Auto wrap policy: automatically decides which submodules to shard.
    For Llama: wrap each LlamaDecoderLayer independently — each layer
    becomes a separate FSDP unit with its own all-gather/reduce-scatter.
    """
    strategy_map = {
        "FULL_SHARD":    ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD":      ShardingStrategy.NO_SHARD,
    }

    # Mixed precision: bf16 for params and gradients
    # bfloat16 chosen over float16: larger dynamic range, no overflow risk
    # A10G natively supports bf16 operations
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,   # gradient reduction in bf16
        buffer_dtype=torch.bfloat16,
    )

    # Auto wrap policy: shard LlamaDecoderLayer independently
    # Each transformer layer ~130MB for Llama 3 8B
    # FSDP shards each layer across all GPUs
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )

    fsdp_model = FSDP(
        model,
        sharding_strategy=strategy_map[sharding_strategy],
        mixed_precision=mixed_precision,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        # BACKWARD_PRE: prefetch next layer's params while computing
        # gradients for current layer — overlaps communication + compute
        cpu_offload=CPUOffload(offload_params=False),
        # cpu_offload=True: move params to CPU when not in use
        # Use only when GPU memory is severely constrained
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        # Prevents too many all-gather ops queued simultaneously
        # Reduces peak memory at cost of slight throughput reduction
    )

    return fsdp_model


def save_fsdp_checkpoint(
    model: FSDP,
    tokenizer,
    output_dir: str,
    rank: int,
):
    """
    Save FSDP model checkpoint.
    FSDP shards params across GPUs — must consolidate before saving.
    Only rank 0 writes to disk — all ranks participate in consolidation.

    FULL_STATE_DICT: consolidates all shards to rank 0 — requires
    enough CPU RAM to hold full model (16GB for Llama 3 8B fp16).
    """
    save_policy = {
        "state_dict_type": StateDictType.FULL_STATE_DICT,
        "offload_to_cpu": True,   # move from GPU to CPU during consolidation
        "rank0_only": True,        # only rank 0 holds the full state dict
    }

    with FSDP.state_dict_type(model, **save_policy):
        state_dict = model.state_dict()

    if rank == 0:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir, state_dict=state_dict)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Checkpoint saved to {output_dir}")


def train_fsdp(args):
    """
    Main FSDP training function.
    Each process trains on its shard of data.
    Gradients synchronized via all-reduce after each backward pass.
    """
    setup_distributed()
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    is_main    = rank == 0

    if is_main:
        logger.info(f"Starting FSDP training: {world_size} GPUs")
        logger.info(f"Sharding strategy: {args.sharding_strategy}")

    # Download model only on rank 0, others wait
    # Prevents N processes simultaneously downloading from S3
    if is_main:
        model_dir = download_model_from_s3(
            args.s3_bucket, args.model_s3_prefix, args.local_model_dir
        )
    dist.barrier()  # All ranks wait until rank 0 finishes download

    if not is_main:
        model_dir = args.local_model_dir

    # Load tokenizer on all ranks
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in bf16 — FSDP handles distribution
    # Do NOT use device_map="auto" with FSDP — FSDP manages placement
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        use_cache=False,  # disable KV cache during training — incompatible with gradient checkpointing
    )

    # Apply LoRA before FSDP wrapping
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing BEFORE FSDP wrapping
    # Recomputes activations during backward instead of storing them
    # Trades compute for memory — reduces VRAM by ~30-40%
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Wrap with FSDP
    model = get_fsdp_model(model, args.sharding_strategy)

    if is_main:
        logger.info(f"Model wrapped with FSDP ({args.sharding_strategy})")

    # Load and shard dataset
    # Each rank loads full dataset but DataLoader shards by rank
    if is_main:
        logger.info("Loading datasets...")

    train_records = load_dataset_from_s3(args.s3_bucket, args.dataset_s3_key)
    val_records   = load_dataset_from_s3(args.s3_bucket, args.val_s3_key)
    train_dataset = tokenize_dataset(train_records, tokenizer, args.max_length)
    val_dataset   = tokenize_dataset(val_records,   tokenizer, args.max_length)

    # DistributedSampler: each rank sees different subset of data
    # Ensures no data duplication across ranks
    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,   # pinned memory: faster CPU→GPU transfer
    )

    # Optimizer — only on trainable (LoRA) params
    # paged AdamW: offloads optimizer states to CPU RAM when GPU is full
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95),  # standard LLM training betas
    )

    # Cosine LR scheduler with warmup
    total_steps   = len(train_loader) * args.num_epochs
    warmup_steps  = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=warmup_steps/total_steps,
        anneal_strategy="cos",
    )

    # MLflow tracking — only on rank 0
    if is_main:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.experiment_name)
        run = mlflow.start_run(
            run_name=f"fsdp-{args.sharding_strategy}-{world_size}gpu"
        )
        mlflow.log_params({
            "base_model":         "llama3-8b",
            "sharding_strategy":  args.sharding_strategy,
            "world_size":         world_size,
            "lora_r":             args.lora_r,
            "learning_rate":      args.learning_rate,
            "num_epochs":         args.num_epochs,
            "batch_size":         args.batch_size,
            "effective_batch":    args.batch_size * args.grad_accum * world_size,
            "max_length":         args.max_length,
        })

    # Training loop
    model.train()
    global_step = 0

    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)  # ensures different shuffle per epoch
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            # Move batch to current GPU
            batch = {k: v.cuda(local_rank) for k, v in batch.items()
                    if isinstance(v, torch.Tensor)}

            outputs = model(**batch)
            loss    = outputs.loss / args.grad_accum  # normalize for accumulation

            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                # Gradient clipping — prevents exploding gradients
                # FSDP handles all-reduce of gradients automatically
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main and global_step % 10 == 0:
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"Epoch {epoch+1} Step {global_step} "
                        f"Loss {loss.item() * args.grad_accum:.4f} "
                        f"LR {lr:.2e}"
                    )
                    mlflow.log_metrics({
                        "train_loss": loss.item() * args.grad_accum,
                        "learning_rate": lr,
                    }, step=global_step)

            epoch_loss += loss.item()

        if is_main:
            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} complete — avg loss: {avg_loss:.4f}")
            mlflow.log_metric("epoch_loss", avg_loss, step=epoch)

    # Save checkpoint — all ranks participate, only rank 0 writes
    if is_main:
        logger.info("Saving checkpoint...")

    save_fsdp_checkpoint(
        model, tokenizer,
        output_dir=f"{args.output_dir}/fsdp-adapter",
        rank=rank,
    )

    # Upload to S3 — only rank 0
    if is_main:
        s3 = boto3.client("s3", region_name="us-east-1")
        adapter_dir = Path(f"{args.output_dir}/fsdp-adapter")
        for f in adapter_dir.rglob("*"):
            if f.is_file():
                s3_key = f"{args.output_s3_prefix}/{f.relative_to(adapter_dir)}"
                s3.upload_file(str(f), args.s3_bucket, s3_key,
                              ExtraArgs={"ServerSideEncryption": "AES256"})
        mlflow.log_param("adapter_s3", f"s3://{args.s3_bucket}/{args.output_s3_prefix}/")
        mlflow.end_run()
        logger.info("FSDP training complete")

    cleanup_distributed()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-s3-prefix",   default="base-models/llama3-8b")
    parser.add_argument("--dataset-s3-key",    default="datasets/v1/sre_ops_train_v1.jsonl")
    parser.add_argument("--val-s3-key",        default="datasets/v1/sre_ops_val_v1.jsonl")
    parser.add_argument("--output-s3-prefix",  default="adapters/llama3-8b-fsdp-v1")
    parser.add_argument("--experiment-name",   default="sre-fsdp-p5")
    parser.add_argument("--s3-bucket",         default="sre-llmops-artifacts")
    parser.add_argument("--mlflow-uri",        default="http://mlflow.mlflow.svc.cluster.local:5000")
    parser.add_argument("--local-model-dir",   default="/tmp/model")
    parser.add_argument("--output-dir",        default="/tmp/training-output")
    parser.add_argument("--sharding-strategy", default="FULL_SHARD",
                        choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"])
    parser.add_argument("--num-epochs",        type=int,   default=3)
    parser.add_argument("--batch-size",        type=int,   default=4)
    parser.add_argument("--grad-accum",        type=int,   default=4)
    parser.add_argument("--learning-rate",     type=float, default=2e-4)
    parser.add_argument("--max-length",        type=int,   default=1024)
    parser.add_argument("--lora-r",            type=int,   default=16)
    parser.add_argument("--lora-alpha",        type=int,   default=32)
    parser.add_argument("--warmup-ratio",      type=float, default=0.03)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_fsdp(args)
