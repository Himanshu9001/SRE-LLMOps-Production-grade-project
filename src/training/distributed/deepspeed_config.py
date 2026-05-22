"""
DeepSpeed ZeRO Stage configurations.
ZeRO (Zero Redundancy Optimizer) eliminates memory redundancy in data parallelism.

ZeRO Stage 1: Partition optimizer states across GPUs
              ~4x memory reduction for optimizer states
ZeRO Stage 2: Partition optimizer states + gradients
              ~8x memory reduction
ZeRO Stage 3: Partition optimizer states + gradients + parameters
              ~Nx memory reduction (N = number of GPUs)
              Enables training models larger than single GPU VRAM

Memory breakdown for Llama 3 8B (bf16):
  Parameters:       16GB
  Gradients:        16GB
  Optimizer states: 48GB (Adam: 2x param size in fp32)
  Total:            80GB — impossible on single 24GB A10G

With ZeRO-3 across 4 GPUs:
  Per GPU:          80GB / 4 = 20GB ✅ fits on A10G
"""


def get_zero2_config(
    train_batch_size: int,
    micro_batch_size: int,
    learning_rate: float,
    warmup_steps: int,
) -> dict:
    """
    ZeRO Stage 2 configuration.
    Best for: models that fit in GPU memory, want faster training.
    Partitions gradients and optimizer states, NOT parameters.
    Lower communication overhead than ZeRO-3.
    """
    return {
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": train_batch_size // micro_batch_size,
        "gradient_clipping": 1.0,

        "bf16": {
            "enabled": True           # A10G natively supports bf16
        },

        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,    # 200MB bucket for all-gather
            "overlap_comm": True,             # overlap communication with computation
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,     # contiguous gradient buffers — faster
            "round_robin_gradients": True,    # distribute gradients evenly across GPUs
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": learning_rate,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.01,
            }
        },

        "scheduler": {
            "type": "WarmupCosineAnnealing",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": learning_rate,
                "warmup_num_steps": warmup_steps,
            }
        },

        "steps_per_print": 10,
        "wall_clock_breakdown": False,
    }


def get_zero3_config(
    train_batch_size: int,
    micro_batch_size: int,
    learning_rate: float,
    warmup_steps: int,
    cpu_offload: bool = False,
) -> dict:
    """
    ZeRO Stage 3 configuration.
    Best for: models that DON'T fit on single GPU.
    Partitions everything: params + gradients + optimizer states.
    Higher communication overhead — params gathered before each forward/backward.

    cpu_offload=True: further reduces GPU memory by offloading
    optimizer states AND parameters to CPU RAM.
    Use when GPU memory is insufficient even with ZeRO-3.
    Cost: 3-5x slower training due to CPU↔GPU data movement.
    """
    config = {
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": train_batch_size // micro_batch_size,
        "gradient_clipping": 1.0,

        "bf16": {
            "enabled": True
        },

        "zero_optimization": {
            "stage": 3,

            # Parameter handling
            "stage3_gather_16bit_weights_on_model_save": True,
            # Consolidates sharded params to fp16 when saving checkpoint
            # Without this, checkpoint is split across GPU files

            "stage3_param_persistence_threshold": 1e4,
            # Params smaller than this stay on GPU (not sharded)
            # Avoids overhead of sharding tiny embedding layers

            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_prefetch_bucket_size": 5e7,   # 50MB prefetch buffer

            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e7,

            # CPU offloading (optional)
            "offload_optimizer": {
                "device": "cpu" if cpu_offload else "none",
                "pin_memory": True,   # pinned memory for faster CPU↔GPU
            },
            "offload_param": {
                "device": "cpu" if cpu_offload else "none",
                "pin_memory": True,
            },
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": learning_rate,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.01,
            }
        },

        "scheduler": {
            "type": "WarmupCosineAnnealing",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": learning_rate,
                "warmup_num_steps": warmup_steps,
            }
        },

        "steps_per_print": 10,
        "wall_clock_breakdown": True,  # ZeRO-3: always profile communication
    }

    return config
