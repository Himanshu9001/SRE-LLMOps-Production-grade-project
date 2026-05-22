"""
Dataset loading and tokenization for SRE QLoRA fine-tuning.
Loads Alpaca-format JSONL from S3, formats as instruction prompt,
tokenizes with Llama 3 tokenizer, returns HuggingFace Dataset.
"""

import json
import boto3
import tempfile
from pathlib import Path
from datasets import Dataset
from transformers import PreTrainedTokenizer
from src.utils.logger import logger


# Alpaca prompt template — wraps instruction + input + output
# This is the exact format the model learns to follow
ALPACA_PROMPT = """Below is an instruction from an SRE engineer describing a problem. Write a response that solves the problem with specific commands and steps.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_PROMPT_INFERENCE = """Below is an instruction from an SRE engineer describing a problem. Write a response that solves the problem with specific commands and steps.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


def load_dataset_from_s3(
    s3_bucket: str,
    s3_key: str,
    region: str = "us-east-1"
) -> list[dict]:
    """Download JSONL dataset from S3 and return as list of dicts."""
    s3 = boto3.client("s3", region_name=region)

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".jsonl", delete=False) as f:
        s3.download_fileobj(s3_bucket, s3_key, f)
        tmp_path = f.name

    records = []
    with open(tmp_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info(f"Loaded {len(records)} records from s3://{s3_bucket}/{s3_key}")
    return records


def format_alpaca_prompt(record: dict, include_response: bool = True) -> str:
    """
    Format a single record as an Alpaca instruction prompt.
    During training: include_response=True — model learns to predict output.
    During inference: include_response=False — model generates the response.
    """
    if include_response:
        return ALPACA_PROMPT.format(
            instruction=record.get("instruction", ""),
            input=record.get("input", ""),
            output=record.get("output", ""),
        )
    else:
        return ALPACA_PROMPT_INFERENCE.format(
            instruction=record.get("instruction", ""),
            input=record.get("input", ""),
        )


def tokenize_dataset(
    records: list[dict],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 1024,
) -> Dataset:
    """
    Tokenize Alpaca-formatted records for causal LM training.

    Key design decisions:
    - max_length=1024: covers 95% of SRE responses (avg ~200 tokens)
    - truncation=True: truncate long sequences rather than OOM
    - padding="max_length": pad all sequences to same length for batching
    - Labels = input_ids with padding masked to -100:
      CrossEntropyLoss ignores -100 positions — model only learns
      to predict non-padded tokens
    """
    prompts = [format_alpaca_prompt(r) for r in records]

    logger.info(f"Tokenizing {len(prompts)} prompts (max_length={max_length})")

    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,    # return lists, not tensors — Dataset handles conversion
    )

    # Mask padding tokens in labels so loss ignores them
    labels = []
    for ids, attention_mask in zip(tokenized["input_ids"], tokenized["attention_mask"]):
        label = [
            token_id if mask == 1 else -100
            for token_id, mask in zip(ids, attention_mask)
        ]
        labels.append(label)

    tokenized["labels"] = labels

    dataset = Dataset.from_dict(tokenized)
    logger.info(f"Dataset ready: {len(dataset)} samples, {max_length} max tokens")
    return dataset
