"""
P9b — LM-Eval Harness Runner
Standard benchmarks: MMLU, HellaSwag, ARC, TruthfulQA.
Compares base model vs fine-tuned vs quantized variants.

EleutherAI LM-Eval Harness:
  Industry standard evaluation framework.
  400+ tasks, reproducible, widely cited.
  Used by Llama, Mistral, Falcon teams for model cards.

Tasks selected:
  MMLU:        57 academic subjects — general knowledge
               SRE-relevant: computer science, engineering
  HellaSwag:   sentence completion — general reasoning
  ARC Easy:    grade school science — basic reasoning
  TruthfulQA:  factual accuracy — important for SRE runbooks
               Model shouldn't hallucinate kubectl flags

Why run standard benchmarks in addition to custom SRE:
  Custom SRE: shows task-specific improvement
  Standard:   shows model didn't catastrophically forget
              general knowledge during fine-tuning (forgetting check)
"""

import subprocess
import json
import mlflow
import boto3
import argparse
from pathlib import Path

from src.utils.logger import logger


LM_EVAL_TASKS = [
    "mmlu",           # 57 subjects, 5-shot
    "hellaswag",      # sentence completion, 10-shot
    "arc_easy",       # science QA, 25-shot
    "truthfulqa_mc1", # factual accuracy, 0-shot
]

LM_EVAL_FEW_SHOTS = {
    "mmlu":           5,
    "hellaswag":      10,
    "arc_easy":       25,
    "truthfulqa_mc1": 0,
}


def run_lm_eval(
    model_path: str,
    model_name: str,
    tasks: list[str] = None,
    output_dir: str = "/tmp/lm-eval-results",
    batch_size: int = 8,
    model_type: str = "hf",
    use_qlora: bool = False,
    adapter_path: str = None,
) -> dict:
    """
    Run LM-Eval Harness on specified model and tasks.

    model_type options:
      "hf":      standard HuggingFace model
      "hf-causal-experimental": for quantized models
      "vllm":    use vLLM backend for faster evaluation

    Returns dict of task -> metric scores.
    """
    if tasks is None:
        tasks = LM_EVAL_TASKS

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = f"{output_dir}/{model_name.replace('/', '_')}_results.json"

    logger.info(f"Running LM-Eval on {model_name}")
    logger.info(f"Tasks: {tasks}")

    # Build lm_eval command
    task_str    = ",".join(tasks)
    few_shots   = ",".join(str(LM_EVAL_FEW_SHOTS.get(t, 0)) for t in tasks)

    cmd = [
        "lm_eval",
        "--model", model_type,
        "--model_args", f"pretrained={model_path},dtype=float16",
        "--tasks", task_str,
        "--num_fewshot", "5",    # default 5-shot for most tasks
        "--batch_size", str(batch_size),
        "--output_path", output_path,
        "--log_samples",
    ]

    # Add LoRA adapter if specified
    if use_qlora and adapter_path:
        cmd[-5] = f"pretrained={model_path},dtype=float16,peft={adapter_path},load_in_4bit=True"

    logger.info(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"LM-Eval failed: {result.stderr}")
        raise RuntimeError(f"LM-Eval failed: {result.stderr[:500]}")

    # Parse results
    with open(output_path) as f:
        results = json.load(f)

    # Extract key metrics
    metrics = {}
    for task in tasks:
        task_results = results.get("results", {}).get(task, {})
        if "acc,none" in task_results:
            metrics[f"{task}_acc"] = round(task_results["acc,none"], 4)
        elif "acc_norm,none" in task_results:
            metrics[f"{task}_acc_norm"] = round(task_results["acc_norm,none"], 4)

    logger.info(f"LM-Eval results for {model_name}:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    return metrics


def compare_models(
    model_configs: list[dict],
    mlflow_uri: str,
    s3_bucket: str,
) -> dict:
    """
    Run LM-Eval comparison across multiple model variants.
    Logs all results to MLflow for comparison.

    model_configs: list of {name, path, type, adapter_path}
    """
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("sre-lm-eval-p9")

    all_results = {}

    with mlflow.start_run(run_name="lm-eval-comparison"):
        for config in model_configs:
            name    = config["name"]
            path    = config["path"]
            mtype   = config.get("type", "hf")
            adapter = config.get("adapter_path")

            logger.info(f"\n{'='*50}")
            logger.info(f"Evaluating: {name}")

            try:
                metrics = run_lm_eval(
                    model_path=path,
                    model_name=name,
                    model_type=mtype,
                    adapter_path=adapter,
                    use_qlora=adapter is not None,
                )

                all_results[name] = metrics

                # Log to MLflow with model name prefix
                prefixed = {f"{name}/{k}": v for k, v in metrics.items()}
                mlflow.log_metrics(prefixed)

            except Exception as e:
                logger.error(f"Evaluation failed for {name}: {e}")
                all_results[name] = {"error": str(e)}

        # Print comparison table
        _print_comparison(all_results)

    return all_results


def _print_comparison(results: dict):
    """Print formatted comparison table."""
    if not results:
        return

    # Get all metric keys
    all_metrics = set()
    for metrics in results.values():
        all_metrics.update(k for k in metrics.keys() if k != "error")
    all_metrics = sorted(all_metrics)

    logger.info(f"\n{'='*80}")
    logger.info("LM-EVAL COMPARISON")
    logger.info(f"{'='*80}")

    # Header
    header = f"{'Model':<30}" + "".join(f"{m:<20}" for m in all_metrics)
    logger.info(header)
    logger.info("-" * len(header))

    for model_name, metrics in results.items():
        row = f"{model_name:<30}"
        for metric in all_metrics:
            val = metrics.get(metric, "N/A")
            if isinstance(val, float):
                row += f"{val:<20.4f}"
            else:
                row += f"{str(val):<20}"
        logger.info(row)

    logger.info(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-bucket",   default="sre-llmops-artifacts")
    parser.add_argument("--mlflow-uri",  default="http://mlflow.mlflow.svc.cluster.local:5000")
    parser.add_argument("--base-path",   default="/tmp/models/base")
    parser.add_argument("--ft-path",     default="/tmp/models/finetuned")
    parser.add_argument("--awq-path",    default="/tmp/models/awq")
    parser.add_argument("--adapter-path", default="/tmp/models/adapter")
    args = parser.parse_args()

    model_configs = [
        {
            "name":    "llama3-8b-base",
            "path":    args.base_path,
            "type":    "hf",
        },
        {
            "name":         "llama3-8b-sre-qlora",
            "path":         args.ft_path,
            "type":         "hf",
            "adapter_path": args.adapter_path,
        },
        {
            "name":  "llama3-8b-awq-4bit",
            "path":  args.awq_path,
            "type":  "hf-causal-experimental",
        },
    ]

    compare_models(model_configs, args.mlflow_uri, args.s3_bucket)


if __name__ == "__main__":
    main()
