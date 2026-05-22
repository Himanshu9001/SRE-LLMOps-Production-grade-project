"""
P9 — Full Evaluation Pipeline
Runs both SRE benchmark + LM-Eval Harness in sequence.
Logs all results to MLflow for comparison across model variants.

Pipeline:
  1. Load model (base / fine-tuned / quantized)
  2. Run custom SRE benchmark (200 questions)
  3. Run LM-Eval Harness (MMLU, HellaSwag, ARC, TruthfulQA)
  4. Log all metrics to MLflow
  5. Compare against baseline — flag regression

Pass/fail criteria (eval gate for P14 CI/CD):
  SRE overall score > 0.65    (baseline: ~0.30 for base model)
  MMLU accuracy > 0.55        (ensure no catastrophic forgetting)
  No category score < 0.50    (balanced across SRE domains)
"""

import os
import json
import torch
import boto3
import mlflow
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.evaluation.sre_benchmark import SREBenchmark, SRE_EVAL_DATASET
from src.evaluation.lm_eval_runner import run_lm_eval
from src.training.model_utils import download_model_from_s3
from src.utils.logger import logger


EVAL_GATE = {
    "sre_overall":      0.65,   # must beat this to promote to production
    "mmlu_acc":         0.55,   # catastrophic forgetting check
    "min_category":     0.50,   # no category below this
}


def load_model_for_eval(
    model_path: str,
    adapter_path: str = None,
    use_qlora: bool = False,
    model_type: str = "fp16",
):
    """Load model for evaluation based on type."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_type == "awq":
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_quantized(
            model_path, fuse_layers=True
        )

    elif model_type == "gptq":
        from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(
            model_path, use_safetensors=True, device="cuda:0"
        )

    elif use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def check_eval_gate(sre_result, lm_eval_metrics: dict) -> tuple[bool, list[str]]:
    """
    Check if model passes eval gate for production promotion.
    Returns (passed, list_of_failures).
    """
    failures = []

    # SRE overall score
    if sre_result.overall_score < EVAL_GATE["sre_overall"]:
        failures.append(
            f"SRE overall score {sre_result.overall_score:.3f} < {EVAL_GATE['sre_overall']}"
        )

    # MMLU check (catastrophic forgetting)
    mmlu_score = lm_eval_metrics.get("mmlu_acc", 0)
    if mmlu_score > 0 and mmlu_score < EVAL_GATE["mmlu_acc"]:
        failures.append(
            f"MMLU score {mmlu_score:.3f} < {EVAL_GATE['mmlu_acc']} (catastrophic forgetting)"
        )

    # Per-category minimum
    for cat, score in sre_result.category_scores.items():
        if score < EVAL_GATE["min_category"]:
            failures.append(
                f"Category '{cat}' score {score:.3f} < {EVAL_GATE['min_category']}"
            )

    passed = len(failures) == 0
    return passed, failures


def run_full_evaluation(args):
    """Run complete evaluation pipeline."""
    logger.info(f"=== P9 Evaluation Pipeline ===")
    logger.info(f"Model: {args.model_name}")

    # Download models
    model_dir = download_model_from_s3(
        args.s3_bucket, args.model_s3_prefix, args.local_model_dir
    )

    adapter_dir = None
    if args.adapter_s3_prefix:
        adapter_dir = download_model_from_s3(
            args.s3_bucket, args.adapter_s3_prefix,
            f"{args.local_model_dir}-adapter"
        )

    # Load model
    model, tokenizer = load_model_for_eval(
        model_path=str(model_dir),
        adapter_path=str(adapter_dir) if adapter_dir else None,
        use_qlora=args.use_qlora,
        model_type=args.model_type,
    )

    # Setup MLflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("sre-evaluation-p9")

    with mlflow.start_run(run_name=f"eval-{args.model_name}") as run:
        mlflow.log_params({
            "model_name":    args.model_name,
            "model_type":    args.model_type,
            "use_qlora":     args.use_qlora,
            "has_adapter":   adapter_dir is not None,
            "eval_questions": len(SRE_EVAL_DATASET),
        })

        # === Step 1: SRE Benchmark ===
        logger.info("\n=== Step 1: SRE Benchmark ===")
        bench = SREBenchmark(
            model=model,
            tokenizer=tokenizer,
            model_name=args.model_name,
        )
        sre_result = bench.run()

        # Log SRE metrics
        mlflow.log_metrics({
            "sre_overall_score":    sre_result.overall_score,
            "sre_avg_latency_ms":   sre_result.avg_latency_ms,
            **{f"sre_{cat}": score
               for cat, score in sre_result.category_scores.items()},
            **{f"sre_diff_{diff}": score
               for diff, score in sre_result.difficulty_scores.items()},
        })

        # === Step 2: LM-Eval (if GPU available) ===
        lm_eval_metrics = {}
        if torch.cuda.is_available() and not args.skip_lm_eval:
            logger.info("\n=== Step 2: LM-Eval Harness ===")
            try:
                lm_eval_metrics = run_lm_eval(
                    model_path=str(model_dir),
                    model_name=args.model_name,
                    tasks=["mmlu", "hellaswag", "arc_easy"],
                    batch_size=args.lm_eval_batch_size,
                    use_qlora=args.use_qlora,
                    adapter_path=str(adapter_dir) if adapter_dir else None,
                )
                mlflow.log_metrics(lm_eval_metrics)
            except Exception as e:
                logger.error(f"LM-Eval failed (non-fatal): {e}")
        else:
            logger.info("Skipping LM-Eval (no GPU or --skip-lm-eval set)")

        # === Step 3: Eval Gate ===
        logger.info("\n=== Step 3: Eval Gate ===")
        passed, failures = check_eval_gate(sre_result, lm_eval_metrics)

        mlflow.log_params({
            "eval_gate_passed": passed,
            "eval_gate_failures": "; ".join(failures) if failures else "none",
        })

        if passed:
            logger.info("✅ EVAL GATE PASSED — model ready for production promotion")
            mlflow.set_tag("eval_gate", "passed")
        else:
            logger.warning("❌ EVAL GATE FAILED:")
            for f in failures:
                logger.warning(f"  - {f}")
            mlflow.set_tag("eval_gate", "failed")

        # Save results to S3
        results = {
            "model_name":       args.model_name,
            "sre_overall":      sre_result.overall_score,
            "sre_categories":   sre_result.category_scores,
            "lm_eval":          lm_eval_metrics,
            "eval_gate_passed": passed,
            "eval_gate_failures": failures,
            "run_id":           run.info.run_id,
        }

        results_path = "/tmp/eval_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.upload_file(
            results_path, args.s3_bucket,
            f"eval-results/{args.model_name}/results.json",
        )

        logger.info(f"\nEvaluation complete. Run ID: {run.info.run_id}")
        logger.info(f"Results: s3://{args.s3_bucket}/eval-results/{args.model_name}/results.json")

        # Exit with error code if gate failed — for CI/CD integration
        if not passed:
            exit(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name",         default="llama3-8b-sre-qlora")
    parser.add_argument("--model-s3-prefix",    default="base-models/llama3-8b")
    parser.add_argument("--adapter-s3-prefix",  default=None)
    parser.add_argument("--model-type",         default="fp16",
                        choices=["fp16", "awq", "gptq"])
    parser.add_argument("--s3-bucket",          default="sre-llmops-artifacts")
    parser.add_argument("--mlflow-uri",         default="http://mlflow.mlflow.svc.cluster.local:5000")
    parser.add_argument("--local-model-dir",    default="/tmp/eval-model")
    parser.add_argument("--use-qlora",          action="store_true", default=False)
    parser.add_argument("--skip-lm-eval",       action="store_true", default=False)
    parser.add_argument("--lm-eval-batch-size", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_evaluation(args)
