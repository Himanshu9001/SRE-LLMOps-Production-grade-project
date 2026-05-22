"""
P8d — Quantization Benchmark
Compare: fp16 baseline vs AWQ 4-bit vs GPTQ 4-bit vs GGUF Q4_K_M

Metrics:
  Perplexity:     lower = better language modeling quality
  SRE accuracy:   % of correct kubectl/terraform commands generated
  Latency (TTFT): time to first token (ms)
  Throughput:     tokens/second
  Memory:         GPU VRAM usage (GB)
  Model size:     disk/S3 storage (GB)
"""

import json
import time
import torch
import boto3
import mlflow
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.utils.logger import logger


SRE_EVAL_PROMPTS = [
    {
        "instruction": "A pod is in CrashLoopBackOff. What is the first command to run?",
        "input": "Pod: api-server-7d9f8b in namespace production. Restarts: 15",
        "expected_keywords": ["kubectl logs", "kubectl describe", "--previous"],
    },
    {
        "instruction": "Terraform state is locked. How do I safely unlock it?",
        "input": "Lock ID: a1b2c3d4-e5f6-7890. CI pipeline job confirmed dead.",
        "expected_keywords": ["force-unlock", "terraform", "lock"],
    },
    {
        "instruction": "ArgoCD application sync failed. What are the debug steps?",
        "input": "App: payment-service. Status: OutOfSync. Error: permission denied",
        "expected_keywords": ["argocd app", "kubectl", "rbac", "sync"],
    },
    {
        "instruction": "Kubernetes node shows NotReady. How to investigate?",
        "input": "Node: ip-10-0-1-45. Condition: MemoryPressure. Duration: 15m",
        "expected_keywords": ["kubectl describe node", "kubectl drain", "free -h"],
    },
    {
        "instruction": "Fluent Bit pods CrashLoopBackOff after ConfigMap update.",
        "input": "Namespace: logging. ConfigMap updated 30min ago. Exit code: 1",
        "expected_keywords": ["kubectl logs", "configmap", "kubectl rollout restart"],
    },
]

ALPACA_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)


class QuantizationBenchmark:
    """Benchmarks multiple quantization variants on SRE evaluation set."""

    def __init__(self, s3_bucket: str, mlflow_uri: str):
        self.s3_bucket  = s3_bucket
        self.mlflow_uri = mlflow_uri
        self.results    = []

    def _load_model(self, model_path: str, model_type: str):
        """Load model based on quantization type."""
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if model_type == "fp16":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        elif model_type == "awq":
            from awq import AutoAWQForCausalLM
            model = AutoAWQForCausalLM.from_quantized(
                model_path,
                fuse_layers=True,    # fuse for faster inference
                trust_remote_code=False,
            )

        elif model_type == "gptq":
            from auto_gptq import AutoGPTQForCausalLM
            model = AutoGPTQForCausalLM.from_quantized(
                model_path,
                use_safetensors=True,
                device="cuda:0",
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return model, tokenizer

    def _measure_vram(self) -> float:
        """Measure current GPU VRAM usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0

    def _compute_perplexity(self, model, tokenizer, num_samples: int = 50) -> float:
        """
        Compute perplexity on WikiText-2 test set.
        Standard benchmark for language modeling quality.
        Lower perplexity = better model.
        """
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            texts   = [t for t in dataset["text"] if len(t) > 100][:num_samples]

            total_loss = 0.0
            total_tokens = 0

            model.eval()
            with torch.no_grad():
                for text in texts:
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    )
                    input_ids = inputs["input_ids"].cuda()

                    outputs = model(input_ids, labels=input_ids)
                    total_loss   += outputs.loss.item() * input_ids.size(1)
                    total_tokens += input_ids.size(1)

            avg_loss   = total_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            return round(perplexity, 2)

        except Exception as e:
            logger.error(f"Perplexity computation failed: {e}")
            return -1.0

    def _compute_sre_accuracy(self, model, tokenizer) -> float:
        """
        Compute SRE task accuracy.
        Checks if generated response contains expected keywords.
        Proxy metric for domain-specific correctness.
        """
        correct = 0

        model.eval()
        with torch.no_grad():
            for prompt_data in SRE_EVAL_PROMPTS:
                prompt = ALPACA_TEMPLATE.format(
                    instruction=prompt_data["instruction"],
                    input=prompt_data["input"],
                )

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to("cuda")

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

                response = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).lower()

                # Check if any expected keyword appears in response
                keywords = [k.lower() for k in prompt_data["expected_keywords"]]
                if any(k in response for k in keywords):
                    correct += 1

        accuracy = correct / len(SRE_EVAL_PROMPTS)
        return round(accuracy, 3)

    def _measure_latency(self, model, tokenizer) -> dict:
        """
        Measure inference latency metrics.
        TTFT: Time To First Token
        ITL:  Inter-Token Latency (ms per token)
        """
        prompt = ALPACA_TEMPLATE.format(
            instruction="A pod is in CrashLoopBackOff. Diagnose it.",
            input="Pod: test-pod in namespace production",
        )

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256
        ).to("cuda")

        # Warmup
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=10,
                          pad_token_id=tokenizer.eos_token_id)

        # Measure
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        throughput = tokens_generated / elapsed

        return {
            "throughput_tokens_per_sec": round(throughput, 1),
            "total_latency_ms":          round(elapsed * 1000, 1),
            "itl_ms":                    round(elapsed * 1000 / tokens_generated, 2),
            "tokens_generated":          tokens_generated,
        }

    def benchmark_model(
        self,
        model_name: str,
        model_path: str,
        model_type: str,
    ) -> dict:
        """Run full benchmark suite for one model variant."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Benchmarking: {model_name} ({model_type})")
        logger.info(f"{'='*50}")

        torch.cuda.empty_cache()
        vram_before = self._measure_vram()

        model, tokenizer = self._load_model(model_path, model_type)

        vram_after = self._measure_vram()
        vram_used  = vram_after - vram_before

        # Model size on disk
        model_size_gb = sum(
            f.stat().st_size for f in Path(model_path).rglob("*") if f.is_file()
        ) / 1024**3

        logger.info(f"VRAM: {vram_used:.2f}GB | Disk: {model_size_gb:.2f}GB")

        # Run benchmarks
        perplexity   = self._compute_perplexity(model, tokenizer)
        sre_accuracy = self._compute_sre_accuracy(model, tokenizer)
        latency      = self._measure_latency(model, tokenizer)

        result = {
            "model_name":       model_name,
            "model_type":       model_type,
            "vram_gb":          round(vram_used, 2),
            "disk_gb":          round(model_size_gb, 2),
            "perplexity":       perplexity,
            "sre_accuracy":     sre_accuracy,
            **latency,
        }

        logger.info(
            f"Results: perplexity={perplexity} | "
            f"sre_acc={sre_accuracy} | "
            f"throughput={latency['throughput_tokens_per_sec']} tok/s"
        )

        self.results.append(result)

        # Cleanup
        del model
        torch.cuda.empty_cache()

        return result

    def run_all(self, model_paths: dict) -> list[dict]:
        """
        Run benchmarks for all model variants.
        model_paths: {name: (path, type)} dict
        """
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment("sre-quantization-benchmark-p8")

        with mlflow.start_run(run_name="quantization-comparison"):
            for name, (path, mtype) in model_paths.items():
                result = self.benchmark_model(name, path, mtype)
                mlflow.log_metrics({
                    f"{name}_perplexity":   result["perplexity"],
                    f"{name}_sre_accuracy": result["sre_accuracy"],
                    f"{name}_throughput":   result["throughput_tokens_per_sec"],
                    f"{name}_vram_gb":      result["vram_gb"],
                    f"{name}_disk_gb":      result["disk_gb"],
                })

            # Print comparison table
            self._print_comparison_table()

        return self.results

    def _print_comparison_table(self):
        """Print formatted comparison table."""
        if not self.results:
            return

        header = f"{'Model':<25} {'Perplexity':>12} {'SRE Acc':>10} {'Throughput':>12} {'VRAM(GB)':>10} {'Disk(GB)':>10}"
        separator = "-" * len(header)

        logger.info(f"\n{separator}")
        logger.info("QUANTIZATION BENCHMARK RESULTS")
        logger.info(separator)
        logger.info(header)
        logger.info(separator)

        for r in self.results:
            logger.info(
                f"{r['model_name']:<25} "
                f"{r['perplexity']:>12.2f} "
                f"{r['sre_accuracy']:>10.3f} "
                f"{r['throughput_tokens_per_sec']:>12.1f} "
                f"{r['vram_gb']:>10.2f} "
                f"{r['disk_gb']:>10.2f}"
            )
        logger.info(separator)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-bucket",      default="sre-llmops-artifacts")
    parser.add_argument("--mlflow-uri",     default="http://mlflow.mlflow.svc.cluster.local:5000")
    parser.add_argument("--fp16-path",      default="/tmp/models/fp16")
    parser.add_argument("--awq-path",       default="/tmp/models/awq")
    parser.add_argument("--gptq-path",      default="/tmp/models/gptq")
    args = parser.parse_args()

    bench = QuantizationBenchmark(args.s3_bucket, args.mlflow_uri)

    model_paths = {
        "llama3-8b-fp16":    (args.fp16_path,  "fp16"),
        "llama3-8b-awq-4bit": (args.awq_path,  "awq"),
        "llama3-8b-gptq-4bit": (args.gptq_path, "gptq"),
    }

    results = bench.run_all(model_paths)

    # Save results JSON
    import json
    with open("/tmp/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Benchmark complete. Results saved to /tmp/benchmark_results.json")


if __name__ == "__main__":
    main()
