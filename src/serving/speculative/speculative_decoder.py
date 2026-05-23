"""
P13 — Speculative Decoding
Draft model generates N candidate tokens → verifier accepts/rejects in parallel.

Why speculative decoding works:
  Standard decode: 1 token per forward pass through 8B model
  Speculative:     draft model generates 5 tokens → verifier checks all 5 in ONE pass
  
  If verifier accepts all 5: 5 tokens generated at cost of ~1.2 forward passes
  If verifier rejects at token 3: 3 tokens accepted, 1 rejected → restart
  Expected speedup: 2-3x for natural language, 1.5-2x for SRE commands

The math:
  Let α = acceptance rate per token (0.7-0.9 for good draft models)
  Let γ = number of draft tokens per step
  Expected accepted tokens per step = γα + (1 - α^γ) / (1 - α)  ← geometric series
  Speedup = expected_accepted / cost_of_one_verifier_pass

  For α=0.8, γ=5:
    Expected accepted = 5×0.8 + ... ≈ 3.2 tokens per step
    Cost ≈ 1 verifier pass + γ × (draft_cost/verifier_cost)
    If draft is 10x smaller: cost ≈ 1 + 5×0.1 = 1.5
    Speedup ≈ 3.2 / 1.5 ≈ 2.1x

Draft model selection for SRE:
  Option A: Llama 3 8B (our fine-tuned) as draft → Llama 3 70B as verifier
            Best quality, requires 70B GPU allocation
  Option B: Llama 3 8B AWQ as draft → Llama 3 8B fp16 as verifier  
            Same model different precision — high acceptance rate
  Option C: Small specialized model (1-3B) → Llama 3 8B as verifier
            Lower acceptance rate but cheaper draft inference

For our SRE use case: Option B
  Draft:    Llama 3 8B AWQ 4-bit (fast, cheap)
  Verifier: Llama 3 8B fp16 (authoritative)
  Expected acceptance: 0.85+ (same model, different precision)
  Speedup: ~2x TTFT improvement

vLLM native support:
  vLLM supports speculative decoding natively since v0.3.0
  Config: --speculative-model + --num-speculative-tokens
  No code changes to inference server needed
"""

import time
import torch
import asyncio
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator
from prometheus_client import Histogram, Counter, Gauge
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Speculative Decoding Metrics
# ---------------------------------------------------------------------------

SPEC_ACCEPTANCE_RATE = Histogram(
    "vllm_speculative_acceptance_rate",
    "Token acceptance rate per speculative decoding step",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

SPEC_TOKENS_ACCEPTED = Counter(
    "vllm_speculative_tokens_accepted_total",
    "Total draft tokens accepted by verifier"
)

SPEC_TOKENS_REJECTED = Counter(
    "vllm_speculative_tokens_rejected_total",
    "Total draft tokens rejected by verifier"
)

SPEC_SPEEDUP = Histogram(
    "vllm_speculative_speedup_ratio",
    "Actual speedup ratio vs non-speculative",
    buckets=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
)

SPEC_DRAFT_LATENCY = Histogram(
    "vllm_speculative_draft_latency_ms",
    "Draft model inference latency per step",
    buckets=[5, 10, 20, 50, 100, 200]
)


# ---------------------------------------------------------------------------
# Core Speculative Decoding Implementation
# ---------------------------------------------------------------------------

@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    draft_model_path:      str
    verifier_model_path:   str
    num_speculative_tokens: int   = 5       # γ: draft tokens per step
    temperature:           float = 1.0
    acceptance_threshold:  float = 0.0     # 0 = standard rejection sampling

    # vLLM engine args for speculative mode
    def to_vllm_args(self) -> dict:
        return {
            "speculative_model":        self.draft_model_path,
            "num_speculative_tokens":   self.num_speculative_tokens,
            "speculative_draft_tensor_parallel_size": 1,
            "enable_chunked_prefill":   False,
            # Chunked prefill incompatible with speculative decoding
        }


@dataclass
class SpecStep:
    """Result of one speculative decoding step."""
    draft_tokens:     list[int]    # γ draft token IDs
    draft_probs:      torch.Tensor # draft probability distributions
    accepted_tokens:  list[int]    # tokens accepted by verifier
    rejected_at:      Optional[int] = None  # position of first rejection
    acceptance_rate:  float = 0.0


class SpeculativeDecoder:
    """
    Manual implementation of speculative decoding.
    For educational/interview purposes — vLLM handles this natively.

    Understanding this implementation is key for interview discussions
    about why speculative decoding works and its failure modes.
    """

    def __init__(
        self,
        draft_model,
        draft_tokenizer,
        verifier_model,
        verifier_tokenizer,
        config: SpeculativeConfig,
    ):
        self.draft    = draft_model
        self.draft_t  = draft_tokenizer
        self.verifier = verifier_model
        self.verify_t = verifier_tokenizer
        self.config   = config

    @torch.no_grad()
    def _draft_step(
        self,
        input_ids: torch.Tensor,
        num_tokens: int,
    ) -> tuple[list[int], torch.Tensor]:
        """
        Draft model generates num_tokens candidate tokens.
        Returns token IDs and their probability distributions.

        Draft model is autoregressive — each token conditions on previous.
        This is the "cheap" step: small model, fast inference.
        """
        draft_token_ids = []
        draft_probs     = []
        current_ids     = input_ids

        for _ in range(num_tokens):
            outputs = self.draft(input_ids=current_ids)
            logits  = outputs.logits[:, -1, :]  # last token logits

            # Sample from draft distribution
            probs     = torch.softmax(logits / self.config.temperature, dim=-1)
            token_id  = torch.multinomial(probs, num_samples=1)

            draft_token_ids.append(token_id.item())
            draft_probs.append(probs)

            # Append to sequence for next draft step
            current_ids = torch.cat([current_ids, token_id.unsqueeze(0)], dim=-1)

        return draft_token_ids, torch.stack(draft_probs, dim=1)

    @torch.no_grad()
    def _verify_step(
        self,
        input_ids: torch.Tensor,
        draft_token_ids: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Verifier checks all draft tokens in ONE forward pass.
        This is the key efficiency gain — O(1) verifier passes regardless of γ.

        Verifier processes: [prompt + all_draft_tokens] in parallel
        Returns probability distributions at each draft position.

        The verifier's distribution is the "correct" distribution.
        Draft tokens are accepted if draft prob / verifier prob >= random uniform.
        """
        # Append draft tokens to input
        draft_tensor = torch.tensor(
            draft_token_ids, device=input_ids.device
        ).unsqueeze(0)
        full_ids = torch.cat([input_ids, draft_tensor], dim=-1)

        # Single forward pass through verifier
        outputs       = self.verifier(input_ids=full_ids)
        verifier_logits = outputs.logits

        # Extract logits at draft token positions
        # Position i in verifier output corresponds to draft token i
        prompt_len = input_ids.shape[1]
        draft_logits = verifier_logits[
            :,
            prompt_len-1 : prompt_len-1+len(draft_token_ids),
            :
        ]

        verifier_probs  = torch.softmax(
            draft_logits / self.config.temperature, dim=-1
        )

        # Also get distribution for the NEXT token (after all draft tokens)
        next_token_logits = verifier_logits[:, -1, :]
        next_token_probs  = torch.softmax(
            next_token_logits / self.config.temperature, dim=-1
        )

        return verifier_probs, next_token_probs

    def _rejection_sample(
        self,
        draft_token_ids: list[int],
        draft_probs: torch.Tensor,
        verifier_probs: torch.Tensor,
    ) -> tuple[list[int], Optional[int]]:
        """
        Standard rejection sampling for speculative decoding.
        DeepMind's algorithm (Chen et al. 2023).

        For each draft token t_i:
          Accept with probability min(1, p_verifier(t_i) / p_draft(t_i))
          If rejected: resample from adjusted distribution
                      p_adjusted = max(0, p_verifier - p_draft) / Z

        This guarantees the OUTPUT distribution exactly matches
        the verifier's distribution — no quality degradation.
        """
        accepted = []

        for i, token_id in enumerate(draft_token_ids):
            p_draft    = draft_probs[0, i, token_id].item()
            p_verifier = verifier_probs[0, i, token_id].item()

            # Acceptance probability
            accept_prob = min(1.0, p_verifier / (p_draft + 1e-10))

            if torch.rand(1).item() < accept_prob:
                # Accept draft token
                accepted.append(token_id)
            else:
                # Reject — resample from adjusted distribution
                adjusted = torch.clamp(
                    verifier_probs[0, i] - draft_probs[0, i],
                    min=0
                )
                adjusted = adjusted / adjusted.sum()

                resampled = torch.multinomial(adjusted, num_samples=1).item()
                accepted.append(resampled)

                # Stop — everything after rejection is invalid
                return accepted, i

        return accepted, None  # all accepted

    def generate_step(
        self,
        input_ids: torch.Tensor,
    ) -> SpecStep:
        """
        One speculative decoding step:
        1. Draft: generate γ candidate tokens
        2. Verify: check all γ tokens in one verifier pass
        3. Accept/reject: standard rejection sampling
        """
        start = time.perf_counter()

        # Draft phase
        draft_ids, draft_probs = self._draft_step(
            input_ids, self.config.num_speculative_tokens
        )

        draft_latency = (time.perf_counter() - start) * 1000
        SPEC_DRAFT_LATENCY.observe(draft_latency)

        # Verify phase
        verifier_probs, next_token_probs = self._verify_step(
            input_ids, draft_ids
        )

        # Rejection sampling
        accepted, rejected_at = self._rejection_sample(
            draft_ids, draft_probs, verifier_probs
        )

        # Metrics
        acceptance_rate = len(accepted) / len(draft_ids)
        SPEC_ACCEPTANCE_RATE.observe(acceptance_rate)
        SPEC_TOKENS_ACCEPTED.inc(len(accepted))
        SPEC_TOKENS_REJECTED.inc(len(draft_ids) - len(accepted))

        return SpecStep(
            draft_tokens=draft_ids,
            draft_probs=draft_probs,
            accepted_tokens=accepted,
            rejected_at=rejected_at,
            acceptance_rate=acceptance_rate,
        )

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_tokens: int = 200,
    ) -> tuple[list[int], dict]:
        """
        Full speculative generation loop.
        Returns generated token IDs and performance stats.
        """
        current_ids  = prompt_ids
        all_tokens   = []
        steps        = 0
        total_drafted = 0
        total_accepted = 0

        start_time = time.perf_counter()

        while len(all_tokens) < max_tokens:
            step = self.generate_step(current_ids)
            steps        += 1
            total_drafted  += len(step.draft_tokens)
            total_accepted += len(step.accepted_tokens)

            all_tokens.extend(step.accepted_tokens)

            # Update context with accepted tokens
            accepted_tensor = torch.tensor(
                step.accepted_tokens, device=current_ids.device
            ).unsqueeze(0)
            current_ids = torch.cat([current_ids, accepted_tensor], dim=-1)

            # Stop on EOS
            if any(t == self.draft_t.eos_token_id for t in step.accepted_tokens):
                break

        total_time = time.perf_counter() - start_time
        avg_acceptance = total_accepted / max(1, total_drafted)

        stats = {
            "total_tokens":      len(all_tokens),
            "speculative_steps": steps,
            "draft_tokens":      total_drafted,
            "accepted_tokens":   total_accepted,
            "acceptance_rate":   round(avg_acceptance, 3),
            "tokens_per_step":   round(total_accepted / max(1, steps), 2),
            "total_time_ms":     round(total_time * 1000, 1),
            "throughput_tps":    round(len(all_tokens) / total_time, 1),
        }

        logger.info(
            f"Speculative generation: {len(all_tokens)} tokens, "
            f"{steps} steps, "
            f"acceptance={avg_acceptance:.2%}, "
            f"{stats['throughput_tps']} tok/s"
        )

        return all_tokens, stats


# ---------------------------------------------------------------------------
# vLLM Native Speculative Decoding Config
# ---------------------------------------------------------------------------

class VLLMSpeculativeConfig:
    """
    Configuration for vLLM's native speculative decoding.
    Much simpler than manual implementation — vLLM handles everything.
    """

    @staticmethod
    def get_server_args(
        verifier_model: str,
        draft_model: str,
        num_spec_tokens: int = 5,
        use_draft_quantization: bool = True,
    ) -> list[str]:
        """
        Get vLLM server args for speculative decoding.

        Launch with:
          python -m vllm.entrypoints.openai.api_server \
            --model {verifier_model} \
            --speculative-model {draft_model} \
            --num-speculative-tokens 5 \
            --speculative-draft-tensor-parallel-size 1

        vLLM handles:
          - Draft model loading and inference
          - Rejection sampling
          - Batch-level speculative decoding
          - Metrics (acceptance rate, speedup)
        """
        args = [
            f"--model={verifier_model}",
            f"--speculative-model={draft_model}",
            f"--num-speculative-tokens={num_spec_tokens}",
            "--speculative-draft-tensor-parallel-size=1",
            "--enable-chunked-prefill=false",  # incompatible
            "--gpu-memory-utilization=0.85",   # leave room for draft model
        ]

        if use_draft_quantization:
            args.append("--speculative-model-quantization=awq")

        return args

    @staticmethod
    def estimate_memory_split(
        verifier_size_gb: float,
        draft_size_gb: float,
        gpu_memory_gb: float = 24.0,
    ) -> dict:
        """
        Estimate memory allocation for speculative decoding.
        Both draft and verifier must fit on the same GPU(s).

        For A10G (24GB):
          Llama 3 8B fp16 (verifier): 16GB
          Llama 3 8B AWQ (draft):      4.5GB
          Total:                       20.5GB → fits in 24GB ✅

          Remaining for KV cache:      ~2.5GB
          Max concurrent seqs:         ~5-10 (limited)

        Recommendation:
          Use AWQ for BOTH draft and verifier to maximize KV cache
          Verifier AWQ: 4.5GB + Draft AWQ: 4.5GB = 9GB
          KV cache: ~13GB → 26 concurrent seqs ✅
        """
        total_model_memory = verifier_size_gb + draft_size_gb
        available_kv       = gpu_memory_gb - total_model_memory - 1.0

        return {
            "verifier_gb":       verifier_size_gb,
            "draft_gb":          draft_size_gb,
            "total_model_gb":    total_model_memory,
            "available_kv_gb":   max(0, available_kv),
            "fits_on_gpu":       total_model_memory < gpu_memory_gb - 2,
            "recommended_config": (
                "Both models AWQ 4-bit for maximum KV cache"
                if total_model_memory > gpu_memory_gb * 0.7
                else "Standard configuration"
            ),
        }


# ---------------------------------------------------------------------------
# Benchmark: Speculative vs Standard
# ---------------------------------------------------------------------------

class SpeculativeBenchmark:
    """
    Compare speculative vs standard decoding on SRE prompts.
    Measures actual speedup in production-like conditions.
    """

    SRE_TEST_PROMPTS = [
        {
            "prompt": "### Instruction:\nA pod is in CrashLoopBackOff. Diagnose it step by step.\n\n### Input:\nPod: api-server in namespace production. Exit code 137.\n\n### Response:\n",
            "expected_length": 150,
        },
        {
            "prompt": "### Instruction:\nTerraform state is locked. How to safely unlock?\n\n### Input:\nLock ID: a1b2c3d4. CI job #1234 confirmed terminated.\n\n### Response:\n",
            "expected_length": 100,
        },
        {
            "prompt": "### Instruction:\nWrite a PromQL query for 5-minute error rate.\n\n### Input:\nMetric: http_requests_total. Label: status='5xx'.\n\n### Response:\n",
            "expected_length": 50,
        },
    ]

    def run_comparison(
        self,
        standard_engine,
        speculative_engine,
        tokenizer,
        num_runs: int = 5,
    ) -> dict:
        """
        Run side-by-side benchmark of standard vs speculative.
        Returns speedup ratios per prompt type.
        """
        import mlflow

        results = {
            "standard":    [],
            "speculative": [],
        }

        for prompt_data in self.SRE_TEST_PROMPTS:
            prompt = prompt_data["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            # Standard decoding
            standard_times = []
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()

                with torch.no_grad():
                    standard_engine.generate(
                        **inputs,
                        max_new_tokens=prompt_data["expected_length"],
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                torch.cuda.synchronize()
                standard_times.append(time.perf_counter() - start)

            avg_standard = sum(standard_times) / len(standard_times)

            # Speculative decoding
            speculative_times = []
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()

                # vLLM speculative is transparent — same API
                with torch.no_grad():
                    speculative_engine.generate(
                        **inputs,
                        max_new_tokens=prompt_data["expected_length"],
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                torch.cuda.synchronize()
                speculative_times.append(time.perf_counter() - start)

            avg_speculative = sum(speculative_times) / len(speculative_times)
            speedup = avg_standard / avg_speculative

            SPEC_SPEEDUP.observe(speedup)

            result = {
                "prompt_preview":    prompt[:50],
                "standard_ms":       round(avg_standard * 1000, 1),
                "speculative_ms":    round(avg_speculative * 1000, 1),
                "speedup":           round(speedup, 2),
                "expected_tokens":   prompt_data["expected_length"],
            }

            results["standard"].append(avg_standard)
            results["speculative"].append(avg_speculative)

            logger.info(
                f"Benchmark: {prompt[:40]}... "
                f"standard={avg_standard*1000:.0f}ms "
                f"speculative={avg_speculative*1000:.0f}ms "
                f"speedup={speedup:.2f}x"
            )

        overall_speedup = (
            sum(results["standard"]) /
            sum(results["speculative"])
        )

        logger.info(f"\nOverall speculative speedup: {overall_speedup:.2f}x")

        return {
            "overall_speedup":    round(overall_speedup, 2),
            "prompt_results":     results,
            "num_runs":           num_runs,
        }
