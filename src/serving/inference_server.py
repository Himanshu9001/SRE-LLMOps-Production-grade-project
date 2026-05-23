"""
P10b — vLLM Inference Server
OpenAI-compatible API with custom SRE middleware.
Adds: request validation, SRE prompt formatting, metrics collection.

Endpoints:
  POST /v1/chat/completions  — OpenAI chat API (primary)
  POST /v1/completions       — OpenAI completion API
  GET  /health               — health check
  GET  /metrics              — Prometheus metrics
  GET  /v1/models            — list available models

Middleware:
  - Request ID injection
  - Latency tracking (TTFT, ITL, total)
  - Token counting
  - SRE context injection (optional)
"""

import os
import time
import uuid
import asyncio
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import boto3

from src.serving.vllm_config import VLLMConfig, CONFIGS
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Prometheus metrics for AI inference
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "vllm_requests_total",
    "Total inference requests",
    ["model", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "vllm_request_latency_seconds",
    "End-to-end request latency",
    ["model", "endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

TTFT_HISTOGRAM = Histogram(
    "vllm_time_to_first_token_seconds",
    "Time to first token (TTFT)",
    ["model"],
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)

ITL_HISTOGRAM = Histogram(
    "vllm_inter_token_latency_ms",
    "Inter-token latency in milliseconds",
    ["model"],
    buckets=[10, 20, 50, 100, 200, 500]
)

TOKENS_GENERATED = Counter(
    "vllm_tokens_generated_total",
    "Total tokens generated",
    ["model"]
)

TOKENS_PROMPT = Counter(
    "vllm_tokens_prompt_total",
    "Total prompt tokens processed",
    ["model"]
)

CONCURRENT_REQUESTS = Gauge(
    "vllm_concurrent_requests",
    "Current number of concurrent requests",
    ["model"]
)

KV_CACHE_USAGE = Gauge(
    "vllm_kv_cache_usage_ratio",
    "KV cache utilization ratio (0-1)",
    ["model"]
)

GPU_MEMORY_USAGE = Gauge(
    "vllm_gpu_memory_usage_gb",
    "GPU memory used in GB",
    ["model", "gpu_id"]
)


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role:    str
    content: str


class ChatCompletionRequest(BaseModel):
    model:             str = "sre-llmops"
    messages:          list[ChatMessage]
    max_tokens:        int   = 512
    temperature:       float = 0.1    # low temp for SRE — want deterministic commands
    top_p:             float = 0.95
    stream:            bool  = False
    stop:              Optional[list[str]] = None
    presence_penalty:  float = 0.0
    frequency_penalty: float = 0.0


class CompletionRequest(BaseModel):
    model:       str   = "sre-llmops"
    prompt:      str
    max_tokens:  int   = 512
    temperature: float = 0.1
    top_p:       float = 0.95
    stream:      bool  = False
    stop:        Optional[list[str]] = None


# ---------------------------------------------------------------------------
# SRE Context Injector
# ---------------------------------------------------------------------------

SRE_SYSTEM_PROMPT = """You are an expert SRE (Site Reliability Engineer) assistant.
You provide specific, actionable responses with exact kubectl commands, 
Terraform snippets, and step-by-step debugging procedures.
Always include the exact commands to run. Be concise and precise."""


def inject_sre_context(messages: list[ChatMessage]) -> list[ChatMessage]:
    """
    Inject SRE system prompt if not already present.
    Ensures model always has domain context for better responses.
    """
    if messages and messages[0].role == "system":
        return messages  # user provided their own system prompt

    return [ChatMessage(role="system", content=SRE_SYSTEM_PROMPT)] + messages


def messages_to_prompt(messages: list[ChatMessage], tokenizer) -> str:
    """
    Convert chat messages to Llama 3 chat format.
    Llama 3 uses special tokens for chat: <|begin_of_text|>, <|eot_id|>, etc.
    """
    # Use tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        chat = [{"role": m.role, "content": m.content} for m in messages]
        return tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback: manual Alpaca format
    prompt = ""
    for msg in messages:
        if msg.role == "system":
            prompt += f"System: {msg.content}\n\n"
        elif msg.role == "user":
            prompt += f"### Instruction:\n{msg.content}\n\n### Response:\n"
        elif msg.role == "assistant":
            prompt += f"{msg.content}\n\n"

    return prompt


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

def create_app(config: VLLMConfig) -> FastAPI:
    """Create FastAPI app with vLLM engine."""

    engine = None
    tokenizer = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize vLLM engine on startup."""
        nonlocal engine, tokenizer

        logger.info("Initializing vLLM engine...")

        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs
            from vllm.sampling_params import SamplingParams
            from transformers import AutoTokenizer

            engine_args = AsyncEngineArgs(**config.to_engine_args())
            engine      = AsyncLLMEngine.from_engine_args(engine_args)
            tokenizer   = AutoTokenizer.from_pretrained(config.model)

            logger.info(f"vLLM engine ready: {config.served_model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            raise

        yield

        logger.info("Shutting down vLLM engine...")

    app = FastAPI(
        title="SRE LLMOps Inference Server",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health():
        """Health check — returns 200 when engine is ready."""
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        return {"status": "healthy", "model": config.served_model_name}

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        # Update GPU memory gauge
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    mem_gb = torch.cuda.memory_allocated(i) / 1024**3
                    GPU_MEMORY_USAGE.labels(
                        model=config.served_model_name,
                        gpu_id=str(i)
                    ).set(mem_gb)
        except Exception:
            pass

        return generate_latest()

    @app.get("/v1/models")
    async def list_models():
        """List available models — OpenAI compatible."""
        return {
            "object": "list",
            "data": [{
                "id":      config.served_model_name,
                "object":  "model",
                "created": int(time.time()),
                "owned_by": "sre-llmops",
            }]
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, http_request: Request):
        """
        OpenAI-compatible chat completions endpoint.
        Supports streaming and non-streaming responses.
        """
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        CONCURRENT_REQUESTS.labels(model=config.served_model_name).inc()
        REQUEST_COUNT.labels(
            model=config.served_model_name,
            endpoint="chat",
            status="started"
        ).inc()

        try:
            from vllm.sampling_params import SamplingParams

            # Inject SRE context
            messages = inject_sre_context(request.messages)

            # Convert messages to prompt
            prompt = messages_to_prompt(messages, tokenizer)

            # Count prompt tokens
            prompt_tokens = len(tokenizer.encode(prompt))
            TOKENS_PROMPT.labels(model=config.served_model_name).inc(prompt_tokens)

            # Sampling parameters
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop or ["<|eot_id|>", "<|end_of_text|>"],
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
            )

            if request.stream:
                return StreamingResponse(
                    _stream_chat_response(
                        engine, prompt, sampling_params,
                        request_id, start_time, config,
                    ),
                    media_type="text/event-stream",
                )

            # Non-streaming response
            results = []
            async for output in engine.generate(prompt, sampling_params, request_id):
                results.append(output)

            final_output = results[-1]
            generated_text = final_output.outputs[0].text

            # Metrics
            total_latency = time.perf_counter() - start_time
            output_tokens = len(final_output.outputs[0].token_ids)

            REQUEST_LATENCY.labels(
                model=config.served_model_name, endpoint="chat"
            ).observe(total_latency)
            TOKENS_GENERATED.labels(model=config.served_model_name).inc(output_tokens)

            REQUEST_COUNT.labels(
                model=config.served_model_name, endpoint="chat", status="success"
            ).inc()

            return {
                "id":      f"chatcmpl-{request_id}",
                "object":  "chat.completion",
                "created": int(time.time()),
                "model":   config.served_model_name,
                "choices": [{
                    "index":         0,
                    "message":       {"role": "assistant", "content": generated_text},
                    "finish_reason": final_output.outputs[0].finish_reason,
                }],
                "usage": {
                    "prompt_tokens":     prompt_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens":      prompt_tokens + output_tokens,
                }
            }

        except Exception as e:
            REQUEST_COUNT.labels(
                model=config.served_model_name, endpoint="chat", status="error"
            ).inc()
            logger.error(f"Request {request_id} failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            CONCURRENT_REQUESTS.labels(model=config.served_model_name).dec()

    async def _stream_chat_response(
        engine,
        prompt: str,
        sampling_params,
        request_id: str,
        start_time: float,
        config: VLLMConfig,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens as Server-Sent Events (SSE).
        Measures TTFT on first token, ITL for subsequent tokens.
        """
        import json

        first_token = True
        prev_time   = start_time
        prev_len    = 0

        async for output in engine.generate(prompt, sampling_params, request_id):
            token_ids   = output.outputs[0].token_ids
            current_len = len(token_ids)

            if current_len > prev_len:
                now = time.perf_counter()

                if first_token:
                    # TTFT measurement
                    ttft = now - start_time
                    TTFT_HISTOGRAM.labels(
                        model=config.served_model_name
                    ).observe(ttft)
                    first_token = False
                else:
                    # ITL measurement
                    itl_ms = (now - prev_time) * 1000
                    ITL_HISTOGRAM.labels(
                        model=config.served_model_name
                    ).observe(itl_ms)

                # Stream delta
                full_text  = output.outputs[0].text
                delta_text = full_text[prev_len:]

                chunk = {
                    "id":      f"chatcmpl-{request_id}",
                    "object":  "chat.completion.chunk",
                    "created": int(time.time()),
                    "model":   config.served_model_name,
                    "choices": [{
                        "index":  0,
                        "delta":  {"content": delta_text},
                        "finish_reason": None,
                    }]
                }

                yield f"data: {json.dumps(chunk)}\n\n"

                prev_time = now
                prev_len  = current_len

        # Final chunk
        yield f"data: {json.dumps({'choices': [{'finish_reason': 'stop', 'delta': {}}]})}\n\n"
        yield "data: [DONE]\n\n"

        TOKENS_GENERATED.labels(
            model=config.served_model_name
        ).inc(prev_len)

    return app


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="production",
                        choices=list(CONFIGS.keys()))
    parser.add_argument("--host",    default="0.0.0.0")
    parser.add_argument("--port",    type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    config = CONFIGS[args.config]
    config.host = args.host
    config.port = args.port

    app = create_app(config)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
