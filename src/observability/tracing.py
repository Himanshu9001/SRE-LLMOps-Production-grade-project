"""
P11d — Distributed Tracing with OpenTelemetry
Traces the complete inference pipeline:
  Client → AI Gateway → vLLM → Tokenizer → Prefill → Decode → Response

Why tracing for LLM inference:
  - Standard metrics tell you WHAT is slow (high P99 latency)
  - Tracing tells you WHERE it's slow (tokenization? prefill? decode?)
  - Critical for diagnosing KV cache misses, queue delays, model load time

Trace structure for one inference request:
  ┌─────────────────────────────────────────────────────┐
  │ gateway.request (total: 3.2s)                       │
  │  ├── gateway.auth_check (2ms)                       │
  │  ├── gateway.rate_limit_check (1ms)                 │
  │  ├── gateway.cache_lookup (3ms)  ← cache miss       │
  │  └── vllm.inference (3.19s)                         │
  │       ├── vllm.tokenize (8ms)                       │
  │       ├── vllm.schedule (2ms)                       │
  │       ├── vllm.prefill (890ms)   ← slow!            │
  │       └── vllm.decode (2.29s)                       │
  │            ├── vllm.decode.step[0] (23ms)           │
  │            ├── vllm.decode.step[1] (22ms)           │
  │            └── ... (100 steps)                      │
  └─────────────────────────────────────────────────────┘

Backend: Jaeger (OpenTelemetry-compatible)
Export: OTLP gRPC to Jaeger collector
"""

import time
import functools
from contextlib import contextmanager
from typing import Optional, Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from src.utils.logger import logger


def setup_tracing(
    service_name: str,
    otlp_endpoint: str = "http://jaeger-collector.monitoring.svc.cluster.local:4317",
    sample_rate: float = 1.0,
) -> trace.Tracer:
    """
    Initialize OpenTelemetry tracing.

    sample_rate:
      1.0 = trace every request (development/debugging)
      0.1 = trace 10% of requests (production, high traffic)
      0.01 = trace 1% (very high traffic)

    For LLM inference: 0.1 is typical
    Each trace adds ~1-2ms overhead — negligible for multi-second requests.
    """
    resource = Resource.create({
        "service.name":    service_name,
        "service.version": "p11",
        "deployment.env":  "production",
        "k8s.cluster":     "sre-llmops-production",
    })

    provider = TracerProvider(resource=resource)

    # OTLP exporter — sends spans to Jaeger/Tempo
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)

    # Auto-instrument FastAPI and httpx
    FastAPIInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()

    tracer = trace.get_tracer(service_name)
    logger.info(f"Tracing initialized: {service_name} → {otlp_endpoint}")
    return tracer


# Global tracer — initialized once at startup
_tracer: Optional[trace.Tracer] = None


def get_tracer() -> trace.Tracer:
    global _tracer
    if _tracer is None:
        _tracer = setup_tracing("sre-llmops-inference")
    return _tracer


@contextmanager
def trace_span(
    name: str,
    attributes: dict = None,
    record_exception: bool = True,
):
    """
    Context manager for creating trace spans.

    Usage:
        with trace_span("vllm.tokenize", {"prompt_length": 100}):
            tokens = tokenizer.encode(prompt)

    Automatically:
    - Records start/end time
    - Captures exceptions as span events
    - Sets span status (OK or ERROR)
    """
    tracer = get_tracer()

    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, str(v))

        try:
            yield span
            span.set_status(Status(StatusCode.OK))

        except Exception as e:
            if record_exception:
                span.record_exception(e)
                span.set_status(
                    Status(StatusCode.ERROR, str(e))
                )
            raise


def trace_inference_request(func):
    """
    Decorator for tracing inference functions.
    Automatically adds prompt/response metadata as span attributes.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        with trace_span(
            f"inference.{func.__name__}",
            attributes={
                "function": func.__name__,
            }
        ) as span:
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000

            span.set_attribute("latency_ms", elapsed_ms)
            return result

    return wrapper


class InferenceTracer:
    """
    Structured tracing for the full inference pipeline.
    Creates parent span for request, child spans for each phase.
    """

    def __init__(self, request_id: str, model_name: str):
        self.request_id = request_id
        self.model_name = model_name
        self.tracer     = get_tracer()
        self._root_span = None

    def __enter__(self):
        self._root_span = self.tracer.start_span(
            "inference.request",
            attributes={
                "request.id":    self.request_id,
                "model.name":    self.model_name,
                "service.name":  "vllm-inference",
            }
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._root_span.record_exception(exc_val)
            self._root_span.set_status(Status(StatusCode.ERROR))
        else:
            self._root_span.set_status(Status(StatusCode.OK))
        self._root_span.end()

    @contextmanager
    def phase(self, phase_name: str, **attributes):
        """
        Record a named phase within the inference pipeline.

        Phases: tokenize, schedule, prefill, decode
        Each phase becomes a child span under the root request span.
        """
        with self.tracer.start_as_current_span(
            f"inference.{phase_name}",
            attributes={
                "phase": phase_name,
                **{k: str(v) for k, v in attributes.items()},
            }
        ) as span:
            start = time.perf_counter()
            yield span
            elapsed = (time.perf_counter() - start) * 1000
            span.set_attribute("duration_ms", elapsed)

    def record_tokenization(self, prompt_tokens: int):
        """Record tokenization result on root span."""
        if self._root_span:
            self._root_span.set_attribute("prompt.tokens", prompt_tokens)

    def record_generation(
        self,
        output_tokens: int,
        ttft_ms: float,
        total_ms: float,
    ):
        """Record generation metrics on root span."""
        if self._root_span:
            self._root_span.set_attribute("output.tokens",  output_tokens)
            self._root_span.set_attribute("ttft_ms",        ttft_ms)
            self._root_span.set_attribute("total_ms",       total_ms)
            self._root_span.set_attribute(
                "throughput_tps",
                output_tokens / (total_ms / 1000)
            )
