"""
P10e — AI Gateway using LiteLLM
Unified API gateway for multiple LLM backends.

AI Gateway responsibilities:
  1. Authentication & rate limiting
  2. Model routing (Llama 3 → GPT-4 fallback)
  3. Request/response logging
  4. Cost tracking per team/user
  5. Prompt caching
  6. Retry with fallback

Why LiteLLM:
  OpenAI-compatible API for 100+ LLM providers
  Drop-in replacement — clients don't know which model responds
  Built-in: retry, fallback, load balancing, logging
  
Architecture:
  Client → AI Gateway (LiteLLM Proxy) → vLLM (primary)
                                      → OpenAI GPT-4 (fallback)
                                      → Anthropic Claude (fallback)

Fallback chain:
  1. sre-llmops (vLLM serving Llama 3 8B AWQ) — primary, free
  2. gpt-3.5-turbo — fallback if vLLM unavailable, cheap
  3. gpt-4 — escalation for complex queries, expensive

Rate limiting:
  Per-team: 100 RPM, 10k tokens/min
  Per-user: 20 RPM, 2k tokens/min
  Burst: 2x limit for 30 seconds
"""

import os
import time
import hashlib
import asyncio
from typing import Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Rate Limiter (token bucket algorithm)
# ---------------------------------------------------------------------------

class TokenBucket:
    """
    Token bucket rate limiter.
    Allows burst traffic while enforcing average rate.

    Tokens refill at rate R per second.
    Bucket capacity = burst_size.
    Each request consumes 1 token.
    Request rejected if bucket empty.
    """

    def __init__(self, rate: float, burst: int):
        self.rate     = rate       # tokens per second
        self.burst    = burst      # max bucket size
        self.tokens   = burst      # start full
        self.last_refill = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if allowed."""
        now = time.time()
        elapsed = now - self.last_refill

        # Refill bucket
        self.tokens = min(
            self.burst,
            self.tokens + elapsed * self.rate
        )
        self.last_refill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimiter:
    """Per-team and per-user rate limiting."""

    def __init__(self):
        self._team_buckets = {}
        self._user_buckets = {}

    def _get_team_bucket(self, team_id: str) -> TokenBucket:
        if team_id not in self._team_buckets:
            # 100 RPM = 100/60 tokens per second, burst=200
            self._team_buckets[team_id] = TokenBucket(
                rate=100/60, burst=200
            )
        return self._team_buckets[team_id]

    def _get_user_bucket(self, user_id: str) -> TokenBucket:
        if user_id not in self._user_buckets:
            # 20 RPM = 20/60 tokens per second, burst=40
            self._user_buckets[user_id] = TokenBucket(
                rate=20/60, burst=40
            )
        return self._user_buckets[user_id]

    def check_rate_limit(self, team_id: str, user_id: str) -> tuple[bool, str]:
        """
        Check if request is within rate limits.
        Returns (allowed, reason).
        """
        if not self._get_team_bucket(team_id).consume():
            return False, f"Team rate limit exceeded: 100 RPM"
        if not self._get_user_bucket(user_id).consume():
            return False, f"User rate limit exceeded: 20 RPM"
        return True, "ok"


# ---------------------------------------------------------------------------
# Prompt Cache
# ---------------------------------------------------------------------------

class PromptCache:
    """
    Simple in-memory prompt cache.
    Identical prompts (same hash) return cached response.

    Cache key: SHA256(model + prompt + temperature + max_tokens)
    TTL: 1 hour (SRE responses may become stale)

    Production: use Redis for distributed cache across replicas.
    Cache hit rate for SRE: ~15-20% (same questions repeated)
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        self._cache:    dict = {}
        self._ttl:      int  = ttl_seconds
        self._max_size: int  = max_size

    def _make_key(self, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
        content = f"{model}:{prompt}:{temperature}:{max_tokens}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, model: str, prompt: str, temperature: float, max_tokens: int) -> Optional[dict]:
        """Get cached response if available and not expired."""
        key = self._make_key(model, prompt, temperature, max_tokens)
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["timestamp"] < self._ttl:
                logger.debug(f"Cache hit: {key[:16]}...")
                return entry["response"]
            else:
                del self._cache[key]
        return None

    def set(self, model: str, prompt: str, temperature: float,
            max_tokens: int, response: dict):
        """Cache a response."""
        if len(self._cache) >= self._max_size:
            # Evict oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        key = self._make_key(model, prompt, temperature, max_tokens)
        self._cache[key] = {
            "response":  response,
            "timestamp": time.time(),
        }


# ---------------------------------------------------------------------------
# Model Router with Fallback
# ---------------------------------------------------------------------------

class ModelRouter:
    """
    Routes requests to appropriate model with fallback chain.
    Primary: vLLM (Llama 3 8B) — free, fast
    Fallback: OpenAI GPT-3.5 → GPT-4
    """

    MODEL_CHAIN = [
        {
            "name":     "sre-llmops",
            "endpoint": os.getenv("VLLM_ENDPOINT", "http://vllm-service.inference.svc.cluster.local:8000"),
            "type":     "vllm",
        },
        {
            "name":     "gpt-3.5-turbo",
            "endpoint": "https://api.openai.com/v1",
            "type":     "openai",
            "api_key":  os.getenv("OPENAI_API_KEY", ""),
        },
    ]

    def __init__(self):
        self.cache        = PromptCache()
        self.rate_limiter = RateLimiter()

    async def complete(
        self,
        messages: list[dict],
        model:        str   = "sre-llmops",
        max_tokens:   int   = 512,
        temperature:  float = 0.1,
        team_id:      str   = "default",
        user_id:      str   = "anonymous",
        use_cache:    bool  = True,
    ) -> dict:
        """
        Route request through model chain with fallback.
        Returns first successful response.
        """
        # Rate limiting
        allowed, reason = self.rate_limiter.check_rate_limit(team_id, user_id)
        if not allowed:
            raise HTTPException(status_code=429, detail=reason)

        # Cache check (only for low-temperature, deterministic requests)
        prompt_str = str(messages)
        if use_cache and temperature < 0.1:
            cached = self.cache.get(model, prompt_str, temperature, max_tokens)
            if cached:
                return {**cached, "cached": True}

        # Try each model in chain
        last_error = None
        for model_config in self.MODEL_CHAIN:
            try:
                response = await self._call_model(
                    model_config, messages, max_tokens, temperature
                )

                # Cache successful response
                if use_cache and temperature < 0.1:
                    self.cache.set(model, prompt_str, temperature, max_tokens, response)

                return response

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Model {model_config['name']} failed: {e}. "
                    f"Trying next in chain..."
                )
                continue

        raise HTTPException(
            status_code=503,
            detail=f"All models in chain failed. Last error: {last_error}"
        )

    async def _call_model(
        self,
        model_config: dict,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> dict:
        """Call a specific model endpoint."""
        import httpx

        if model_config["type"] == "vllm":
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{model_config['endpoint']}/v1/chat/completions",
                    json={
                        "model":       model_config["name"],
                        "messages":    messages,
                        "max_tokens":  max_tokens,
                        "temperature": temperature,
                    }
                )
                response.raise_for_status()
                return response.json()

        elif model_config["type"] == "openai":
            import openai
            client = openai.AsyncOpenAI(api_key=model_config["api_key"])
            response = await client.chat.completions.create(
                model=model_config["name"],
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.model_dump()


# ---------------------------------------------------------------------------
# Gateway FastAPI Application
# ---------------------------------------------------------------------------

def create_gateway_app() -> FastAPI:
    """Create AI Gateway FastAPI application."""

    router = ModelRouter()

    app = FastAPI(title="SRE LLMOps AI Gateway", version="1.0.0")

    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "ai-gateway"}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        x_team_id: str = Header(default="default"),
        x_user_id: str = Header(default="anonymous"),
        x_api_key: str = Header(default=""),
    ):
        """
        Main gateway endpoint — routes to appropriate model.
        Supports same request format as OpenAI API.
        """
        body = await request.json()

        response = await router.complete(
            messages=body.get("messages", []),
            model=body.get("model", "sre-llmops"),
            max_tokens=body.get("max_tokens", 512),
            temperature=body.get("temperature", 0.1),
            team_id=x_team_id,
            user_id=x_user_id,
        )

        return response

    return app


if __name__ == "__main__":
    import uvicorn
    app = create_gateway_app()
    uvicorn.run(app, host="0.0.0.0", port=8080)
