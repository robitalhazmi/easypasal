"""Asynchronous client for benchmarking vLLM and OpenAI-compatible serving engines."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time

import httpx

from experiments.e6_vllm.models import BenchmarkRequestResult

logger = logging.getLogger("experiments.e6_vllm.client")


class AsyncVLLMBenchmarkClient:
    """High-precision asynchronous streaming client for measuring TTFT and throughput."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        mock: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.mock = mock
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def generate_stream(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int = 160,
        temperature: float = 0.1,
        stop: list[str] | None = None,
        concurrency_context: int = 1,
    ) -> BenchmarkRequestResult:
        """Send a streaming request, tracking exact TTFT and inter-token arrival."""
        if self.mock:
            return await self._simulate_mock_request(
                request_id=request_id,
                prompt=prompt,
                max_tokens=max_tokens,
                concurrency_context=concurrency_context,
            )

        start_time = time.perf_counter()
        first_token_time: float | None = None
        chunks: list[str] = []
        prompt_tokens = len(prompt.split())  # Approximation if usage omitted

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stop": stop or ["<|im_end|>", "<|eot_id|>", "</answer>"],
        }

        url = f"{self.base_url}/chat/completions"
        try:
            async with (
                httpx.AsyncClient(headers=self.headers, timeout=self.timeout) as client,
                client.stream("POST", url, json=payload) as response,
            ):
                if response.status_code != 200:
                    error_body = await response.aread()
                    return BenchmarkRequestResult(
                        request_id=request_id,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,
                        ttft_ms=0.0,
                        total_latency_ms=(time.perf_counter() - start_time) * 1000.0,
                        itl_ms=0.0,
                        tokens_per_second=0.0,
                        status="error",
                        error_message=f"HTTP {response.status_code}: {error_body.decode('utf-8', errors='ignore')}",
                    )

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        raw_data = line[6:].strip()
                        if raw_data == "[DONE]":
                            break
                        try:
                            chunk_json = json.loads(raw_data)
                            choices = chunk_json.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    if first_token_time is None:
                                        first_token_time = time.perf_counter()
                                    chunks.append(content)
                        except json.JSONDecodeError:
                            continue

            end_time = time.perf_counter()
            total_latency_ms = (end_time - start_time) * 1000.0

            if first_token_time is None:
                first_token_time = end_time

            ttft_ms = (first_token_time - start_time) * 1000.0
            generated_text = "".join(chunks)
            completion_tokens = len(generated_text.split())  # approximate token count

            itl_ms = (
                (total_latency_ms - ttft_ms) / max(1, completion_tokens - 1)
                if completion_tokens > 1
                else 0.0
            )
            tokens_per_sec = (
                completion_tokens / (total_latency_ms / 1000.0) if total_latency_ms > 0 else 0.0
            )

            return BenchmarkRequestResult(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ttft_ms=round(ttft_ms, 2),
                total_latency_ms=round(total_latency_ms, 2),
                itl_ms=round(itl_ms, 2),
                tokens_per_second=round(tokens_per_sec, 2),
                status="success",
                response_text=generated_text,
            )

        except Exception as e:
            return BenchmarkRequestResult(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                ttft_ms=0.0,
                total_latency_ms=(time.perf_counter() - start_time) * 1000.0,
                itl_ms=0.0,
                tokens_per_second=0.0,
                status="error",
                error_message=str(e),
            )

    async def _simulate_mock_request(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        concurrency_context: int,
    ) -> BenchmarkRequestResult:
        """Simulate realistic vLLM PagedAttention continuous batching dynamics on T4 GPU."""
        prompt_tokens = len(prompt.split())
        c = max(1, concurrency_context)

        # Base TTFT on T4 GPU: ~45ms for ~1500 tokens with prefill scheduling
        # c=1: ~45ms, c=4: ~75ms, c=16: ~165ms, c=32: ~310ms
        ttft_scaling = {1: 45.0, 4: 76.0, 16: 168.0, 32: 315.0}.get(
            c, 45.0 * (1.0 + 0.35 * (c**0.65))
        )
        base_ttft_ms = ttft_scaling + random.uniform(-3.5, 4.5)
        base_ttft_ms = max(20.0, base_ttft_ms)

        # Output tokens: realistic gold answers in 75 - 115 tokens range
        sim_tokens = min(max_tokens, random.randint(75, 115))

        # Decode phase ITL (Time Per Output Token):
        # Continuous batching: ITL is ~22ms at c=1, scaling smoothly to ~33ms at c=32
        itl_scaling = {1: 22.0, 4: 24.5, 16: 28.5, 32: 33.5}.get(c, 20.0 + (0.42 * c))
        sim_itl_ms = itl_scaling + random.uniform(-1.2, 1.8)
        sim_decode_duration_ms = sim_itl_ms * (sim_tokens - 1)

        # Additional queuing overhead under load:
        queue_overhead_ms = {1: 0.0, 4: 250.0, 16: 1800.0, 32: 4500.0}.get(c, float(c * 120))
        queue_jitter = random.uniform(0.85, 1.15) * queue_overhead_ms

        sim_total_ms = base_ttft_ms + sim_decode_duration_ms + queue_jitter

        # Brief async yield to simulate concurrency
        await asyncio.sleep(0.005)

        tokens_per_second = round(sim_tokens / (sim_total_ms / 1000.0), 2)

        return BenchmarkRequestResult(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=sim_tokens,
            ttft_ms=round(base_ttft_ms, 2),
            total_latency_ms=round(sim_total_ms, 2),
            itl_ms=round(sim_itl_ms, 2),
            tokens_per_second=tokens_per_second,
            status="success",
            response_text=f"[Mock vLLM Response {request_id}] Ketentuan kepatuhan diatur dalam POJK.",
        )
