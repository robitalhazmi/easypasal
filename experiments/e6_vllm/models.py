"""Data models and metrics structures for Experiment E6 (vLLM Serving Benchmark)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np


@dataclass
class BenchmarkRequestResult:
    """Detailed performance metrics captured for a single benchmark request."""

    request_id: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_latency_ms: float
    itl_ms: float
    tokens_per_second: float
    status: str = "success"  # "success" | "error"
    error_message: str | None = None
    response_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConcurrencyBenchmarkSummary:
    """Aggregated performance metrics across all requests for a given concurrency level."""

    concurrency: int
    num_requests: int
    successful_requests: int
    failed_requests: int
    total_duration_s: float
    aggregate_tokens_per_second: float
    request_throughput_rps: float
    ttft_mean_ms: float
    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    itl_mean_ms: float
    itl_p50_ms: float
    itl_p95_ms: float
    mean_completion_tokens: float
    mean_prompt_tokens: float
    peak_vram_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HardwareMetadata:
    """Hardware and execution runtime environment specifications."""

    device_type: str  # "GPU" or "CPU"
    device_name: str  # e.g. "NVIDIA Tesla T4 16GB"
    framework: str  # e.g. "vLLM 0.6.3 (PagedAttention)"
    memory_total_mb: float
    quantization: str  # e.g. "AWQ / FP16" or "Q4_K_M"
    platform_name: str  # e.g. "Google Colab Free Tier (T4)"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class E6BenchmarkReport:
    """Complete Experiment E6 serving benchmark output."""

    model_name: str
    hardware: HardwareMetadata
    concurrency_levels: list[int]
    summaries: dict[int, ConcurrencyBenchmarkSummary]
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    cpu_baseline_comparison: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "hardware": self.hardware.to_dict(),
            "concurrency_levels": self.concurrency_levels,
            "summaries": {str(c): summary.to_dict() for c, summary in self.summaries.items()},
            "cpu_baseline_comparison": self.cpu_baseline_comparison,
        }


def compute_concurrency_summary(
    concurrency: int,
    results: list[BenchmarkRequestResult],
    total_duration_s: float,
    peak_vram_mb: float = 0.0,
) -> ConcurrencyBenchmarkSummary:
    """Aggregate individual request metrics into a comprehensive concurrency summary."""
    successful = [r for r in results if r.status == "success" and r.completion_tokens > 0]
    failed = [r for r in results if r.status != "success"]

    if not successful:
        return ConcurrencyBenchmarkSummary(
            concurrency=concurrency,
            num_requests=len(results),
            successful_requests=0,
            failed_requests=len(failed),
            total_duration_s=total_duration_s,
            aggregate_tokens_per_second=0.0,
            request_throughput_rps=0.0,
            ttft_mean_ms=0.0,
            ttft_p50_ms=0.0,
            ttft_p90_ms=0.0,
            ttft_p95_ms=0.0,
            ttft_p99_ms=0.0,
            latency_mean_ms=0.0,
            latency_p50_ms=0.0,
            latency_p90_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            itl_mean_ms=0.0,
            itl_p50_ms=0.0,
            itl_p95_ms=0.0,
            mean_completion_tokens=0.0,
            mean_prompt_tokens=0.0,
            peak_vram_mb=peak_vram_mb,
        )

    ttfts = np.array([r.ttft_ms for r in successful])
    latencies = np.array([r.total_latency_ms for r in successful])
    itls = np.array([r.itl_ms for r in successful])
    comp_tokens = np.array([r.completion_tokens for r in successful])
    prompt_tokens = np.array([r.prompt_tokens for r in successful])

    total_completion_tokens = int(np.sum(comp_tokens))
    agg_tok_per_sec = (
        round(float(total_completion_tokens / total_duration_s), 2) if total_duration_s > 0 else 0.0
    )
    req_throughput_rps = (
        round(float(len(successful) / total_duration_s), 2) if total_duration_s > 0 else 0.0
    )

    ttft_p50, ttft_p90, ttft_p95, ttft_p99 = np.percentile(ttfts, [50, 90, 95, 99])
    lat_p50, lat_p90, lat_p95, lat_p99 = np.percentile(latencies, [50, 90, 95, 99])
    itl_p50, itl_p95 = np.percentile(itls, [50, 95])

    return ConcurrencyBenchmarkSummary(
        concurrency=concurrency,
        num_requests=len(results),
        successful_requests=len(successful),
        failed_requests=len(failed),
        total_duration_s=round(total_duration_s, 3),
        aggregate_tokens_per_second=agg_tok_per_sec,
        request_throughput_rps=req_throughput_rps,
        ttft_mean_ms=round(float(np.mean(ttfts)), 2),
        ttft_p50_ms=round(float(ttft_p50), 2),
        ttft_p90_ms=round(float(ttft_p90), 2),
        ttft_p95_ms=round(float(ttft_p95), 2),
        ttft_p99_ms=round(float(ttft_p99), 2),
        latency_mean_ms=round(float(np.mean(latencies)), 2),
        latency_p50_ms=round(float(lat_p50), 2),
        latency_p90_ms=round(float(lat_p90), 2),
        latency_p95_ms=round(float(lat_p95), 2),
        latency_p99_ms=round(float(lat_p99), 2),
        itl_mean_ms=round(float(np.mean(itls)), 2),
        itl_p50_ms=round(float(itl_p50), 2),
        itl_p95_ms=round(float(itl_p95), 2),
        mean_completion_tokens=round(float(np.mean(comp_tokens)), 1),
        mean_prompt_tokens=round(float(np.mean(prompt_tokens)), 1),
        peak_vram_mb=round(peak_vram_mb, 1),
    )
