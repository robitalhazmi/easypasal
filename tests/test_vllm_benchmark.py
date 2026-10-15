"""Unit tests for Experiment E6 vLLM serving benchmark harness and metrics."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.e6_vllm.client import AsyncVLLMBenchmarkClient
from experiments.e6_vllm.models import (
    BenchmarkRequestResult,
    HardwareMetadata,
    compute_concurrency_summary,
)
from experiments.e6_vllm.plot_benchmark import generate_benchmark_plots
from experiments.e6_vllm.run_benchmark import (
    CPU_E4_BASELINE,
    load_test_prompts,
    run_concurrency_sweep,
    run_e6_benchmark,
)


def test_benchmark_models_serialization() -> None:
    """Verify dataclasses and serialization for request results and summaries."""
    res = BenchmarkRequestResult(
        request_id="req_001",
        prompt_tokens=120,
        completion_tokens=80,
        ttft_ms=45.2,
        total_latency_ms=2100.5,
        itl_ms=25.8,
        tokens_per_second=38.1,
        status="success",
        response_text="Test response",
    )
    d = res.to_dict()
    assert d["request_id"] == "req_001"
    assert d["ttft_ms"] == 45.2
    assert d["tokens_per_second"] == 38.1
    assert d["status"] == "success"

    hw = HardwareMetadata(
        device_type="GPU",
        device_name="NVIDIA Tesla T4",
        framework="vLLM 0.6.3",
        memory_total_mb=15360.0,
        quantization="AWQ / FP16",
        platform_name="Google Colab",
    )
    hw_d = hw.to_dict()
    assert hw_d["device_type"] == "GPU"
    assert hw_d["memory_total_mb"] == 15360.0


def test_compute_concurrency_summary_percentiles() -> None:
    """Verify metric aggregation, percentiles, and aggregate throughput calculation."""
    results = [
        BenchmarkRequestResult(
            request_id=f"r_{i}",
            prompt_tokens=100,
            completion_tokens=50,
            ttft_ms=40.0 + i * 2.0,
            total_latency_ms=1000.0 + i * 50.0,
            itl_ms=20.0,
            tokens_per_second=50.0,
            status="success",
        )
        for i in range(10)
    ]

    total_duration = 5.0
    summary = compute_concurrency_summary(
        concurrency=4,
        results=results,
        total_duration_s=total_duration,
        peak_vram_mb=4096.0,
    )

    assert summary.concurrency == 4
    assert summary.num_requests == 10
    assert summary.successful_requests == 10
    assert summary.failed_requests == 0
    # Total tokens = 10 * 50 = 500, duration = 5.0s -> 100 tok/s
    assert summary.aggregate_tokens_per_second == 100.0
    assert summary.request_throughput_rps == 2.0
    assert summary.ttft_mean_ms > 40.0
    assert summary.latency_p50_ms > 1000.0
    assert summary.latency_p95_ms > summary.latency_p50_ms
    assert summary.peak_vram_mb == 4096.0

    # Test empty / all-failed handling
    failed_results = [
        BenchmarkRequestResult(
            request_id="fail_1",
            prompt_tokens=10,
            completion_tokens=0,
            ttft_ms=0.0,
            total_latency_ms=500.0,
            itl_ms=0.0,
            tokens_per_second=0.0,
            status="error",
            error_message="timeout",
        )
    ]
    empty_summary = compute_concurrency_summary(
        concurrency=1, results=failed_results, total_duration_s=1.0
    )
    assert empty_summary.successful_requests == 0
    assert empty_summary.failed_requests == 1
    assert empty_summary.aggregate_tokens_per_second == 0.0


def test_async_vllm_client_mock() -> None:
    """Verify AsyncVLLMBenchmarkClient behavior in simulation/mock mode."""

    async def _test() -> None:
        client = AsyncVLLMBenchmarkClient(mock=True)
        res = await client.generate_stream(
            request_id="req_test",
            prompt="Pertanyaan tentang batas modal bank",
            max_tokens=120,
            concurrency_context=4,
        )

        assert res.status == "success"
        assert res.completion_tokens > 0
        assert res.ttft_ms > 0.0
        assert res.total_latency_ms > res.ttft_ms
        assert res.tokens_per_second > 0.0
        assert "Mock vLLM Response" in res.response_text

    import asyncio

    asyncio.run(_test())


def test_run_concurrency_sweep_mock() -> None:
    """Verify concurrency sweep execution with semaphore throttling in mock mode."""

    async def _test() -> None:
        client = AsyncVLLMBenchmarkClient(mock=True)
        prompts = ["Pertanyaan 1", "Pertanyaan 2"]
        summary = await run_concurrency_sweep(
            client=client,
            concurrency=2,
            num_requests=4,
            prompts=prompts,
            max_tokens=80,
        )

        assert summary.concurrency == 2
        assert summary.num_requests == 4
        assert summary.successful_requests == 4
        assert summary.aggregate_tokens_per_second > 0.0
        assert summary.ttft_p95_ms > 0.0

    import asyncio

    asyncio.run(_test())


def test_plot_generation(tmp_path: Path) -> None:
    """Verify publication plot generation runs without display errors."""
    mock_report_dict = {
        "concurrency_levels": [1, 4, 16, 32],
        "summaries": {
            "1": {
                "aggregate_tokens_per_second": 38.4,
                "latency_p95_ms": 4200.0,
                "ttft_p95_ms": 48.0,
            },
            "4": {
                "aggregate_tokens_per_second": 148.2,
                "latency_p95_ms": 4600.0,
                "ttft_p95_ms": 82.0,
            },
            "16": {
                "aggregate_tokens_per_second": 365.8,
                "latency_p95_ms": 7100.0,
                "ttft_p95_ms": 185.0,
            },
            "32": {
                "aggregate_tokens_per_second": 482.5,
                "latency_p95_ms": 11800.0,
                "ttft_p95_ms": 340.0,
            },
        },
        "cpu_baseline_comparison": CPU_E4_BASELINE,
    }

    plots = generate_benchmark_plots(mock_report_dict, output_dir=tmp_path)
    assert len(plots) == 4
    for p in plots:
        assert p.exists()
        assert p.stat().st_size > 1000  # Non-trivial image content


def test_load_test_prompts() -> None:
    """Verify prompt loading fallback when path does not exist."""
    prompts = load_test_prompts("non_existent_file.jsonl", fallback_count=5)
    assert len(prompts) == 5
    assert all("Kueri Regulasi Keuangan" in p for p in prompts)


def test_run_e6_benchmark_end_to_end_mock(tmp_path: Path) -> None:
    """Verify complete E6 benchmark workflow produces valid JSON and plot artifacts."""

    async def _test() -> None:
        out_json = tmp_path / "e6_results.json"
        plot_dir = tmp_path / "assets"

        report = await run_e6_benchmark(
            concurrencies=[1, 2],
            num_requests_per_level=2,
            max_tokens=60,
            mock=True,
            output_file=out_json,
            plot_dir=plot_dir,
            prompts_file="non_existent.jsonl",
        )

        assert out_json.exists()
        assert report.concurrency_levels == [1, 2]
        assert 1 in report.summaries
        assert 2 in report.summaries

        with open(out_json, encoding="utf-8") as f:
            data = json.load(f)
        assert "summaries" in data
        assert "cpu_baseline_comparison" in data

    import asyncio

    asyncio.run(_test())
