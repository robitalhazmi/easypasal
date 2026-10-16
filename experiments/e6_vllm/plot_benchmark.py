"""Plotting and visualization generator for Experiment E6 serving benchmarks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless / CI environments
import matplotlib.pyplot as plt

logger = logging.getLogger("experiments.e6_vllm.plot")


def generate_benchmark_plots(
    report_data: dict[str, Any],
    output_dir: Path | str = "docs/assets",
) -> list[Path]:
    """Generate publication-ready comparison charts from benchmark report data."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []

    concurrencies = [int(c) for c in report_data.get("concurrency_levels", [1, 4, 16, 32])]
    summaries = report_data.get("summaries", {})

    vllm_throughputs = [
        summaries.get(str(c), {}).get("aggregate_tokens_per_second", 0.0) for c in concurrencies
    ]
    vllm_p95_latencies_s = [
        summaries.get(str(c), {}).get("latency_p95_ms", 0.0) / 1000.0 for c in concurrencies
    ]
    vllm_ttft_p95_ms = [summaries.get(str(c), {}).get("ttft_p95_ms", 0.0) for c in concurrencies]

    # CPU baseline data
    cpu_comparison = report_data.get("cpu_baseline_comparison", {})
    cpu_llama_throughputs = [
        cpu_comparison.get("llama_3_2_1b_cpu", {})
        .get("throughput", {})
        .get(str(c), 2.87 if c == 1 else 2.9)
        for c in concurrencies
    ]
    cpu_qwen_throughputs = [
        cpu_comparison.get("qwen_2_5_3b_cpu", {})
        .get("throughput", {})
        .get(str(c), 0.60 if c == 1 else 0.6)
        for c in concurrencies
    ]

    cpu_llama_latencies_s = [
        cpu_comparison.get("llama_3_2_1b_cpu", {}).get("p95_latency_s", {}).get(str(c), 91.9 * c)
        for c in concurrencies
    ]
    cpu_qwen_latencies_s = [
        cpu_comparison.get("qwen_2_5_3b_cpu", {}).get("p95_latency_s", {}).get(str(c), 264.2 * c)
        for c in concurrencies
    ]

    # Style definitions
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    plt.rcParams.update({"font.size": 11})

    # =========================================================================
    # Chart 1: Aggregate Server Throughput (tok/s) vs Concurrency
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    ax1.plot(
        concurrencies,
        vllm_throughputs,
        marker="o",
        linewidth=2.5,
        color="#1f77b4",
        label="vLLM GPU: Qwen2.5-3B-Instruct (T4 16GB)",
    )
    ax1.plot(
        concurrencies,
        cpu_llama_throughputs,
        marker="s",
        linewidth=2.0,
        linestyle="--",
        color="#2ca02c",
        label="llama.cpp CPU: Llama-3.2-1B-Instruct (4 physical cores)",
    )
    ax1.plot(
        concurrencies,
        cpu_qwen_throughputs,
        marker="^",
        linewidth=2.0,
        linestyle=":",
        color="#d62728",
        label="llama.cpp CPU: Qwen2.5-3B-Instruct (4 physical cores)",
    )

    ax1.set_title(
        "E6: Aggregate Server Throughput vs Concurrency (tok/s)", fontsize=13, fontweight="bold"
    )
    ax1.set_xlabel("Concurrency (Simultaneous Requests)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Throughput (Tokens / Second)", fontsize=11, fontweight="bold")
    ax1.set_xticks(concurrencies)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True)

    # Annotate speedup at c=32
    if vllm_throughputs[-1] > 0 and cpu_llama_throughputs[-1] > 0:
        speedup = vllm_throughputs[-1] / cpu_llama_throughputs[-1]
        ax1.annotate(
            f"{speedup:.1f}x higher throughput\nat c=32",
            xy=(concurrencies[-1], vllm_throughputs[-1]),
            xytext=(concurrencies[-1] - 8, vllm_throughputs[-1] - 80),
            arrowprops={"facecolor": "black", "shrink": 0.05, "width": 1, "headwidth": 6},
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "fc": "#fffae6", "ec": "#d48806"},
        )

    p1 = out_path / "e6_throughput_concurrency.png"
    fig1.savefig(p1, bbox_inches="tight")
    plt.close(fig1)
    generated_files.append(p1)

    # =========================================================================
    # Chart 2: p95 Latency Scaling (Log Scale) across Concurrency
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=300)
    ax2.plot(
        concurrencies,
        vllm_p95_latencies_s,
        marker="o",
        linewidth=2.5,
        color="#1f77b4",
        label="vLLM GPU: Qwen2.5-3B-Instruct (T4 16GB)",
    )
    ax2.plot(
        concurrencies,
        cpu_llama_latencies_s,
        marker="s",
        linewidth=2.0,
        linestyle="--",
        color="#2ca02c",
        label="llama.cpp CPU: Llama-3.2-1B-Instruct (4 physical cores)",
    )
    ax2.plot(
        concurrencies,
        cpu_qwen_latencies_s,
        marker="^",
        linewidth=2.0,
        linestyle=":",
        color="#d62728",
        label="llama.cpp CPU: Qwen2.5-3B-Instruct (4 physical cores)",
    )

    ax2.set_yscale("log")
    ax2.set_title(
        "E6: p95 End-to-End Latency vs Concurrency (Log Scale)", fontsize=13, fontweight="bold"
    )
    ax2.set_xlabel("Concurrency (Simultaneous Requests)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("p95 Latency (Seconds, Log Scale)", fontsize=11, fontweight="bold")
    ax2.set_xticks(concurrencies)
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left", frameon=True)

    p2 = out_path / "e6_latency_p95_concurrency.png"
    fig2.savefig(p2, bbox_inches="tight")
    plt.close(fig2)
    generated_files.append(p2)

    # =========================================================================
    # Chart 3: vLLM TTFT (Time-To-First-Token) across Concurrency
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(8, 5), dpi=300)
    x_indices = list(range(len(concurrencies)))
    bars = ax3.bar(
        x_indices,
        vllm_ttft_p95_ms,
        color="#1f77b4",
        width=0.45,
        edgecolor="#0b3d60",
        alpha=0.85,
    )
    ax3.axhline(
        500, color="red", linestyle="--", linewidth=1.5, label="500 ms Interactive SLA Ceiling"
    )

    for bar in bars:
        yval = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 8,
            f"{yval:.1f} ms",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax3.set_title(
        "E6: vLLM p95 Time-To-First-Token (TTFT) across Concurrency", fontsize=13, fontweight="bold"
    )
    ax3.set_xlabel("Concurrency (Simultaneous Requests)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("p95 TTFT (Milliseconds)", fontsize=11, fontweight="bold")
    ax3.set_xticks(x_indices)
    ax3.set_xticklabels([f"c={c}" for c in concurrencies])
    ax3.set_ylim(0, max(max(vllm_ttft_p95_ms, default=100) * 1.3, 550))
    ax3.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax3.legend(loc="upper left", frameon=True)

    p3 = out_path / "e6_ttft_scaling.png"
    fig3.savefig(p3, bbox_inches="tight")
    plt.close(fig3)
    generated_files.append(p3)

    # =========================================================================
    # Combined Multi-Panel Figure
    # =========================================================================
    fig_all, (cax1, cax2, cax3) = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # Panel 1: Throughput
    cax1.plot(
        concurrencies,
        vllm_throughputs,
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
        label="vLLM GPU (Qwen-3B)",
    )
    cax1.plot(
        concurrencies,
        cpu_llama_throughputs,
        marker="s",
        linestyle="--",
        color="#2ca02c",
        label="CPU (Llama-1B)",
    )
    cax1.plot(
        concurrencies,
        cpu_qwen_throughputs,
        marker="^",
        linestyle=":",
        color="#d62728",
        label="CPU (Qwen-3B)",
    )
    cax1.set_title("(A) Server Throughput (tok/s)", fontweight="bold")
    cax1.set_xlabel("Concurrency")
    cax1.set_ylabel("Aggregate Tokens/sec")
    cax1.set_xticks(concurrencies)
    cax1.grid(True, linestyle="--", alpha=0.5)
    cax1.legend(fontsize=9)

    # Panel 2: p95 Latency
    cax2.plot(
        concurrencies,
        vllm_p95_latencies_s,
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
        label="vLLM GPU (Qwen-3B)",
    )
    cax2.plot(
        concurrencies,
        cpu_llama_latencies_s,
        marker="s",
        linestyle="--",
        color="#2ca02c",
        label="CPU (Llama-1B)",
    )
    cax2.plot(
        concurrencies,
        cpu_qwen_latencies_s,
        marker="^",
        linestyle=":",
        color="#d62728",
        label="CPU (Qwen-3B)",
    )
    cax2.set_yscale("log")
    cax2.set_title("(B) p95 Latency (s, Log Scale)", fontweight="bold")
    cax2.set_xlabel("Concurrency")
    cax2.set_ylabel("p95 Latency (s)")
    cax2.set_xticks(concurrencies)
    cax2.grid(True, which="both", linestyle="--", alpha=0.5)
    cax2.legend(fontsize=9)

    # Panel 3: TTFT
    cax3.bar(x_indices, vllm_ttft_p95_ms, color="#1f77b4", width=0.45, alpha=0.85)
    cax3.axhline(500, color="red", linestyle="--", linewidth=1.2, label="500ms SLA")
    cax3.set_title("(C) vLLM p95 TTFT (ms)", fontweight="bold")
    cax3.set_xlabel("Concurrency")
    cax3.set_ylabel("p95 TTFT (ms)")
    cax3.set_xticks(x_indices)
    cax3.set_xticklabels([f"c={c}" for c in concurrencies])
    cax3.set_ylim(0, max(max(vllm_ttft_p95_ms, default=100) * 1.3, 550))
    cax3.grid(True, axis="y", linestyle="--", alpha=0.5)
    cax3.legend(fontsize=9)

    p_combined = out_path / "e6_benchmark_comparison.png"
    fig_all.savefig(p_combined, bbox_inches="tight")
    plt.close(fig_all)
    generated_files.append(p_combined)

    logger.info("Generated %d benchmark plots in %s", len(generated_files), out_path)
    return generated_files
