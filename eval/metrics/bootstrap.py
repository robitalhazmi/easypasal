"""Non-parametric Bootstrap Confidence Intervals and paired significance testing.

Provides percentile bootstrap CIs, Monte Carlo sign-flip permutation tests,
and Cohen's d_z effect sizes for comparing RAG pipeline variants and experiments.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    """Non-parametric percentile bootstrap confidence interval."""

    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float = 0.95

    def __str__(self) -> str:
        """Format as point_estimate [lower, upper]."""
        return f"{self.point_estimate:.4f} [{self.lower_bound:.4f}, {self.upper_bound:.4f}]"

    def to_dict(self) -> dict[str, Any]:
        """Convert CI to serializable dictionary."""
        return {
            "point_estimate": round(self.point_estimate, 4),
            "lower_bound": round(self.lower_bound, 4),
            "upper_bound": round(self.upper_bound, 4),
            "confidence_level": self.confidence_level,
            "formatted": str(self),
        }


@dataclass(frozen=True)
class PairedComparisonResult:
    """Statistical summary of paired variant comparison on a single metric."""

    metric: str
    baseline_mean: float
    baseline_ci: BootstrapCI
    candidate_mean: float
    candidate_ci: BootstrapCI
    delta_mean: float
    delta_ci: BootstrapCI
    p_value: float
    cohens_d: float
    is_significant: bool
    status: str
    status_badge: str

    def to_dict(self) -> dict[str, Any]:
        """Convert comparison result to serializable dictionary."""
        return {
            "metric": self.metric,
            "baseline_mean": round(self.baseline_mean, 4),
            "baseline_ci": self.baseline_ci.to_dict(),
            "candidate_mean": round(self.candidate_mean, 4),
            "candidate_ci": self.candidate_ci.to_dict(),
            "delta_mean": round(self.delta_mean, 4),
            "delta_ci": self.delta_ci.to_dict(),
            "p_value": round(self.p_value, 5),
            "cohens_d": round(self.cohens_d, 4),
            "is_significant": self.is_significant,
            "status": self.status,
            "status_badge": self.status_badge,
        }


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Compute non-parametric percentile bootstrap confidence interval of the sample mean.

    Args:
        values: Sequence of numerical metric observations across queries.
        n_resamples: Number of bootstrap iterations (default 2000).
        confidence: Desired confidence level (default 0.95 for 95% CI).
        seed: Random seed for exact reproducibility.

    Returns:
        BootstrapCI containing sample mean, lower bound, and upper bound.
    """
    if not values:
        return BootstrapCI(
            point_estimate=0.0,
            lower_bound=0.0,
            upper_bound=0.0,
            confidence_level=confidence,
        )

    arr = np.asarray(values, dtype=np.float64)
    point_estimate = float(np.mean(arr))

    # Constant sequence or single observation
    if len(arr) == 1 or np.all(arr == arr[0]):
        return BootstrapCI(
            point_estimate=point_estimate,
            lower_bound=point_estimate,
            upper_bound=point_estimate,
            confidence_level=confidence,
        )

    rng = np.random.default_rng(seed)
    n = len(arr)
    indices = rng.integers(0, n, size=(n_resamples, n))
    resample_means = np.mean(arr[indices], axis=1)

    alpha = 1.0 - confidence
    lower = float(np.percentile(resample_means, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(resample_means, 100.0 * (1.0 - alpha / 2.0)))

    return BootstrapCI(
        point_estimate=point_estimate,
        lower_bound=lower,
        upper_bound=upper,
        confidence_level=confidence,
    )


def paired_permutation_test(
    baseline: Sequence[float],
    candidate: Sequence[float],
    n_permutations: int = 5000,
    seed: int = 42,
) -> float:
    """Perform a two-sided Monte Carlo paired permutation test on sample differences.

    Tests the null hypothesis H0: mu_delta = 0 by randomly flipping the signs of
    paired differences delta_i = candidate_i - baseline_i.

    Args:
        baseline: Baseline metric scores per query.
        candidate: Candidate metric scores per query (must match length of baseline).
        n_permutations: Number of random sign-flip permutations (default 5000).
        seed: Random seed for exact reproducibility.

    Returns:
        Two-sided empirical p-value.
    """
    if len(baseline) != len(candidate):
        raise ValueError(
            f"Baseline and candidate sequences must have equal length: "
            f"{len(baseline)} vs {len(candidate)}"
        )
    if not baseline:
        raise ValueError("Cannot perform permutation test on empty sequences")

    base_arr = np.asarray(baseline, dtype=np.float64)
    cand_arr = np.asarray(candidate, dtype=np.float64)
    deltas = cand_arr - base_arr
    obs_delta = float(np.mean(deltas))

    # Identical runs have zero difference under all permutations
    if np.all(deltas == 0.0):
        return 1.0

    rng = np.random.default_rng(seed)
    n = len(deltas)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    perm_deltas = np.mean(signs * deltas, axis=1)

    count_extreme = int(np.sum(np.abs(perm_deltas) >= np.abs(obs_delta)))
    # Standard Monte Carlo p-value with continuity correction
    p_val = float((count_extreme + 1) / (n_permutations + 1))
    return min(1.0, p_val)


def paired_bootstrap_comparison(
    baseline: Sequence[float],
    candidate: Sequence[float],
    metric_name: str,
    n_resamples: int = 2000,
    n_permutations: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
    higher_is_better: bool = True,
) -> PairedComparisonResult:
    """Compute complete paired variant comparison with 95% CIs and significance test.

    Computes:
    - Baseline 95% Bootstrap CI
    - Candidate 95% Bootstrap CI
    - Paired difference Delta = Candidate - Baseline 95% Bootstrap CI
    - Two-sided Monte Carlo paired permutation p-value
    - Paired Cohen's d_z effect size: mean(delta) / std(delta)
    - Significance determination (p < 0.05 and 0 not in delta CI)
    - Status badge: SIG GAIN, REGRESSION, or NEUTRAL

    Args:
        baseline: Baseline scores per query.
        candidate: Candidate scores per query.
        metric_name: Name of the metric (e.g., 'Recall@5', 'MRR', 'Latency_ms').
        n_resamples: Number of bootstrap resamples (default 2000).
        n_permutations: Number of permutation iterations (default 5000).
        confidence: Confidence level (default 0.95).
        seed: Random seed for reproducibility.
        higher_is_better: True if higher is improved (e.g. recall), False for latency.

    Returns:
        PairedComparisonResult with full statistical breakdown.
    """
    if len(baseline) != len(candidate):
        raise ValueError(
            f"Baseline and candidate sequences must have equal length: "
            f"{len(baseline)} vs {len(candidate)}"
        )
    if not baseline:
        raise ValueError("Cannot perform paired comparison on empty sequences")

    base_ci = bootstrap_ci(baseline, n_resamples=n_resamples, confidence=confidence, seed=seed)
    cand_ci = bootstrap_ci(candidate, n_resamples=n_resamples, confidence=confidence, seed=seed + 1)

    base_arr = np.asarray(baseline, dtype=np.float64)
    cand_arr = np.asarray(candidate, dtype=np.float64)
    deltas = cand_arr - base_arr
    delta_mean = float(np.mean(deltas))

    # Bootstrap CI on paired difference
    delta_ci = bootstrap_ci(
        deltas.tolist(), n_resamples=n_resamples, confidence=confidence, seed=seed + 2
    )

    # Permutation test p-value
    p_val = paired_permutation_test(
        baseline, candidate, n_permutations=n_permutations, seed=seed + 3
    )

    # Cohen's d_z effect size
    if len(deltas) > 1:
        delta_std = float(np.std(deltas, ddof=1))
        cohens_d = float(delta_mean / delta_std) if delta_std > 1e-12 else 0.0
    else:
        cohens_d = 0.0

    # Statistical significance: p < 0.05 AND 0 not enclosed within the 95% bootstrap CI of delta
    alpha = 1.0 - confidence
    zero_in_ci = delta_ci.lower_bound <= 0.0 <= delta_ci.upper_bound
    is_significant = (p_val < alpha) and (not zero_in_ci)

    if higher_is_better:
        if is_significant and delta_mean > 0:
            status = "SIG_GAIN"
            badge = "✅ SIG GAIN"
        elif is_significant and delta_mean < 0:
            status = "REGRESSION"
            badge = "❌ REGRESSION"
        else:
            status = "NEUTRAL"
            badge = "ℹ️ NEUTRAL"
    else:
        # Lower is better (e.g. latency)
        if is_significant and delta_mean < 0:
            status = "SIG_GAIN"
            badge = "✅ SIG GAIN"
        elif is_significant and delta_mean > 0:
            status = "REGRESSION"
            badge = "❌ REGRESSION"
        else:
            status = "NEUTRAL"
            badge = "ℹ️ NEUTRAL"

    return PairedComparisonResult(
        metric=metric_name,
        baseline_mean=base_ci.point_estimate,
        baseline_ci=base_ci,
        candidate_mean=cand_ci.point_estimate,
        candidate_ci=cand_ci,
        delta_mean=delta_mean,
        delta_ci=delta_ci,
        p_value=p_val,
        cohens_d=cohens_d,
        is_significant=is_significant,
        status=status,
        status_badge=badge,
    )
