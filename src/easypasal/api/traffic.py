"""Traffic splitting and variant routing engine for EasyPasal (Task T5.3).

Provides deterministic hashing, canonical variant alias mapping, and request-level
variant resolution for side-by-side serving (A/B evaluation and releases).
"""

from __future__ import annotations

import hashlib
import logging

from easypasal.config import TrafficSplitConfig

logger = logging.getLogger("easypasal.api.traffic")

# Canonical variant identifiers
VARIANT_BASELINE_V0_1 = "baseline_v0.1"
VARIANT_GEPA_V0_2 = "gepa_v0.2"

CANONICAL_VARIANTS: frozenset[str] = frozenset(
    {
        VARIANT_BASELINE_V0_1,
        VARIANT_GEPA_V0_2,
    }
)

# Friendly alias mapping to canonical variant IDs
VARIANT_ALIASES: dict[str, str] = {
    "a": VARIANT_BASELINE_V0_1,
    "baseline": VARIANT_BASELINE_V0_1,
    "baseline_v0.1": VARIANT_BASELINE_V0_1,
    "b": VARIANT_GEPA_V0_2,
    "gepa": VARIANT_GEPA_V0_2,
    "gepa_v0.2": VARIANT_GEPA_V0_2,
}

# Mapping canonical variant to underlying prompt variant ("baseline" vs "gepa")
VARIANT_TO_PROMPT: dict[str, str] = {
    VARIANT_BASELINE_V0_1: "baseline",
    VARIANT_GEPA_V0_2: "gepa",
}


def normalize_variant(variant_name: str | None) -> str | None:
    """Normalize user- or client-supplied variant string to a canonical variant ID.

    Args:
        variant_name: Variant string (e.g., 'A', 'baseline', 'gepa_v0.2').

    Returns:
        Canonical variant ID string, or None if input was None or empty.

    Raises:
        ValueError: If input is not a recognized variant or alias.
    """
    if not variant_name or not variant_name.strip():
        return None

    cleaned = variant_name.strip().lower()
    if cleaned in VARIANT_ALIASES:
        return VARIANT_ALIASES[cleaned]

    # Check case-insensitive against canonical variants directly
    for canonical in CANONICAL_VARIANTS:
        if cleaned == canonical.lower():
            return canonical

    supported = sorted(list(CANONICAL_VARIANTS) + ["A", "B", "baseline", "gepa"])
    raise ValueError(
        f"Unknown variant '{variant_name}'. Supported variants and aliases: {supported}"
    )


def compute_deterministic_variant(
    key: str,
    weights: dict[str, float],
    salt: str = "easypasal-traffic-split-v1",
) -> str:
    """Deterministically assign a variant using salted SHA-256 hashing.

    The hash bucket calculation produces a uniform distribution over [0, 1.0)
    that is strictly repeatable for identical input keys without needing database state.

    Args:
        key: Input identifier (e.g., client IP, user ID, or question text).
        weights: Dictionary mapping variant IDs to non-negative relative weights.
        salt: Salt string to prevent hash correlation across experiments.

    Returns:
        The deterministically selected variant ID.
    """
    if not weights:
        return VARIANT_BASELINE_V0_1

    # Clean and normalize weights
    valid_weights: dict[str, float] = {}
    for var, w in weights.items():
        canonical_var = normalize_variant(var) or var
        valid_weights[canonical_var] = max(0.0, float(w))

    total_weight = sum(valid_weights.values())
    if total_weight <= 0.0:
        return next(iter(valid_weights.keys()), VARIANT_BASELINE_V0_1)

    # Hash input key with salt
    hash_payload = f"{key}:{salt}".encode()
    hex_digest = hashlib.sha256(hash_payload).hexdigest()

    # Use first 8 hex characters (32-bit unsigned int) normalized to [0, 1)
    bucket = (int(hex_digest[:8], 16) % 10000) / 10000.0

    # Match against cumulative intervals (sorted by variant name for absolute determinism)
    cumulative = 0.0
    for var_id in sorted(valid_weights.keys()):
        weight = valid_weights[var_id]
        if weight <= 0.0:
            continue
        cumulative += weight / total_weight
        if bucket < cumulative:
            return var_id

    # Fallback to the last variant
    return sorted(valid_weights.keys())[-1]


def resolve_variant(
    payload_variant: str | None = None,
    header_variant: str | None = None,
    client_key: str | None = None,
    traffic_config: TrafficSplitConfig | None = None,
) -> str:
    """Resolve active variant using hierarchical precedence rules.

    Precedence:
    1. Explicit payload override (`payload_variant`)
    2. Explicit request header override (`header_variant`)
    3. Deterministic traffic splitting (if enabled in `traffic_config` and `client_key` given)
    4. Configured `default_variant`
    5. Fallback baseline `baseline_v0.1`

    Args:
        payload_variant: Variant override from request body.
        header_variant: Variant override from HTTP header (e.g. X-EasyPasal-Variant).
        client_key: Identifier for deterministic hashing (e.g. IP or query text).
        traffic_config: Active TrafficSplitConfig instance.

    Returns:
        Canonical variant ID string.

    Raises:
        ValueError: If an explicit variant was supplied but cannot be recognized.
    """
    # 1. Payload override
    norm_payload = normalize_variant(payload_variant)
    if norm_payload is not None:
        return norm_payload

    # 2. Header override
    norm_header = normalize_variant(header_variant)
    if norm_header is not None:
        return norm_header

    # 3. Deterministic traffic splitting
    if traffic_config is not None and traffic_config.enabled and client_key:
        return compute_deterministic_variant(
            key=client_key,
            weights=traffic_config.weights,
            salt=traffic_config.salt,
        )

    # 4. Configured default
    if traffic_config is not None and traffic_config.default_variant:
        norm_default = normalize_variant(traffic_config.default_variant)
        if norm_default is not None:
            return norm_default

    # 5. Global baseline fallback
    return VARIANT_BASELINE_V0_1


def map_variant_to_prompt(variant_id: str) -> str:
    """Map canonical variant ID to pipeline prompt variant name ('baseline' or 'gepa').

    Args:
        variant_id: Canonical variant ID.

    Returns:
        Prompt variant string ('baseline' or 'gepa').
    """
    return VARIANT_TO_PROMPT.get(variant_id, "gepa" if "gepa" in variant_id else "baseline")
