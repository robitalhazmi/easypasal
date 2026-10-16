"""Data models and schemas for Experiment E7 (QLoRA Router & Guardrail Classifier)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CANONICAL_THEMES = [
    "Consumer Protection",
    "IT & Cyber Risk",
    "AML / CFT",
    "Data Privacy & Secrecy",
    "Capital Markets",
    "General Financial Regulation",
    "Out of Scope",
]

ThemeType = Literal[
    "Consumer Protection",
    "IT & Cyber Risk",
    "AML / CFT",
    "Data Privacy & Secrecy",
    "Capital Markets",
    "General Financial Regulation",
    "Out of Scope",
]


class RouterTarget(BaseModel):
    """Structured classification output produced by the router/classifier."""

    theme: str = Field(
        ...,
        description="Regulatory theme (one of canonical themes or 'Out of Scope')",
    )
    is_answerable: bool = Field(
        ...,
        description="True if question is answerable by current Indonesian financial regulations, False otherwise",
    )
    is_injection: bool = Field(
        ...,
        description="True if prompt injection, jailbreak, or system override attempt detected",
    )


class RouterSample(BaseModel):
    """Individual item formatted for router training and evaluation."""

    id: str = Field(..., description="Unique sample identifier")
    question: str = Field(..., description="User query text")
    category: str = Field(..., description="Question category from gold taxonomy")
    target: RouterTarget = Field(..., description="Ground truth router target")
    split: str = Field(..., description="Dataset partition: 'train', 'dev', or 'test'")


class RouterPrediction(BaseModel):
    """Result of inference on a single router sample."""

    id: str
    question: str
    category: str
    ground_truth: RouterTarget
    predicted_target: RouterTarget
    raw_response: str = ""
    latency_ms: float = 0.0
    parse_success: bool = True

    @property
    def exact_match(self) -> bool:
        """True if all three prediction fields match ground truth exactly."""
        return (
            self.predicted_target.theme == self.ground_truth.theme
            and self.predicted_target.is_answerable == self.ground_truth.is_answerable
            and self.predicted_target.is_injection == self.ground_truth.is_injection
        )


class MetricSummary(BaseModel):
    """Precision, Recall, F1, and Accuracy summary for a binary or class label."""

    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1: float = Field(..., ge=0.0, le=1.0)
    accuracy: float = Field(..., ge=0.0, le=1.0)
    support: int = Field(..., ge=0)


class CategoryBreakdown(BaseModel):
    """Regression audit metrics across individual question categories."""

    category: str
    count: int
    theme_accuracy: float
    answerability_accuracy: float
    injection_accuracy: float
    exact_match_rate: float


class E7EvaluationReport(BaseModel):
    """Comprehensive evaluation report for Experiment E7."""

    model_id: str
    mode: str
    split: str
    sample_count: int
    theme_macro_f1: float
    theme_accuracy: float
    theme_breakdown: dict[str, MetricSummary]
    answerability_metrics: MetricSummary
    injection_metrics: MetricSummary
    overall_exact_match: float
    category_breakdown: list[CategoryBreakdown]
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    honest_claims_notice: str = (
        "HONEST-CLAIMS NOTICE (Plan §7 & §13): This evaluation evaluates a narrow "
        "parameter-efficient QLoRA adapter on an offline benchmark dataset. "
        "It does not represent generalized model fine-tuning or live multi-user production telemetry."
    )
