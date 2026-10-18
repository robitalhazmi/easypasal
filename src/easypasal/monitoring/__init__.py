"""Monitoring package: request logging, PII redaction, drift reporting, alerting, and telemetry schemas."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from easypasal.monitoring.alerts import (
        evaluate_threshold_alerts,
        run_alert_drill,
    )
    from easypasal.monitoring.drift import (
        build_current_dataset_from_db,
        build_current_dataset_from_file,
        build_gold_reference_dataset,
        generate_drift_report,
    )
    from easypasal.monitoring.logger import RequestLogger
    from easypasal.monitoring.redaction import has_pii, redact_pii
    from easypasal.monitoring.schemas import (
        AlertCheckResult,
        AlertEvent,
        AlertThresholds,
        FeedbackLogRecord,
        FeedbackPayload,
        RequestLogRecord,
    )

__all__ = [
    "AlertCheckResult",
    "AlertEvent",
    "AlertThresholds",
    "FeedbackLogRecord",
    "FeedbackPayload",
    "RequestLogRecord",
    "RequestLogger",
    "build_current_dataset_from_db",
    "build_current_dataset_from_file",
    "build_gold_reference_dataset",
    "evaluate_threshold_alerts",
    "generate_drift_report",
    "has_pii",
    "redact_pii",
    "run_alert_drill",
]

_ALERT_SYMBOLS = {"evaluate_threshold_alerts", "run_alert_drill"}
_DRIFT_SYMBOLS = {
    "build_current_dataset_from_db",
    "build_current_dataset_from_file",
    "build_gold_reference_dataset",
    "generate_drift_report",
}
_LOGGER_SYMBOLS = {"RequestLogger"}
_REDACTION_SYMBOLS = {"has_pii", "redact_pii"}
_SCHEMA_SYMBOLS = {
    "AlertCheckResult",
    "AlertEvent",
    "AlertThresholds",
    "FeedbackLogRecord",
    "FeedbackPayload",
    "RequestLogRecord",
}


def __getattr__(name: str) -> object:
    if name in _ALERT_SYMBOLS:
        import easypasal.monitoring.alerts as _alerts

        return getattr(_alerts, name)
    if name in _DRIFT_SYMBOLS:
        import easypasal.monitoring.drift as _drift

        return getattr(_drift, name)
    if name in _LOGGER_SYMBOLS:
        import easypasal.monitoring.logger as _logger

        return getattr(_logger, name)
    if name in _REDACTION_SYMBOLS:
        import easypasal.monitoring.redaction as _redaction

        return getattr(_redaction, name)
    if name in _SCHEMA_SYMBOLS:
        import easypasal.monitoring.schemas as _schemas

        return getattr(_schemas, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
