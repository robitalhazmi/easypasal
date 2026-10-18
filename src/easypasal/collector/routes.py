"""FastAPI route handlers for VPS log collector (Plan §9 / Task T7.2).

Provides endpoints for request ingestion, batch streaming, feedback aggregation,
real-time telemetry querying, and unauthenticated health and resource monitoring.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse

from easypasal.collector.auth import get_collector_config, verify_collector_token
from easypasal.collector.config import CollectorConfig
from easypasal.collector.db import CollectorDB
from easypasal.monitoring.schemas import (
    AlertCheckResult,
    AlertEvent,
    AlertThresholds,
    FeedbackLogRecord,
    RequestLogRecord,
)

logger = logging.getLogger("easypasal.collector.routes")

router = APIRouter(tags=["collector"])


def get_db(request: Request) -> CollectorDB:
    """Retrieve or initialize the CollectorDB instance stored on app state."""
    db: CollectorDB | None = getattr(request.app.state, "db", None)
    if db is None:
        config = get_collector_config(request)
        db = CollectorDB(db_path=config.db_path, enable_wal=config.enable_wal)
        request.app.state.db = db
    return db


def get_memory_usage() -> dict[str, float]:
    """Retrieve process memory usage (RSS) in megabytes without external dependencies."""
    rss_mb = 0.0
    peak_rss_mb = 0.0

    try:
        if os.path.exists("/proc/self/status"):
            with open("/proc/self/status", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            rss_mb = round(float(parts[1]) / 1024.0, 2)
                        break
    except Exception:
        pass

    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            peak_rss_mb = round(ru / (1024.0 * 1024.0), 2)
        else:
            peak_rss_mb = round(ru / 1024.0, 2)
    except Exception:
        pass

    if rss_mb == 0.0:
        rss_mb = peak_rss_mb

    return {
        "rss_mb": rss_mb,
        "peak_rss_mb": peak_rss_mb,
    }


@router.get("/health", summary="Service health and resource usage probe")
def health_probe(
    db: Annotated[CollectorDB, Depends(get_db)],
) -> dict[str, Any]:
    """Public unauthenticated health and telemetry probe.

    Returns memory budget metrics and SQLite storage diagnostics.
    """
    mem = get_memory_usage()
    db_info = db.get_db_info()
    return {
        "status": "healthy",
        "service": "easypasal-collector",
        "memory": {
            "current_rss_mb": mem["rss_mb"],
            "peak_rss_mb": mem["peak_rss_mb"],
            "budget_mb": 90.0,
            "within_budget": mem["rss_mb"] <= 90.0,
        },
        "database": db_info,
    }


@router.post(
    "/v1/logs/requests",
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a single request log record",
)
def ingest_request(
    record: RequestLogRecord,
    db: Annotated[CollectorDB, Depends(get_db)],
    _token: Annotated[str, Depends(verify_collector_token)],
) -> dict[str, Any]:
    """Persist an individual request log entry into the SQLite relational store."""
    try:
        db.insert_request(record)
        return {"status": "ok", "request_id": record.request_id}
    except Exception as exc:
        logger.error("Failed to insert request log %s: %s", record.request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database insertion failed: {exc}",
        ) from exc


@router.post(
    "/v1/logs/batch",
    status_code=status.HTTP_201_CREATED,
    summary="Atomically ingest a batch of request logs",
)
def ingest_batch(
    records: list[RequestLogRecord],
    db: Annotated[CollectorDB, Depends(get_db)],
    config: Annotated[CollectorConfig, Depends(get_collector_config)],
    _token: Annotated[str, Depends(verify_collector_token)],
) -> dict[str, Any]:
    """Atomically insert up to max_batch_size request logs in a single transaction."""
    if len(records) > config.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Batch size {len(records)} exceeds maximum allowed limit of {config.max_batch_size}."
            ),
        )

    try:
        count = db.insert_requests_batch(records)
        return {"status": "ok", "inserted": count}
    except Exception as exc:
        logger.error("Failed to insert batch of %d records: %s", len(records), exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch insertion failed: {exc}",
        ) from exc


@router.post(
    "/v1/logs/feedback",
    status_code=status.HTTP_201_CREATED,
    summary="Ingest user feedback",
)
def ingest_feedback(
    feedback: FeedbackLogRecord,
    db: Annotated[CollectorDB, Depends(get_db)],
    _token: Annotated[str, Depends(verify_collector_token)],
) -> dict[str, Any]:
    """Persist user feedback and link rating/comments to the original request record."""
    try:
        db.insert_feedback(feedback)
        return {"status": "ok", "request_id": feedback.request_id}
    except Exception as exc:
        logger.error("Failed to insert feedback for %s: %s", feedback.request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback insertion failed: {exc}",
        ) from exc


@router.get(
    "/v1/logs/requests",
    summary="Query request logs with filtering and pagination",
)
def query_requests(
    db: Annotated[CollectorDB, Depends(get_db)],
    _token: Annotated[str, Depends(verify_collector_token)],
    variant: Annotated[str | None, Query(description="Filter by pipeline variant ID")] = None,
    theme: Annotated[str | None, Query(description="Filter by legal theme")] = None,
    abstained: Annotated[bool | None, Query(description="Filter by abstention status")] = None,
    start_time: Annotated[str | None, Query(description="ISO 8601 start timestamp filter")] = None,
    end_time: Annotated[str | None, Query(description="ISO 8601 end timestamp filter")] = None,
    limit: Annotated[int, Query(ge=1, le=1000, description="Page size")] = 100,
    offset: Annotated[int, Query(ge=0, description="Offset")] = 0,
) -> dict[str, Any]:
    """Retrieve paginated request log records matching search parameters."""
    items, total = db.get_requests(
        variant=variant,
        theme=theme,
        abstained=abstained,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset,
    )
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get(
    "/v1/logs/feedback",
    summary="Query user feedback logs",
)
def query_feedback(
    db: Annotated[CollectorDB, Depends(get_db)],
    _token: Annotated[str, Depends(verify_collector_token)],
    request_id: Annotated[str | None, Query(description="Filter by request ID")] = None,
    rating: Annotated[int | None, Query(description="Filter by rating (+1 or -1)")] = None,
    limit: Annotated[int, Query(ge=1, le=1000, description="Page size")] = 100,
    offset: Annotated[int, Query(ge=0, description="Offset")] = 0,
) -> dict[str, Any]:
    """Retrieve feedback entries matching query criteria."""
    items = db.get_feedback(
        request_id=request_id,
        rating=rating,
        limit=limit,
        offset=offset,
    )
    return {
        "items": items,
        "limit": limit,
        "offset": offset,
    }


@router.get(
    "/v1/stats/summary",
    summary="Retrieve telemetry summary and latency percentiles",
)
def get_stats_summary(
    db: Annotated[CollectorDB, Depends(get_db)],
    _token: Annotated[str, Depends(verify_collector_token)],
) -> dict[str, Any]:
    """Compute aggregate request counts, p50/p95/p99 latencies, token consumption, and abstain rates."""
    return db.get_stats_summary()


@router.get(
    "/v1/reports/drift",
    summary="Compute or retrieve Evidently data drift summary",
)
def get_drift_report_summary(
    db: Annotated[CollectorDB, Depends(get_db)],
    _token: Annotated[str, Depends(verify_collector_token)],
) -> dict[str, Any]:
    """Return structured data drift summary comparing production query logs to the gold dataset."""
    from easypasal.monitoring.drift import (
        build_current_dataset_from_db,
        build_gold_reference_dataset,
        generate_drift_report,
    )

    try:
        ref_df = build_gold_reference_dataset()
        cur_df = build_current_dataset_from_db(db)
        summary = generate_drift_report(
            reference_df=ref_df,
            current_df=cur_df,
            output_html_path="reports/drift_report.html",
            output_json_path="reports/drift_summary.json",
        )
        return summary
    except Exception as exc:
        logger.error("Failed to generate drift report: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drift report generation failed: {exc}",
        ) from exc


@router.get(
    "/reports/drift",
    response_class=HTMLResponse,
    summary="Interactive Evidently Data Drift HTML Report",
)
def get_drift_html_report(
    db: Annotated[CollectorDB, Depends(get_db)],
) -> HTMLResponse:
    """Serve the interactive HTML Evidently drift report."""
    html_path = Path("reports/drift_report.html")
    if html_path.exists():
        with open(html_path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    from easypasal.monitoring.drift import (
        build_current_dataset_from_db,
        build_gold_reference_dataset,
        generate_drift_report,
    )

    ref_df = build_gold_reference_dataset()
    cur_df = build_current_dataset_from_db(db)
    generate_drift_report(
        reference_df=ref_df,
        current_df=cur_df,
        output_html_path=html_path,
        output_json_path="reports/drift_summary.json",
    )
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@router.post(
    "/v1/alerts/check",
    summary="Evaluate operational metrics against alert thresholds",
    response_model=AlertCheckResult,
)
def check_threshold_alerts(
    db: Annotated[CollectorDB, Depends(get_db)],
    _token: Annotated[str, Depends(verify_collector_token)],
    thresholds: AlertThresholds | None = None,
) -> AlertCheckResult:
    """Evaluate current telemetry metrics against thresholds, persisting any triggered alert events."""
    from easypasal.monitoring.alerts import evaluate_threshold_alerts

    return evaluate_threshold_alerts(db=db, thresholds=thresholds)


@router.post(
    "/v1/alerts/drill",
    summary="Execute an automated alert drill test",
    response_model=AlertEvent,
)
def execute_alert_drill(
    db: Annotated[CollectorDB, Depends(get_db)],
    _token: Annotated[str, Depends(verify_collector_token)],
    metric: Annotated[
        str,
        Query(description="Target metric: abstain_rate, p95_latency_ms, or low_evidence_rate"),
    ] = "abstain_rate",
) -> AlertEvent:
    """Trigger a synthetic alert breach to test end-to-end alerting infrastructure (Plan §9 / Task T7.5 AC)."""
    from easypasal.monitoring.alerts import run_alert_drill

    return run_alert_drill(db=db, metric_to_test=metric)


@router.get(
    "/v1/alerts",
    summary="Query persisted alert events",
)
def query_alerts(
    db: Annotated[CollectorDB, Depends(get_db)],
    _token: Annotated[str, Depends(verify_collector_token)],
    metric_name: Annotated[str | None, Query(description="Filter by metric name")] = None,
    severity: Annotated[
        str | None, Query(description="Filter by severity (WARNING or CRITICAL)")
    ] = None,
    is_drill: Annotated[bool | None, Query(description="Filter by drill flag")] = None,
    limit: Annotated[int, Query(ge=1, le=500, description="Page limit")] = 50,
    offset: Annotated[int, Query(ge=0, description="Offset")] = 0,
) -> dict[str, Any]:
    """Retrieve historical alert events with pagination and filtering."""
    items, total = db.get_alerts(
        metric_name=metric_name,
        severity=severity,
        is_drill=is_drill,
        limit=limit,
        offset=offset,
    )
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get(
    "/v1/alerts/active",
    summary="Query active alerts within recent lookback window",
)
def get_active_alerts(
    db: Annotated[CollectorDB, Depends(get_db)],
    _token: Annotated[str, Depends(verify_collector_token)],
    lookback_minutes: Annotated[
        int, Query(ge=1, le=1440, description="Lookback window in minutes")
    ] = 60,
    is_drill: Annotated[bool | None, Query(description="Filter by drill flag")] = None,
) -> dict[str, Any]:
    """Retrieve active alerts triggered in the last N minutes."""
    alerts = db.get_active_alerts(lookback_minutes=lookback_minutes, is_drill=is_drill)
    return {
        "active_alerts": alerts,
        "count": len(alerts),
        "lookback_minutes": lookback_minutes,
    }
