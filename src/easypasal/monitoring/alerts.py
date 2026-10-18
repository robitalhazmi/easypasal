"""Simple threshold alerting engine and drill harness (Plan §9 / Task T7.5).

Monitors operational telemetry (abstain rate, p95 latency, low-evidence score rate)
against configurable threshold boundaries and persists flagged alert events in SQLite
and JSONL audit logs.
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Literal

from easypasal.collector.db import CollectorDB
from easypasal.monitoring.schemas import (
    AlertCheckResult,
    AlertEvent,
    AlertThresholds,
)

logger = logging.getLogger("easypasal.monitoring.alerts")


def dispatch_webhook(webhook_url: str, alert: AlertEvent, timeout: float = 5.0) -> bool:
    """Dispatch an alert event notification to an external webhook (e.g. Discord, Slack, custom webhook).

    Args:
        webhook_url: Target HTTP/HTTPS endpoint.
        alert: Flagged alert event model.
        timeout: Network timeout in seconds.

    Returns:
        True if notification delivered successfully, False otherwise.
    """
    try:
        payload = {
            "text": f"🚨 EasyPasal Alert [{alert.severity}]: {alert.message}",
            "alert_id": alert.alert_id,
            "metric": alert.metric_name,
            "severity": alert.severity,
            "current_value": alert.current_value,
            "threshold_value": alert.threshold_value,
            "timestamp": alert.timestamp.isoformat(),
            "is_drill": alert.is_drill,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "EasyPasal-Alerts/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status_code = response.getcode()
            if 200 <= status_code < 300:
                logger.info("Dispatched alert %s to webhook %s", alert.alert_id, webhook_url)
                return True
            logger.warning("Webhook returned unexpected status code: %d", status_code)
            return False
    except Exception as exc:
        logger.warning("Failed to dispatch alert %s to webhook: %s", alert.alert_id, exc)
        return False


def _append_alert_log(alert: AlertEvent, log_path: Path | str) -> None:
    """Append a flagged alert event to the JSON Lines alert log file."""
    try:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(alert.model_dump_json() + "\n")
    except Exception as exc:
        logger.error("Failed to append alert to %s: %s", log_path, exc)


def evaluate_threshold_alerts(
    db: CollectorDB | Path | str,
    thresholds: AlertThresholds | None = None,
    alerts_log_path: Path | str | None = "logs/alerts.jsonl",
) -> AlertCheckResult:
    """Evaluate operational metrics against configured threshold boundaries.

    Args:
        db: CollectorDB instance or filesystem path to SQLite database.
        thresholds: Boundary thresholds for metric alerts.
        alerts_log_path: Optional path to JSON Lines alert sink file.

    Returns:
        AlertCheckResult containing triggered alerts and evaluation snapshot.
    """
    collector_db = CollectorDB(db_path=db) if isinstance(db, (str, Path)) else db
    cfg = thresholds or AlertThresholds()

    stats = collector_db.get_stats_summary()
    total_req = stats.get("total_requests", 0)

    # Insufficient data check to avoid false alarms during cold start
    if total_req < cfg.min_sample_size:
        logger.info(
            "Evaluation sample size (%d) is below minimum threshold (%d); skipping checks.",
            total_req,
            cfg.min_sample_size,
        )
        return AlertCheckResult(
            sample_size=total_req,
            status="INSUFFICIENT_DATA",
            alerts_triggered=[],
            summary=stats,
        )

    alerts: list[AlertEvent] = []

    # 1. Abstention Rate Threshold Check
    abstain_rate = float(stats.get("abstain_rate", 0.0))
    if abstain_rate > cfg.abstain_rate_max:
        abstain_sev: Literal["WARNING", "CRITICAL"] = (
            "CRITICAL" if abstain_rate >= 0.50 else "WARNING"
        )
        msg = (
            f"Abstention rate {abstain_rate * 100:.1f}% exceeded threshold "
            f"{cfg.abstain_rate_max * 100:.1f}% across {total_req} requests."
        )
        alerts.append(
            AlertEvent(
                metric_name="abstain_rate",
                severity=abstain_sev,
                current_value=round(abstain_rate, 4),
                threshold_value=cfg.abstain_rate_max,
                condition=">",
                sample_size=total_req,
                message=msg,
                is_drill=False,
            )
        )

    # 2. p95 Latency Threshold Check
    p95_lat = float(stats.get("p95_latency_ms", 0.0))
    if p95_lat > cfg.p95_latency_ms_max:
        lat_sev: Literal["WARNING", "CRITICAL"] = (
            "CRITICAL" if p95_lat >= 2.0 * cfg.p95_latency_ms_max else "WARNING"
        )
        msg = (
            f"p95 latency {p95_lat:.1f}ms exceeded threshold "
            f"{cfg.p95_latency_ms_max:.1f}ms across {total_req} requests."
        )
        alerts.append(
            AlertEvent(
                metric_name="p95_latency_ms",
                severity=lat_sev,
                current_value=round(p95_lat, 2),
                threshold_value=cfg.p95_latency_ms_max,
                condition=">",
                sample_size=total_req,
                message=msg,
                is_drill=False,
            )
        )

    # 3. Low-Evidence Answer Rate Check (Top-1 retrieval score < 0.60)
    low_ev_rate = 0.0
    items, total_scored = collector_db.get_requests(limit=min(total_req, 1000))
    scored_items = [i for i in items if i.get("retrieval_score_top1") is not None]
    if scored_items:
        low_count = sum(1 for i in scored_items if float(i["retrieval_score_top1"]) < 0.60)
        low_ev_rate = low_count / len(scored_items)
        if low_ev_rate > cfg.low_evidence_rate_max:
            msg = (
                f"Low-evidence query rate {low_ev_rate * 100:.1f}% exceeded threshold "
                f"{cfg.low_evidence_rate_max * 100:.1f}%."
            )
            alerts.append(
                AlertEvent(
                    metric_name="low_evidence_rate",
                    severity="WARNING",
                    current_value=round(low_ev_rate, 4),
                    threshold_value=cfg.low_evidence_rate_max,
                    condition=">",
                    sample_size=len(scored_items),
                    message=msg,
                    is_drill=False,
                )
            )

    stats["low_evidence_rate"] = round(low_ev_rate, 4)

    # Persist and dispatch triggered alerts
    for alert in alerts:
        collector_db.insert_alert(alert)
        if alerts_log_path:
            _append_alert_log(alert, alerts_log_path)
        if cfg.webhook_url:
            dispatch_webhook(cfg.webhook_url, alert)

    outcome_status: Literal["OK", "ALERT", "INSUFFICIENT_DATA"] = "ALERT" if alerts else "OK"
    return AlertCheckResult(
        sample_size=total_req,
        status=outcome_status,
        alerts_triggered=alerts,
        summary=stats,
    )


def run_alert_drill(
    db: CollectorDB | Path | str,
    metric_to_test: str = "abstain_rate",
    thresholds: AlertThresholds | None = None,
    alerts_log_path: Path | str | None = "logs/alerts.jsonl",
) -> AlertEvent:
    """Execute an automated synthetic alert drill to verify alerting pipelines (Plan §9 / Task T7.5 AC).

    Simulates a threshold violation, persists the flagged event with is_drill=True,
    appends to the alert audit log, and optionally dispatches to webhooks.

    Args:
        db: CollectorDB instance or path to SQLite file.
        metric_to_test: Target metric to trigger ('abstain_rate', 'p95_latency_ms', 'low_evidence_rate').
        thresholds: Configured alert thresholds.
        alerts_log_path: Optional JSONL log destination.

    Returns:
        The generated AlertEvent.
    """
    collector_db = CollectorDB(db_path=db) if isinstance(db, (str, Path)) else db
    cfg = thresholds or AlertThresholds()

    if metric_to_test == "p95_latency_ms":
        current_val = round(cfg.p95_latency_ms_max * 1.5, 2)
        threshold_val = cfg.p95_latency_ms_max
        msg = (
            f"DRILL ALERT: Synthetic p95 latency spike {current_val}ms "
            f"breached threshold {threshold_val}ms."
        )
    elif metric_to_test == "low_evidence_rate":
        current_val = min(1.0, round(cfg.low_evidence_rate_max + 0.15, 4))
        threshold_val = cfg.low_evidence_rate_max
        msg = (
            f"DRILL ALERT: Synthetic low-evidence retrieval rate spike {current_val * 100:.1f}% "
            f"breached threshold {threshold_val * 100:.1f}%."
        )
    else:  # Default to abstain_rate
        metric_to_test = "abstain_rate"
        current_val = min(1.0, round(cfg.abstain_rate_max + 0.20, 4))
        threshold_val = cfg.abstain_rate_max
        msg = (
            f"DRILL ALERT: Synthetic abstention rate spike {current_val * 100:.1f}% "
            f"breached threshold {threshold_val * 100:.1f}%."
        )

    alert = AlertEvent(
        metric_name=metric_to_test,
        severity="WARNING",
        current_value=current_val,
        threshold_value=threshold_val,
        condition=">",
        sample_size=50,
        message=msg,
        is_drill=True,
    )

    collector_db.insert_alert(alert)
    if alerts_log_path:
        _append_alert_log(alert, alerts_log_path)
    if cfg.webhook_url:
        dispatch_webhook(cfg.webhook_url, alert)

    logger.info(
        "Successfully executed alert drill for %s (alert_id=%s)", metric_to_test, alert.alert_id
    )
    return alert


def main() -> None:
    """CLI entrypoint for alert checking and drill testing."""
    parser = argparse.ArgumentParser(description="EasyPasal Monitoring Alerting & Drill Tool")
    parser.add_argument(
        "--action",
        choices=["check", "drill", "list"],
        default="check",
        help="Action to perform: 'check' (evaluate metrics), 'drill' (trigger synthetic alert), 'list' (show alerts)",
    )
    parser.add_argument(
        "--db-path",
        default="data/collector/collector.db",
        help="Path to SQLite collector database (default: data/collector/collector.db)",
    )
    parser.add_argument(
        "--metric",
        choices=["abstain_rate", "p95_latency_ms", "low_evidence_rate"],
        default="abstain_rate",
        help="Target metric for alert drill (default: abstain_rate)",
    )
    parser.add_argument(
        "--threshold-abstain",
        type=float,
        default=0.35,
        help="Abstention rate max threshold (default: 0.35)",
    )
    parser.add_argument(
        "--threshold-latency",
        type=float,
        default=5000.0,
        help="p95 latency max threshold in ms (default: 5000.0)",
    )
    parser.add_argument(
        "--webhook-url",
        default=None,
        help="Optional webhook URL for alert notification dispatch",
    )
    parser.add_argument(
        "--alerts-log",
        default="logs/alerts.jsonl",
        help="Path for JSONL alert log sink (default: logs/alerts.jsonl)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = AlertThresholds(
        abstain_rate_max=args.threshold_abstain,
        p95_latency_ms_max=args.threshold_latency,
        webhook_url=args.webhook_url,
    )

    db = CollectorDB(db_path=args.db_path)

    if args.action == "drill":
        alert = run_alert_drill(
            db=db,
            metric_to_test=args.metric,
            thresholds=cfg,
            alerts_log_path=args.alerts_log,
        )
        print("\n" + "=" * 60)
        print("🚨 EASYPASAL ALERT DRILL TRIGGERED")
        print("=" * 60)
        print(f"Alert ID:       {alert.alert_id}")
        print(f"Metric:         {alert.metric_name}")
        print(f"Severity:       {alert.severity}")
        print(f"Current Value:  {alert.current_value}")
        print(f"Threshold:      {alert.threshold_value}")
        print(f"Message:        {alert.message}")
        print(f"Drill Flag:     {alert.is_drill}")
        print("=" * 60 + "\n")

    elif args.action == "list":
        alerts, total = db.get_alerts(limit=20)
        print("\n" + "=" * 60)
        print(f"EasyPasal Stored Alerts ({total} total, showing {len(alerts)}):")
        print("=" * 60)
        for a in alerts:
            drill_tag = "[DRILL] " if a.get("is_drill") else ""
            print(f"• {a['timestamp'][:19]} | {a['severity']} | {drill_tag}{a['message']}")
        print("=" * 60 + "\n")

    else:
        result = evaluate_threshold_alerts(
            db=db,
            thresholds=cfg,
            alerts_log_path=args.alerts_log,
        )
        print("\n" + "=" * 60)
        print(f"EasyPasal Operational Health Check: {result.status}")
        print("=" * 60)
        print(f"Sample Size:        {result.sample_size}")
        print(f"Alerts Triggered:   {len(result.alerts_triggered)}")
        if result.alerts_triggered:
            for triggered_alert in result.alerts_triggered:
                print(
                    f"  🚨 [{triggered_alert.severity}] {triggered_alert.metric_name}: {triggered_alert.message}"
                )
        else:
            print("  ✅ All monitored metrics within normal operational bounds.")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
