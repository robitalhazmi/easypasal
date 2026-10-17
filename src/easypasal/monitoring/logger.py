"""Request logging and telemetry storage engine (Plan §9 / Task T7.1).

Validates every logged request against RequestLogRecord schema and persists
structured JSON Lines entries for operational monitoring, drift analysis, and auditability.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from easypasal.monitoring.schemas import (
    FeedbackLogRecord,
    FeedbackPayload,
    RequestLogRecord,
)

logger = logging.getLogger("easypasal.monitoring.logger")


class RequestLogger:
    """Thread-safe structured request and feedback logger for EasyPasal."""

    def __init__(
        self,
        log_file: Path | str | None = "logs/requests.jsonl",
        feedback_file: Path | str | None = "logs/feedback.jsonl",
        redact_pii: bool = True,
        salt: str = "easypasal-monitoring-salt-v1",
    ) -> None:
        self.log_file = Path(log_file) if log_file else None
        self.feedback_file = Path(feedback_file) if feedback_file else None
        self.redact_pii = redact_pii
        self.salt = salt
        self._lock = threading.Lock()

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if self.feedback_file:
            self.feedback_file.parent.mkdir(parents=True, exist_ok=True)

    def hash_client_ip(self, ip: str | None) -> str | None:
        """Compute a salted, one-way SHA-256 hash of a client IP address."""
        if not ip or not ip.strip() or ip == "unknown":
            return None
        raw = f"{self.salt}:{ip.strip()}".encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def log_request(self, record: RequestLogRecord | dict[str, Any]) -> RequestLogRecord:
        """Validate and append a structured request log entry.

        Args:
            record: RequestLogRecord instance or dictionary adhering to schema.

        Returns:
            Validated RequestLogRecord.

        Raises:
            ValidationError: If record fails schema validation.
        """
        if isinstance(record, dict):
            validated = RequestLogRecord.model_validate(record)
        elif isinstance(record, RequestLogRecord):
            validated = record
        else:
            raise TypeError(f"Expected RequestLogRecord or dict, got {type(record)}")

        if self.log_file is not None:
            json_line = validated.model_dump_json()
            with self._lock:
                try:
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(json_line + "\n")
                except OSError as exc:
                    logger.error("Failed to append request log to %s: %s", self.log_file, exc)
                    raise

        return validated

    def log_feedback(
        self,
        feedback: FeedbackPayload | FeedbackLogRecord | dict[str, Any],
    ) -> FeedbackLogRecord:
        """Validate and record user feedback entry.

        Args:
            feedback: Feedback entry.

        Returns:
            Validated FeedbackLogRecord.
        """
        if isinstance(feedback, FeedbackPayload):
            record = FeedbackLogRecord(
                request_id=feedback.request_id,
                rating=feedback.rating,
                comment_redacted=feedback.comment,
                tags=feedback.tags,
            )
        elif isinstance(feedback, dict):
            record = FeedbackLogRecord.model_validate(feedback)
        elif isinstance(feedback, FeedbackLogRecord):
            record = feedback
        else:
            raise TypeError(f"Expected FeedbackPayload or FeedbackLogRecord, got {type(feedback)}")

        if self.feedback_file is not None:
            json_line = record.model_dump_json()
            with self._lock:
                try:
                    with open(self.feedback_file, "a", encoding="utf-8") as f:
                        f.write(json_line + "\n")
                except OSError as exc:
                    logger.error("Failed to append feedback log to %s: %s", self.feedback_file, exc)
                    raise

        return record

    def read_logs(self, limit: int | None = None) -> list[RequestLogRecord]:
        """Read logged request records from disk.

        Args:
            limit: Optional maximum number of recent records to return.

        Returns:
            List of validated RequestLogRecord objects in chronological order.
        """
        if not self.log_file or not self.log_file.exists():
            return []

        records: list[RequestLogRecord] = []
        with self._lock, open(self.log_file, encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    records.append(RequestLogRecord.model_validate(data))
                except (json.JSONDecodeError, ValidationError) as exc:
                    logger.warning("Corrupt log line skipped in %s: %s", self.log_file, exc)

        if limit is not None and limit > 0:
            return records[-limit:]
        return records

    def read_feedback(self, limit: int | None = None) -> list[FeedbackLogRecord]:
        """Read logged user feedback entries from disk."""
        if not self.feedback_file or not self.feedback_file.exists():
            return []

        entries: list[FeedbackLogRecord] = []
        with self._lock, open(self.feedback_file, encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    entries.append(FeedbackLogRecord.model_validate(data))
                except (json.JSONDecodeError, ValidationError) as exc:
                    logger.warning(
                        "Corrupt feedback line skipped in %s: %s", self.feedback_file, exc
                    )

        if limit is not None and limit > 0:
            return entries[-limit:]
        return entries
