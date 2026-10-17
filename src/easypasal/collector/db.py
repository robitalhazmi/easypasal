"""SQLite database storage engine for VPS log collector (Plan §9 / Task T7.2).

Provides high-throughput, low-memory persistence using WAL mode and optimized indices.
Memory footprint is constrained to <30 MB RAM, well inside the 2 GB VPS budget.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from easypasal.monitoring.schemas import (
    FeedbackLogRecord,
    RequestLogRecord,
)

logger = logging.getLogger("easypasal.collector.db")

_CREATE_REQUEST_LOGS_SQL = """
CREATE TABLE IF NOT EXISTS request_logs (
    request_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    variant TEXT NOT NULL,
    language TEXT NOT NULL,
    question_length INTEGER NOT NULL,
    question_redacted TEXT NOT NULL,
    retrieval_score_top1 REAL,
    retrieval_scores_topk TEXT,
    retrieved_doc_ids TEXT,
    abstained INTEGER NOT NULL,
    routed INTEGER NOT NULL DEFAULT 0,
    theme TEXT,
    retrieval_latency_ms REAL,
    generation_latency_ms REAL,
    total_latency_ms REAL NOT NULL,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    citation_count INTEGER DEFAULT 0,
    grounded_citation_count INTEGER,
    ungrounded_citation_count INTEGER,
    feedback_rating INTEGER,
    feedback_comment TEXT,
    client_ip_hash TEXT,
    model_name TEXT,
    created_at TEXT NOT NULL
);
"""

_CREATE_FEEDBACK_LOGS_SQL = """
CREATE TABLE IF NOT EXISTS feedback_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    rating INTEGER NOT NULL,
    comment_redacted TEXT,
    tags TEXT,
    created_at TEXT NOT NULL
);
"""

_CREATE_INDICES_SQL = """
CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON request_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_requests_variant ON request_logs(variant);
CREATE INDEX IF NOT EXISTS idx_requests_abstained ON request_logs(abstained);
CREATE INDEX IF NOT EXISTS idx_requests_theme ON request_logs(theme);
CREATE INDEX IF NOT EXISTS idx_feedback_request_id ON feedback_logs(request_id);
"""


class CollectorDB:
    """Thread-safe SQLite storage engine for request logs and user feedback."""

    def __init__(
        self,
        db_path: Path | str = "data/collector/collector.db",
        enable_wal: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        self.enable_wal = enable_wal
        self._lock = threading.Lock()

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and configure a new SQLite connection with performance pragmas."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row

        if self.enable_wal:
            conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        # Limit page cache to ~4 MB to respect VPS memory ceiling
        conn.execute("PRAGMA cache_size = -4000;")
        return conn

    def _init_db(self) -> None:
        """Initialize relational database tables and performance indices."""
        with self._lock, self._get_connection() as conn:
            conn.execute(_CREATE_REQUEST_LOGS_SQL)
            conn.execute(_CREATE_FEEDBACK_LOGS_SQL)
            conn.executescript(_CREATE_INDICES_SQL)
            conn.commit()
        logger.info("Collector SQLite database initialized at %s", self.db_path)

    def insert_request(self, record: RequestLogRecord) -> None:
        """Insert or replace a single request log record in SQLite."""
        sql = """
        INSERT OR REPLACE INTO request_logs (
            request_id, timestamp, variant, language, question_length, question_redacted,
            retrieval_score_top1, retrieval_scores_topk, retrieved_doc_ids,
            abstained, routed, theme,
            retrieval_latency_ms, generation_latency_ms, total_latency_ms,
            prompt_tokens, completion_tokens, total_tokens,
            citation_count, grounded_citation_count, ungrounded_citation_count,
            feedback_rating, feedback_comment, client_ip_hash, model_name, created_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """
        now_utc = datetime.now(UTC).isoformat()
        params = (
            record.request_id,
            record.timestamp.isoformat(),
            record.variant,
            record.language,
            record.question_length,
            record.question_redacted,
            record.retrieval_score_top1,
            json.dumps(record.retrieval_scores_topk),
            json.dumps(record.retrieved_doc_ids),
            1 if record.abstained else 0,
            1 if record.routed else 0,
            record.theme,
            record.retrieval_latency_ms,
            record.generation_latency_ms,
            record.total_latency_ms,
            record.prompt_tokens,
            record.completion_tokens,
            record.total_tokens,
            record.citation_count,
            record.grounded_citation_count,
            record.ungrounded_citation_count,
            record.feedback_rating,
            record.feedback_comment,
            record.client_ip_hash,
            record.model_name,
            now_utc,
        )

        with self._lock, self._get_connection() as conn:
            conn.execute(sql, params)
            conn.commit()

    def insert_requests_batch(self, records: list[RequestLogRecord]) -> int:
        """Atomically insert a batch of request log records within a single transaction."""
        if not records:
            return 0

        sql = """
        INSERT OR REPLACE INTO request_logs (
            request_id, timestamp, variant, language, question_length, question_redacted,
            retrieval_score_top1, retrieval_scores_topk, retrieved_doc_ids,
            abstained, routed, theme,
            retrieval_latency_ms, generation_latency_ms, total_latency_ms,
            prompt_tokens, completion_tokens, total_tokens,
            citation_count, grounded_citation_count, ungrounded_citation_count,
            feedback_rating, feedback_comment, client_ip_hash, model_name, created_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """
        now_utc = datetime.now(UTC).isoformat()
        batch_params = [
            (
                r.request_id,
                r.timestamp.isoformat(),
                r.variant,
                r.language,
                r.question_length,
                r.question_redacted,
                r.retrieval_score_top1,
                json.dumps(r.retrieval_scores_topk),
                json.dumps(r.retrieved_doc_ids),
                1 if r.abstained else 0,
                1 if r.routed else 0,
                r.theme,
                r.retrieval_latency_ms,
                r.generation_latency_ms,
                r.total_latency_ms,
                r.prompt_tokens,
                r.completion_tokens,
                r.total_tokens,
                r.citation_count,
                r.grounded_citation_count,
                r.ungrounded_citation_count,
                r.feedback_rating,
                r.feedback_comment,
                r.client_ip_hash,
                r.model_name,
                now_utc,
            )
            for r in records
        ]

        with self._lock, self._get_connection() as conn:
            conn.executemany(sql, batch_params)
            conn.commit()
        return len(records)

    def insert_feedback(self, feedback: FeedbackLogRecord) -> None:
        """Insert user feedback entry and update request_logs record if present."""
        sql_feedback = """
        INSERT INTO feedback_logs (
            request_id, timestamp, rating, comment_redacted, tags, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """
        sql_update_request = """
        UPDATE request_logs
        SET feedback_rating = ?, feedback_comment = ?
        WHERE request_id = ?
        """
        now_utc = datetime.now(UTC).isoformat()

        with self._lock, self._get_connection() as conn:
            conn.execute(
                sql_feedback,
                (
                    feedback.request_id,
                    feedback.timestamp.isoformat(),
                    feedback.rating,
                    feedback.comment_redacted,
                    json.dumps(feedback.tags),
                    now_utc,
                ),
            )
            conn.execute(
                sql_update_request,
                (feedback.rating, feedback.comment_redacted, feedback.request_id),
            )
            conn.commit()

    def get_requests(
        self,
        variant: str | None = None,
        theme: str | None = None,
        abstained: bool | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Query request logs with optional filtering and pagination."""
        clauses: list[str] = []
        params: list[Any] = []

        if variant:
            clauses.append("variant = ?")
            params.append(variant)
        if theme:
            clauses.append("theme = ?")
            params.append(theme)
        if abstained is not None:
            clauses.append("abstained = ?")
            params.append(1 if abstained else 0)
        if start_time:
            clauses.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            clauses.append("timestamp <= ?")
            params.append(end_time)

        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._lock, self._get_connection() as conn:
            # Count total matching rows
            count_sql = f"SELECT COUNT(*) AS total FROM request_logs {where_clause}"
            total_count = conn.execute(count_sql, params).fetchone()["total"]

            # Query paginated rows ordered by timestamp descending
            query_sql = f"""
            SELECT * FROM request_logs
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """
            rows = conn.execute(query_sql, params + [limit, offset]).fetchall()

            results: list[dict[str, Any]] = []
            for row in rows:
                d = dict(row)
                d["abstained"] = bool(d["abstained"])
                d["routed"] = bool(d["routed"])
                d["retrieval_scores_topk"] = (
                    json.loads(d["retrieval_scores_topk"]) if d.get("retrieval_scores_topk") else []
                )
                d["retrieved_doc_ids"] = (
                    json.loads(d["retrieved_doc_ids"]) if d.get("retrieved_doc_ids") else []
                )
                results.append(d)

            return results, total_count

    def get_feedback(
        self,
        request_id: str | None = None,
        rating: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query feedback logs with optional filtering."""
        clauses: list[str] = []
        params: list[Any] = []

        if request_id:
            clauses.append("request_id = ?")
            params.append(request_id)
        if rating is not None:
            clauses.append("rating = ?")
            params.append(rating)

        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query_sql = f"""
        SELECT * FROM feedback_logs
        {where_clause}
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
        """

        with self._lock, self._get_connection() as conn:
            rows = conn.execute(query_sql, params + [limit, offset]).fetchall()
            results: list[dict[str, Any]] = []
            for row in rows:
                d = dict(row)
                d["tags"] = json.loads(d["tags"]) if d.get("tags") else []
                results.append(d)
            return results

    def get_stats_summary(self) -> dict[str, Any]:
        """Compute aggregate metrics, latency percentiles, and telemetry statistics."""
        with self._lock, self._get_connection() as conn:
            total_req = conn.execute("SELECT COUNT(*) AS c FROM request_logs").fetchone()["c"]
            if total_req == 0:
                return {
                    "total_requests": 0,
                    "total_feedback": 0,
                    "abstain_count": 0,
                    "abstain_rate": 0.0,
                    "mean_latency_ms": 0.0,
                    "p50_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "p99_latency_ms": 0.0,
                    "variant_distribution": {},
                    "theme_distribution": {},
                    "feedback_counts": {"positive": 0, "negative": 0},
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_tokens": 0,
                    "mean_retrieval_score_top1": None,
                }

            # Abstentions
            abstain_c = conn.execute(
                "SELECT COUNT(*) AS c FROM request_logs WHERE abstained = 1"
            ).fetchone()["c"]

            # Token sums
            tokens_row = conn.execute(
                """
                SELECT
                    SUM(prompt_tokens) AS p,
                    SUM(completion_tokens) AS c,
                    SUM(total_tokens) AS t,
                    AVG(retrieval_score_top1) AS avg_s1
                FROM request_logs
                """
            ).fetchone()

            # Latency statistics & percentiles
            lat_rows = [
                r["total_latency_ms"]
                for r in conn.execute(
                    "SELECT total_latency_ms FROM request_logs ORDER BY total_latency_ms ASC"
                ).fetchall()
            ]

            n = len(lat_rows)
            mean_lat = sum(lat_rows) / n if n > 0 else 0.0
            p50 = lat_rows[int(n * 0.50)] if n > 0 else 0.0
            p95 = lat_rows[min(int(n * 0.95), n - 1)] if n > 0 else 0.0
            p99 = lat_rows[min(int(n * 0.99), n - 1)] if n > 0 else 0.0

            # Variant distribution
            var_dist = {
                r["variant"]: r["c"]
                for r in conn.execute(
                    "SELECT variant, COUNT(*) AS c FROM request_logs GROUP BY variant"
                ).fetchall()
            }

            # Theme distribution
            theme_dist = {
                r["theme"] or "Unknown": r["c"]
                for r in conn.execute(
                    "SELECT theme, COUNT(*) AS c FROM request_logs GROUP BY theme"
                ).fetchall()
            }

            # Feedback counts
            fb_c = conn.execute("SELECT COUNT(*) AS c FROM feedback_logs").fetchone()["c"]
            pos_fb = conn.execute(
                "SELECT COUNT(*) AS c FROM feedback_logs WHERE rating > 0"
            ).fetchone()["c"]
            neg_fb = conn.execute(
                "SELECT COUNT(*) AS c FROM feedback_logs WHERE rating < 0"
            ).fetchone()["c"]

            return {
                "total_requests": total_req,
                "total_feedback": fb_c,
                "abstain_count": abstain_c,
                "abstain_rate": round(abstain_c / total_req, 4),
                "mean_latency_ms": round(mean_lat, 2),
                "p50_latency_ms": round(p50, 2),
                "p95_latency_ms": round(p95, 2),
                "p99_latency_ms": round(p99, 2),
                "variant_distribution": var_dist,
                "theme_distribution": theme_dist,
                "feedback_counts": {"positive": pos_fb, "negative": neg_fb},
                "total_prompt_tokens": int(tokens_row["p"] or 0),
                "total_completion_tokens": int(tokens_row["c"] or 0),
                "total_tokens": int(tokens_row["t"] or 0),
                "mean_retrieval_score_top1": (
                    round(float(tokens_row["avg_s1"]), 4)
                    if tokens_row["avg_s1"] is not None
                    else None
                ),
            }

    def get_db_info(self) -> dict[str, Any]:
        """Return diagnostic storage metrics."""
        size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        with self._lock, self._get_connection() as conn:
            page_count = conn.execute("PRAGMA page_count;").fetchone()[0]
            page_size = conn.execute("PRAGMA page_size;").fetchone()[0]
            req_count = conn.execute("SELECT COUNT(*) FROM request_logs;").fetchone()[0]
            fb_count = conn.execute("SELECT COUNT(*) FROM feedback_logs;").fetchone()[0]

        return {
            "path": str(self.db_path),
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 3),
            "page_count": page_count,
            "page_size": page_size,
            "request_logs_count": req_count,
            "feedback_logs_count": fb_count,
        }
