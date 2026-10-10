"""Durable SQLite feedback provider with conflict-safe idempotency."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

from .._contracts import (
    FeedbackConflictError,
    canonical_event_bytes,
    feedback_event_request_hash,
    normalize_page_revision,
    normalize_site_id,
    parse_feedback_event,
    validate_request_hash,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback_events (
    feedback_id TEXT PRIMARY KEY,
    request_hash TEXT NOT NULL,
    site_id TEXT NOT NULL,
    page_id TEXT NOT NULL,
    event_json TEXT NOT NULL
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS feedback_events_page_idx
ON feedback_events(site_id, page_id);
"""


class SQLiteFeedbackStore:
    """Small transactional event store; no participant/network identity columns."""

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 5000) -> None:
        self.path = Path(path)
        self.busy_timeout_ms = max(100, min(int(busy_timeout_ms), 30_000))

    def _setup_with_retry(self, conn: sqlite3.Connection) -> None:
        """
        Establish WAL/schema with bounded retry for first-open races.

        SQLite's busy timeout does not reliably cover every journal-mode transition.
        Concurrent first connections can therefore observe ``database is locked``
        before ordinary transactional retry behavior applies. Keep that race inside
        this initialization boundary rather than surfacing it as a failed feedback
        submission.
        """
        deadline = time.monotonic() + self.busy_timeout_ms / 1000
        delay = 0.005
        while True:
            try:
                mode = str(
                    conn.execute("PRAGMA journal_mode").fetchone()[0],
                ).lower()
                if mode != "wal":
                    mode = str(
                        conn.execute("PRAGMA journal_mode=WAL").fetchone()[0],
                    ).lower()
                    if mode != "wal":
                        raise sqlite3.OperationalError(
                            "unable to enable WAL journal mode",
                        )
                conn.executescript(_SCHEMA)
                return
            except sqlite3.OperationalError as exc:  # ruff: ignore[try-except-in-loop]
                locked = "locked" in str(exc).lower() or "busy" in str(exc).lower()
                if not locked or time.monotonic() >= deadline:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 0.1)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1000,
            isolation_level=None,
        )
        try:
            conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
            self._setup_with_retry(conn)
            conn.execute("PRAGMA synchronous=FULL")
            return conn
        except Exception:
            conn.close()
            raise

    def put(
        self, *, feedback_id: str, request_hash: str, event: dict[str, Any]
    ) -> dict[str, Any]:
        normalized = parse_feedback_event(event)
        if feedback_id != normalized["feedback"]["id"]:
            raise ValueError("feedback_id does not match the durable event")
        validate_request_hash(request_hash)
        expected_hash = feedback_event_request_hash(normalized)
        if request_hash != expected_hash:
            raise ValueError("request_hash does not match the durable feedback event")
        event_json = canonical_event_bytes(normalized).decode("utf-8").rstrip("\n")
        # closing() releases the handle; the inner `with conn` keeps the
        # rollback-on-error transaction semantics.
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT request_hash, event_json FROM feedback_events WHERE feedback_id = ?",
                (feedback_id,),
            ).fetchone()
            if row is not None:
                if row[0] != request_hash or row[1] != event_json:
                    conn.rollback()
                    raise FeedbackConflictError(
                        "feedback_id was already used for different feedback content"
                    )
                conn.commit()
                return {
                    "status": "replay",
                    "provider": "sqlite",
                    "feedback_id": feedback_id,
                    "request_hash": request_hash,
                }
            conn.execute(
                "INSERT INTO feedback_events(feedback_id, request_hash, site_id, page_id, event_json) VALUES(?,?,?,?,?)",
                (
                    feedback_id,
                    request_hash,
                    normalized["site_id"],
                    normalized["page_id"],
                    event_json,
                ),
            )
            conn.commit()
            return {
                "status": "accepted",
                "provider": "sqlite",
                "feedback_id": feedback_id,
                "request_hash": request_hash,
            }

    def events(self, *, site_id: str | None = None) -> list[dict[str, Any]]:
        if site_id is not None:
            site_id = normalize_site_id(site_id)
        # closing() releases the handle; the inner `with conn` keeps the
        # rollback-on-error transaction semantics.
        with closing(self._connect()) as conn, conn:
            if site_id is None:
                rows = conn.execute(
                    "SELECT feedback_id, request_hash, site_id, page_id, event_json "
                    "FROM feedback_events ORDER BY site_id, page_id, feedback_id"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT feedback_id, request_hash, site_id, page_id, event_json "
                    "FROM feedback_events WHERE site_id = ? ORDER BY page_id, feedback_id",
                    (site_id,),
                ).fetchall()
        events: list[dict[str, Any]] = []
        for stored_id, stored_hash, stored_site, stored_page, raw_json in rows:
            try:
                decoded = json.loads(raw_json)
                event = parse_feedback_event(decoded)
                canonical = canonical_event_bytes(event).decode("utf-8").rstrip("\n")
                expected_hash = feedback_event_request_hash(event)
            except (json.JSONDecodeError, ValueError, RecursionError) as exc:
                raise sqlite3.DatabaseError(
                    "stored feedback event is invalid",
                ) from exc
            if canonical != raw_json:
                raise sqlite3.DatabaseError(
                    "stored feedback event is not canonical",
                )
            if (
                stored_id != event["feedback"]["id"]
                or stored_site != event["site_id"]
                or stored_page != event["page_id"]
                or stored_hash != expected_hash
            ):
                raise sqlite3.DatabaseError(
                    "stored feedback event metadata does not match canonical event content"
                )
            events.append(event)
        return events

    def aggregate(
        self, *, site_id: str, page_revision: str | None = None
    ) -> dict[str, dict[str, int]]:
        site_id = normalize_site_id(site_id)
        revision_filter: str | None = None
        if page_revision not in (None, ""):
            revision_filter = normalize_page_revision(page_revision)
            if not revision_filter:
                raise ValueError(
                    "revision-scoped aggregate requires a non-empty page_revision",
                )
        pages: dict[str, dict[str, int]] = {}
        for event in self.events(site_id=site_id):
            if (
                revision_filter is not None
                and event.get("page_revision", "") != revision_filter
            ):
                continue
            state = pages.setdefault(
                event["page_id"],
                {
                    "count": 0,
                    "score": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "neutral_count": 0,
                },
            )
            rating = int(event["feedback"]["rating"])
            state["count"] += 1
            state["score"] += rating
            if rating > 0:
                state["positive_count"] += 1
            elif rating < 0:
                state["negative_count"] += 1
            else:
                state["neutral_count"] += 1
        return pages
