"""
Mutable contribution receipt lifecycle control plane.

The contribution data path has two very different storage needs:

* pending review rows must live in a mutable store so a participant can delete
  them before promotion and reviewers can atomically claim exactly one promotion;
* promoted rows may be copied to append-only/versioned providers, but the receipt
  lifecycle must remain mutable so a later withdrawal can be represented
  truthfully without pretending that Git history was physically erased.

This module therefore stores only the *control plane*.  Pending canonical rows are
kept only until promotion/deletion/expiry.  After promotion the raw rows are
removed from the ledger and only bounded lifecycle metadata, deduplication keys,
a digest of the participant delete/withdraw capability, and provider record-path
metadata remain.

Two backends are bundled:

``memory``
    Compatibility/development backend. Process-local and intentionally not
    durable.

``sqlite``
    Local transactional durable backend using the Python standard library.
    It survives process restarts when its file lives on durable storage and
    prevents duplicate promotion with transactional state transitions. It is
    **not** a shared multi-replica database; operators must not represent it as
    one.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any

from ._redis_security import RedisSecurityError, redis_connection_kwargs

_TERMINAL = {"deleted", "expired"}
_ACTIVE_PENDING = {"quarantined", "promoting", "promotion_uncertain", "withdrawing"}
_MANAGED = {
    "quarantined",
    "promoting",
    "promotion_uncertain",
    "eligible",
    "withdrawing",
    "withdrawal_uncertain",
    "withdrawn",
}


class ContributionLedgerError(RuntimeError):
    """Stable, non-sensitive control-plane error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _copy_entry(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if entry is None:
        return None
    # JSON round-trip prevents callers from mutating nested records/storage maps.
    return json.loads(json.dumps(entry, ensure_ascii=False))


def _now() -> float:
    return time.time()


class MemoryContributionLedger:
    """Bounded process-local compatibility ledger."""

    backend = "memory"
    durability = "process_local"
    durable = False
    shared = False

    def __init__(
        self,
        *,
        max_pending_entries: int,
        max_pending_bytes: int,
        max_receipts: int,
        terminal_retention_seconds: int = 86_400,
    ) -> None:
        self.max_pending_entries = max_pending_entries
        self.max_pending_bytes = max_pending_bytes
        self.max_receipts = max_receipts
        self.terminal_retention_seconds = max(60, int(terminal_retention_seconds))
        self.entries: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def manifest(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "durability": self.durability,
            "durable": self.durable,
            "shared": self.shared,
        }

    def _sweep_locked(self, now: float) -> None:
        retire_before = now - self.terminal_retention_seconds
        retired: list[str] = []
        for receipt_id, entry in self.entries.items():
            if (
                entry.get("state") == "quarantined"
                and float(entry.get("expiresAt") or 0) <= now
            ):
                entry["state"] = "expired"
                entry["records"] = []
                entry["bytes"] = 0
                entry["updatedAt"] = now
            if (
                entry.get("state") in _TERMINAL | {"withdrawn"}
                and float(entry.get("updatedAt") or 0) <= retire_before
            ):
                retired.append(receipt_id)
        for receipt_id in retired:
            self.entries.pop(receipt_id, None)

    def _pending_counts_locked(self) -> tuple[int, int]:
        pending = [
            e for e in self.entries.values() if e.get("state") in _ACTIVE_PENDING
        ]
        return len(pending), sum(int(e.get("bytes") or 0) for e in pending)

    async def create(self, entry: dict[str, Any]) -> None:
        async with self._lock:
            now = _now()
            self._sweep_locked(now)
            if entry["receiptId"] in self.entries:
                raise ContributionLedgerError("DUPLICATE_RECEIPT")
            if len(self.entries) >= self.max_receipts:
                raise ContributionLedgerError("RECEIPT_CAPACITY")
            count, total = self._pending_counts_locked()
            if count >= self.max_pending_entries:
                raise ContributionLedgerError("PENDING_CAPACITY")
            if total + int(entry.get("bytes") or 0) > self.max_pending_bytes:
                raise ContributionLedgerError("PENDING_BYTE_CAPACITY")
            self.entries[entry["receiptId"]] = _copy_entry(entry) or {}

    async def get(self, receipt_id: str) -> dict[str, Any] | None:
        async with self._lock:
            self._sweep_locked(_now())
            return _copy_entry(self.entries.get(receipt_id))

    async def replace_pending_payload(
        self,
        receipt_id: str,
        *,
        records: list[dict[str, Any]],
        byte_count: int,
        dedup_keys: list[str],
        payload_digest: str,
        row_count: int,
        storage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Replace one quarantined payload without changing receipt authority."""
        async with self._lock:
            now = _now()
            self._sweep_locked(now)
            entry = self.entries.get(receipt_id)
            if entry is None:
                raise ContributionLedgerError("NOT_FOUND")
            if entry.get("state") == "expired":
                raise ContributionLedgerError("EXPIRED")
            if entry.get("state") != "quarantined":
                raise ContributionLedgerError("NOT_PENDING")
            _, total = self._pending_counts_locked()
            old_bytes = int(entry.get("bytes") or 0)
            if total - old_bytes + int(byte_count) > self.max_pending_bytes:
                raise ContributionLedgerError("PENDING_BYTE_CAPACITY")
            operation = dict(entry.get("operation") or {})
            operation["payloadDigest"] = str(payload_digest)
            operation["reviewRevision"] = int(operation.get("reviewRevision") or 1) + 1
            entry["records"] = _copy_entry({"records": records})["records"]
            entry["bytes"] = int(byte_count)
            entry["dedupKeys"] = list(dedup_keys)
            entry["operation"] = operation
            entry["rowCount"] = int(row_count)
            if storage is not None:
                entry["storage"] = _copy_entry(storage) or {}
            entry["lastError"] = ""
            entry["updatedAt"] = now
            return _copy_entry(entry) or {}

    async def set_pending_storage(
        self, receipt_id: str, *, storage: dict[str, Any]
    ) -> dict[str, Any]:
        """Attach provider-review metadata while the receipt is quarantined."""
        async with self._lock:
            now = _now()
            self._sweep_locked(now)
            entry = self.entries.get(receipt_id)
            if entry is None:
                raise ContributionLedgerError("NOT_FOUND")
            if entry.get("state") == "expired":
                raise ContributionLedgerError("EXPIRED")
            if entry.get("state") != "quarantined":
                raise ContributionLedgerError("NOT_PENDING")
            entry["storage"] = _copy_entry(storage) or {}
            entry["updatedAt"] = now
            return _copy_entry(entry) or {}

    async def begin_promotion(self, receipt_id: str) -> dict[str, Any]:
        async with self._lock:
            now = _now()
            self._sweep_locked(now)
            entry = self.entries.get(receipt_id)
            if entry is None:
                raise ContributionLedgerError("NOT_FOUND")
            state = entry.get("state")
            if state == "expired":
                raise ContributionLedgerError("EXPIRED")
            if state == "promoting":
                raise ContributionLedgerError("PROMOTION_IN_PROGRESS")
            if state != "quarantined":
                raise ContributionLedgerError("NOT_PENDING")
            entry["state"] = "promoting"
            entry["updatedAt"] = now
            return _copy_entry(entry) or {}

    async def promotion_failed(
        self, receipt_id: str, code: str, *, claim_token: str | None = None
    ) -> None:
        async with self._lock:
            entry = self.entries.get(receipt_id)
            if not entry or entry.get("state") != "promoting":
                return
            now = _now()
            if float(entry.get("expiresAt") or 0) <= now:
                entry["state"] = "expired"
                entry["records"] = []
                entry["bytes"] = 0
            else:
                entry["state"] = "quarantined"
            entry["lastError"] = str(code or "PROMOTION_FAILED")[:64]
            entry["updatedAt"] = now

    async def mark_promotion_uncertain(
        self, receipt_id: str, code: str, *, claim_token: str | None = None
    ) -> dict[str, Any]:
        async with self._lock:
            entry = self.entries.get(receipt_id)
            if entry is None or entry.get("state") != "promoting":
                raise ContributionLedgerError("PROMOTION_STATE")
            entry["state"] = "promotion_uncertain"
            entry["lastError"] = str(code or "PROMOTION_OUTCOME_UNCERTAIN")[:64]
            entry["updatedAt"] = _now()
            return _copy_entry(entry) or {}

    async def mark_promoted(
        self,
        receipt_id: str,
        *,
        storage: dict[str, Any],
        claim_token: str | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            entry = self.entries.get(receipt_id)
            if entry is None or entry.get("state") != "promoting":
                raise ContributionLedgerError("PROMOTION_STATE")
            now = _now()
            entry["state"] = "eligible"
            entry["promotedAt"] = now
            entry["storage"] = _copy_entry(storage) or {}
            entry["records"] = []
            entry["bytes"] = 0
            entry["lastError"] = ""
            entry["updatedAt"] = now
            return _copy_entry(entry) or {}

    async def delete_pending(self, receipt_id: str) -> dict[str, Any]:
        async with self._lock:
            now = _now()
            self._sweep_locked(now)
            entry = self.entries.get(receipt_id)
            if entry is None:
                raise ContributionLedgerError("NOT_FOUND")
            state = entry.get("state")
            if state == "expired":
                raise ContributionLedgerError("EXPIRED")
            if state in {"promoting", "withdrawing"}:
                raise ContributionLedgerError("BUSY")
            if state != "quarantined":
                raise ContributionLedgerError("NOT_PENDING")
            entry["state"] = "deleted"
            entry["records"] = []
            entry["bytes"] = 0
            entry["deletedAt"] = now
            entry["updatedAt"] = now
            return _copy_entry(entry) or {}

    async def begin_withdrawal(self, receipt_id: str) -> dict[str, Any]:
        async with self._lock:
            entry = self.entries.get(receipt_id)
            if entry is None:
                raise ContributionLedgerError("NOT_FOUND")
            state = entry.get("state")
            if state == "withdrawn":
                return _copy_entry(entry) or {}
            if state == "withdrawing":
                raise ContributionLedgerError("WITHDRAWAL_IN_PROGRESS")
            if state not in {"eligible", "promotion_uncertain", "withdrawal_uncertain"}:
                raise ContributionLedgerError("NOT_ELIGIBLE")
            entry["state"] = "withdrawing"
            entry["updatedAt"] = _now()
            return _copy_entry(entry) or {}

    async def withdrawal_failed(
        self, receipt_id: str, code: str, *, claim_token: str | None = None
    ) -> None:
        async with self._lock:
            entry = self.entries.get(receipt_id)
            if not entry or entry.get("state") != "withdrawing":
                return
            entry["state"] = (
                "eligible" if entry.get("promotedAt") else "promotion_uncertain"
            )
            entry["lastError"] = str(code or "WITHDRAWAL_FAILED")[:64]
            entry["updatedAt"] = _now()

    async def mark_withdrawn(
        self,
        receipt_id: str,
        *,
        withdrawal_storage: dict[str, Any],
        current_view_removal: dict[str, str],
        claim_token: str | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            entry = self.entries.get(receipt_id)
            if entry is None or entry.get("state") != "withdrawing":
                raise ContributionLedgerError("WITHDRAWAL_STATE")
            now = _now()
            entry["state"] = "withdrawn"
            entry["records"] = []
            entry["bytes"] = 0
            entry["withdrawnAt"] = now
            entry["withdrawalStorage"] = _copy_entry(withdrawal_storage) or {}
            entry["currentViewRemoval"] = dict(current_view_removal)
            entry["lastError"] = ""
            entry["updatedAt"] = now
            return _copy_entry(entry) or {}

    def clear_for_tests(self) -> None:
        self.entries.clear()


class SQLiteContributionLedger:
    """
    Local ACID receipt ledger backed by SQLite.

    SQLite is transactional and process-restart durable when the configured file
    resides on durable storage.  It is deliberately advertised as *local*, not
    shared/distributed, so a multi-replica deployment cannot accidentally claim
    one authoritative review ledger.
    """

    backend = "sqlite"
    durability = "local_transactional"
    durable = True
    shared = False

    def __init__(
        self,
        path: str,
        *,
        max_pending_entries: int,
        max_pending_bytes: int,
        max_receipts: int,
        terminal_retention_seconds: int = 86_400,
    ) -> None:
        self.path = str(Path(path).expanduser())
        self.max_pending_entries = max_pending_entries
        self.max_pending_bytes = max_pending_bytes
        self.max_receipts = max_receipts
        self.terminal_retention_seconds = max(60, int(terminal_retention_seconds))
        self._lock = asyncio.Lock()

    def manifest(self) -> dict[str, Any]:
        # Do not expose the filesystem path in public discovery/logs.
        return {
            "backend": self.backend,
            "durability": self.durability,
            "durable": self.durable,
            "shared": self.shared,
        }

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        # Defense in depth for sensitive pending rows.  This reduces forensic
        # remnants in ordinary SQLite table pages; it is not a global erasure
        # guarantee because WAL/filesystem snapshots/backups may exist.
        conn.execute("PRAGMA secure_delete=ON")
        conn.execute("PRAGMA journal_size_limit=0")
        return conn

    def _init_sync(self) -> None:
        parent = Path(self.path).parent
        parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS contribution_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    records_json TEXT NOT NULL DEFAULT '[]',
                    bytes INTEGER NOT NULL DEFAULT 0,
                    delete_token_hash TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    received_at REAL NOT NULL,
                    promoted_at REAL,
                    withdrawn_at REAL,
                    deleted_at REAL,
                    dedup_keys_json TEXT NOT NULL DEFAULT '[]',
                    storage_json TEXT NOT NULL DEFAULT '{}',
                    withdrawal_storage_json TEXT NOT NULL DEFAULT '{}',
                    current_view_removal_json TEXT NOT NULL DEFAULT '{}',
                    last_error TEXT NOT NULL DEFAULT '',
                    operation_json TEXT NOT NULL DEFAULT '{}',
                    row_count INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                )
            """)
            # Additive migration for pre-Run-18 ledgers.  Existing receipt
            # lifecycle state is preserved; only recovery metadata is new.
            _columns = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(contribution_receipts)"
                ).fetchall()
            }
            if "operation_json" not in _columns:
                conn.execute(
                    "ALTER TABLE contribution_receipts ADD COLUMN operation_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "row_count" not in _columns:
                conn.execute(
                    "ALTER TABLE contribution_receipts ADD COLUMN row_count INTEGER NOT NULL DEFAULT 0"
                )
            # Any transient operation state present during startup belongs to a
            # previous process. Promotion is replay-safe because the app writes
            # the reviewed payload to a receipt-stable provider path derived from
            # receivedAt. Withdrawal tombstones are also idempotent under dataset
            # last-write-wins, so both states can be reclaimed rather than left
            # permanently BUSY after a crash.
            now = _now()
            conn.execute(
                """UPDATE contribution_receipts
                   SET state=CASE WHEN expires_at <= ? THEN 'expired' ELSE 'quarantined' END,
                       records_json=CASE WHEN expires_at <= ? THEN '[]' ELSE records_json END,
                       bytes=CASE WHEN expires_at <= ? THEN 0 ELSE bytes END,
                       last_error='RECOVERED_AFTER_RESTART',updated_at=?
                   WHERE state='promoting'""",
                (now, now, now, now),
            )
            conn.execute(
                """UPDATE contribution_receipts
                   SET state=CASE WHEN promoted_at IS NULL THEN 'promotion_uncertain' ELSE 'eligible' END,last_error='RECOVERED_AFTER_RESTART',updated_at=?
                   WHERE state='withdrawing'""",
                (now,),
            )
            self._sweep_sync(conn, now)
            conn.commit()
            self._checkpoint_sensitive(conn)
        finally:
            conn.close()

    async def initialize(self) -> None:
        await asyncio.to_thread(self._init_sync)

    async def close(self) -> None:
        return None

    @staticmethod
    def _row_to_entry(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "receiptId": row["receipt_id"],
            "state": row["state"],
            "records": json.loads(row["records_json"] or "[]"),
            "bytes": int(row["bytes"] or 0),
            "deleteTokenHash": row["delete_token_hash"],
            "expiresAt": float(row["expires_at"] or 0),
            "receivedAt": float(row["received_at"] or 0),
            "promotedAt": row["promoted_at"],
            "withdrawnAt": row["withdrawn_at"],
            "deletedAt": row["deleted_at"],
            "dedupKeys": json.loads(row["dedup_keys_json"] or "[]"),
            "storage": json.loads(row["storage_json"] or "{}"),
            "withdrawalStorage": json.loads(row["withdrawal_storage_json"] or "{}"),
            "currentViewRemoval": json.loads(row["current_view_removal_json"] or "{}"),
            "lastError": row["last_error"] or "",
            "operation": json.loads(row["operation_json"] or "{}"),
            "rowCount": int(row["row_count"] or 0),
            "updatedAt": float(row["updated_at"] or 0),
        }

    def _sweep_sync(self, conn: sqlite3.Connection, now: float) -> None:
        conn.execute(
            """UPDATE contribution_receipts
               SET state='expired', records_json='[]', bytes=0, updated_at=?
               WHERE state='quarantined' AND expires_at <= ?""",
            (now, now),
        )
        # Terminal lifecycle tombstones are useful for a bounded status window,
        # but keeping them forever turns max_receipts into a permanent denial of
        # future intake. Eligible receipts are intentionally retained until the
        # participant withdraws or an external control-plane policy supersedes
        # this single-instance backend.
        conn.execute(
            """DELETE FROM contribution_receipts
               WHERE state IN ('deleted','expired','withdrawn') AND updated_at <= ?""",
            (now - self.terminal_retention_seconds,),
        )

    @staticmethod
    def _checkpoint_sensitive(conn: sqlite3.Connection) -> None:
        """Best-effort truncate WAL after content-clearing lifecycle writes."""
        try:  # ruff: ignore[suppressible-exception]
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.DatabaseError:
            # Checkpoint may be busy when another process/connection is active.
            # The lifecycle transaction is already committed; never reinterpret
            # a checkpoint limitation as proof that the user content was erased.
            pass

    async def create(self, entry: dict[str, Any]) -> None:
        async with self._lock:

            def _op() -> None:
                conn = self._connect()
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    now = _now()
                    self._sweep_sync(conn, now)
                    total_rows = int(
                        conn.execute(
                            "SELECT COUNT(*) FROM contribution_receipts"
                        ).fetchone()[0]
                    )
                    if total_rows >= self.max_receipts:
                        raise ContributionLedgerError("RECEIPT_CAPACITY")
                    pending_count, pending_bytes = conn.execute(
                        "SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM contribution_receipts WHERE state IN ('quarantined','promoting','promotion_uncertain','withdrawing')"
                    ).fetchone()
                    if int(pending_count) >= self.max_pending_entries:
                        raise ContributionLedgerError("PENDING_CAPACITY")
                    if (
                        int(pending_bytes) + int(entry.get("bytes") or 0)
                        > self.max_pending_bytes
                    ):
                        raise ContributionLedgerError("PENDING_BYTE_CAPACITY")
                    conn.execute(
                        """INSERT INTO contribution_receipts
                           (receipt_id,state,records_json,bytes,delete_token_hash,expires_at,received_at,
                            dedup_keys_json,operation_json,row_count,updated_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            entry["receiptId"],
                            entry["state"],
                            json.dumps(
                                entry.get("records") or [],
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            int(entry.get("bytes") or 0),
                            entry["deleteTokenHash"],
                            float(entry["expiresAt"]),
                            float(entry["receivedAt"]),
                            json.dumps(
                                entry.get("dedupKeys") or [], separators=(",", ":")
                            ),
                            json.dumps(
                                entry.get("operation") or {}, separators=(",", ":")
                            ),
                            int(entry.get("rowCount") or 0),
                            now,
                        ),
                    )
                    conn.commit()
                except sqlite3.IntegrityError as exc:
                    conn.rollback()
                    raise ContributionLedgerError("DUPLICATE_RECEIPT") from exc
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()

            await asyncio.to_thread(_op)

    async def get(self, receipt_id: str) -> dict[str, Any] | None:
        async with self._lock:

            def _op() -> dict[str, Any] | None:
                conn = self._connect()
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    self._sweep_sync(conn, _now())
                    row = conn.execute(
                        "SELECT * FROM contribution_receipts WHERE receipt_id=?",
                        (receipt_id,),
                    ).fetchone()
                    conn.commit()
                    self._checkpoint_sensitive(conn)
                    return self._row_to_entry(row)
                finally:
                    conn.close()

            return await asyncio.to_thread(_op)

    async def _transition(
        self,
        receipt_id: str,
        *,
        allowed: set[str],
        to_state: str,
        busy_code: str | None = None,
    ) -> dict[str, Any]:
        async with self._lock:

            def _op() -> dict[str, Any]:
                conn = self._connect()
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    now = _now()
                    self._sweep_sync(conn, now)
                    row = conn.execute(
                        "SELECT * FROM contribution_receipts WHERE receipt_id=?",
                        (receipt_id,),
                    ).fetchone()
                    entry = self._row_to_entry(row)
                    if entry is None:
                        raise ContributionLedgerError("NOT_FOUND")
                    state = str(entry["state"])
                    if state == "expired":
                        raise ContributionLedgerError("EXPIRED")
                    if state not in allowed:
                        if busy_code and state in {"promoting", "withdrawing"}:
                            raise ContributionLedgerError(busy_code)
                        raise ContributionLedgerError(
                            "NOT_PENDING" if to_state == "promoting" else "NOT_ELIGIBLE"
                        )
                    conn.execute(
                        "UPDATE contribution_receipts SET state=?, updated_at=? WHERE receipt_id=?",
                        (to_state, now, receipt_id),
                    )
                    row = conn.execute(
                        "SELECT * FROM contribution_receipts WHERE receipt_id=?",
                        (receipt_id,),
                    ).fetchone()
                    conn.commit()
                    return self._row_to_entry(row) or {}
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()

            return await asyncio.to_thread(_op)

    async def replace_pending_payload(
        self,
        receipt_id: str,
        *,
        records: list[dict[str, Any]],
        byte_count: int,
        dedup_keys: list[str],
        payload_digest: str,
        row_count: int,
        storage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        async with self._lock:

            def _op() -> dict[str, Any]:
                conn = self._connect()
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    now = _now()
                    self._sweep_sync(conn, now)
                    row = conn.execute(
                        "SELECT * FROM contribution_receipts WHERE receipt_id=?",
                        (receipt_id,),
                    ).fetchone()
                    entry = self._row_to_entry(row)
                    if entry is None:
                        raise ContributionLedgerError("NOT_FOUND")
                    if entry.get("state") == "expired":
                        raise ContributionLedgerError("EXPIRED")
                    if entry.get("state") != "quarantined":
                        raise ContributionLedgerError("NOT_PENDING")
                    pending_bytes = int(
                        conn.execute(
                            "SELECT COALESCE(SUM(bytes), 0) FROM contribution_receipts WHERE state IN ('quarantined','promoting','promotion_uncertain','withdrawing')"
                        ).fetchone()[0]
                    )
                    if (
                        pending_bytes - int(entry.get("bytes") or 0) + int(byte_count)
                        > self.max_pending_bytes
                    ):
                        raise ContributionLedgerError("PENDING_BYTE_CAPACITY")
                    operation = dict(entry.get("operation") or {})
                    operation["payloadDigest"] = str(payload_digest)
                    operation["reviewRevision"] = (
                        int(operation.get("reviewRevision") or 1) + 1
                    )
                    storage_json = (
                        json.dumps(storage, separators=(",", ":"))
                        if storage is not None
                        else json.dumps(
                            entry.get("storage") or {}, separators=(",", ":")
                        )
                    )
                    conn.execute(
                        """UPDATE contribution_receipts
                           SET records_json=?,bytes=?,dedup_keys_json=?,operation_json=?,row_count=?,storage_json=?,last_error='',updated_at=?
                           WHERE receipt_id=?""",
                        (
                            json.dumps(
                                records, ensure_ascii=False, separators=(",", ":")
                            ),
                            int(byte_count),
                            json.dumps(dedup_keys, separators=(",", ":")),
                            json.dumps(operation, separators=(",", ":")),
                            int(row_count),
                            storage_json,
                            now,
                            receipt_id,
                        ),
                    )
                    row = conn.execute(
                        "SELECT * FROM contribution_receipts WHERE receipt_id=?",
                        (receipt_id,),
                    ).fetchone()
                    conn.commit()
                    return self._row_to_entry(row) or {}
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()

            return await asyncio.to_thread(_op)

    async def set_pending_storage(
        self, receipt_id: str, *, storage: dict[str, Any]
    ) -> dict[str, Any]:
        """Persist provider-review metadata without changing lifecycle state."""
        async with self._lock:

            def _op() -> dict[str, Any]:
                conn = self._connect()
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    self._sweep_sync(conn, _now())
                    row = conn.execute(
                        "SELECT state FROM contribution_receipts WHERE receipt_id=?",
                        (receipt_id,),
                    ).fetchone()
                    if row is None:
                        raise ContributionLedgerError("NOT_FOUND")
                    if row["state"] == "expired":
                        raise ContributionLedgerError("EXPIRED")
                    if row["state"] != "quarantined":
                        raise ContributionLedgerError("NOT_PENDING")
                    now = _now()
                    conn.execute(
                        "UPDATE contribution_receipts SET storage_json=?,updated_at=? WHERE receipt_id=?",
                        (json.dumps(storage, separators=(",", ":")), now, receipt_id),
                    )
                    row = conn.execute(
                        "SELECT * FROM contribution_receipts WHERE receipt_id=?",
                        (receipt_id,),
                    ).fetchone()
                    conn.commit()
                    self._checkpoint_sensitive(conn)
                    return self._row_to_entry(row) or {}
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()

            return await asyncio.to_thread(_op)

    async def begin_promotion(self, receipt_id: str) -> dict[str, Any]:
        return await self._transition(
            receipt_id,
            allowed={"quarantined"},
            to_state="promoting",
            busy_code="PROMOTION_IN_PROGRESS",
        )

    async def promotion_failed(
        self, receipt_id: str, code: str, *, claim_token: str | None = None
    ) -> None:
        async with self._lock:

            def _op() -> None:
                conn = self._connect()
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    row = conn.execute(
                        "SELECT state,expires_at FROM contribution_receipts WHERE receipt_id=?",
                        (receipt_id,),
                    ).fetchone()
                    if row and row["state"] == "promoting":
                        now = _now()
                        if float(row["expires_at"] or 0) <= now:
                            conn.execute(
                                "UPDATE contribution_receipts SET state='expired',records_json='[]',bytes=0,last_error=?,updated_at=? WHERE receipt_id=?",
                                (str(code or "PROMOTION_FAILED")[:64], now, receipt_id),
                            )
                        else:
                            conn.execute(
                                "UPDATE contribution_receipts SET state='quarantined',last_error=?,updated_at=? WHERE receipt_id=?",
                                (str(code or "PROMOTION_FAILED")[:64], now, receipt_id),
                            )
                    conn.commit()
                    self._checkpoint_sensitive(conn)
                finally:
                    conn.close()

            await asyncio.to_thread(_op)

    async def mark_promotion_uncertain(
        self, receipt_id: str, code: str, *, claim_token: str | None = None
    ) -> dict[str, Any]:
        async with self._lock:

            def _op() -> dict[str, Any]:
                conn = self._connect()
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    row = conn.execute(
                        "SELECT state FROM contribution_receipts WHERE receipt_id=?",
                        (receipt_id,),
                    ).fetchone()
                    if row is None or row["state"] != "promoting":
                        raise ContributionLedgerError("PROMOTION_STATE")
                    now = _now()
                    conn.execute(
                        "UPDATE contribution_receipts SET state='promotion_uncertain',last_error=?,updated_at=? WHERE receipt_id=?",
                        (
                            str(code or "PROMOTION_OUTCOME_UNCERTAIN")[:64],
                            now,
                            receipt_id,
                        ),
                    )
                    row = conn.execute(
                        "SELECT * FROM contribution_receipts WHERE receipt_id=?",
                        (receipt_id,),
                    ).fetchone()
                    conn.commit()
                    return self._row_to_entry(row) or {}
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()

            return await asyncio.to_thread(_op)

    async def mark_promoted(
        self,
        receipt_id: str,
        *,
        storage: dict[str, Any],
        claim_token: str | None = None,
    ) -> dict[str, Any]:
        async with self._lock:

            def _op() -> dict[str, Any]:
                conn = self._connect()
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    row = conn.execute(
                        "SELECT state FROM contribution_receipts WHERE receipt_id=?",
                        (receipt_id,),
                    ).fetchone()
                    if row is None or row["state"] != "promoting":
                        raise ContributionLedgerError("PROMOTION_STATE")
                    now = _now()
                    conn.execute(
                        """UPDATE contribution_receipts
                           SET state='eligible',records_json='[]',bytes=0,promoted_at=?,storage_json=?,last_error='',updated_at=?
                           WHERE receipt_id=?""",
                        (
                            now,
                            json.dumps(storage, separators=(",", ":")),
                            now,
                            receipt_id,
                        ),
                    )
                    row = conn.execute(
                        "SELECT * FROM contribution_receipts WHERE receipt_id=?",
                        (receipt_id,),
                    ).fetchone()
                    conn.commit()
                    self._checkpoint_sensitive(conn)
                    return self._row_to_entry(row) or {}
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()

            return await asyncio.to_thread(_op)

    async def delete_pending(self, receipt_id: str) -> dict[str, Any]:
        async with self._lock:

            def _op() -> dict[str, Any]:
                conn = self._connect()
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    now = _now()
                    self._sweep_sync(conn, now)
                    row = conn.execute(
                        "SELECT * FROM contribution_receipts WHERE receipt_id=?",
                        (receipt_id,),
                    ).fetchone()
                    entry = self._row_to_entry(row)
                    if entry is None:
                        raise ContributionLedgerError("NOT_FOUND")
                    state = entry["state"]
                    if state == "expired":
                        raise ContributionLedgerError("EXPIRED")
                    if state in {"promoting", "withdrawing"}:
                        raise ContributionLedgerError("BUSY")
                    if state != "quarantined":
                        raise ContributionLedgerError("NOT_PENDING")
                    conn.execute(
                        """UPDATE contribution_receipts SET state='deleted',records_json='[]',bytes=0,deleted_at=?,updated_at=?
                           WHERE receipt_id=?""",
                        (now, now, receipt_id),
                    )
                    row = conn.execute(
                        "SELECT * FROM contribution_receipts WHERE receipt_id=?",
                        (receipt_id,),
                    ).fetchone()
                    conn.commit()
                    self._checkpoint_sensitive(conn)
                    return self._row_to_entry(row) or {}
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()

            return await asyncio.to_thread(_op)

    async def begin_withdrawal(self, receipt_id: str) -> dict[str, Any]:
        current = await self.get(receipt_id)
        if current and current.get("state") == "withdrawn":
            return current
        return await self._transition(
            receipt_id,
            allowed={"eligible", "promotion_uncertain", "withdrawal_uncertain"},
            to_state="withdrawing",
            busy_code="WITHDRAWAL_IN_PROGRESS",
        )

    async def withdrawal_failed(
        self, receipt_id: str, code: str, *, claim_token: str | None = None
    ) -> None:
        async with self._lock:

            def _op() -> None:
                conn = self._connect()
                try:
                    now = _now()
                    conn.execute(
                        """UPDATE contribution_receipts SET state=CASE WHEN promoted_at IS NULL THEN 'promotion_uncertain' ELSE 'eligible' END,last_error=?,updated_at=?
                           WHERE receipt_id=? AND state='withdrawing'""",
                        (str(code or "WITHDRAWAL_FAILED")[:64], now, receipt_id),
                    )
                    conn.commit()
                finally:
                    conn.close()

            await asyncio.to_thread(_op)

    async def mark_withdrawn(
        self,
        receipt_id: str,
        *,
        withdrawal_storage: dict[str, Any],
        current_view_removal: dict[str, str],
        claim_token: str | None = None,
    ) -> dict[str, Any]:
        async with self._lock:

            def _op() -> dict[str, Any]:
                conn = self._connect()
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    row = conn.execute(
                        "SELECT state FROM contribution_receipts WHERE receipt_id=?",
                        (receipt_id,),
                    ).fetchone()
                    if row is None or row["state"] != "withdrawing":
                        raise ContributionLedgerError("WITHDRAWAL_STATE")
                    now = _now()
                    conn.execute(
                        """UPDATE contribution_receipts
                           SET state='withdrawn',records_json='[]',bytes=0,withdrawn_at=?,withdrawal_storage_json=?,current_view_removal_json=?,last_error='',updated_at=?
                           WHERE receipt_id=?""",
                        (
                            now,
                            json.dumps(withdrawal_storage, separators=(",", ":")),
                            json.dumps(current_view_removal, separators=(",", ":")),
                            now,
                            receipt_id,
                        ),
                    )
                    row = conn.execute(
                        "SELECT * FROM contribution_receipts WHERE receipt_id=?",
                        (receipt_id,),
                    ).fetchone()
                    conn.commit()
                    return self._row_to_entry(row) or {}
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()

            return await asyncio.to_thread(_op)


# Redis scripts intentionally keep all index keys in one ``{contribution}``
# hash slot.  This makes the lifecycle operations compatible with a Redis
# Cluster consistency domain without scattering one receipt transition across
# slots.  Receipt identifiers are HMACed before becoming Redis key material.
_REDIS_CREATE_LUA = r"""
local now = tonumber(ARGV[1])
local member = ARGV[2]
local payload = ARGV[3]
local expires_at = tonumber(ARGV[4])
local live_until = tonumber(ARGV[5])
local max_receipts = tonumber(ARGV[6])
local max_pending = tonumber(ARGV[7])
local max_bytes = tonumber(ARGV[8])
local bytes = tonumber(ARGV[9])
local ttl = tonumber(ARGV[10])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
local expired = redis.call('ZRANGEBYSCORE', KEYS[2], '-inf', now)
if #expired > 0 then
  redis.call('ZREM', KEYS[2], unpack(expired))
  redis.call('HDEL', KEYS[3], unpack(expired))
end
if redis.call('EXISTS', KEYS[4]) == 1 then return {0, 'DUPLICATE_RECEIPT'} end
if redis.call('ZCARD', KEYS[1]) >= max_receipts then return {0, 'RECEIPT_CAPACITY'} end
if redis.call('ZCARD', KEYS[2]) >= max_pending then return {0, 'PENDING_CAPACITY'} end
local values = redis.call('HVALS', KEYS[3])
local pending_bytes = 0
for _, value in ipairs(values) do pending_bytes = pending_bytes + tonumber(value) end
if pending_bytes + bytes > max_bytes then return {0, 'PENDING_BYTE_CAPACITY'} end
local created = redis.call('SET', KEYS[4], payload, 'EX', ttl, 'NX')
if not created then return {0, 'DUPLICATE_RECEIPT'} end
redis.call('ZADD', KEYS[1], live_until, member)
redis.call('ZADD', KEYS[2], expires_at, member)
redis.call('HSET', KEYS[3], member, bytes)
return {1, payload}
""".strip()

_REDIS_GET_LUA = r"""
local now = tonumber(ARGV[1])
local member = ARGV[2]
local terminal_retention = tonumber(ARGV[3])
local immortal = tonumber(ARGV[4])
local raw = redis.call('GET', KEYS[4])
if not raw then
  redis.call('ZREM', KEYS[1], member)
  redis.call('ZREM', KEYS[2], member)
  redis.call('HDEL', KEYS[3], member)
  return {1, ''}
end
local entry = cjson.decode(raw)
local state = tostring(entry.state or '')
local expires_at = tonumber(entry.expiresAt or 0)
local lease_until = tonumber(entry.operationLeaseUntil or 0)
if state == 'quarantined' and expires_at <= now then
  entry.state = 'expired'; entry.records = {}; entry.bytes = 0
  entry.lastError = ''; entry.updatedAt = now
  entry.operationClaimHash = ''; entry.operationLeaseUntil = 0
  raw = cjson.encode(entry)
  redis.call('SET', KEYS[4], raw, 'EX', terminal_retention)
  redis.call('ZREM', KEYS[2], member); redis.call('HDEL', KEYS[3], member)
  redis.call('ZADD', KEYS[1], now + terminal_retention, member)
elseif state == 'promoting' and lease_until > 0 and lease_until <= now then
  entry.state = 'promotion_uncertain'; entry.lastError = 'CLAIM_EXPIRED_RECONCILIATION_REQUIRED'; entry.updatedAt = now
  entry.operationClaimHash = ''; entry.operationLeaseUntil = 0
  raw = cjson.encode(entry)
  redis.call('SET', KEYS[4], raw); redis.call('PERSIST', KEYS[4])
  redis.call('ZADD', KEYS[1], immortal, member); redis.call('ZADD', KEYS[2], immortal, member)
  redis.call('HSET', KEYS[3], member, tonumber(entry.bytes or 0))
elseif state == 'withdrawing' and lease_until > 0 and lease_until <= now then
  entry.state = 'withdrawal_uncertain'; entry.lastError = 'CLAIM_EXPIRED_RECONCILIATION_REQUIRED'; entry.updatedAt = now
  entry.operationClaimHash = ''; entry.operationLeaseUntil = 0
  raw = cjson.encode(entry)
  redis.call('SET', KEYS[4], raw); redis.call('PERSIST', KEYS[4]); redis.call('ZADD', KEYS[1], immortal, member)
end
return {1, raw}
""".strip()

_REDIS_REPLACE_PENDING_LUA = r"""
local now=tonumber(ARGV[1]); local member=ARGV[2]; local records_json=ARGV[3]
local new_bytes=tonumber(ARGV[4]); local dedup_json=ARGV[5]; local payload_digest=ARGV[6]
local row_count=tonumber(ARGV[7]); local max_bytes=tonumber(ARGV[8]); local terminal_retention=tonumber(ARGV[9]); local storage_json=ARGV[10]
local raw=redis.call('GET', KEYS[4]); if not raw then return {0, 'NOT_FOUND'} end
local entry=cjson.decode(raw); local state=tostring(entry.state or ''); local expires_at=tonumber(entry.expiresAt or 0)
if state == 'quarantined' and expires_at <= now then
  entry.state='expired'; entry.records={}; entry.bytes=0; entry.updatedAt=now
  raw=cjson.encode(entry); redis.call('SET', KEYS[4], raw, 'EX', terminal_retention)
  redis.call('ZREM', KEYS[2], member); redis.call('HDEL', KEYS[3], member); redis.call('ZADD', KEYS[1], now+terminal_retention, member)
  return {0, 'EXPIRED'}
end
if state ~= 'quarantined' then return {0, 'NOT_PENDING'} end
local values=redis.call('HVALS', KEYS[3]); local total=0
for _, value in ipairs(values) do total=total+tonumber(value) end
local old_bytes=tonumber(entry.bytes or 0)
if total-old_bytes+new_bytes > max_bytes then return {0, 'PENDING_BYTE_CAPACITY'} end
entry.records=cjson.decode(records_json); entry.bytes=new_bytes; entry.dedupKeys=cjson.decode(dedup_json); entry.rowCount=row_count
local op=entry.operation or {}; op.payloadDigest=payload_digest; op.reviewRevision=tonumber(op.reviewRevision or 1)+1; entry.operation=op
if storage_json ~= '' then entry.storage=cjson.decode(storage_json) end
entry.lastError=''; entry.updatedAt=now
raw=cjson.encode(entry); redis.call('SET', KEYS[4], raw, 'KEEPTTL'); redis.call('HSET', KEYS[3], member, new_bytes)
return {1, raw}
""".strip()

_REDIS_SET_PENDING_STORAGE_LUA = r"""
local now=tonumber(ARGV[1]); local member=ARGV[2]; local storage_json=ARGV[3]; local terminal_retention=tonumber(ARGV[4])
local raw=redis.call('GET', KEYS[4]); if not raw then return {0, 'NOT_FOUND'} end
local entry=cjson.decode(raw); local state=tostring(entry.state or ''); local expires_at=tonumber(entry.expiresAt or 0)
if state == 'quarantined' and expires_at <= now then
  entry.state='expired'; entry.records={}; entry.bytes=0; entry.updatedAt=now
  raw=cjson.encode(entry); redis.call('SET', KEYS[4], raw, 'EX', terminal_retention)
  redis.call('ZREM', KEYS[2], member); redis.call('HDEL', KEYS[3], member); redis.call('ZADD', KEYS[1], now+terminal_retention, member)
  return {0, 'EXPIRED'}
end
if state ~= 'quarantined' then return {0, 'NOT_PENDING'} end
entry.storage=cjson.decode(storage_json); entry.updatedAt=now
raw=cjson.encode(entry); redis.call('SET', KEYS[4], raw, 'KEEPTTL')
return {1, raw}
""".strip()

_REDIS_BEGIN_PROMOTION_LUA = r"""
local now = tonumber(ARGV[1]); local member = ARGV[2]; local claim_hash = ARGV[3]
local lease_until = tonumber(ARGV[4]); local terminal_retention = tonumber(ARGV[5])
local raw = redis.call('GET', KEYS[4]); if not raw then return {0, 'NOT_FOUND'} end
local entry = cjson.decode(raw); local state = tostring(entry.state or '')
local expires_at = tonumber(entry.expiresAt or 0); local old_lease = tonumber(entry.operationLeaseUntil or 0)
if state == 'promoting' and old_lease > now then return {0, 'PROMOTION_IN_PROGRESS'} end
if state == 'promoting' and old_lease <= now then
  entry.state='promotion_uncertain'; entry.lastError='CLAIM_EXPIRED_RECONCILIATION_REQUIRED'
  entry.operationClaimHash=''; entry.operationLeaseUntil=0; entry.updatedAt=now
  raw=cjson.encode(entry); redis.call('SET', KEYS[4], raw); redis.call('PERSIST', KEYS[4])
  redis.call('ZADD', KEYS[1], 253402300799, member); redis.call('ZADD', KEYS[2], 253402300799, member)
  redis.call('HSET', KEYS[3], member, tonumber(entry.bytes or 0))
  return {0, 'RECONCILIATION_REQUIRED'}
end
if state == 'quarantined' and expires_at <= now then
  entry.state='expired'; entry.records={}; entry.bytes=0; entry.updatedAt=now
  entry.operationClaimHash=''; entry.operationLeaseUntil=0
  raw=cjson.encode(entry); redis.call('SET', KEYS[4], raw, 'EX', terminal_retention)
  redis.call('ZREM', KEYS[2], member); redis.call('HDEL', KEYS[3], member)
  redis.call('ZADD', KEYS[1], now + terminal_retention, member)
  return {0, 'EXPIRED'}
end
if state ~= 'quarantined' then return {0, 'NOT_PENDING'} end
entry.state='promoting'; entry.operationClaimHash=claim_hash; entry.operationLeaseUntil=lease_until
entry.lastError=''; entry.updatedAt=now
raw=cjson.encode(entry); redis.call('SET', KEYS[4], raw, 'KEEPTTL')
local pending_until=expires_at; if lease_until > pending_until then pending_until=lease_until end
redis.call('ZADD', KEYS[2], pending_until, member); redis.call('HSET', KEYS[3], member, tonumber(entry.bytes or 0))
return {1, raw}
""".strip()

_REDIS_PROMOTION_FAILED_LUA = r"""
local now=tonumber(ARGV[1]); local member=ARGV[2]; local claim_hash=ARGV[3]; local code=ARGV[4]
local terminal_retention=tonumber(ARGV[5])
local raw=redis.call('GET', KEYS[4]); if not raw then return {1, ''} end
local entry=cjson.decode(raw)
if tostring(entry.state or '') ~= 'promoting' or tostring(entry.operationClaimHash or '') ~= claim_hash then return {1, raw} end
local expires_at=tonumber(entry.expiresAt or 0)
entry.operationClaimHash=''; entry.operationLeaseUntil=0; entry.lastError=string.sub(code,1,64); entry.updatedAt=now
if expires_at <= now then
  entry.state='expired'; entry.records={}; entry.bytes=0
  raw=cjson.encode(entry); redis.call('SET', KEYS[4], raw, 'EX', terminal_retention)
  redis.call('ZREM', KEYS[2], member); redis.call('HDEL', KEYS[3], member)
  redis.call('ZADD', KEYS[1], now + terminal_retention, member)
else
  entry.state='quarantined'; raw=cjson.encode(entry); redis.call('SET', KEYS[4], raw, 'KEEPTTL')
  redis.call('ZADD', KEYS[2], expires_at, member); redis.call('HSET', KEYS[3], member, tonumber(entry.bytes or 0))
end
return {1, raw}
""".strip()

_REDIS_MARK_PROMOTION_UNCERTAIN_LUA = r"""
local now=tonumber(ARGV[1]); local member=ARGV[2]; local claim_hash=ARGV[3]; local code=ARGV[4]
local immortal=tonumber(ARGV[5]); local raw=redis.call('GET', KEYS[4]); if not raw then return {0, 'NOT_FOUND'} end
local entry=cjson.decode(raw)
if tostring(entry.state or '') ~= 'promoting' then return {0, 'PROMOTION_STATE'} end
if tostring(entry.operationClaimHash or '') ~= claim_hash then return {0, 'STALE_CLAIM'} end
entry.state='promotion_uncertain'; entry.lastError=string.sub(code,1,64); entry.updatedAt=now
entry.operationClaimHash=''; entry.operationLeaseUntil=0
raw=cjson.encode(entry); redis.call('SET', KEYS[4], raw); redis.call('PERSIST', KEYS[4])
redis.call('ZADD', KEYS[1], immortal, member); redis.call('ZADD', KEYS[2], immortal, member)
redis.call('HSET', KEYS[3], member, tonumber(entry.bytes or 0))
return {1, raw}
""".strip()

_REDIS_MARK_PROMOTED_LUA = r"""
local now=tonumber(ARGV[1]); local member=ARGV[2]; local claim_hash=ARGV[3]; local storage_json=ARGV[4]
local immortal=tonumber(ARGV[5]); local raw=redis.call('GET', KEYS[4]); if not raw then return {0, 'NOT_FOUND'} end
local entry=cjson.decode(raw)
if tostring(entry.state or '') ~= 'promoting' then return {0, 'PROMOTION_STATE'} end
if tostring(entry.operationClaimHash or '') ~= claim_hash then return {0, 'STALE_CLAIM'} end
entry.state='eligible'; entry.records={}; entry.bytes=0; entry.promotedAt=now; entry.storage=cjson.decode(storage_json)
entry.lastError=''; entry.updatedAt=now; entry.operationClaimHash=''; entry.operationLeaseUntil=0
raw=cjson.encode(entry); redis.call('SET', KEYS[4], raw); redis.call('PERSIST', KEYS[4])
redis.call('ZREM', KEYS[2], member); redis.call('HDEL', KEYS[3], member); redis.call('ZADD', KEYS[1], immortal, member)
return {1, raw}
""".strip()

_REDIS_DELETE_PENDING_LUA = r"""
local now=tonumber(ARGV[1]); local member=ARGV[2]; local terminal_retention=tonumber(ARGV[3])
local raw=redis.call('GET', KEYS[4]); if not raw then return {0, 'NOT_FOUND'} end
local entry=cjson.decode(raw); local state=tostring(entry.state or '')
local expires_at=tonumber(entry.expiresAt or 0); local lease_until=tonumber(entry.operationLeaseUntil or 0)
if state == 'promoting' and lease_until > now then return {0, 'BUSY'} end
if state == 'promoting' and lease_until <= now then
  entry.state='promotion_uncertain'; entry.lastError='CLAIM_EXPIRED_RECONCILIATION_REQUIRED'; entry.updatedAt=now
  entry.operationClaimHash=''; entry.operationLeaseUntil=0
  raw=cjson.encode(entry); redis.call('SET', KEYS[4], raw)
  return {0, 'RECONCILIATION_REQUIRED'}
end
if state == 'quarantined' and expires_at <= now then
  entry.state='expired'; entry.records={}; entry.bytes=0; entry.updatedAt=now
  entry.operationClaimHash=''; entry.operationLeaseUntil=0
  raw=cjson.encode(entry); redis.call('SET', KEYS[4], raw, 'EX', terminal_retention)
  redis.call('ZREM', KEYS[2], member); redis.call('HDEL', KEYS[3], member); redis.call('ZADD', KEYS[1], now+terminal_retention, member)
  return {0, 'EXPIRED'}
end
if state == 'withdrawing' then return {0, 'BUSY'} end
if state ~= 'quarantined' then return {0, 'NOT_PENDING'} end
entry.state='deleted'; entry.records={}; entry.bytes=0; entry.deletedAt=now; entry.updatedAt=now
entry.operationClaimHash=''; entry.operationLeaseUntil=0
raw=cjson.encode(entry); redis.call('SET', KEYS[4], raw, 'EX', terminal_retention)
redis.call('ZREM', KEYS[2], member); redis.call('HDEL', KEYS[3], member); redis.call('ZADD', KEYS[1], now+terminal_retention, member)
return {1, raw}
""".strip()

_REDIS_BEGIN_WITHDRAWAL_LUA = r"""
local now=tonumber(ARGV[1]); local member=ARGV[2]; local claim_hash=ARGV[3]; local lease_until=tonumber(ARGV[4])
local immortal=tonumber(ARGV[5]); local raw=redis.call('GET', KEYS[4]); if not raw then return {0, 'NOT_FOUND'} end
local entry=cjson.decode(raw); local state=tostring(entry.state or ''); local old_lease=tonumber(entry.operationLeaseUntil or 0)
if state == 'withdrawn' then return {1, raw} end
if state == 'withdrawing' and old_lease > now then return {0, 'WITHDRAWAL_IN_PROGRESS'} end
if state == 'withdrawing' and old_lease <= now then state='withdrawal_uncertain' end
if state ~= 'eligible' and state ~= 'promotion_uncertain' and state ~= 'withdrawal_uncertain' then return {0, 'NOT_ELIGIBLE'} end
entry.operationPriorState=state
entry.state='withdrawing'; entry.operationClaimHash=claim_hash; entry.operationLeaseUntil=lease_until
entry.lastError=''; entry.updatedAt=now
raw=cjson.encode(entry); redis.call('SET', KEYS[4], raw); redis.call('PERSIST', KEYS[4]); redis.call('ZADD', KEYS[1], immortal, member)
return {1, raw}
""".strip()

_REDIS_WITHDRAWAL_FAILED_LUA = r"""
local now=tonumber(ARGV[1]); local claim_hash=ARGV[2]; local code=ARGV[3]
local raw=redis.call('GET', KEYS[1]); if not raw then return {1, ''} end
local entry=cjson.decode(raw)
if tostring(entry.state or '') ~= 'withdrawing' or tostring(entry.operationClaimHash or '') ~= claim_hash then return {1, raw} end
local prior=tostring(entry.operationPriorState or 'eligible')
if prior == 'promotion_uncertain' or prior == 'withdrawal_uncertain' then entry.state=prior else entry.state='eligible' end
entry.operationClaimHash=''; entry.operationLeaseUntil=0; entry.operationPriorState=''
entry.lastError=string.sub(code,1,64); entry.updatedAt=now
raw=cjson.encode(entry); redis.call('SET', KEYS[1], raw); redis.call('PERSIST', KEYS[1]); return {1, raw}
""".strip()

_REDIS_MARK_WITHDRAWN_LUA = r"""
local now=tonumber(ARGV[1]); local member=ARGV[2]; local claim_hash=ARGV[3]
local withdrawal_json=ARGV[4]; local removal_json=ARGV[5]; local terminal_retention=tonumber(ARGV[6])
local raw=redis.call('GET', KEYS[2]); if not raw then return {0, 'NOT_FOUND'} end
local entry=cjson.decode(raw)
if tostring(entry.state or '') ~= 'withdrawing' then return {0, 'WITHDRAWAL_STATE'} end
if tostring(entry.operationClaimHash or '') ~= claim_hash then return {0, 'STALE_CLAIM'} end
entry.state='withdrawn'; entry.records={}; entry.bytes=0; entry.withdrawnAt=now; entry.withdrawalStorage=cjson.decode(withdrawal_json)
entry.currentViewRemoval=cjson.decode(removal_json); entry.lastError=''; entry.updatedAt=now
entry.operationClaimHash=''; entry.operationLeaseUntil=0; entry.operationPriorState=''
raw=cjson.encode(entry); redis.call('SET', KEYS[4], raw, 'EX', terminal_retention); redis.call('ZADD', KEYS[1], now+terminal_retention, member)
redis.call('ZREM', KEYS[2], member); redis.call('HDEL', KEYS[3], member)
return {1, raw}
""".strip()


class RedisContributionLedger:
    """
    Shared transactional receipt authority backed by one Redis domain.

    The Redis backend closes the *multi-replica coordination* gap: create,
    promotion claims, pending delete, withdrawal claims, and terminal transitions
    are atomic server-side operations.  It does **not** infer the operator's
    Redis persistence/backup policy; therefore ``durable`` remains false and
    ``CONTRIBUTION_REQUIRE_DURABLE`` must be satisfied separately when crash/power
    loss durability is a deployment requirement.
    """

    backend = "redis"
    durability = "shared_transactional_external"
    durable = False
    shared = True
    authoritative = True
    consistency_scope = "single_redis_consistency_domain"
    _IMMORTAL_SCORE = 253402300799.0

    def __init__(
        self,
        url: str,
        *,
        key_secret: str,
        key_prefix: str,
        max_pending_entries: int,
        max_pending_bytes: int,
        max_receipts: int,
        terminal_retention_seconds: int = 86_400,
        operation_lease_seconds: int = 120,
        socket_timeout_seconds: float = 2.0,
        client: Any | None = None,
        require_tls: bool = False,
    ) -> None:
        if not str(url or "").strip():
            raise ContributionLedgerError("REDIS_URL_REQUIRED")
        if len(str(key_secret or "").encode("utf-8")) < (
            32  # ruff: ignore[magic-value-comparison]
        ):
            raise ContributionLedgerError("REDIS_KEY_SECRET_TOO_SHORT")
        self.url = str(url).strip()
        self.require_tls = bool(require_tls)
        try:
            self._transport, self._connection_kwargs = redis_connection_kwargs(
                self.url,
                require_tls=self.require_tls,
                socket_timeout_seconds=socket_timeout_seconds,
            )
        except RedisSecurityError as exc:
            raise ContributionLedgerError(exc.code) from exc
        self._secret = str(key_secret).encode("utf-8")
        safe_prefix = "".join(
            ch
            for ch in str(key_prefix or "sphinx-ai-assistant").lower()
            if ch.isalnum() or ch in "_-:"
        )
        self.key_prefix = safe_prefix[:64] or "sphinx-ai-assistant"
        self.max_pending_entries = int(max_pending_entries)
        self.max_pending_bytes = int(max_pending_bytes)
        self.max_receipts = int(max_receipts)
        self.terminal_retention_seconds = max(60, int(terminal_retention_seconds))
        self.operation_lease_seconds = max(30, min(int(operation_lease_seconds), 900))
        self.socket_timeout_seconds = max(
            0.25, min(float(socket_timeout_seconds), 10.0)
        )
        self._client = client
        self._owns_client = client is None
        self._init_lock = asyncio.Lock()
        tag = f"{self.key_prefix}:{{contribution}}"
        self._all_key = f"{tag}:all"
        self._pending_key = f"{tag}:pending"
        self._pending_bytes_key = f"{tag}:pending-bytes"
        self._receipt_prefix = f"{tag}:receipt:"

    def manifest(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "durability": self.durability,
            "durable": self.durable,
            "shared": self.shared,
            "authoritative": self.authoritative,
            "consistency_scope": self.consistency_scope,
            "receipt_id_externalized": "hmac_sha256",
            "operation_claims": "leased_sha256",
            **self._transport.manifest(),
        }

    async def initialize(self) -> None:
        async with self._init_lock:
            if self._client is None:
                try:
                    import redis.asyncio as redis_async  # type: ignore[import-not-found]  # ruff: ignore[import-outside-top-level]
                except Exception as exc:  # pragma: no cover - deployment dependency
                    raise ContributionLedgerError(
                        "REDIS_DEPENDENCY_UNAVAILABLE"
                    ) from exc
                self._client = redis_async.from_url(self.url, **self._connection_kwargs)
            try:
                await self._client.ping()
            except Exception as exc:
                raise ContributionLedgerError("REDIS_UNAVAILABLE") from exc

    async def close(self) -> None:
        if self._client is None or not self._owns_client:
            return
        closer = getattr(self._client, "aclose", None) or getattr(
            self._client, "close", None
        )
        if closer is not None:
            result = closer()
            if hasattr(result, "__await__"):
                await result
        self._client = None

    def _member(self, receipt_id: str) -> str:
        return hmac.new(
            self._secret, str(receipt_id).encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def _receipt_key(self, member: str) -> str:
        return f"{self._receipt_prefix}{member}"

    @staticmethod
    def _claim_hash(claim: str) -> str:
        return hashlib.sha256(str(claim).encode("utf-8")).hexdigest()

    @staticmethod
    def _encode(entry: dict[str, Any]) -> str:
        private = {
            k: v for k, v in entry.items() if k not in {"receiptId", "operationClaim"}
        }
        private.setdefault("operationClaimHash", "")
        private.setdefault("operationLeaseUntil", 0)
        return json.dumps(private, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _decode(raw: Any, receipt_id: str) -> dict[str, Any] | None:
        if raw in {None, b"", ""}:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        entry = json.loads(str(raw))
        entry.pop("operationClaimHash", None)
        entry.pop("operationLeaseUntil", None)
        entry.pop("operationPriorState", None)
        entry["receiptId"] = receipt_id
        return entry

    @staticmethod
    def _result_parts(result: Any) -> tuple[int, Any]:
        if not isinstance(result, (list, tuple)) or len(result) < (
            2  # ruff: ignore[magic-value-comparison]
        ):
            raise ContributionLedgerError("REDIS_PROTOCOL_ERROR")
        ok = int(result[0])
        value = result[1]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return ok, value

    async def _eval(
        self, script: str, keys: list[str], args: list[Any]
    ) -> tuple[int, Any]:
        if self._client is None:
            raise ContributionLedgerError("REDIS_NOT_INITIALIZED")
        try:
            result = await self._client.eval(script, len(keys), *keys, *args)
        except ContributionLedgerError:
            raise
        except Exception as exc:
            raise ContributionLedgerError("REDIS_OPERATION_FAILED") from exc
        return self._result_parts(result)

    def _keys(self, receipt_id: str) -> tuple[str, str]:
        member = self._member(receipt_id)
        return member, self._receipt_key(member)

    async def create(self, entry: dict[str, Any]) -> None:
        now = _now()
        member, receipt_key = self._keys(entry["receiptId"])
        expires_at = float(entry["expiresAt"])
        live_until = expires_at + self.terminal_retention_seconds
        ttl = max(1, int(live_until - now + 0.999))
        ok, value = await self._eval(
            _REDIS_CREATE_LUA,
            [self._all_key, self._pending_key, self._pending_bytes_key, receipt_key],
            [
                now,
                member,
                self._encode(entry),
                expires_at,
                live_until,
                self.max_receipts,
                self.max_pending_entries,
                self.max_pending_bytes,
                int(entry.get("bytes") or 0),
                ttl,
            ],
        )
        if not ok:
            raise ContributionLedgerError(str(value))

    async def get(self, receipt_id: str) -> dict[str, Any] | None:
        member, receipt_key = self._keys(receipt_id)
        ok, value = await self._eval(
            _REDIS_GET_LUA,
            [self._all_key, self._pending_key, self._pending_bytes_key, receipt_key],
            [_now(), member, self.terminal_retention_seconds, self._IMMORTAL_SCORE],
        )
        if not ok:
            raise ContributionLedgerError(str(value))
        return self._decode(value, receipt_id)

    async def replace_pending_payload(
        self,
        receipt_id: str,
        *,
        records: list[dict[str, Any]],
        byte_count: int,
        dedup_keys: list[str],
        payload_digest: str,
        row_count: int,
        storage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        member, receipt_key = self._keys(receipt_id)
        ok, value = await self._eval(
            _REDIS_REPLACE_PENDING_LUA,
            [self._all_key, self._pending_key, self._pending_bytes_key, receipt_key],
            [
                _now(),
                member,
                json.dumps(records, ensure_ascii=False, separators=(",", ":")),
                int(byte_count),
                json.dumps(dedup_keys, separators=(",", ":")),
                str(payload_digest),
                int(row_count),
                self.max_pending_bytes,
                self.terminal_retention_seconds,
                (
                    json.dumps(storage, ensure_ascii=False, separators=(",", ":"))
                    if storage is not None
                    else ""
                ),
            ],
        )
        if not ok:
            raise ContributionLedgerError(str(value))
        return self._decode(value, receipt_id) or {}

    async def set_pending_storage(
        self, receipt_id: str, *, storage: dict[str, Any]
    ) -> dict[str, Any]:
        member, receipt_key = self._keys(receipt_id)
        ok, value = await self._eval(
            _REDIS_SET_PENDING_STORAGE_LUA,
            [self._all_key, self._pending_key, self._pending_bytes_key, receipt_key],
            [
                _now(),
                member,
                json.dumps(storage, ensure_ascii=False, separators=(",", ":")),
                self.terminal_retention_seconds,
            ],
        )
        if not ok:
            raise ContributionLedgerError(str(value))
        return self._decode(value, receipt_id) or {}

    async def begin_promotion(self, receipt_id: str) -> dict[str, Any]:
        member, receipt_key = self._keys(receipt_id)
        now = _now()
        claim = secrets.token_urlsafe(24)
        claim_hash = self._claim_hash(claim)
        ok, value = await self._eval(
            _REDIS_BEGIN_PROMOTION_LUA,
            [self._all_key, self._pending_key, self._pending_bytes_key, receipt_key],
            [
                now,
                member,
                claim_hash,
                now + self.operation_lease_seconds,
                self.terminal_retention_seconds,
            ],
        )
        if not ok:
            raise ContributionLedgerError(str(value))
        entry = self._decode(value, receipt_id) or {}
        entry["operationClaim"] = claim
        return entry

    async def promotion_failed(
        self, receipt_id: str, code: str, *, claim_token: str | None = None
    ) -> None:
        member, receipt_key = self._keys(receipt_id)
        claim_hash = self._claim_hash(claim_token or "")
        await self._eval(
            _REDIS_PROMOTION_FAILED_LUA,
            [self._all_key, self._pending_key, self._pending_bytes_key, receipt_key],
            [
                _now(),
                member,
                claim_hash,
                str(code or "PROMOTION_FAILED")[:64],
                self.terminal_retention_seconds,
            ],
        )

    async def mark_promotion_uncertain(
        self, receipt_id: str, code: str, *, claim_token: str | None = None
    ) -> dict[str, Any]:
        member, receipt_key = self._keys(receipt_id)
        ok, value = await self._eval(
            _REDIS_MARK_PROMOTION_UNCERTAIN_LUA,
            [self._all_key, self._pending_key, self._pending_bytes_key, receipt_key],
            [
                _now(),
                member,
                self._claim_hash(claim_token or ""),
                str(code or "PROMOTION_OUTCOME_UNCERTAIN")[:64],
                self._IMMORTAL_SCORE,
            ],
        )
        if not ok:
            raise ContributionLedgerError(str(value))
        return self._decode(value, receipt_id) or {}

    async def mark_promoted(
        self,
        receipt_id: str,
        *,
        storage: dict[str, Any],
        claim_token: str | None = None,
    ) -> dict[str, Any]:
        member, receipt_key = self._keys(receipt_id)
        ok, value = await self._eval(
            _REDIS_MARK_PROMOTED_LUA,
            [self._all_key, self._pending_key, self._pending_bytes_key, receipt_key],
            [
                _now(),
                member,
                self._claim_hash(claim_token or ""),
                json.dumps(storage, separators=(",", ":")),
                self._IMMORTAL_SCORE,
            ],
        )
        if not ok:
            raise ContributionLedgerError(str(value))
        return self._decode(value, receipt_id) or {}

    async def delete_pending(self, receipt_id: str) -> dict[str, Any]:
        member, receipt_key = self._keys(receipt_id)
        ok, value = await self._eval(
            _REDIS_DELETE_PENDING_LUA,
            [self._all_key, self._pending_key, self._pending_bytes_key, receipt_key],
            [_now(), member, self.terminal_retention_seconds],
        )
        if not ok:
            raise ContributionLedgerError(str(value))
        return self._decode(value, receipt_id) or {}

    async def begin_withdrawal(self, receipt_id: str) -> dict[str, Any]:
        member, receipt_key = self._keys(receipt_id)
        now = _now()
        claim = secrets.token_urlsafe(24)
        claim_hash = self._claim_hash(claim)
        ok, value = await self._eval(
            _REDIS_BEGIN_WITHDRAWAL_LUA,
            [self._all_key, self._pending_key, self._pending_bytes_key, receipt_key],
            [
                now,
                member,
                claim_hash,
                now + self.operation_lease_seconds,
                self._IMMORTAL_SCORE,
            ],
        )
        if not ok:
            raise ContributionLedgerError(str(value))
        entry = self._decode(value, receipt_id) or {}
        if entry.get("state") != "withdrawn":
            entry["operationClaim"] = claim
        return entry

    async def withdrawal_failed(
        self, receipt_id: str, code: str, *, claim_token: str | None = None
    ) -> None:
        _member, receipt_key = self._keys(receipt_id)
        await self._eval(
            _REDIS_WITHDRAWAL_FAILED_LUA,
            [receipt_key],
            [
                _now(),
                self._claim_hash(claim_token or ""),
                str(code or "WITHDRAWAL_FAILED")[:64],
            ],
        )

    async def mark_withdrawn(
        self,
        receipt_id: str,
        *,
        withdrawal_storage: dict[str, Any],
        current_view_removal: dict[str, str],
        claim_token: str | None = None,
    ) -> dict[str, Any]:
        member, receipt_key = self._keys(receipt_id)
        ok, value = await self._eval(
            _REDIS_MARK_WITHDRAWN_LUA,
            [self._all_key, self._pending_key, self._pending_bytes_key, receipt_key],
            [
                _now(),
                member,
                self._claim_hash(claim_token or ""),
                json.dumps(withdrawal_storage, separators=(",", ":")),
                json.dumps(current_view_removal, separators=(",", ":")),
                self.terminal_retention_seconds,
            ],
        )
        if not ok:
            raise ContributionLedgerError(str(value))
        return self._decode(value, receipt_id) or {}


def build_contribution_ledger(
    backend: str,
    *,
    sqlite_path: str,
    redis_url: str = "",
    redis_key_secret: str = "",
    redis_key_prefix: str = "sphinx-ai-assistant",
    redis_timeout_seconds: float = 2.0,
    operation_lease_seconds: int = 120,
    max_pending_entries: int,
    max_pending_bytes: int,
    max_receipts: int,
    terminal_retention_seconds: int = 86_400,
    require_redis_tls: bool = False,
):
    """Construct the configured receipt ledger without reading any credentials."""
    mode = str(backend or "memory").strip().lower()
    if mode == "redis":
        return RedisContributionLedger(
            redis_url,
            key_secret=redis_key_secret,
            key_prefix=redis_key_prefix,
            max_pending_entries=max_pending_entries,
            max_pending_bytes=max_pending_bytes,
            max_receipts=max_receipts,
            terminal_retention_seconds=terminal_retention_seconds,
            operation_lease_seconds=operation_lease_seconds,
            socket_timeout_seconds=redis_timeout_seconds,
            require_tls=require_redis_tls,
        )
    if mode == "sqlite":
        return SQLiteContributionLedger(
            sqlite_path,
            max_pending_entries=max_pending_entries,
            max_pending_bytes=max_pending_bytes,
            max_receipts=max_receipts,
            terminal_retention_seconds=terminal_retention_seconds,
        )
    if mode != "memory":
        raise ContributionLedgerError("UNSUPPORTED_BACKEND")
    return MemoryContributionLedger(
        max_pending_entries=max_pending_entries,
        max_pending_bytes=max_pending_bytes,
        max_receipts=max_receipts,
        terminal_retention_seconds=terminal_retention_seconds,
    )


__all__ = [
    "_REDIS_BEGIN_PROMOTION_LUA",
    "_REDIS_CREATE_LUA",
    "_REDIS_MARK_PROMOTED_LUA",
    "_REDIS_MARK_PROMOTION_UNCERTAIN_LUA",
    "ContributionLedgerError",
    "MemoryContributionLedger",
    "RedisContributionLedger",
    "SQLiteContributionLedger",
    "build_contribution_ledger",
]
