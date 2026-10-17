# scikitplot/agent/_audit.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
An append-only, hash-chained audit trail: what a run did, and why.

Each record is one event of one run (a decision, a denial, an approval, a tool
call and its result, the run's end). Records of a run are numbered from 0, and
each carries the digest of the one before it, so a removed, reordered or
altered record is found by :func:`verify_audit`.

Notes
-----
**User.** ``JsonlAuditSink(path)`` appends one canonical JSON line per record;
several runs may share one file. ``read_audit(path)`` reads it back and
``verify_audit(records)`` checks every run's chain.

**Developer.** The chain shows tampering with records that were written; it
cannot show a trail truncated after its last record. Compare the run state's
``audit_head`` (the last record's sequence number and digest) with the trail
for that. A sink that raises stops the run: an action is audited *before* it
runs, so an action that could not be recorded does not run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ._canonical import canonical_json, digest, plain

__all__ = [
    "AuditSink",
    "JsonlAuditSink",
    "MemoryAuditSink",
    "read_audit",
    "verify_audit",
]

#: ``prev`` of a run's first record.
GENESIS = "0" * 64
_FIELDS = ("seq", "run_id", "kind", "time", "data", "prev")


class AuditSink:
    """Where audit records go. Subclasses implement :meth:`write`."""

    def write(self, record: Mapping[str, Any]) -> None:
        """Append one record; raise if it could not be kept."""
        raise NotImplementedError

    def head(self, run_id: str) -> dict[str, Any] | None:
        """
        Return ``{"seq", "digest"}`` of the last record of ``run_id``.

        ``None`` means the sink holds no record of the run, or cannot tell
        (the default: a sink that only writes). The agent uses it to refuse a
        run id already used, and at resume to compare the trail with the
        saved state.
        """
        return None

    def digest_at(self, run_id: str, seq: int) -> str | None:
        """
        Return the digest of record ``seq`` of ``run_id``, or ``None``.

        ``None`` means the sink holds no such record, or cannot tell.
        """
        return None


def _is_entry(record: Any) -> bool:
    """Whether ``record`` has the fields a head is read from."""
    return (
        isinstance(record, Mapping)
        and isinstance(record.get("run_id"), str)
        and isinstance(record.get("seq"), int)
        and not isinstance(record.get("seq"), bool)
        and isinstance(record.get("digest"), str)
    )


def _last(records: Iterable[Any], run_id: str) -> dict[str, Any] | None:
    found = None
    for record in records:
        if _is_entry(record) and record["run_id"] == run_id:
            found = {"seq": record["seq"], "digest": record["digest"]}
    return found


def _at(records: Iterable[Any], run_id: str, seq: int) -> str | None:
    found = None
    for record in records:
        if _is_entry(record) and record["run_id"] == run_id and record["seq"] == seq:
            found = record["digest"]
    return found


class MemoryAuditSink(AuditSink):
    """
    Keep records in a list, :attr:`records`.

    Examples
    --------
    >>> from scikitplot.agent import MemoryAuditSink
    >>> MemoryAuditSink().records
    []
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write(self, record: Mapping[str, Any]) -> None:
        """Append a copy of ``record``."""
        self.records.append(plain(dict(record)))

    def head(self, run_id: str) -> dict[str, Any] | None:
        """Return the last record of ``run_id``, as ``{"seq", "digest"}``."""
        return _last(self.records, run_id)

    def digest_at(self, run_id: str, seq: int) -> str | None:
        """Return the digest of record ``seq`` of ``run_id``."""
        return _at(self.records, run_id, seq)


class JsonlAuditSink(AuditSink):
    """
    Append records to a JSON Lines file, one canonical line each.

    Parameters
    ----------
    path : str or os.PathLike
        The file; created if missing, never truncated.
    fsync : bool, default=False
        Call :func:`os.fsync` after each record, so it survives a crash of
        the machine and not only of the process.
    """

    def __init__(self, path: str | os.PathLike[str], *, fsync: bool = False) -> None:
        if fsync is not True and fsync is not False:
            raise TypeError("JsonlAuditSink.fsync must be a bool.")
        self.path = Path(path)
        self.fsync = fsync
        self._heads: dict[str, dict[str, Any]] = {}
        self._scanned: tuple[int, int] | None = None  # (size, mtime_ns) read

    def write(self, record: Mapping[str, Any]) -> None:
        """Append ``record`` as one line and flush it."""
        line = canonical_json(dict(record)) + "\n"
        before = self._signature()
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
            if self.fsync:
                os.fsync(handle.fileno())
        if self._scanned is not None and before == self._scanned and _is_entry(record):
            # Nobody else wrote since the last read: the cache stays exact.
            self._heads[record["run_id"]] = {
                "seq": record["seq"],
                "digest": record["digest"],
            }
            self._scanned = self._signature()
        else:
            self._scanned = None  # read the file again at the next head()

    def _signature(self) -> tuple[int, int] | None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return stat.st_size, stat.st_mtime_ns

    def _entries(self) -> list[dict[str, Any]]:
        """Read every record that parses; a torn or foreign line is skipped.

        :func:`verify_audit` over :func:`read_audit` is where such a line is
        reported; reading a head must not stop unrelated runs.
        """
        out = []
        with open(self.path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if _is_entry(record):
                    out.append(record)
        return out

    def head(self, run_id: str) -> dict[str, Any] | None:
        """Return the last record of ``run_id`` in the file.

        The file is read once and then only again when another writer has
        changed it (its size or modification time differ).
        """
        signature = self._signature()
        if signature is None:
            return None
        if signature != self._scanned:
            heads: dict[str, dict[str, Any]] = {}
            for record in self._entries():
                heads[record["run_id"]] = {
                    "seq": record["seq"],
                    "digest": record["digest"],
                }
            self._heads, self._scanned = heads, signature
        found = self._heads.get(run_id)
        return None if found is None else dict(found)

    def digest_at(self, run_id: str, seq: int) -> str | None:
        """Return the digest of record ``seq`` of ``run_id`` (reads the file)."""
        if self._signature() is None:
            return None
        return _at(self._entries(), run_id, seq)


def record_digest(record: Mapping[str, Any]) -> str:
    """Return the digest of a record's fields other than ``digest``."""
    return digest({name: record[name] for name in _FIELDS})


class AuditTrail:
    """
    The writer for one run: numbers, chains and writes its records.

    Parameters
    ----------
    sink : AuditSink
    run_id : str
    head : mapping or None
        ``{"seq", "digest"}`` of the run's last record, to continue its
        chain; ``None`` for a new run.
    now : callable
        Returns the time recorded in each record (``time.time``).
    """

    def __init__(
        self,
        sink: AuditSink,
        run_id: str,
        head: Mapping[str, Any] | None,
        now: Callable[[], float],
    ) -> None:
        self.sink = sink
        self.run_id = run_id
        self.now = now
        self.seq = -1 if head is None else int(head["seq"])
        self.prev = GENESIS if head is None else str(head["digest"])

    @property
    def head(self) -> dict[str, Any] | None:
        """``{"seq", "digest"}`` of the last record written, or ``None``."""
        return None if self.seq < 0 else {"seq": self.seq, "digest": self.prev}

    def emit(self, kind: str, data: Mapping[str, Any]) -> dict[str, Any]:
        """Write one record and return it."""
        record: dict[str, Any] = {
            "seq": self.seq + 1,
            "run_id": self.run_id,
            "kind": kind,
            "time": float(self.now()),
            "data": plain(dict(data)),
            "prev": self.prev,
        }
        record["digest"] = record_digest(record)
        self.sink.write(record)
        self.seq += 1
        self.prev = record["digest"]
        return record


def read_audit(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """
    Read a JSON Lines audit file.

    Parameters
    ----------
    path : str or os.PathLike
        A file written by :class:`JsonlAuditSink`.

    Returns
    -------
    list of dict

    Raises
    ------
    ValueError
        If a line is not a JSON object, naming the line.
    """
    records = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: not JSON: {exc}") from None
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{number}: not a JSON object")  # noqa: TRY004
            records.append(record)
    return records


def verify_audit(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """
    Check every run's hash chain in ``records``.

    Parameters
    ----------
    records : iterable of mapping
        Records as written, in order; several runs may be interleaved.

    Returns
    -------
    list of str
        The problems found; empty when every chain is whole. A run's first
        record must be number 0 and name :data:`GENESIS` as ``prev``; each
        next record must be numbered one more and name the digest of the one
        before; every digest must match its record.

    Examples
    --------
    >>> from scikitplot.agent import verify_audit
    >>> verify_audit([])
    []
    """
    problems: list[str] = []
    last: dict[str, tuple[int, str]] = {}
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            problems.append(f"record {position}: not an object")
            continue
        missing = [k for k in (*_FIELDS, "digest") if k not in record]
        if missing:
            problems.append(f"record {position}: missing {missing}")
            continue
        run_id = record["run_id"]
        if not isinstance(run_id, str):
            problems.append(f"record {position}: run_id is not a string")
            continue
        try:
            expected_digest = record_digest(record)
        except (TypeError, ValueError) as exc:
            problems.append(f"record {position}: not plain JSON: {exc}")
            continue
        if record["digest"] != expected_digest:
            problems.append(
                f"record {position} (run {run_id}, seq {record['seq']}): "
                "digest does not match its content"
            )
        if isinstance(record["seq"], bool) or not isinstance(record["seq"], int):
            problems.append(f"record {position} (run {run_id}): seq is not an integer")
            continue
        seq, prev = last.get(run_id, (-1, GENESIS))
        if record["seq"] != seq + 1:
            problems.append(
                f"record {position} (run {run_id}): seq {record['seq']}, "
                f"expected {seq + 1}"
            )
        if record["prev"] != prev:
            problems.append(
                f"record {position} (run {run_id}, seq {record['seq']}): prev does "
                "not name the record before it"
            )
        last[run_id] = (record["seq"], record["digest"])
    return problems
