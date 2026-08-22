# scikitplot/corpus/tests/test__diagnostics.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`scikitplot.corpus._diagnostics`.

These lock the two properties that motivated :class:`ErrorRecord`: it must be
serialisable (finding F-R02-03) and it must not retain the originating
exception's frame graph (finding F-R11-02, measured at 139x).
"""

from __future__ import annotations

import gc
import json
import sys

import pytest

from .._diagnostics import ErrorCategory, ErrorRecord
from .._schema import ErrorPolicy

__all__: "list[str]" = [
    "TestErrorRecord",
    "TestErrorPolicy",
]


def _raise_with_big_local(size: int) -> None:
    """Raise with a large object bound in the failing frame."""
    payload = bytes(size)  # noqa: F841 - deliberately retained by the traceback
    raise ValueError("boom")


class TestErrorRecord:
    """Construction, serialisation and non-retention."""

    def test_is_json_serialisable(self) -> None:
        """F-R02-03: a BuildResult must be writable to a run manifest."""
        rec = ErrorRecord(
            code="PARSE_FAILED",
            category=ErrorCategory.PARSE,
            message="malformed XML",
            stage="read",
            source_id="doc-1",
        )
        payload = json.dumps(rec.to_dict())
        assert "PARSE_FAILED" in payload

    def test_round_trips_through_dict(self) -> None:
        rec = ErrorRecord(
            code="C",
            category=ErrorCategory.SECURITY,
            message="refused",
            details={"member": "../etc/passwd"},
        )
        assert ErrorRecord.from_dict(rec.to_dict()) == rec

    def test_from_exception_keeps_type_name_not_object(self) -> None:
        """The exception itself must not be reachable from the record."""
        try:
            raise ValueError("boom")
        except ValueError as exc:
            rec = ErrorRecord.from_exception(
                exc, code="X", category=ErrorCategory.INTERNAL
            )

        assert rec.exception_type == "ValueError"
        assert rec.message == "boom"
        assert rec.traceback_text is None
        # No field anywhere holds a BaseException.
        assert not any(
            isinstance(v, BaseException) for v in rec.to_dict().values()
        )

    def test_traceback_is_opt_in_and_is_text(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError as exc:
            rec = ErrorRecord.from_exception(
                exc,
                code="X",
                category=ErrorCategory.INTERNAL,
                include_traceback=True,
            )
        assert isinstance(rec.traceback_text, str)
        assert "ValueError" in rec.traceback_text
        json.dumps(rec.to_dict())  # still serialisable

    def test_does_not_retain_the_failing_frame(self) -> None:
        """F-R11-02: holding the exception retained 139x the needed memory.

        The record must not keep the traceback alive, so the large object bound
        in the failing frame becomes collectable once the ``except`` block ends.
        """
        records = []
        for _ in range(50):
            try:
                _raise_with_big_local(200_000)
            except ValueError as exc:
                records.append(
                    ErrorRecord.from_exception(
                        exc, code="X", category=ErrorCategory.INTERNAL
                    )
                )
        gc.collect()

        # Nothing reachable from the records holds a frame or traceback.
        for rec in records:
            for value in vars(rec).values():
                assert not isinstance(value, BaseException)
                assert type(value).__name__ not in {"traceback", "frame"}

        # A conservative size bound: 50 records must be far smaller than the
        # 10 MB of frame locals they were built from.
        total = sum(sys.getsizeof(json.dumps(r.to_dict())) for r in records)
        assert total < 100_000

    def test_str_is_compact_and_names_the_subject(self) -> None:
        rec = ErrorRecord(
            code="ARCHIVE_MEMBER_REFUSED",
            category=ErrorCategory.SECURITY,
            message="ZipSlip",
            source_id="../x",
        )
        assert str(rec) == "[security/ARCHIVE_MEMBER_REFUSED] ../x: ZipSlip"

    @pytest.mark.parametrize(
        "category",
        list(ErrorCategory),
        ids=[c.value for c in ErrorCategory],
    )
    def test_every_category_round_trips(self, category: ErrorCategory) -> None:
        rec = ErrorRecord(code="C", category=category, message="m")
        assert ErrorRecord.from_dict(rec.to_dict()).category is category


class TestErrorPolicy:
    """The extended policy set from ADR-R02-001."""

    def test_members(self) -> None:
        assert {p.value for p in ErrorPolicy} == {
            "raise",
            "skip",
            "retry",
            "collect",
            "fallback",
        }

    def test_log_member_is_gone(self) -> None:
        """F-R02-01: LOG conflated behaviour with visibility.

        It shared the SKIP dispatch branch and differed only in whether a log
        line fired, so it encoded a logging decision as a behaviour value.
        Callers wanting "skip, and tell me" use COLLECT, which yields a
        structured record rather than only unstructured text.
        """
        assert not hasattr(ErrorPolicy, "LOG")
        with pytest.raises(ValueError, match="log"):
            ErrorPolicy("log")

    def test_collect_and_fallback_exist(self) -> None:
        assert ErrorPolicy.COLLECT == "collect"
        assert ErrorPolicy.FALLBACK == "fallback"
