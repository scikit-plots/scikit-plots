# scikitplot/corpus/_storage/tests/test__resource_hygiene.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Resource hygiene (Run 3, R3-8).

The corpus suite on Python 3.13 left 80 SQLite connections open at exit:
``SQLiteStorage`` had ``close()`` but no context manager, and tests did not
close what they opened. Every store is now a context manager, and the
package's autouse guard fails the test that leaves a connection open.
"""

from __future__ import annotations

import sqlite3

import pytest

from ... import conftest
from .._storage import InMemoryStorage, JSONLStorage, SQLiteStorage


@pytest.fixture(params=["memory", "jsonl", "sqlite"])
def make_store(request, tmp_path):
    def make():
        if request.param == "memory":
            return InMemoryStorage()
        if request.param == "jsonl":
            return JSONLStorage(tmp_path / "c.jsonl")
        return SQLiteStorage(tmp_path / "c.db")

    return make


class TestContextManager:
    def test_with_returns_the_store_and_closes_it(self, make_store):
        store = make_store()
        with store as entered:
            assert entered is store
            assert entered.count() == 0
        store.close()  # idempotent

    def test_an_error_inside_still_closes(self, tmp_path):
        store = SQLiteStorage(tmp_path / "c.db")
        with pytest.raises(RuntimeError, match="boom"), store:
            raise RuntimeError("boom")
        with pytest.raises(sqlite3.ProgrammingError):
            store.count()

    def test_close_twice(self):
        store = SQLiteStorage(":memory:")
        store.close()
        store.close()


class TestLeakGuard:
    def test_a_connection_left_open_is_found_and_closed(self, monkeypatch):
        guard = conftest.SQLiteLeakGuard(monkeypatch)
        leaked_store = SQLiteStorage(":memory:")
        with SQLiteStorage(":memory:"):
            pass
        leaks = guard.close_leaks()
        assert len(leaks) == 1
        assert any("test__resource_hygiene.py" in site for site in leaks[0])
        with pytest.raises(sqlite3.ProgrammingError):
            leaked_store.count()
        assert guard.close_leaks() == []


class TestReviewFindings:
    """Review of 3e: leaks the first version left."""

    def test_a_refused_runtime_closes_the_store_it_opened(self):
        """Review 1: the owned store of a failed materialize stayed open."""
        from ... import FluentCorpus  # noqa: PLC0415

        for build in (
            lambda fc: fc.storage("sqlite").export(123),
            lambda fc: fc.storage("sqlite").retrieval(123),
        ):
            with pytest.raises((TypeError, ValueError)):
                build(FluentCorpus()).materialize()
        # The autouse guard fails this test if a connection is still open.

    def test_a_file_that_is_not_a_database_closes_its_connection(self, tmp_path):
        """Review 2: the PRAGMAs ran outside the try that closes."""
        junk = tmp_path / "junk.db"
        junk.write_bytes(b"this is not a sqlite database" * 64)
        with pytest.raises(sqlite3.DatabaseError):
            SQLiteStorage(junk)

    def test_the_guard_reports_a_connection_of_another_thread(self, monkeypatch):
        """Review 4: closing it raised, and stopped the loop."""
        import threading  # noqa: PLC0415

        guard = conftest.SQLiteLeakGuard(monkeypatch)
        opened, release = threading.Event(), threading.Event()

        def worker():
            conn = sqlite3.connect(":memory:")  # check_same_thread=True
            opened.set()
            release.wait(10)
            conn.close()  # only its own thread can

        thread = threading.Thread(target=worker)
        thread.start()
        opened.wait(10)
        leftover = sqlite3.connect(":memory:")
        try:
            assert len(guard.close_leaks()) == 2  # both reported, no raise
            with pytest.raises(sqlite3.ProgrammingError):
                leftover.total_changes  # noqa: B018 (closed by the guard)
        finally:
            release.set()
            thread.join(10)
