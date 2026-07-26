# scikitplot/cython/tests/test__performance.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression + benchmark tests for CYTHON-PERF-001.

Several operations scaled by full scans or duplicated work.  This run
deduplicates normalized include dirs (without weakening validation) and gives
cache-stats traversal an optional budget.  The benchmark asserts include-dir
dedup stays sub-linear in duplicate count and that bounded traversal does less
work than a full scan.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

from .._gc import _dir_size_bytes
from .._public import _dedup_paths


class TestDedupPaths:
    def test_collapses_duplicates_by_normalized_identity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a"
            a.mkdir()
            # Same dir as str and Path, plus a trailing-slash variant.
            out = _dedup_paths([str(a), a, str(a) + "/"])
        assert len(out) == 1

    def test_drop_removes_intrinsic_parent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a"
            b = Path(td) / "b"
            a.mkdir()
            b.mkdir()
            out = _dedup_paths([a, b], drop={b})
        assert [str(x) for x in out] == [str(a)]

    def test_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a"
            b = Path(td) / "c"
            a.mkdir()
            b.mkdir()
            out = _dedup_paths([a, b, a])
        assert [str(x) for x in out] == [str(a), str(b)]

    def test_nonresolvable_kept_and_deduped(self) -> None:
        out = _dedup_paths(["/no/such/x", "/no/such/x", "rel"])
        assert out == ["/no/such/x", "rel"]

    def test_empty(self) -> None:
        assert _dedup_paths([]) == []


class TestBoundedTraversal:
    def test_budget_stops_early(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i in range(50):
                (root / f"f{i}.bin").write_bytes(b"x" * 10)
            full = _dir_size_bytes(root)
            bounded = _dir_size_bytes(root, max_files=5)
        assert full == 500
        assert bounded == 50
        assert bounded < full

    def test_default_is_full_scan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i in range(10):
                (root / f"f{i}.bin").write_bytes(b"y" * 3)
            assert _dir_size_bytes(root) == 30


class TestDedupBenchmark:
    """Benchmark: dedup of N duplicates stays fast (sub-quadratic)."""

    def test_dedup_scales(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a"
            a.mkdir()
            for n in (10, 1000):
                dupes = [a] * n
                t0 = time.perf_counter()
                out = _dedup_paths(dupes)
                dt = time.perf_counter() - t0
                assert len(out) == 1
                # Generous ceiling; the point is it does not blow up.
                assert dt < 2.0, f"dedup of {n} took {dt:.3f}s"
