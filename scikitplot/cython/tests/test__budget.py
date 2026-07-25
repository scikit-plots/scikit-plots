# scikitplot/cython/tests/test__budget.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-RES-001 (in-process budget portion).

Build execution had no deadline or output cap.  ``_budget`` adds a build
deadline (``run_with_deadline`` / ``build_timeout_s``) and a bounded output
buffer.  These tests pin the budget primitives and that ``build_timeout_s`` is
enforced by the real build path.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from .._budget import (
    BoundedBuffer,
    BuildBudget,
    BuildTimeoutError,
    run_with_deadline,
)


class TestRunWithDeadline:
    def test_timeout_raises(self) -> None:
        with pytest.raises(BuildTimeoutError, match="deadline"):
            run_with_deadline(lambda: time.sleep(2.0), timeout_s=0.05, what="x")

    def test_result_returned(self) -> None:
        assert run_with_deadline(lambda: 7, timeout_s=5.0) == 7

    def test_none_timeout_runs_inline(self) -> None:
        assert run_with_deadline(lambda: 9, timeout_s=None) == 9

    def test_exception_propagates(self) -> None:
        def boom() -> None:
            raise ValueError("kaboom")

        with pytest.raises(ValueError, match="kaboom"):
            run_with_deadline(boom, timeout_s=5.0)


class TestBoundedBuffer:
    def test_keeps_tail_within_budget(self) -> None:
        b = BoundedBuffer(max_bytes=20)
        for i in range(100):
            b.write(f"line{i}\n")
        v = b.getvalue()
        assert b.truncated is True
        # Retained content (excluding the marker) is within budget.
        assert "line99" in v  # newest retained
        assert "line0\n" not in v  # oldest dropped

    def test_small_output_not_truncated(self) -> None:
        b = BoundedBuffer(max_bytes=1000)
        b.write("hello\n")
        assert b.truncated is False
        assert b.getvalue() == "hello\n"

    def test_rejects_bad_size(self) -> None:
        with pytest.raises(ValueError):
            BoundedBuffer(max_bytes=0)


class TestBuildBudget:
    def test_validates(self) -> None:
        with pytest.raises(ValueError):
            BuildBudget(compile_timeout_s=-1.0)
        with pytest.raises(ValueError):
            BuildBudget(max_output_bytes=0)

    def test_defaults(self) -> None:
        b = BuildBudget()
        assert b.compile_timeout_s is None
        assert b.max_output_bytes == 1024 * 1024


class TestBuildTimeoutWired:
    _KW = dict(
        code="def add(int a, int b):\n    return a + b\n",
        source_path=None,
        use_cache=True,
        force_rebuild=False,
        verbose=-1,
        annotate=False,
        view_annotate=False,
        numpy_support=False,
        numpy_required=False,
        include_dirs=None,
        library_dirs=None,
        libraries=None,
        define_macros=None,
        extra_compile_args=None,
        extra_link_args=None,
        compiler_directives=None,
    )

    def test_tiny_timeout_raises_from_real_build(self) -> None:
        from .._builder import build_extension_module_result

        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(BuildTimeoutError):
                build_extension_module_result(
                    module_name="res001_timeout",
                    cache_dir=Path(td),
                    build_timeout_s=0.001,
                    **self._KW,
                )
