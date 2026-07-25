# scikitplot/cython/tests/test__batch.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-BATCH-001.

``cython_import_all`` was fail-fast and returned nothing on a mid-batch failure,
leaving earlier compiled/imported items committed with no report.  The batch
path now returns a structured ``BatchBuildResult`` (ordered successes/failures,
committed side effects, policy) and, under the fail-fast policy, raises
``BatchBuildError`` carrying the partial result and a resume token.  These tests
mock ``cython_import_result`` so they exercise the batch orchestration without a
compiler.
"""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from .. import _public
from .._public import cython_import_all, cython_import_all_result
from .._result import BatchBuildError, BatchBuildResult, BatchFailure


def _make_pyx(dir_: Path, *stems: str) -> None:
    for s in stems:
        (dir_ / f"{s}.pyx").write_text("x", encoding="utf-8")


def _fake_import(fail: set[str]):
    def _inner(f: Path, **kwargs):
        if f.stem in fail:
            raise ValueError(f"boom-{f.stem}")
        return f"built-{f.stem}"

    return _inner


class TestCollectPolicy:
    def test_all_succeed(self, tmp_path: Path) -> None:
        _make_pyx(tmp_path, "a", "b", "c")
        with mock.patch.object(_public, "cython_import_result", _fake_import(set())):
            res = cython_import_all_result(tmp_path, collect=True)
        assert isinstance(res, BatchBuildResult)
        assert res.ok is True
        assert list(res.successes) == ["a", "b", "c"]
        assert res.n_failed == 0
        assert res.policy == "collect"

    def test_collect_records_failures_and_continues(self, tmp_path: Path) -> None:
        _make_pyx(tmp_path, "a", "b", "c")
        with mock.patch.object(_public, "cython_import_result", _fake_import({"b"})):
            res = cython_import_all_result(tmp_path, collect=True)
        assert res.ok is False
        assert list(res.successes) == ["a", "c"]  # continued past b
        assert [f.name for f in res.failures] == ["b"]
        assert isinstance(res.failures[0], BatchFailure)
        assert res.failures[0].error_type == "ValueError"
        assert res.committed == ["a", "c"]


class TestFailFastPolicy:
    def test_fail_fast_raises_with_partial(self, tmp_path: Path) -> None:
        _make_pyx(tmp_path, "a", "b", "c")
        with mock.patch.object(_public, "cython_import_result", _fake_import({"b"})):
            with pytest.raises(BatchBuildError) as ei:
                cython_import_all_result(tmp_path, collect=False)
        result = ei.value.result
        assert result.committed == ["a"]  # only a committed before b failed
        assert result.n_failed == 1
        assert result.ok is False
        # Resume token = the not-yet-attempted stems.
        assert ei.value.resume_token == ("c",)

    def test_wrapper_is_fail_fast(self, tmp_path: Path) -> None:
        _make_pyx(tmp_path, "a", "b")
        with mock.patch.object(_public, "cython_import_result", _fake_import({"b"})):
            with pytest.raises(BatchBuildError):
                cython_import_all(tmp_path)

    def test_wrapper_returns_dict_on_success(self, tmp_path: Path) -> None:
        _make_pyx(tmp_path, "a", "b")
        with mock.patch.object(_public, "cython_import_result", _fake_import(set())):
            out = cython_import_all(tmp_path)
        assert set(out) == {"a", "b"}


class TestResume:
    def test_only_restricts_batch(self, tmp_path: Path) -> None:
        _make_pyx(tmp_path, "a", "b", "c")
        with mock.patch.object(_public, "cython_import_result", _fake_import(set())):
            res = cython_import_all_result(tmp_path, collect=True, only=["c"])
        assert list(res.successes) == ["c"]

    def test_resume_after_failfast(self, tmp_path: Path) -> None:
        _make_pyx(tmp_path, "a", "b", "c")
        # First pass fails at b; then fix b and resume with the token.
        with mock.patch.object(_public, "cython_import_result", _fake_import({"b"})):
            with pytest.raises(BatchBuildError) as ei:
                cython_import_all_result(tmp_path, collect=False)
        token = ei.value.resume_token
        with mock.patch.object(_public, "cython_import_result", _fake_import(set())):
            res = cython_import_all_result(tmp_path, collect=True, only=list(token))
        assert list(res.successes) == ["c"]


class TestMissingDir:
    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            cython_import_all_result(tmp_path / "nope", collect=True)
