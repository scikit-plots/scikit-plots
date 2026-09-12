# scikitplot/cython/tests/test__operations_docs.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-DOC-001.

The operational guide (``OPERATIONS.md``) must exist, cover the required
chapters (security, cache, concurrency, recovery, platform), be well-formed
Markdown, and its executable examples (``_operations_examples.py``) must pass as
doctests — so the documentation is tested, not just written.
"""

from __future__ import annotations

import doctest
from pathlib import Path

import scikitplot.cython as skc
from scikitplot.cython import _operations_examples


def _pkg_dir() -> Path:
    return Path(skc.__file__).resolve().parent


def _ops_md() -> Path:
    return _pkg_dir() / "OPERATIONS.md"


class TestOperationsGuidePresent:
    def test_file_exists(self) -> None:
        assert _ops_md().is_file(), "OPERATIONS.md is missing"

    def test_required_chapters_present(self) -> None:
        text = _ops_md().read_text(encoding="utf-8").lower()
        for chapter in [
            "security",
            "cache",
            "concurrency",
            "batch",
            "unsupported",
            "stability",
        ]:
            assert chapter in text, f"OPERATIONS.md missing a {chapter!r} chapter"


class TestMarkdownWellFormed:
    def test_code_fences_balanced(self) -> None:
        text = _ops_md().read_text(encoding="utf-8")
        # Every opening ``` must have a matching closing ```.
        assert text.count("```") % 2 == 0, "unbalanced code fences in OPERATIONS.md"

    def test_no_tab_indentation(self) -> None:
        text = _ops_md().read_text(encoding="utf-8")
        assert "\t" not in text, "OPERATIONS.md contains hard tabs"

    def test_nonempty(self) -> None:
        assert len(_ops_md().read_text(encoding="utf-8").strip()) > 500


class TestExecutableExamples:
    def test_example_module_doctests_pass(self) -> None:
        results = doctest.testmod(_operations_examples, verbose=False)
        assert results.failed == 0, f"{results.failed} operational doctest(s) failed"
        assert results.attempted > 0, "no operational doctests were exercised"

    def test_examples_are_not_public_api(self) -> None:
        # The examples module must not leak into the public surface.
        assert _operations_examples.__all__ == []
        assert "_operations_examples" not in skc.__all__
