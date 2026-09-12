# scikitplot/cython/tests/test__maintainer_docs.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Guard tests for the in-module maintainer knowledge base.

``MAINTAINING.md`` is the durable home for the change workflow, prevention
rules, and the review log.  These tests keep it honest: it must exist, be
well-formed, reference only documents that actually ship, and never silently
drop a reviewed finding.  This is the same "docs must be verified against the
code" discipline used for the stub parity guard (CYTHON-TYP-001) and the
operational guide (CYTHON-DOC-001).
"""

from __future__ import annotations

import re
from pathlib import Path

import scikitplot.cython as skc

# The complete set of findings closed by the R1-R30 review.  Adding a new
# reviewed finding means adding it here and to MAINTAINING.md — the test makes
# that coupling explicit rather than letting the log quietly fall behind.
_ALL_FINDINGS = frozenset(
    {
        "CYTHON-CON-001",
        "CYTHON-CACHE-001",
        "CYTHON-GC-001",
        "CYTHON-LOAD-002",
        "CYTHON-SEC-001",
        "CYTHON-API-001",
        "CYTHON-CACHE-002",
        "CYTHON-PIN-001",
        "CYTHON-WASM-001",
        "CYTHON-LOAD-001",
        "CYTHON-CACHE-003",
        "CYTHON-TPL-001",
        "CYTHON-PKG-001",
        "CYTHON-CON-002",
        "CYTHON-RES-001",
        "CYTHON-CACHE-004",
        "CYTHON-COMP-001",
        "CYTHON-TYP-001",
        "CYTHON-SEC-002",
        "CYTHON-SCH-001",
        "CYTHON-API-002",
        "CYTHON-API-003",
        "CYTHON-BATCH-001",
        "CYTHON-DOC-001",
        "CYTHON-PORT-001",
        "CYTHON-TPL-002",
        "CYTHON-OBS-001",
        "CYTHON-ABI-001",
        "CYTHON-TEST-001",
        "CYTHON-PERF-001",
    }
)


def _pkg_dir() -> Path:
    return Path(skc.__file__).resolve().parent


def _doc() -> Path:
    return _pkg_dir() / "MAINTAINING.md"


class TestPresence:
    def test_file_exists(self) -> None:
        assert _doc().is_file(), "MAINTAINING.md is missing"

    def test_nontrivial(self) -> None:
        assert len(_doc().read_text(encoding="utf-8").strip()) > 1000


class TestWellFormed:
    def test_code_fences_balanced(self) -> None:
        assert _doc().read_text(encoding="utf-8").count("```") % 2 == 0

    def test_no_hard_tabs(self) -> None:
        assert "\t" not in _doc().read_text(encoding="utf-8")

    def test_required_sections(self) -> None:
        text = _doc().read_text(encoding="utf-8").lower()
        for section in (
            "document map",
            "change workflow",
            "prevention rules",
            "review log",
            "adding an adr",
        ):
            assert section in text, f"MAINTAINING.md missing section: {section!r}"


class TestCompleteReviewLog:
    def test_every_finding_logged(self) -> None:
        text = _doc().read_text(encoding="utf-8")
        found = set(re.findall(r"CYTHON-[A-Z]+-\d+", text))
        missing = _ALL_FINDINGS - found
        assert not missing, f"MAINTAINING.md review log is missing: {sorted(missing)}"

    def test_no_unknown_finding_codes(self) -> None:
        text = _doc().read_text(encoding="utf-8")
        found = set(re.findall(r"CYTHON-[A-Z]+-\d+", text))
        unknown = found - _ALL_FINDINGS
        assert not unknown, (
            f"MAINTAINING.md references unknown findings: {sorted(unknown)}"
        )


class TestNoDanglingReferences:
    def test_referenced_docs_exist(self) -> None:
        text = _doc().read_text(encoding="utf-8")
        # Any backtick-quoted token ending in .md that looks like a sibling doc.
        referenced = set(re.findall(r"`([A-Za-z0-9_./-]+\.md)`", text))
        pkg = _pkg_dir()
        missing = [
            name for name in referenced if "/" not in name and not (pkg / name).exists()
        ]
        assert not missing, f"MAINTAINING.md links to missing docs: {sorted(missing)}"
