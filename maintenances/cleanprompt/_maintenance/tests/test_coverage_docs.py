"""The user guide's coverage table equals what the registries produce (CP-116)."""

from __future__ import annotations

import runpy
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent / "tools"
generator = runpy.run_path(str(TOOLS / "generate_coverage_docs.py"))


def test_the_committed_table_is_current():
    root = generator["_repository_root"]()
    from scikitplot.cleanprompt._diagnostics import (  # ruff: ignore[import-outside-top-level]
        coverage_matrix,
    )

    expected = generator["render"](coverage_matrix())
    committed = (root / generator["TARGET"]).read_text(encoding="utf-8")
    assert committed == expected, (
        "the coverage table is stale; run "
        "maintenances/cleanprompt/_maintenance/tools/generate_coverage_docs.py"
    )


def test_how_it_works_includes_it():
    root = generator["_repository_root"]()
    page = root / "docs/source/user_guide/cleanprompt/how_it_works.rst"
    assert ".. include:: coverage_matrix.inc" in page.read_text(encoding="utf-8")
