from __future__ import annotations

import ast
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parents[3]
CORE = ROOT / "scikitplot" / "levenshtein" / "_core.py"
GALLERY = ROOT / "galleries" / "examples" / "levenshtein"


def _core_public_names() -> set[str]:
    tree = ast.parse(CORE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return {ast.literal_eval(item) for item in node.value.elts}
    raise AssertionError("_core.__all__ not found")


def _all_docs_text() -> str:
    return "\n".join(
        p.read_text(encoding="utf-8") for p in sorted(DOCS.glob("*.rst"))
    )


def test_every_public_name_is_documented() -> None:
    text = _all_docs_text()
    missing = sorted(name for name in _core_public_names() if name not in text)
    assert not missing, missing


def test_backend_names_are_documented() -> None:
    text = _all_docs_text()
    for name in ("internal", "rapidfuzz", "levenshtein", "python", "auto"):
        assert name in text


def test_license_safe_auto_order_is_documented() -> None:
    text = (DOCS / "backends_and_fallbacks.rst").read_text(encoding="utf-8")
    assert "GPL-2.0-or-later" in text
    assert "explicit only" in text.lower()


def test_score_cutoff_performance_limit_is_documented() -> None:
    text = " ".join(_all_docs_text().split())
    assert "post-computation" in text
    assert "not a performance" in text


def test_open_maintenance_findings_are_explained() -> None:
    text = _all_docs_text()
    assert "LV-001" in text
    assert "LV-002" in text


def test_gallery_entry_point_exists() -> None:
    assert (GALLERY / "README.txt").is_file()


def test_index_links_all_guide_pages() -> None:
    index = (DOCS / "index.rst").read_text(encoding="utf-8")
    for stem in (
        "getting_started",
        "how_it_works",
        "backends_and_fallbacks",
        "ranking_and_matching",
        "corpus_integration",
        "python_api",
        "performance_and_limits",
        "troubleshooting",
    ):
        assert stem in index
