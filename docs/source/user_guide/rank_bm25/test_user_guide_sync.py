"""
Static drift checks for the Rank-BM25 user guide.

The checks intentionally avoid importing :mod:`scikitplot`.  They derive the
public scorer inventory and selected constructor defaults from source syntax so
the guide remains checkable in a source-only checkout.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _repository_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "scikitplot" / "rank_bm25" / "_rank_bm25.py"
        ).is_file():
            return candidate
    raise AssertionError("could not locate repository root")


ROOT = _repository_root()
SOURCE = ROOT / "scikitplot" / "rank_bm25"
GUIDE = ROOT / "docs" / "source" / "user_guide" / "rank_bm25" / "index.rst"
USER_GUIDE_INDEX = ROOT / "docs" / "source" / "user_guide" / "index.rst"
AFFILIATED_TEMPLATE = ROOT / "docs" / "source" / "affiliated" / "index.rst.in"
GALLERY = ROOT / "galleries" / "examples" / "rank_bm25"


def _tree() -> ast.Module:
    path = SOURCE / "_rank_bm25.py"
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _public_scorers() -> tuple[str, ...]:
    for node in _tree().body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        return tuple(value)
    raise AssertionError("could not statically find rank_bm25.__all__")


def _constructor_defaults(class_name: str) -> dict[str, object]:
    for node in _tree().body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for child in node.body:
            if not isinstance(child, ast.FunctionDef) or child.name != "__init__":
                continue
            positional = child.args.args
            defaults = child.args.defaults
            names = [arg.arg for arg in positional[-len(defaults) :]] if defaults else []
            return {name: ast.literal_eval(default) for name, default in zip(names, defaults)}
    raise AssertionError(f"could not find {class_name}.__init__")


def _guide() -> str:
    return GUIDE.read_text(encoding="utf-8")


def test_guide_covers_every_public_scorer() -> None:
    guide = _guide()
    missing = [name for name in _public_scorers() if f"``{name}``" not in guide]
    assert not missing, f"Rank-BM25 guide is missing public scorers: {missing}"


def test_documented_scoring_defaults_match_source() -> None:
    guide = _guide()
    expected = {
        "BM25Okapi": {"k1": 1.5, "b": 0.75, "epsilon": 0.25},
        "BM25L": {"k1": 1.5, "b": 0.75, "delta": 0.5},
        "BM25Plus": {"k1": 1.5, "b": 0.75, "delta": 1},
    }
    for class_name, parameters in expected.items():
        actual = _constructor_defaults(class_name)
        for name, value in parameters.items():
            assert actual[name] == value
            assert f"``{name}={value}``" in guide


def test_guide_keeps_result_binding_and_artifact_privacy_explicit() -> None:
    guide = " ".join(_guide().split())
    required = (
        "prefer ``get_top_ids``",
        "exactly the same row order",
        "does **not** make the artifact non-sensitive",
        "per-document term frequencies",
        "do not build application code on its private import path",
    )
    for phrase in required:
        assert phrase in guide


def test_guide_does_not_teach_private_analyzer_import() -> None:
    guide = _guide()
    assert "from scikitplot.rank_bm25._identity import Analyzer" not in guide


def test_rank_bm25_is_reachable_from_primary_navigation() -> None:
    user_index = USER_GUIDE_INDEX.read_text(encoding="utf-8")
    affiliated = AFFILIATED_TEMPLATE.read_text(encoding="utf-8")
    assert "Rank-BM25 <./rank_bm25/index.rst>" in user_index
    assert "../user_guide/rank_bm25/index" in affiliated
    assert "no dedicated user-guide chapter" not in affiliated



def test_rank_bm25_gallery_covers_the_primary_workflows() -> None:
    readme = (GALLERY / "README.txt").read_text(encoding="utf-8")
    expected = {
        "plot_rank_bm25_quickstart_script.py": ("doc_ids", "get_top_ids"),
        "plot_rank_bm25_recipes_script.py": ("BM25Okapi", "BM25L", "BM25Plus"),
        "plot_rank_bm25_sparse_dense_script.py": (
            "candidate_count",
            "get_top_ids",
            "get_scores",
        ),
        "plot_rank_bm25_persistence_script.py": ("save", "load", "TemporaryDirectory"),
    }
    for filename, required in expected.items():
        path = GALLERY / filename
        assert path.is_file(), f"missing Rank-BM25 gallery example: {filename}"
        source = path.read_text(encoding="utf-8")
        assert f"sphx_glr_auto_examples_rank_bm25_{filename}" in readme
        for name in required:
            assert name in source, f"{filename} does not exercise {name}"


def test_rank_bm25_gallery_uses_only_the_public_import_boundary() -> None:
    for path in GALLERY.glob("plot_*.py"):
        source = path.read_text(encoding="utf-8")
        assert "from scikitplot.rank_bm25 import" in source
        assert "scikitplot.rank_bm25._" not in source


def test_rank_bm25_guide_links_to_the_gallery() -> None:
    assert ":ref:`rank_bm25_examples`" in _guide()
