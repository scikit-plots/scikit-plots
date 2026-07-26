# scikitplot/cython/tests/test__stub_parity.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression guard for CYTHON-TYP-001: runtime/stub (.pyi) parity.

The packaged type stub ``__init__.pyi`` drifted behind the runtime ``__all__``
(31 public symbols were undeclared).  This test pins the contract: **every**
name in the runtime ``__all__`` must be declared in the stub.  It fails if a
future change adds a public export without a matching stub declaration, so the
stub can never silently fall out of date again.
"""
from __future__ import annotations

import ast
from pathlib import Path

import scikitplot.cython as skc

# Type aliases / helpers that legitimately live in the stub but not in __all__.
_ALLOWED_STUB_ONLY = {"PathLikeAny", "ProfileName", "PathLikeSeq"}


def _stub_path() -> Path:
    return Path(skc.__file__).with_name("__init__.pyi")


def _stub_top_level_names() -> set[str]:
    tree = ast.parse(_stub_path().read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


def test_stub_exists() -> None:
    assert _stub_path().is_file(), "packaged __init__.pyi is missing"


def test_stub_parses() -> None:
    ast.parse(_stub_path().read_text(encoding="utf-8"))


def test_every_public_symbol_declared_in_stub() -> None:
    runtime = set(skc.__all__)
    stub = _stub_top_level_names()
    missing = sorted(runtime - stub)
    assert missing == [], (
        f"{len(missing)} public symbol(s) missing from __init__.pyi: {missing}"
    )


def test_no_unexpected_stub_only_public_names() -> None:
    """Stub should not declare public (non-underscore) names absent from __all__."""
    runtime = set(skc.__all__)
    stub = _stub_top_level_names()
    extra = sorted(
        n
        for n in (stub - runtime)
        if not n.startswith("_") and n not in _ALLOWED_STUB_ONLY
    )
    assert extra == [], f"stub declares public names not in __all__: {extra}"
