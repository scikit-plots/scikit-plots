# scikitplot/annoy/_mixins/tests/test__pickle.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests of ``scikitplot/annoy/_mixins/_pickle.py``: its module-level names.

The behaviour of :class:`~scikitplot.annoy._mixins._pickle.PickleMixin` is
covered in ``test_mixins.py``. These tests cover what importing the module
does, which is where it failed on Python 3.14.
"""

from __future__ import annotations

import ast
import typing
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "_pickle.py"
ALIASES = ("CompressMode", "PickleMode")


def _module_level_doc_assignments(tree: ast.Module) -> list[str]:
    """Return ``name`` for every module-level ``name.__doc__ = ...``."""
    found = []
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        for target in targets:
            if isinstance(target, ast.Attribute) and target.attr == "__doc__":
                found.append(ast.unparse(target.value))
    return found


class TestTypeAliasesAreNotMutated:
    """A type alias is a shared object; the module must not write to it."""

    def test_no_module_level_doc_assignment(self):
        # ``(X | None).__doc__ = ...`` raises AttributeError on Python 3.14,
        # where ``X | None`` is a ``typing.Union`` with a read-only ``__doc__``.
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
        assert _module_level_doc_assignments(tree) == []

    @pytest.mark.parametrize("name", ALIASES)
    def test_each_alias_is_documented_by_an_autodoc_comment(self, name):
        lines = SOURCE.read_text(encoding="utf-8").splitlines()
        (index,) = [i for i, line in enumerate(lines) if line.startswith(f"{name}:")]
        assert lines[index - 1].startswith("#: "), lines[index - 1]

    @pytest.mark.parametrize("name", ALIASES)
    def test_importing_the_module_writes_no_doc_onto_an_alias(self, name):
        # What must hold is that the module stored no ``__doc__`` on the alias
        # object: an alias keeps the documentation of its type. This is asked
        # of the object the module binds. It is *not* compared by identity
        # with a freshly written ``Literal[...]``: whether two equal literals
        # are one object depends on a cache inside ``typing`` that a test run
        # may clear (it did in CI), so identity is not a contract.
        module = pytest.importorskip("scikitplot.annoy._mixins._pickle")
        alias = getattr(module, name)
        assert "__doc__" not in getattr(alias, "__dict__", {})
        assert alias.__doc__ == type(alias).__doc__

    def test_the_literal_alias_equals_the_literal_it_names(self):
        module = pytest.importorskip("scikitplot.annoy._mixins._pickle")
        assert module.PickleMode == typing.Literal["auto", "disk", "byte"]

    @pytest.mark.parametrize("name", ALIASES)
    def test_the_aliases_are_exported(self, name):
        module = pytest.importorskip("scikitplot.annoy._mixins._pickle")
        assert name in module.__all__
        assert hasattr(module, name)
