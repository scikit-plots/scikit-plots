"""
Every deliberately slow regular expression in test data is wrapped.

Notes
-----
**User notes.** The pattern-risk check is tested on patterns that must
backtrack catastrophically. Written as plain literals, they reach
:func:`re.compile` and CodeQL's ``py/redos`` query reports each one as a
high-severity alert on the pull request (alert 227 on PR 864, round 27).
They are written ``regex_fixture(r"(a+)+")`` instead
(``scikitplot/cleanprompt/tests/_regex_fixtures.py`` explains why that keeps
the test data out of the scan without changing what the test receives).

**Developer notes.** The check is the subsystem's own analyser: every string
constant in the test and probe files that compiles as a regular expression
and gets a finding (other than ``not-analysed``) must be the argument of a
``regex_fixture(...)`` call. Docstrings are prose and are skipped. Runtime
modules are out of scope: their regular expressions are checked by
``test__pattern_risk.TestBuiltinsAreClean``, and their prose suggestions
merely *contain* pattern text. The gallery is out of scope because its
example pattern is pack data written to a file, never compiled in place, and
a reader should see it as written.
"""

from __future__ import annotations

import ast
import re
import sys
import types
import warnings
from pathlib import Path

import pytest

CHECKOUT = Path(__file__).resolve().parents[4]
PACKAGE = CHECKOUT / "scikitplot" / "cleanprompt"
SCOPES = (
    PACKAGE / "tests",
    CHECKOUT / "maintenances" / "cleanprompt" / "_maintenance" / "evidence",
)


def _analyse():
    """Return ``analyse_pattern`` under a stand-in parent, as other plane tests do."""
    if not PACKAGE.is_dir():
        pytest.skip(f"package not present in this checkout: {PACKAGE}")
    if "scikitplot" not in sys.modules:
        stand_in = types.ModuleType("scikitplot")
        stand_in.__path__ = [str(CHECKOUT / "scikitplot")]
        sys.modules["scikitplot"] = stand_in
    from scikitplot.cleanprompt._pattern_risk import (  # noqa: PLC0415 - after the stand-in
        analyse_pattern,
    )

    return analyse_pattern


def _docstrings(tree):
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                found.add(id(first.value))
    return found


def _risky(analyse, text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            re.compile(text)
        except (re.error, OverflowError, RecursionError):
            return False
    return any(risk.rule != "not-analysed" for risk in analyse(text))


def _unwrapped():
    analyse = _analyse()
    problems = []
    for scope in SCOPES:
        for path in sorted(scope.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            prose = _docstrings(tree)
            wrapped = {
                id(node.args[0])
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "regex_fixture"
                and node.args
            }
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in prose
                    and id(node) not in wrapped
                    and _risky(analyse, node.value)
                ):
                    rel = path.relative_to(CHECKOUT)
                    problems.append(f"{rel}:{node.lineno}: {node.value!r}")
    return problems


def test_every_slow_fixture_is_wrapped():
    problems = _unwrapped()
    assert not problems, (
        "write these as regex_fixture(...) so static ReDoS scanners do not "
        "report test data (see scikitplot/cleanprompt/tests/_regex_fixtures.py):\n"
        + "\n".join(problems)
    )


def test_the_wrapper_returns_the_text_unchanged():
    _analyse()
    from scikitplot.cleanprompt.tests._regex_fixtures import (  # noqa: PLC0415
        regex_fixture,
    )

    for text in ("(a+)+", r"(?x) (?: \w+ \s? )+", "\u0430+"):
        assert regex_fixture(text) == text
        assert type(regex_fixture(text)) is str
