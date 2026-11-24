"""
First-party code never extracts an archive without a filter.

Notes
-----
**User notes.** Extracting a tar archive without ``filter=`` follows
absolute paths, ``..`` and links out of the target directory (the class of
CVE-2007-4559). Every first-party ``extractall`` call passes ``filter=`` or
carries ``# noqa: S202`` on its line, which marks members validated just
before (an explicit decision a reviewer can find).

**Developer notes.** Checked on the AST of every first-party source file,
vendored copies (``externals``, ``cexternals``) and tests left out, so the
result does not depend on the installed Python or on what is imported.
``tarfile_extractall`` in ``utils._fixes`` fell back to an unfiltered call on
Pythons without extraction filters and had no caller; it was removed.
"""

from __future__ import annotations

import ast
import pathlib

PACKAGE = pathlib.Path(__file__).resolve().parents[1]
SKIP = ("externals", "cexternals")


def _sources():
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE)
        if rel.parts[0] in SKIP or "tests" in rel.parts:
            continue
        yield rel, path.read_text(encoding="utf-8")


def _unfiltered(text):
    lines = text.splitlines()
    for node in ast.walk(ast.parse(text)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "extractall"
            and not any(k.arg == "filter" for k in node.keywords)
            and "noqa: S202" not in lines[node.lineno - 1]
        ):
            yield node.lineno


def test_every_extractall_is_filtered_or_acknowledged():
    found = [f"{rel}:{line}" for rel, text in _sources() for line in _unfiltered(text)]
    assert not found, f"extractall without filter= or '# noqa: S202': {found}"


def test_the_check_sees_an_unfiltered_call():
    bad = "def f(tar, p):\n    tar.extractall(p)\n"
    ok = (
        "def f(tar, p, m):\n"
        "    tar.extractall(p, filter='data')\n"
        "    tar.extractall(p, members=m)  # noqa: S202 - validated\n"
    )
    assert list(_unfiltered(bad)) == [2]
    assert list(_unfiltered(ok)) == []


def test_the_unfiltered_fallback_is_gone():
    text = (PACKAGE / "utils" / "_fixes.py").read_text(encoding="utf-8")
    assert "def tarfile_extractall" not in text
