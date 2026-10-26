# scikitplot/tests/test_session_diagnostics.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
No test or conftest changes where a crash of the session reports itself.

pytest enables :mod:`faulthandler` for the whole session on a copy of stderr.
A call of ``faulthandler.enable()`` inside a test re-targets it to that
test's captured stderr, so every later native crash of the session dies with
no traceback at all: CI run 40244733381 ended with ``Segmentation fault`` and
exit status 139 and no Python frame, because
``annoy/tests/angular_index_test.py`` had made that call earlier in the run.

No test outside the TensorFlow suites loads TensorFlow in-process either:
with triton installed, TensorFlow loaded first makes a later import of
triton crash the session (Rule 158), and whether that happens depends on
test order, so only a scan of the source proves a tree clean.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: The faulthandler functions that change where, or whether, it reports.
FORBIDDEN = frozenset(
    {
        "enable",
        "disable",
        "register",
        "unregister",
        "dump_traceback_later",
        "cancel_dump_traceback_later",
    }
)

#: pytest.ini's norecursedirs below the package: vendored, never collected.
NOT_COLLECTED = ("externals", "cexternals/_astropy")

PACKAGE = Path(__file__).resolve().parents[1]

#: Fewer test files than this means the scan looked in the wrong place.
MINIMUM_FILES = 100


def faulthandler_calls(source: str) -> list[int]:
    """
    Return the lines that call a :data:`FORBIDDEN` faulthandler function.

    Parameters
    ----------
    source : str
        Python source.

    Returns
    -------
    list of int
        Line numbers: ``faulthandler.<name>(...)`` calls, and imports of a
        forbidden name from ``faulthandler``.
    """
    lines = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "faulthandler"
            and node.func.attr in FORBIDDEN
        ):
            lines.append(node.lineno)
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "faulthandler"
            and any(alias.name in FORBIDDEN for alias in node.names)
        ):
            lines.append(node.lineno)
    return sorted(lines)


#: The suites that run in a TensorFlow session of their own (the workflows'
#: ``TENSORFLOW_SUITES``); only their tests may import TensorFlow in-process.
TENSORFLOW_SUITES = ("visualkeras",)


def tensorflow_loads(source: str) -> list[int]:
    """
    Return the lines that load TensorFlow into the test process.

    Parameters
    ----------
    source : str
        Python source.

    Returns
    -------
    list of int
        Line numbers of ``import tensorflow``, ``from tensorflow import ...``
        and ``importorskip("tensorflow")`` (also as ``pytest.importorskip``).
        Code a test runs in a child interpreter is a string, not an import,
        and is not counted.
    """
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import) and any(
            alias.name.split(".")[0] == "tensorflow" for alias in node.names
        ):
            lines.append(node.lineno)
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and (node.module or "").split(".")[0] == "tensorflow"
        ):
            lines.append(node.lineno)
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", None)
            )
            if (
                name == "importorskip"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and str(node.args[0].value).split(".")[0] == "tensorflow"
            ):
                lines.append(node.lineno)
    return sorted(lines)


def _test_sources() -> list[Path]:
    skipped = tuple(PACKAGE / part for part in NOT_COLLECTED)
    found = {
        path
        for pattern in ("test_*.py", "*_test.py", "conftest.py")
        for path in PACKAGE.rglob(pattern)
        if not any(parent in path.parents for parent in skipped)
    }
    return sorted(found)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import faulthandler\nfaulthandler.enable()\n", [2]),
        ("import faulthandler\nfaulthandler.dump_traceback_later(5)\n", [2]),
        ("from faulthandler import enable\n", [1]),
        ("import faulthandler\nfaulthandler.dump_traceback()\n", []),
        ("x.enable()\n", []),
    ],
)
def test_the_scanner_finds_what_it_must_and_nothing_else(source, expected):
    """Calibration (Rule 56): known-bad and known-good sources."""
    assert faulthandler_calls(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import tensorflow as tf\n", [1]),
        ("import tensorflow.keras\n", [1]),
        ("from tensorflow import keras\n", [1]),
        ("tf = pytest.importorskip('tensorflow')\n", [1]),
        ("from pytest import importorskip\nimportorskip('tensorflow.keras')\n", [2]),
        ("code = 'import tensorflow as tf'\n", []),
        ("find_spec('tensorflow')\n", []),
        ("torch = pytest.importorskip('torch')\n", []),
        ("from .tensorflow import x\n", []),
    ],
)
def test_the_tensorflow_scanner_finds_what_it_must_and_nothing_else(source, expected):
    """Calibration (Rule 56): known-bad and known-good sources."""
    assert tensorflow_loads(source) == expected


def test_only_tensorflow_suites_load_tensorflow_in_process():
    """
    Rule 158: TensorFlow loaded before triton crashes the session later.

    Whether it crashes depends on test order (triton loaded first is safe),
    so the runtime warning alone cannot prove a tree clean; this scan does.
    """
    suites = tuple(PACKAGE / name for name in TENSORFLOW_SUITES)
    sources = [
        path
        for path in _test_sources()
        if not any(suite in path.parents for suite in suites)
    ]
    assert len(sources) > MINIMUM_FILES, "the scan found too few test files"
    offenders = []
    for path in sources:
        text = path.read_text(encoding="utf-8")
        if "tensorflow" in text:
            offenders.extend(
                f"{path.relative_to(PACKAGE).as_posix()}:{line}"
                for line in tensorflow_loads(text)
            )
    assert offenders == [], (
        "these tests load TensorFlow into a session that may load triton "
        "later (Rule 158); run the TensorFlow part in a child interpreter or "
        f"move it to a suite in TENSORFLOW_SUITES: {offenders}"
    )


def test_no_test_or_conftest_retargets_faulthandler():
    sources = _test_sources()
    assert (
        len(sources) > MINIMUM_FILES
    ), "the scan found too few test files to mean anything"
    texts = {path: path.read_text(encoding="utf-8") for path in sources}
    offenders = [
        f"{path.relative_to(PACKAGE).as_posix()}:{line}"
        for path, text in texts.items()
        if "faulthandler" in text  # no parse for the many files without it
        for line in faulthandler_calls(text)
    ]
    assert offenders == [], (
        "these calls re-target pytest's faulthandler, so a later crash of the "
        f"session prints no traceback: {offenders}"
    )
