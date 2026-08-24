# scikitplot/corpus/tests/test__deprecation_import_hygiene.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Import-hygiene gates for the ``scikitplot.corpus`` deprecation shims.

These tests permanently lock the contract established by finding
``CORPUS-R00-F001``: importing :mod:`scikitplot.corpus` must never emit a
:exc:`DeprecationWarning` attributable to a deprecated symbol that the caller
did not reference.

Notes
-----
**User-focused.**  If one of these tests fails, ``import scikitplot.corpus``
has started warning on a plain import.  Downstream projects that run their test
suites under ``-W error`` or ``PYTHONWARNINGS=error`` -- a common strict
setting -- will be unable to import the package at all.

**Developer-focused.**  The regression this guards against is subtle.  A
deprecated ``Enum`` cannot emit its warning from ``__new__``: ``Enum`` invokes
``__new__`` once per member while the *class body* executes, so the warning
fires at module import rather than on member access.  Serving the name lazily
from a module-level ``__getattr__`` does not help either, because
``corpus/__init__.py`` performs ``from ._types import *`` and a star-import
invokes ``__getattr__`` for every name in ``__all__``.

The supported pattern for a deprecated symbol that is bound in the module body
is therefore a ``.. deprecated::`` docstring directive plus an entry in
``_DEPRECATED_NAMES`` -- never a runtime side effect at import.

Each check runs in a fresh subprocess.  A warning raised during the *first*
import of a module cannot be observed in-process once that module is already in
:data:`sys.modules`, which every other test in this suite has already caused.

Compatibility
-------------
Written against the full supported range, Python 3.8 through 3.15+.  Uses only
:mod:`subprocess`, :mod:`sys` and :mod:`textwrap`; no walrus operator, no
positional-only parameters, no ``X | Y`` runtime annotations, and no
``tomllib``.  ``subprocess.run(..., capture_output=True)`` requires Python 3.7+.

See Also
--------
scikitplot.corpus._types : formerly hosted the deprecated shims, now deleted.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

__all__: "list[str]" = [
    "test_import_corpus_emits_no_deprecation_warning",
    "test_import_types_emits_no_deprecation_warning",
    "test_deprecation_machinery_is_fully_removed",
    "test_canonical_replacements_are_reachable",
]


def _run_isolated(body):
    """Execute ``body`` in a fresh interpreter with warnings escalated.

    Parameters
    ----------
    body : str
        Python source executed after ``warnings.simplefilter("error")``.

    Returns
    -------
    subprocess.CompletedProcess
        The completed process, with ``stdout`` and ``stderr`` captured as text.

    Notes
    -----
    ``-I`` (isolated mode) is deliberately not used: the test run may rely on
    ``PYTHONPATH`` to locate an unbuilt source tree, and ``-I`` would discard it.
    """
    source = "import warnings\nwarnings.simplefilter('error')\n" + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "module",
    ["scikitplot.corpus", "scikitplot.corpus._types"],
    ids=["package", "types_module"],
)
def test_import_emits_no_deprecation_warning(module):
    """Importing ``module`` under ``-W error`` must succeed.

    This is the primary gate for ``CORPUS-R00-F001``.
    """
    result = _run_isolated("import {}\n".format(module))
    assert result.returncode == 0, (
        "CORPUS-R00-F001 regression: `import {}` raised under "
        "warnings-as-errors. A deprecated symbol is emitting a warning at "
        "import time.\n\n{}".format(module, result.stderr)
    )


# Retained under the exact names promised in ``__all__`` so the gate is
# discoverable by symbol as well as by parametrised id.
def test_import_corpus_emits_no_deprecation_warning():
    """Alias gate: ``import scikitplot.corpus`` stays warning-free."""
    test_import_emits_no_deprecation_warning("scikitplot.corpus")


def test_import_types_emits_no_deprecation_warning():
    """Alias gate: ``import scikitplot.corpus._types`` stays warning-free."""
    test_import_emits_no_deprecation_warning("scikitplot.corpus._types")


def test_deprecation_machinery_is_fully_removed():
    """ADR-C22 / DEC-157: all three shims are deleted, not deprecated.

    The original CORPUS-R00-F001 defect was an import-time ``DeprecationWarning``
    emitted by a shim.  With backward compatibility withdrawn the shims are gone
    outright, so the deprecation machinery that carried them must be gone too --
    otherwise a future shim could reintroduce the same import-time side effect.
    """
    from scikitplot.corpus import _types

    for gone in ("ChunkStrategy", "Document", "LegacyPipelineResult", "PipelineResult"):
        assert not hasattr(_types, gone), f"{gone} should have been deleted"

    for machinery in ("_DEPRECATED_NAMES", "_DEPRECATED_WARNED", "__getattr__"):
        assert not hasattr(_types, machinery), (
            f"_types.{machinery} survived the shim deletion; the deprecation "
            "mechanism should have been removed with its last consumer"
        )


def test_canonical_replacements_are_reachable():
    """Each deleted shim's canonical replacement remains public."""
    import scikitplot.corpus as c
    from scikitplot.corpus._schema import ChunkingStrategy, CorpusDocument

    assert c.ChunkingStrategy is ChunkingStrategy
    assert c.CorpusDocument is CorpusDocument
    assert len(list(ChunkingStrategy)) >= 1
