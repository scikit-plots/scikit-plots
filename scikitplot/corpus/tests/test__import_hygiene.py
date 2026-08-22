# scikitplot/corpus/tests/test__import_hygiene.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Import-cost gates for :mod:`scikitplot.corpus`.

These tests lock the contract measured in review run R15 (disproof D-23):
importing :mod:`scikitplot.corpus` must not pull in any *optional* backend.

Notes
-----
**User-focused.**  These gates are why ``corpus.capabilities()`` and CLI
``--help``/``--version`` stay fast: asking what Corpus *can* do never loads the
machinery that does it.

**Developer-focused.**  The property is currently maintained by roughly 288
deferred imports across 41 modules, each marked ``# noqa: PLC0415``.  Nothing
enforces it globally, and a *single* module-scope ``import torch`` anywhere in
the package would silently undo all of them -- while every existing test kept
passing.  That is precisely the kind of property that decays unnoticed, so it is
made a gate here (proposal P-I0-12).

Each check runs in a fresh subprocess: once a heavyweight is in
:data:`sys.modules` from an unrelated test, an in-process check cannot tell who
imported it.

Compatibility
-------------
Written against the full supported range, Python 3.8 through 3.15+.  Uses only
:mod:`subprocess`, :mod:`sys` and :mod:`textwrap`.

See Also
--------
scikitplot.corpus._capabilities : the lightweight capability probe these gates protect.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

__all__: "list[str]" = [
    "test_importing_corpus_loads_no_optional_backend",
    "test_capability_snapshot_loads_no_optional_backend",
]

#: Optional heavyweights that must never be imported as a side effect of
#: importing Corpus.  Each is either a large native dependency, a model runtime,
#: or a parser that Corpus reaches for only when a specific reader runs.
OPTIONAL_HEAVYWEIGHTS = (
    "torch",
    "transformers",
    "sentence_transformers",
    "tensorflow",
    "nltk",
    "gensim",
    "spacy",
    "annoy",
    "faiss",
    "voyager",
    "lxml",
    "bs4",
    "requests",
    "PIL",
)


def _loaded_after(statement):
    """Import in a fresh interpreter and report which heavyweights loaded.

    Parameters
    ----------
    statement : str
        Python source executed in the child process.

    Returns
    -------
    list of str
        Names from :data:`OPTIONAL_HEAVYWEIGHTS` present in ``sys.modules``.

    Raises
    ------
    AssertionError
        If the child process fails, with its stderr attached.
    """
    source = textwrap.dedent(
        """
        import sys
        {statement}
        watched = {watched!r}
        print(",".join(sorted(m for m in watched if m in sys.modules)))
        """
    ).format(statement=statement, watched=OPTIONAL_HEAVYWEIGHTS)

    proc = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stderr}"
    out = proc.stdout.strip().splitlines()
    last = out[-1] if out else ""
    return [name for name in last.split(",") if name]


@pytest.mark.parametrize(
    "statement",
    [
        "import scikitplot.corpus",
        "from scikitplot.corpus import _capabilities",
    ],
    ids=["package", "capabilities_module"],
)
def test_no_optional_backend_is_imported(statement):
    """Importing Corpus must not load any optional heavyweight."""
    loaded = _loaded_after(statement)
    assert loaded == [], (
        "P-I0-12 regression: importing Corpus loaded optional "
        "dependencies {0}. Corpus keeps these out of the import path via "
        "deferred (call-time) imports so that capability queries and CLI "
        "--help stay cheap; a module-scope import of one of these undoes "
        "that for every consumer.".format(loaded)
    )


def test_importing_corpus_loads_no_optional_backend():
    """Alias gate: ``import scikitplot.corpus`` pulls in no optional backend."""
    test_no_optional_backend_is_imported("import scikitplot.corpus")


def test_capability_snapshot_loads_no_optional_backend():
    """Probing capabilities must not load the backends it reports on.

    This is the sharper half of the contract: ``capability_snapshot()`` reports
    whether ``annoy``/``faiss``/``voyager`` are available, and it must answer
    that *without* importing them.
    """
    loaded = _loaded_after(
        "from scikitplot.corpus._capabilities import capability_snapshot\n"
        "capability_snapshot()"
    )
    assert loaded == [], (
        "capability_snapshot() imported {0}; availability must be probed "
        "without loading the backend itself.".format(loaded)
    )
