# scikitplot/lean/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Evidence on a Lean project, read from its source text.

:class:`LeanIndex` indexes a Lean 4 (or Lean 3) project's declarations, the
names each one uses and the audit flags its code raises, and answers:

- :meth:`LeanIndex.summary`: the project (sizes, flags, duplicate names,
  imports no file provides, syntax extensions, generated declarations);
- :meth:`LeanIndex.dossier`: the evidence on one theorem (its statement and
  the definitions it unfolds to, what its proof names and the path to each
  flag, what it could use without naming, checks with a verdict);
- :meth:`LeanIndex.discover` and :meth:`LeanIndex.export`: candidate rules,
  patterns, claims and anomalies, written as documents for people and
  models.

:func:`defined_commands` returns the command keywords a file defines, to
read files one at a time as an index reads a folder.

Every item names its file and lines; nothing here is a proof. Lean's kernel
decides proofs (``lake build``, ``#print axioms``, a replay).

From the shell: ``python -m scikitplot.lean FOLDER [--target NAME]
[--fail-on sorry]``.

Modules
-------
- ``_syntax``: the Lean language core (lexer, command splitter,
  :data:`LEAN_FLAGS`), shared with :class:`scikitplot.corpus.LeanReader`.
- ``_index``: :class:`LeanIndex` and the command line.
- ``_discovery``: discovery and the helper documents, loaded on first use.

Notes
-----
**Developer.** Standard library only; nothing is imported from
:mod:`scikitplot.corpus` (the corpus reader depends on this package, never
the reverse).
"""  # noqa: D205, D400

from __future__ import annotations

from ._index import AUDIT_FLAGS, LeanIndex, defined_commands
from ._syntax import LEAN_FLAGS

__all__ = ["AUDIT_FLAGS", "LEAN_FLAGS", "LeanIndex", "defined_commands"]
