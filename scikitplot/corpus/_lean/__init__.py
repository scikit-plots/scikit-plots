# scikitplot/corpus/_lean/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.corpus._lean
=======================
Evidence and discovery over a Lean project, read from source text.

- :mod:`._evidence`: :class:`~scikitplot.corpus.LeanIndex` (the index,
  ``summary``, ``dossier``) and the command line.
- :mod:`._discovery`: ``discover`` and ``export`` (rules, patterns, claims,
  anomalies, helper documents), loaded on first use.

Reading a single file stays in :mod:`scikitplot.corpus._readers._lean`
(:class:`~scikitplot.corpus.LeanReader`), with the other readers.

Run ``python -m scikitplot.corpus._lean FOLDER --target NAME``.
"""  # noqa: D205, D400

from __future__ import annotations

from ._evidence import *  # noqa: F403
from ._evidence import __all__ as _evidence_all

__all__ = list(_evidence_all)
