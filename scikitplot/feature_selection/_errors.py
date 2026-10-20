# scikitplot/feature_selection/_errors.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Error types of :mod:`scikitplot.feature_selection`.

Notes
-----
**User note.** Each error says what was found, what was expected and what to
change. Catch :class:`FeatureSelectionError` to handle every refusal of this
package at once.

**Developer note.** Every class also derives from the built-in type a caller
would expect (``ValueError`` for bad input, ``RuntimeError`` for an
inconsistent fitted object), so existing ``except ValueError`` handlers keep
working. :class:`IncompatibleSchemaError` is separate from a malformed record
so a reader never mistakes data of another generation for corruption
(Rule 108).
"""

from __future__ import annotations

__all__ = [
    "FeatureSelectionError",
    "IncompatibleSchemaError",
    "LineageError",
    "PolicyError",
    "SchemaMismatchError",
    "SchemaMismatchWarning",
]


class FeatureSelectionError(Exception):
    """Base class of every error raised by :mod:`scikitplot.feature_selection`."""


class PolicyError(FeatureSelectionError, ValueError):
    """A policy or configuration document is off contract (unknown key, wrong type, bad range)."""


class SchemaMismatchError(FeatureSelectionError, ValueError):
    """Data given after fitting does not match the fitted schema (missing, extra or retyped columns)."""


class IncompatibleSchemaError(FeatureSelectionError, ValueError):
    """A persisted artefact was written by another, unsupported schema generation."""


class LineageError(FeatureSelectionError, RuntimeError):
    """A pipeline's feature topology cannot be traced or does not reconcile with its output."""


class SchemaMismatchWarning(UserWarning):
    """Later data differs from the fitted schema and ``on_mismatch="warn"`` let it through."""
