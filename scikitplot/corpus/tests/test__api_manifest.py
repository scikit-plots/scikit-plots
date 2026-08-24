# corpus/tests/test__api_manifest.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Public API manifest and canonical-identity gate (CORPUS-API-001)
================================================================

The facade re-exports ~20 subpackages via wildcard imports. Because
``from X import *`` binds the *last* import of a shared name, canonical symbols
were being shadowed by deprecated ``_types`` aliases, and the aggregated
``__all__`` contained duplicates. These are the permanent regressions for that
fix; they are the reproduction named in the finding.

Run with::

    pytest scikitplot/corpus/tests/test__api_manifest.py -v
"""

from __future__ import annotations

import scikitplot.corpus as c


def test_top_level_all_is_unique():
    """Aggregated ``__all__`` must contain no duplicate names."""
    dupes = [n for n in set(c.__all__) if c.__all__.count(n) > 1]
    assert len(c.__all__) == len(set(c.__all__)), f"duplicate exports: {sorted(dupes)}"


def test_pipeline_result_is_canonical():
    """Top-level ``PipelineResult`` is the canonical ``_pipeline`` type."""
    from scikitplot.corpus._pipeline import PipelineResult as Canonical

    assert c.PipelineResult is Canonical


def test_all_names_resolve_on_this_platform():
    """Every advertised name must resolve as a real attribute of the facade."""
    missing = [n for n in c.__all__ if not hasattr(c, n)]
    assert not missing, f"names in __all__ with no attribute: {missing}"

def test_pipeline_result_has_exactly_one_definition() -> None:
    """
    ADR-C22 / DEC-157: the LegacyPipelineResult shim and its ``_types`` alias
    are deleted, so ``PipelineResult`` is defined once, in ``_pipeline``.
    """
    import scikitplot.corpus as c
    from scikitplot.corpus import _pipeline, _types

    assert c.PipelineResult is _pipeline.PipelineResult
    assert not hasattr(_types, "PipelineResult")
    assert not hasattr(_types, "LegacyPipelineResult")
