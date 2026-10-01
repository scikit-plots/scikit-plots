# scikitplot/_externals/_sphinx_ext/_sphinx_ai_learn/__init__.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""JSON-first AI Learn content materialization for Sphinx.

Canonical Learn content is stored as validated JSON.  During ``config-inited``
the extension deterministically materializes sibling RST sources, then normal
Sphinx builders render those sources to HTML or other output formats.

Build-time materialization is deliberately local and deterministic: it performs
no model calls, network access, Git operations, telemetry, or publication.
"""

from __future__ import annotations

from ._materialize import load_content_tree, materialize
from ._schema import LearnValidationError, validate_contribution

__version__ = "0.47.0"
__all__ = [
    "LearnValidationError",
    "load_content_tree",
    "materialize",
    "setup",
    "validate_contribution",
]


def setup(app):
    """Register the extension lazily so importing data helpers needs no Sphinx."""
    from ._sphinx import setup_extension  # noqa: PLC0415 -- keep Sphinx optional

    return setup_extension(app)
