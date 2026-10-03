# scikitplot/_externals/_sphinx_ext/_sphinx_feedback/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Privacy-minimal page feedback for Sphinx and static HTML.

The dependency-free helpers in this package can be imported without Sphinx.
Sphinx-specific integration is imported lazily by :func:`setup`.
"""

from __future__ import annotations

from ._contracts import (
    FeedbackConflictError,
    FeedbackValidationError,
    build_feedback_event,
    canonical_feedback_bytes,
    decode_feedback_request,
    feedback_event_request_hash,
    feedback_request_from_event,
    feedback_request_hash,
    parse_feedback_request,
)

__version__ = "0.6.1"

__all__ = [
    "FeedbackConflictError",
    "FeedbackValidationError",
    "build_feedback_event",
    "canonical_feedback_bytes",
    "decode_feedback_request",
    "feedback_event_request_hash",
    "feedback_request_from_event",
    "feedback_request_hash",
    "parse_feedback_request",
    "setup",
]


def setup(app):
    """Register the Sphinx adapter lazily so helpers remain dependency-free."""
    from ._sphinx import setup_extension  # noqa: PLC0415 -- Sphinx is optional

    return setup_extension(app)
