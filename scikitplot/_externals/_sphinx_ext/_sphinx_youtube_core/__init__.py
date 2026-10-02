# scikitplot/_externals/_sphinx_ext/_sphinx_youtube_core/__init__.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Dependency-free YouTube provider primitives shared by Sphinx adapters.

This package owns provider grammar and leaf-player option validation only.
It intentionally owns no browser/search controls: collection search UI belongs
to ``_sphinx_collection`` and is consumed by gallery adapters. Keeping this
package UI-free lets catalog tooling, the gallery adapter, and the standalone
player share one provider contract without creating an extension dependency
cycle. Result-count placement is likewise outside this provider layer: the
shared collection/gallery renderer owns the V4 controls -> status -> cards
structure and its bounded 12-card display/load-more controller; this provider
layer must never create, paginate, hide, or reposition collection cards itself.
"""

from __future__ import annotations

__all__: list[str] = []
