# scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_utils/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Private implementation helpers for the Hugging Face proxy service.

The public/deployment entrypoints intentionally remain at the parent level:
``app.py`` and ``deduplicate_dataset.py``.  Keep this package import-light: do
not eagerly import helper modules here, because several helpers have optional
runtime dependencies and deployment-specific initialization.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
