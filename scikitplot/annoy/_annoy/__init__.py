# scikitplot/annoy/_annoy/__init__.py
#
# ruff: noqa: F401,F405
# flake8: noqa: F403
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# python tools/cython/cython_generate_v3.py  --src-dir scikitplot/annoy/_annoy/
"""
Random.
"""

from __future__ import annotations

from . import annoylib
from .annoylib import *  # noqa: F403

# spotify/annoy Backward compatibility helper
AnnoyIndex = Index

__all__ = ["AnnoyIndex"]
__all__ += annoylib.__all__
