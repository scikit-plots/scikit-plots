# scikitplot/externals/_probscale/__init__.py

# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# This module was copied from the matplotlib project.
# https://github.com/matplotlib/mpl-probscale
"""Real probability scales for matplotlib.

Registering this module (via ``import``) adds a ``'prob'`` scale to
matplotlib that is available to all Axes in the process.

Notes
-----
``scale.register_scale(ProbScale)`` is called **once** at import time.
Re-importing the module is safe: matplotlib's ``_scale_mapping`` is a
plain dict and repeated registration simply overwrites the same entry.
"""

from matplotlib import scale as _mpl_scale

from .probscale import ProbScale, _minimal_norm
from .viz import *

__all__ = [
    "ProbScale",
    "_minimal_norm",
]

# Register once at import time.  Idempotent: repeated import just
# overwrites the same key in matplotlib's internal _scale_mapping dict.
_mpl_scale.register_scale(ProbScale)
del _mpl_scale

# ---------------------------------------------------------------------------
# Version metadata
# ---------------------------------------------------------------------------
# https://github.com/matplotlib/mpl-probscale/blob/master/probscale/__init__.py
__version__ = "0.2.6dev"
__author__ = "Paul Hobson (Herrera Environmental Consultants)"
__author_email__ = "phobson@herrerainc.com"

from ..._build_utils.gitversion import git_remote_version

# __git_hash__ = git_remote_version(url="https://github.com/scikit-plots/mpl-probscale")[0]
__git_hash__ = "be697c65ecaa223032ad2f7364ef350d684f73c0"
del git_remote_version
