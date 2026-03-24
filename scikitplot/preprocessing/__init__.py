# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Methods for scaling, centering, normalization, binarization, and more."""

from __future__ import annotations

from . import _encoders
from ._encoders import *  # noqa: F403

__all__ = []
__all__ += _encoders.__all__
