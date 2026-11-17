# scikitplot/_brand/_kit/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
A small kit for generating brand assets from code.

A brand is described as data (geometry, palettes, text) and drawn by these
helpers into static, safe, reproducible files:

- :mod:`._color` - colours, tonal scales, WCAG contrast, contrast-safe choice;
- :mod:`._svg` - an SVG writer that refuses scripts and external references;
- :mod:`._text` - font lookup, measured advances, glyph outlines;
- :mod:`._io` - atomic, idempotent writing and checksummed manifests.

See Also
--------
scikitplot._brand._eukrina : the Eukrina Mind AI brand, built on this kit.

Notes
-----
**User notes:** nothing here reads an image; every asset is drawn from its
description, so the same inputs give the same bytes.

**Developer notes:** the kit uses the standard library only, except
:mod:`._text`, which imports Matplotlib when called. A brand package depends
on the kit, never the other way round.
"""

from __future__ import annotations

from . import _color, _io, _svg, _text  # noqa: F401
from ._color import *  # noqa: F403
from ._io import *  # noqa: F403
from ._svg import *  # noqa: F403
from ._text import *  # noqa: F403

__all__ = []
__all__ += _color.__all__
__all__ += _io.__all__
__all__ += _svg.__all__
__all__ += _text.__all__
