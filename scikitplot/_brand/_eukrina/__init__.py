# scikitplot/_brand/_eukrina/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The Eukrina Mind AI brand, generated from code.

The mark (a ``U`` crossed by a diagonal accent), its two colour editions
(Charcoal / Bronze and Deep Navy / Blue / Forest Green), its type and its
layouts are described as data in :mod:`._spec`; :func:`render` draws one
asset as SVG and :func:`build` writes a whole set with colour tokens and a
checksummed manifest.

Layouts: ``symbol``, ``horizontal`` (mark beside the text), ``stacked`` (mark
above the text), ``wordmark``, ``app-icon`` and ``round-icon``; each for
``light`` and ``dark`` backgrounds, in ``full`` colour or ``mono``.

See Also
--------
scikitplot._brand._kit : the generic kit this brand is built on.

Notes
-----
**User notes:**

- Colours follow a contrast rule by default: a brand colour is kept where it
  reaches 3:1 (mark, title) or 4.5:1 (subtitle) on the background, and
  otherwise moved to the nearest step of its own scale that does. Pass
  ``contrast=False`` for the exact brand colours.
- Text is live (``<text>``, Inter font stack) by default; ``text="outline"``
  draws it as paths from a font file, for files that must look the same in
  every viewer.
- One-colour versions cut a small gap into the ``U`` around the diagonal, so
  the two strokes stay apart.

**Developer notes:** nothing reads an image. Layout positions come from the
mark's bounds and the type metrics through the rules in :data:`RULES`.

Examples
--------
>>> from scikitplot._brand._eukrina import render
>>> svg = render("stacked", "blue-green", "dark")
>>> "MIND AI" in svg
True
"""

from __future__ import annotations

from . import _build, _render, _spec  # noqa: F401
from ._build import *  # noqa: F403
from ._render import *  # noqa: F403
from ._spec import *  # noqa: F403

__all__ = []
__all__ += _build.__all__
__all__ += _render.__all__
__all__ += _spec.__all__
