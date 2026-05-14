# scikitplot/corpus/_compat.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.corpus._compat
=========================
Python version compatibility shims for the corpus package.

Single source of truth for backports that would otherwise be duplicated
across ``_schema.py``, ``_base.py``, and other modules.

Supports Python 3.8 through 3.15.

Notes
-----
**MEDIUM-07 fix:** ``_StrEnumBase`` was previously defined independently
in both ``_schema.py`` and would have been needed in ``_base.py``.
This module centralises the shim so every consumer does::

    from ._compat import StrEnum as _StrEnumBase

and the backport is maintained in exactly one place.
"""  # noqa: D205, D400

from __future__ import annotations

import sys
from enum import Enum

__all__: list[str] = ["StrEnum"]

if sys.version_info >= (3, 11):  # pragma: >=3.11
    from enum import StrEnum  # type: ignore[attr-defined]
else:  # pragma: <3.11

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        """
        Backport of ``enum.StrEnum`` for Python < 3.11.

        Ensures ``str(member) == member.value`` and that members are
        valid ``str`` instances — the same contract as the stdlib version.
        """
