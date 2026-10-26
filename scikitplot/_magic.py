# scikitplot/_magic.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Deprecated: ``%load_ext scikitplot._magic`` now loads :py:mod:`scikitplot._ipymagic`.

Notes
-----
**User notes.** Use ``%load_ext scikitplot``. The old ``%scikitplot`` of this
module (fuzzy-matched commands, YAML plot recipes, a classic-notebook toolbar
button) was replaced: ``%scikitplot`` now runs the real ``scikitplot``
commands, and a mistyped command is refused with suggestions rather than
replaced by a guess.
"""

from __future__ import annotations

import warnings
from typing import Any

__all__ = ["load_ipython_extension", "unload_ipython_extension"]


def load_ipython_extension(ipython: Any) -> None:
    """
    Load :py:mod:`scikitplot._ipymagic`, warning that this name is deprecated.

    Parameters
    ----------
    ipython : InteractiveShell
        The shell IPython passes.
    """
    warnings.warn(
        "'%load_ext scikitplot._magic' is deprecated; use '%load_ext scikitplot'",
        FutureWarning,  # shown by default (DeprecationWarning is hidden here)
        stacklevel=2,
    )
    from ._ipymagic import load_ipython_extension as _load  # noqa: PLC0415

    _load(ipython)


def unload_ipython_extension(ipython: Any) -> None:
    """
    Unload :py:mod:`scikitplot._ipymagic`.

    Parameters
    ----------
    ipython : InteractiveShell
        The shell IPython passes.
    """
    from ._ipymagic import unload_ipython_extension as _unload  # noqa: PLC0415

    _unload(ipython)
