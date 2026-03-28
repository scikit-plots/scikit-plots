# scikitplot/_utils/env_manager.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Environment-manager identifier constants and validation.

This module defines the canonical string identifiers for the Python
environment managers supported by scikit-plots, and provides a single
validation entry-point used throughout the package.

Supported managers
------------------
LOCAL      : system-level or in-project interpreter with no isolation layer
CONDA      : conda / mamba environment
VIRTUALENV : ``virtualenv``-managed environment
UV         : ``uv venv``-managed environment

Notes
-----
**User note:** Import the constants rather than hard-coding the strings.
This ensures your code stays correct if an identifier is ever renamed.

**Developer note:** ``validate`` is the sole authority on which values are
legal.  Any new manager must be added to both the constant block *and* the
``_ALLOWED`` tuple inside ``validate``; the tuple is the single source of
truth for membership testing.
"""

from __future__ import annotations

from ..exceptions import ScikitplotException

__all__ = [
    "LOCAL",
    "CONDA",
    "VIRTUALENV",
    "UV",
    "validate",
]

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

LOCAL: str = "local"
"""Identifier for a system-level or in-project Python interpreter."""

CONDA: str = "conda"
"""Identifier for a conda / mamba-managed environment."""

VIRTUALENV: str = "virtualenv"
"""Identifier for a ``virtualenv``-managed environment."""

UV: str = "uv"
"""Identifier for a ``uv venv``-managed environment."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(env_manager: object) -> None:
    """Validate that *env_manager* is a recognised environment-manager identifier.

    Raises :class:`~scikitplot.exceptions.ScikitplotException` when the value
    is not one of the four supported string identifiers.  The check is
    deliberately strict: type, case, and surrounding whitespace all matter.

    Parameters
    ----------
    env_manager : object
        The value to validate.  Must be exactly one of :data:`LOCAL`,
        :data:`CONDA`, :data:`VIRTUALENV`, or :data:`UV`.

    Returns
    -------
    None
        Returned on success so callers may assert ``validate(x) is None``.

    Raises
    ------
    ScikitplotException
        When *env_manager* is not one of the four allowed string identifiers.
        ``error_code`` is set to ``0`` and the message names the invalid value
        together with the full list of allowed values.

    See Also
    --------
    LOCAL, CONDA, VIRTUALENV, UV : The four allowed identifier constants.

    Notes
    -----
    **User note:** All comparisons are exact (``not in`` against a tuple of
    the four constants).  Uppercase variants, whitespace-padded strings, and
    non-string types will all raise.

    **Developer note:** ``_ALLOWED`` is the single source of truth for
    membership.  Do not duplicate the list anywhere else in this module.

    Examples
    --------
    >>> from scikitplot._utils.env_manager import validate, LOCAL, CONDA
    >>> validate(LOCAL)   # returns None — no exception
    >>> validate(CONDA)   # returns None — no exception
    >>> validate("pipenv")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    scikitplot.exceptions.ScikitplotException: Invalid value for `env_manager`: ...
    """
    _ALLOWED = (LOCAL, CONDA, VIRTUALENV, UV)
    if env_manager not in _ALLOWED:
        raise ScikitplotException(
            f"Invalid value for `env_manager`: {env_manager!r}. "
            f"Must be one of {list(_ALLOWED)}",
            error_code=0,
        )
