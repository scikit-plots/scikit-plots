# scikitplot/logging/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
:py:mod:`~.logging` (alias, :py:obj:`~.logger`) module provide unified both Python :py:mod:`logging` and :py:class:`logging.Logger` utilities.

.. dropdown:: View aliases

    **Main aliases**

    `scikitplot.logging`

    **Compat aliases**

    `scikitplot.logger`

Inspired by `"Tensorflow's logging system"
<https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/tf_logging.py#L94>`_ [1]_.

This module provides advanced logging utilities for Python applications,
including support for singleton-based logging with customizable formatters,
handlers, and thread-safety.

It extends Python's standard logging library to enhance usability
and flexibility for large-scale projects.

Scikit-plots logging helpers, supports vendoring.

Module Dependencies:
- Python standard library: :py:mod:`logging`

.. seealso::
  * https://github.com/python/cpython/blob/main/Lib/logging/__init__.py

References
----------
.. [1] `Tensorflow contributors. (2025).
   "Tensorflow's logging system"
   Tensorflow. https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/tf_logging.py#L94
   <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/tf_logging.py#L94>`_
"""  # noqa: D205, D400

# scikitplot.logging
# │
# ├── __init__.py
# │   └── public facade / compatibility contract
# │
# └── _logging.py
#     └── logger implementation / handlers / formatters / env policy

from __future__ import annotations

from typing import Iterable

from . import _logging
from ._logging import *  # noqa: F403

__all__ = []
__all__ += _logging.__all__

######################################################################
## logging falling back to python logging
######################################################################


def __getattr__(name: str) -> any:
    """
    Dynamic attribute resolver for this module.

    If an attribute is not found in this module, resolve it strictly from
    the standard library `logging` module.

    Parameters
    ----------
    name : str
        The attribute name being accessed.

    Returns
    -------
    Any
        The corresponding attribute from the stdlib `logging` module.

    Raises
    ------
    AttributeError
        If the attribute does not exist in stdlib `logging`.

    Notes
    -----
    - Never proxies dunder names to avoid breaking introspection tools.
    - Never returns None for missing attributes.
    - No side effects (no logging/configuration) during attribute resolution.

    Examples
    --------
    This function makes it possible to do things like:

    >>> from scikitplot.logging import DEBUG, warning
    >>> warning("This will behave like logging.warning")

    >>> hasattr(logging, "INFO")
    True  # Delegated to logging.INFO

    >>> logging.NonexistentAttribute
    AttributeError: Module 'logging' has no attribute 'NonexistentAttribute'...
    """
    # Never proxy dunder names; tooling probes these (e.g., __mro__).
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)

    # Strict lookup: no default, so missing names raise AttributeError.
    try:
        attr = getattr(_logging, name)
    except AttributeError:
        # strict logger fallback (only if you want it)
        lg = _logging.get_logger()  # must be import-safe
        attr = getattr(lg, name)  # raises AttributeError if missing

    # Cache on this module for speed + stable introspection.
    globals()[name] = attr
    return attr


def __dir__() -> Iterable[str]:
    """
    Improve `dir(scikitplot.logging)` results.

    Returns
    -------
    Iterable[str]
        Combined names from this module and stdlib logging.
    """
    return sorted(set(globals().keys()) | set(dir(_logging)))
