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

from . import _logging
from ._logging import *  # noqa: F403

__all__ = []
__all__ += _logging.__all__
