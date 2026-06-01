# scikitplot/cexternals/_editdistance/__init__.py
#
# flake8: noqa: D213
#
# Authors: Hiroyuki Tanaka
# SPDX-License-Identifier: MIT
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
C-External libraries editdistance.
"""

from __future__ import annotations

from .bycython import eval, eval_criterion


def distance(*args, **kwargs):
    """"An alias to eval"""
    return eval(*args, **kwargs)

def distance_le_than(*args, **kwargs):
    """"An alias to eval"""
    return eval_criterion(*args, **kwargs)

__all__ = (
    'eval',
    'distance',
    "eval_criterion",
    "distance_le_than",
)
