"""python built-in compat."""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=unused-import

# ruff: noqa: F401

import functools
import importlib
import sys
from sys import version_info

PYTHON_VERSION = f"{version_info.major}.{version_info.minor}.{version_info.micro}"

if sys.version_info >= (3, 9):  # noqa: UP036
    # Use @cache if you want unlimited cache (Python 3.9+).
    # Use @lru_cache(maxsize=...) if you want size-limited cache.
    from functools import cache, lru_cache
else:
    # Fallback to `lru_cache` for Python < 3.9 (acts like cache)
    from functools import lru_cache

    cache = lru_cache  # alias cache to lru_cache for older Python versions
