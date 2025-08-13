# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=invalid-name

# ruff: noqa: F401

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""python built-in compat."""

# import importlib
import functools
import sys

# from sys import version_info
from .. import logger

PYTHON_VERSION = (
    f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
)

# 3.9

if sys.version_info >= (3, 9):  # noqa: UP036
    # Use @cache if you want unlimited cache (Python 3.9+).
    # Use @lru_cache(maxsize=...) if you want size-limited cache.
    from functools import cache as _cache
    from functools import lru_cache as _lru_cache
else:
    # Fallback to `lru_cache` for Python < 3.9 (acts like cache)
    from functools import lru_cache as _lru_cache

    _cache = _lru_cache  # alias cache to lru_cache for older Python versions


def lru_cache(maxsize=128):
    """Preserve Func Signature and Docstring."""

    def decorator(func):
        cached_func = _lru_cache(maxsize=maxsize)(func)

        @functools.wraps(func)  # This preserves the docstring and signature
        def wrapper(*args, **kwargs):
            return cached_func(*args, **kwargs)

        return wrapper

    return decorator


def cache(func):
    """Preserve Func Signature and Docstring."""
    return functools.wraps(func)(_cache(func))
