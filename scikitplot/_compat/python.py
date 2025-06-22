"""python built-in compat."""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=invalid-name

# ruff: noqa: F401

# import importlib
import functools
import sys
from sys import version_info

from .. import logger

PYTHON_VERSION = f"{version_info.major}.{version_info.minor}.{version_info.micro}"

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


# 3.11

# `tomllib` and `tomli` require binary read mode (`'rb'`), while `toml` uses text mode.
# Track TOML support
TOML_READ_SUPPORT = False
TOML_WRITE_SUPPORT = False
TOML_SOURCE = None

# Try importing tomllib (Python 3.11+)
if sys.version_info >= (3, 11):
    try:
        import tomllib  # Python 3.11+ builtin, read-only

        TOML_READ_SUPPORT = True
        TOML_SOURCE = "tomllib"
    except ImportError as e:
        logger.exception("Failed to import built-in `tomllib`: %s", e)
        tomllib = None

# Fallback to `tomli` (read-only)
if not TOML_READ_SUPPORT:
    try:
        import tomli as tomllib  # External tomli, API-compatible with tomllib

        TOML_READ_SUPPORT = True
        TOML_SOURCE = "tomli"
    except ImportError:
        logger.info(
            "TOML read support requires `tomli` (for Python < 3.11) or `tomllib`."
        )
        tomllib = None

# Fallback to `toml` (read/write)
try:
    import toml  # Supports both read & write

    TOML_WRITE_SUPPORT = True
    if not TOML_READ_SUPPORT:
        TOML_READ_SUPPORT = True
        TOML_SOURCE = "toml"
except ImportError:
    logger.info(
        "TOML write support requires `toml` package. Install via `pip install toml`."
    )
    toml = None
