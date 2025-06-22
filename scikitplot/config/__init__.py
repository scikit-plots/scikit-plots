# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# Flake8: noqa: F401,F403,F405
# type: ignore[]
# pylint: disable=import-error,import-self,undefined-all-variable

"""
Configuration module for the package.

This module consolidates configuration-related components.
"""

from .._testing._pytesttester import PytestTester  # Pytest testing
from .__config__ import *
from ._citation import *
from ._config import *

test = PytestTester(__name__)
del PytestTester

# __all__ = [s for s in dir() if not s.startswith("_")]  # Remove dunders.
__all__ = [
    # "_citation",
    "__bibtex__",
    "__citation__",
    # "_config",
    "config_context",
    "get_config",
    "set_config",
    # "__config__",
    "show_config",
    "test",
]
