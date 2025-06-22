# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error,import-self

"""
Configuration module for the package.

This module consolidates configuration-related components.
"""

from . import (
    __config__,  # type: ignore[]
    _citation,  # type: ignore[]
    _config,
)
from .__config__ import *  # noqa: F401,F403  # type: ignore[]
from ._citation import *  # noqa: F401,F403  # type: ignore[]
from ._config import *  # noqa: F401,F403

__all__ = sorted(
    (
        *__config__.__all__,
        *_citation.__all__,
        *_config.__all__,
    )
)

# ['__bibtex__',
#  '__citation__',
#  'config_context',
#  'get_config',
#  'set_config',
#  'show_config']
