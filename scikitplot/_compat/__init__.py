# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Importing this module will also install monkey-patches defined in it

"""
The content of this module is solely for internal use of ``scikitplot``.

Subject to changes without deprecations. Do not use it in external
packages or code.
"""

from __future__ import annotations

from contextlib import suppress as _suppress

from .. import logger as _logger
from ..exceptions import ScikitplotException as _ScikitplotException
from ..externals._seaborn._compat import *  # noqa: F403

try:
    with _suppress(ImportError):
        from .numpycompat import *  # noqa: F403
except _ScikitplotException:
    _logger.warning(
        "Failed to import numpycompat. Some features may not work as expected."
    )
