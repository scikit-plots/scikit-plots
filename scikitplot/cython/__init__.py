# scikitplot/cython/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.cython
==================
A lightweight runtime Cython development kit with caching, pinning,
garbage collection, and templating support.

:mod:`scikitplot.cython` enables real-time, in-place (in-situ) generation of
low-level Cython packages and modules for immediate use and testing.

.. seealso::
   * https://github.com/cython/cython
   * https://cython.readthedocs.io/en/latest/index.html
   * https://doc.sagemath.org/html/en/reference/misc/sage/misc/cython.html
"""  # noqa: D205, D400
# A small runtime Cython devkit with caching, pinning, GC, and templates.

from __future__ import annotations

from . import (
    _builder,
    _cache,
    _custom_compiler,
    _gc,
    _loader,
    _lock,
    _pins,
    _profiles,
    _public,
    _result,
    _security,
    _templates_api,
    _utils,
)
from ._builder import *  # noqa: F403
from ._cache import *  # noqa: F403
from ._custom_compiler import *  # noqa: F403
from ._gc import *  # noqa: F403
from ._loader import *  # noqa: F403
from ._lock import *  # noqa: F403
from ._pins import *  # noqa: F403
from ._profiles import *  # noqa: F403
from ._public import *  # noqa: F403
from ._result import *  # noqa: F403
from ._security import *  # noqa: F403
from ._templates_api import *  # noqa: F403
from ._utils import *  # noqa: F403

__all__: list[str] = []
__all__ += _builder.__all__
__all__ += _cache.__all__
__all__ += _custom_compiler.__all__
__all__ += _gc.__all__
__all__ += _loader.__all__
__all__ += _lock.__all__
__all__ += _pins.__all__
__all__ += _profiles.__all__
__all__ += _public.__all__
__all__ += _result.__all__
__all__ += _security.__all__
__all__ += _templates_api.__all__
__all__ += _utils.__all__
