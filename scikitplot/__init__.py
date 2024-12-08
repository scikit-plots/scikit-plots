"""
scikit-plots

An intuitive library to add plotting functionality to scikit-learn objects

Documentation is available in the docstrings and
online at https://scikit-plots.github.io.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

######################################################################
## scikit-plots version
## PEP0440 compatible formatted version, see:
## https://peps.python.org/pep-0440/#version-scheme
## https://www.python.org/dev/peps/pep-0440/
## Version scheme: [N!]N(.N)*[{a|b|rc}N][.postN][.devN]
## Generic release markers (.devN, aN, bN, rcN, <no suffix>, .postN):
#   X.Y.devN             # 'Development release' 1.0.dev1
#   X.YrcN.devM          # Developmental release of a release candidate
#   X.Y.postN.devM       # Developmental release of a post-release
#   X.YrcN.postN.devN    # Developmental release of a post-release of a release candidate
#   X.Y{a|b|rc|c}N       # 'Pre-release' 1.0a1
#   X.Y==X.Y.0==N(.N)*   # For first 'Release' after an increment in Y
#   X.Y{post|rev|r}N     # 'Post-release' 1.0.post1
#   X.YrcN.postM         # Post-release of a release candidate
#   X.Y.N                # 'Bug fixes' 1.0.1
## setuptools-scm extracts Python package versions
######################################################################

# https://packaging.python.org/en/latest/discussions/versioning/#valid-version-numbers
# Admissible pre-release markers:
#   Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
#   X.Y.dev0     # is the canonical version of 'X.Y.dev'
#   X.Y.ZaN      # Alpha release
#   X.Y.ZbN      # Beta release
#   X.Y.ZrcN     # Release Candidate
#   X.Y.Z        # Final release
#   X.Y.Z.postM  # Post release
__version__ = '0.4.0.post0'
# from ._version import get_versions
# __version__ = get_versions()['version']
# del get_versions

######################################################################
## scikit-plots configuration
######################################################################

py_set = set  # keep python set, 'scikitplot.set' Override then raise error

import os
import sys
import pathlib
import warnings

# import logging; log=logging.getLogger(__name__); del logging;
from ._log import log

try:
  from scikitplot.__config__ import show as show_config
except (ImportError, ModuleNotFoundError) as e:
  sys.stderr.write('Running on source directory: %s\n' % 'scikitplot')
  msg = (
    "Error importing scikitplot: you cannot import scikitplot while "
    "being in scikitplot source directory; please exit the scikitplot source "
    "tree first and relaunch your Python interpreter."
  )
  # raise ImportError(msg) from e
    
  log.error('BOOM! :: %s', msg)
  _BUILT_WITH_MESON = False
else:
  _BUILT_WITH_MESON = True

# Capture the original matplotlib rcParams, delete the mpl module reference
import matplotlib as mpl; _orig_rc_params=mpl.rcParams.copy(); del mpl

# Import scikitplot objects
from ._globals import ModuleDeprecationWarning, VisibleDeprecationWarning
from ._globals import _Default, _NoValue, _Deprecated
from ._utils._show_versions import show_versions  # noqa: E402
from .colors import xkcd_rgb, crayons  # noqa: F401
from .rcmod import reset_orig, reset_defaults  # noqa: F401,F403

# Export Modules
from .colors import xkcd_rgb, crayons  # noqa: F401
from .miscplot import *  # noqa: F401,F403
from .palettes import *  # noqa: F401,F403
from .rcmod import *  # noqa: F401,F403
from .utils._utils import *  # noqa: F401,F403
from .widgets import *  # noqa: F401,F403

from .api import *
from .api.axisgrid import *  # noqa: F401,F403
from .api.categorical import *  # noqa: F401,F403
from .api.distributions import *  # noqa: F401,F403
from .api.matrix import *  # noqa: F401,F403
from .api.regression import *  # noqa: F401,F403
from .api.relational import *  # noqa: F401,F403

from ._citation import _get_bibtex
__citation__ = __bibtex__ = _get_bibtex()

######################################################################
## scikit-plots modules
######################################################################

# Modules
from . import (  # noqa: F401
  _api,
  _core,
  _factory_api,
  _marks,
  _numcpp_api,
  _xp_core_lib,
  api,
  backends,
  cm,
  colors,
  miscplot,
  palettes,
  probscale,
  rcmod,
  stats,
  utils,
  widgets,
  _compat,
  _docstring,
  _docstrings,
  # _globals,
  _preprocess,
  _statistics,
  algorithms,
  cbook
)

# Pytest testing
from ._testing._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester

# Remove symbols imported for internal use
del os, sys, pathlib, warnings
# Define __all__ to control what gets imported with 'from module import *'
# Combine global names (explicitly defined in the module) and dynamically available names
__all__ = [
  name for name in map(str, py_set(globals()).union(dir()))
  # Exclude private/internal names (those starting with '_')
  if not name.startswith('_') or name not in [
    'externals',
  ]
] + [
  '__dir__',
  '__getattr__',
]

######################################################################
## additional
######################################################################

def __dir__():
  # Custom attributes we want to be displayed when dir() is called
  return sorted(__all__)

# Define how undefined attributes should be handled in this module
def __getattr__(name):
  import importlib
  # Submodules dir()
  if name in dir():
    return importlib.import_module(f'scikitplot.{name}')
  else:
    try:
      # Explicitly defined names
      return globals()[name]
    except KeyError:
      raise AttributeError(
        f"Module 'scikitplot' has no attribute '{name}'"
      )