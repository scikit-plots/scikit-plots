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

# If a version with git hash was stored, use that instead
from . import version
from .version import __githash__#, __version__

from ._citation import __citation__, __bibtex__

__array_api_version__ = "2023.12"

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
  msg = (
    "Error importing scikitplot: you cannot import scikitplot while "
    "being in scikitplot source directory; please exit the scikitplot source "
    "tree first and relaunch your Python interpreter."
  )
  # raise ImportError(msg) from e
    
  log.error('BOOM! :: %s', msg)
  sys.stderr.write('Running on source directory: %s\n' % 'scikitplot')
  show_config = _BUILT_WITH_MESON = None; del msg;
else:
  _BUILT_WITH_MESON = True

######################################################################
## scikit-plots modules and objects
######################################################################

# Import scikitplot objects
from ._globals import _Default, _NoValue, _Deprecated
from ._globals import ModuleDeprecationWarning, VisibleDeprecationWarning
from ._utils._show_versions import show_versions  # noqa: E402
# from .rcmod import reset_orig, reset_defaults  # noqa: F401,F403

# Export Modules
from ._seaborn import *
from .api import *
from . import (
  _api,
  _astropy,
  _build_utils,
  _compat,
  _externals,
  _factory_api,
  _kds,
  _modelplotpy,
  _seaborn,
  _testing,
  _tweedie,
  _utils,
  _xp_core_lib,
  api,
  experimental,
  misc,
  probscale,
  stats,
  typing,
  utils,
  __config__,
  _citation,
  _config,
  _docstring,
  _globals,
  _log,
  _preprocess,
  cbook,
  version,
)

# Pytest testing
from ._testing._pytesttester import PytestTester
test = PytestTester(__name__); del PytestTester;

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
  'online_help',
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
######################################################################
## online search helper scikit-plots
######################################################################

def online_docs(
    query: str = '',
    docs_root_url: str = 'https://scikit-plots.github.io/',
    search_page: str = 'search.html',
    new_window: int = 0,  # default
) -> bool:
    """
    Open the online documentation search page for a given query in the default web browser.

    This function constructs a search URL based on the provided query and opens it in the web browser.
    It detects whether the version is in a development or stable state and directs the user to the correct documentation.

    Parameters
    ----------
    query : str, optional
        The search query to find relevant documentation. Defaults to an empty string.
    docs_root_url : str, optional
        The base URL of the documentation website. Defaults to 'https://scikit-plots.github.io/'.
    search_page : str, optional
        The search page URL (relative to `docs_root_url`). Defaults to 'search.html'.
    new_window : int, optional
        Controls how the URL is opened in the browser:
        - 0: Open in the same browser window.
        - 1: Open in a new browser window.
        - 2: Open in a new browser tab (default).

    Returns
    -------
    bool
        Returns True if the browser was successfully launched, False otherwise.

    Notes
    -----
    - The function automatically switches between the 'dev' and 'stable' versions of the documentation based on the value of `__version__`.
    - Requires an active internet connection.

    Examples
    --------
    >>> search_docs(query='installation')
    True
    """
    import os
    import webbrowser
    from urllib.parse import urlparse, urlencode, quote
    # Determine if the current version is in development or stable
    version_type = 'dev' if 'dev' in __version__ else 'stable'

    # Construct the base documentation URL, appending the version type
    docs_root_url = os.getenv('DOCS_ROOT_URL', docs_root_url).strip().strip('/')
    docs_root_url = f"{docs_root_url}/{version_type}/"

    # Build the search URL with query parameters
    search_url = f'{docs_root_url}/{search_page}'
    params = {
        # 'lang': 'en',
        'q': query,
    }
    full_url = f"{search_url}{('&' if urlparse(search_url).query else '?')}{urlencode(params)}"

    # Open the URL in the browser
    return webbrowser.open(full_url, new=new_window)