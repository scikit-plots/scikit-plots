"""
scikit-plots

An intuitive library to add plotting functionality to scikit-learn objects

Documentation is available in the docstrings and
online at https://scikit-plots.github.io.
"""
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

__version__ = '0.4.0rc0'
__array_api_version__ = "2023.12"

py_set = set  # 'seaborn.set' override raise error
import os
import sys
import pathlib
import warnings

from scikitplot import sp_logging
from scikitplot.sp_logging import get_logger, SpLogger, sp_logger
try:
  # Trt to import meson builded files, modules (etc. *.in)
  from .__config__ import show as show_config
  from ._citation import __citation__, __bibtex__
  # If a version with git hash was stored, use that instead
  # from . import version
  from .version import __git_hash__, __version__  # Override git version
except (ImportError, ModuleNotFoundError) as e:
  msg = (
    "Error importing scikitplot: you cannot import scikitplot while "
    "being in scikitplot source directory; please exit the scikitplot source "
    "tree first and relaunch your Python interpreter."
  )
  get_logger().warning('BOOM! :: %s', msg)
  show_config = _BUILT_WITH_MESON = None; del msg;
  # raise ImportError(msg) from e
else:
  _BUILT_WITH_MESON = True

######################################################################
## scikit-plots modules and objects
######################################################################

# Import scikitplot objects
from ._globals import _Default, _NoValue, _Deprecated
from ._globals import ModuleDeprecationWarning, VisibleDeprecationWarning
from ._utils._show_versions import show_versions  # noqa: E402

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
  _seaborn,
  _testing,
  _tweedie,
  _utils,
  _xp_core_lib,
  api,
  experimental,
  kds,
  misc,
  modelplotpy,
  probscale,
  sp_logging,
  stats,
  typing,
  utils,
  visualkeras,
  __config__,
  _citation,
  _config,
  _docstring,
  _globals,
  _preprocess,
  cbook,
  version,
)

# Pytest testing
from ._testing._pytesttester import PytestTester
test = PytestTester(__name__); del PytestTester;

######################################################################
## Public Interface define explicitly `__all__`
######################################################################

# Don't pollute namespace. Imported for internal use.
del os, sys, pathlib, warnings

# Define __all__ to control what gets imported with 'from module import *'
# Combine global names (explicitly defined in the module) and dynamically available names
__all__ = [
  name for name in map(str, py_set(globals()).union(dir()))
  # Exclude private/internal names (those starting with '_')
  if not ( name.startswith('...') or name in ['py_set',])
] + [
  '__dir__',
  '__getattr__',
  'online_help',
]

######################################################################
## additional
######################################################################

# Custom attributes we want to be displayed when dir() is called
def __dir__():
    """
    Returns a sorted list of custom attributes for the module.

    This function overrides the default `dir()` behavior to display
    a custom list of attributes defined in the `__all__` variable.

    Returns
    -------
    list of str
        A sorted list of attribute names defined in `__all__`.

    Examples
    --------
    >>> from scikitplot import __dir__
    >>> __dir__()
    ['attribute1', 'attribute2', 'attribute3']  # Example output
    """
    return sorted(__all__)

# Define how undefined attributes should be handled in this module
def __getattr__(name):
    """
    Dynamically handles undefined attributes in the 'scikitplot' module.

    This function allows for dynamic imports of submodules or retrieval of 
    explicitly defined attributes within the 'scikitplot' module. If the 
    attribute cannot be resolved, an AttributeError is raised.

    Parameters
    ----------
    name : str
        The name of the attribute being accessed.

    Returns
    -------
    Any
        The dynamically imported submodule or the explicitly defined attribute.

    Raises
    ------
    AttributeError
        If the attribute does not exist as a submodule or in the global namespace.

    Examples
    --------
    >>> from scikitplot import __getattr__
    >>> plot = __getattr__('metrics')
    >>> print(plot)
    <module 'scikitplot.metrics' from '.../scikitplot/metrics.py'>

    """
    import importlib

    # Attempt to import submodule dynamically, submodules dir()
    try:
        return importlib.import_module(f'scikitplot.{name}')
    except ModuleNotFoundError:
        pass

    # Attempt to retrieve from global namespace
    try:
        return globals()[name]
    except KeyError:
        raise AttributeError(
            f"Module 'scikitplot' has no attribute '{name}'. "
            "Ensure the attribute or submodule exists and is correctly named."
        )

######################################################################
## online search helper scikit-plots
######################################################################

def online_help(
    query: str = '',
    docs_root_url: str = 'https://scikit-plots.github.io/',
    search_page: str = 'search.html',
    new_window: int = 0,
) -> bool:
    """\
    Open the online documentation search page
    for a given query in the default web browser.

    This function constructs a search URL based on the provided query
    and opens it in the web browser.
    It detects whether the version is in development or stable state
    and directs the user to the appropriate documentation.

    Parameters
    ----------
    query : str, optional
        The search query to find relevant documentation.
        Defaults to an empty string.
    docs_root_url : str, optional
        The base URL of the documentation website.
        Defaults to `https://scikit-plots.github.io/`.
    search_page : str, optional
        The search page URL (relative to `docs_root_url`).
        Defaults to `search.html`.
    new_window : int, optional
        Controls how the URL is opened in the browser:
        
        - 0: Open in the same browser window.
        - 1: Open in a new browser window.
        - 2: Open in a new browser tab.

    Returns
    -------
    bool
        Returns True if the browser was successfully launched, False otherwise.

    Notes
    -----
    - The function automatically switches between the 'dev' and 'stable'
      versions of the documentation based on the value of `__version__`.
    - Requires an active internet connection.
    - If the environment variable `DOCS_ROOT_URL` is set,
      it overrides the `docs_root_url` argument.

    Examples
    --------
    .. jupyter-execute::
    
        >>> import scikitplot
        >>> scikitplot.online_help('installation')
    """
    try:
        import os
        import webbrowser
        from urllib.parse import urlparse, urlencode
        # from scikitplot import __version__

        # Determine if the current version is in development or stable
        version_type = 'dev' if 'dev' in __version__ else 'stable'

        # Construct the base documentation URL, appending the version type
        docs_root_url = os.getenv('DOCS_ROOT_URL', docs_root_url).strip().strip('/')
        docs_root_url = f"{docs_root_url}/{version_type}"

        # Build the search URL with query parameters
        search_url = f"{docs_root_url}/{search_page}"
        params = {'q': query}
        full_url = f"{search_url}{('&' if urlparse(search_url).query else '?')}{urlencode(params)}"

        # Open the URL in the browser
        return webbrowser.open(full_url, new=new_window)
    except Exception as e:
        print(f"Error opening documentation: {e}")
        return False

######################################################################
##
######################################################################