"""
scikit-plots

An intuitive library to add plotting functionality to scikit-learn objects

Documentation is available in the docstrings and
online at https://scikit-plots.github.io.
"""
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

__version__ = '0.5.dev0'

# import os
# import sys
# import pathlib
# import warnings
# py_set = set  # 'seaborn.set' override raise error or use builtins.set
import builtins

######################################################################
## scikit-plots modules and objects
######################################################################

# import logging
from .sp_logging import get_logger, SpLogger, sp_logger
try:
  # Trt to import meson builded files, modules (etc. *.in)
  from .__config__ import show as show_config
  from ._citation import __citation__, __bibtex__
  # If a version with git hash was stored, use that instead.
  from .version import __git_hash__, __version__  # Override version if any.
except (ImportError, ModuleNotFoundError) as e:
  msg = (
    "Error importing scikitplot: you cannot import scikitplot while "
    "being in scikitplot source directory; please exit the scikitplot source "
    "tree first and relaunch your Python interpreter."
  )
  # raise ImportError(msg) from e
  get_logger().warning('BOOM! :: %s', msg); del msg;
  _BUILT_WITH_MESON = show_config = None
else:
  _BUILT_WITH_MESON = True

from ._config import config_context, get_config, set_config
from ._globals import _Default, _Deprecated, _NoValue
from ._seaborn import _orig_rc_params
from ._utils._show_versions import show_versions  # noqa: E402
from ._xp_core_lib import __array_api_version__

# Export modules
from ._seaborn import *
from .api import *

# Sub-modules:
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
  stats,
  typing,
  visualkeras,
  __config__,
  _citation,
  _config,
  _docstring,
  _globals,
  _preprocess,
  cbook,
  sp_logging,
  version,
)

# Pytest testing
from ._testing._pytesttester import PytestTester
test = PytestTester(__name__); del PytestTester

######################################################################
## Public Interface define explicitly `__all__`
######################################################################

# Don't pollute namespace. Imported for internal use.
# del os, sys, pathlib, warnings

# public submodules are imported lazily, therefore are accessible from
# __getattr__.
_submodules = {
  # Sub-modules:
  '_api',
  '_astropy',
  '_build_utils',
  '_compat',
  '_externals',
  '_factory_api',
  '_seaborn',
  '_testing',
  '_tweedie',
  '_utils',
  '_xp_core_lib',
  'api',
  'experimental',
  'kds',
  'misc',
  'modelplotpy',
  'probscale',
  'stats',
  'typing',
  'utils',
  'visualkeras',
  '__config__',
  '_citation',
  '_config',
  '_docstring',
  '_globals',
  '_preprocess',
  'cbook',
  'sp_logging',
  'version',
  # Non-modules:
  "show_versions",
  'test',
  '__dir__',
  '__getattr__',
  'online_help',
  'setup_module',
}
_discard = {
  '_discard',
  '_submodules',
  'py_set',
  'builtins',
}
## Define __all__ to control what gets imported with 'from module import *'.
## If __all__ is not defined, Python falls back to using the module's global namespace
## (as returned by dir() or globals().keys()) for 'from module import *'.
## This means that all names in the global namespace, except those starting with '_',
## will be imported by default.
## Reference: https://docs.python.org/3/tutorial/modules.html#importing-from-a-package
__all__ = sorted([
  name for name in builtins.set(globals()).union(_submodules).difference(_discard)
  # Exclude private/internal names (those starting with '_')
  if not ( (name.startswith('...') and not name.endswith('...')) )
])

######################################################################
## Customize lazy-loading mechanism `__dir__` with `__getattr__`
######################################################################

## Customize dir(object) behavior to control what attributes are displayed.
## By default, dir() lists all names in the module's global namespace,
## including functions, classes, variables, and special attributes like '__doc__'.
## Reference: https://docs.python.org/3/library/functions.html#dir
## dir() default behavior: Similar to globals().keys().
## To customize what dir() returns, define a custom __dir__() function within the module.
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
    from . import (
      _seaborn,
      api,
    )
    return sorted(
      builtins
      .set(globals())
      .union(_submodules)
      .union(dir(_seaborn))
      .union(dir(api))
      .difference(_discard)
    )
  
## Dynamically import submodules only when they are accessed (lazy-loading mechanism).
## Avoid loading unnecessary submodules, reducing initial import overhead.
## Provide clear error handling and suggestions for unresolved attributes.
def __getattr__(name):
    """
    Dynamically handle undefined attributes in the module.

    This function is called when an attribute is accessed but not found in the module's
    global namespace. It allows for dynamic imports of submodules or the generation of 
    attributes at runtime. If the attribute cannot be resolved, it raises an AttributeError.

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
    """
    try:
        if name in dir():
            import importlib
            return importlib.import_module(f'.{name}', package=__name__)
          
        return globals()[name]
    except (ModuleNotFoundError,KeyError):
        pass  # Submodule not found; proceed to error handling.

    import difflib
    # Generate suggestions for mistyped names.
    all_names = dir()
    suggestions = difflib.get_close_matches(name, all_names)
    
    # Raise an error indicating the attribute could not be found, with suggestions if any.
    suggestion_msg = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'.{suggestion_msg}")

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
## globally seeding
######################################################################

def setup_module(module):
    """Fixture for the tests to assure globally controllable seeding of RNGs"""
    import os
    import random
    import numpy as np

    # Check if a random seed exists in the environment, if not create one.
    _random_seed = os.environ.get("SKPLT_SEED", None)
    if _random_seed is None:
        _random_seed = np.random.uniform() * np.iinfo(np.int32).max
    _random_seed = int(_random_seed)
    print("I: Seeding RNGs with %r" % _random_seed)
    np.random.seed(_random_seed)
    random.seed(_random_seed)

######################################################################
##
######################################################################