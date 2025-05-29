"""
scikit-plots.

An intuitive library to add plotting functionality to scikit-learn objects

Documentation is available in the docstrings and
online at https://scikit-plots.github.io.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# mypy: disallow-any-generics

# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=wrong-import-position
# pylint: disable=import-outside-toplevel
# pylint: disable=broad-exception-caught

# ruff: noqa: F401

# import os as _os
# import sys as _sys
# import importlib as _importlib
# import pathlib as _pathlib
# import warnings as _warnings
# _set = set  # 'seaborn.set' can be override raise error
import builtins as _builtins  # noqa: I001
from numpy import (  # type: ignore[reportMissingModuleSource]
    __version__ as __numpy_version__,
)

######################################################################
## scikit-plots modules and objects
######################################################################

## Format: MAJOR.MINOR.PATCH.devN
## Format: MAJOR.MINOR.PATCHrcN
## Format: MAJOR.MINOR.PATCH
# __version__ = "0.3.7.post0"
# __version__ = "0.3.9rc3"
# __version__ = "0.4.0rc4"
# __version__ = "0.4.0"
__version__ = "0.5.0.dev0"

# import logging as _logging
from . import sp_logging as logger

logger.setLevel(logger.DEBUG)

try:  # Trt to import meson built files, modules (etc. *.in)
    from ._lib import __array_api_version__
    from ._lib._array_api import gpu_libraries
    from ._lib._ccallback import LowLevelCallable  # Low-level callback function
    from .config import __bibtex__, __citation__, show_config
    from .version import (  # type: ignore[reportMissingModuleSource]
        # If a version with git hash was stored,
        # use that instead so override version if any.
        __git_hash__,
        __version__,
    )
except (ImportError, ModuleNotFoundError):
    _BUILT_WITH_MESON = show_config = None
    logger.warning(
        "BOOM! :: %s",
        (
            "Error importing scikitplot: you cannot import scikitplot while "
            "being in scikitplot source directory; please exit the scikitplot source "
            "tree first and relaunch your Python interpreter."
        ),
    )
    # raise ImportError(_msg) from e
else:
    _BUILT_WITH_MESON = True

# Export some module objects
from ._globals import _Default, _Deprecated, _NoValue
from ._testing._pytesttester import PytestTester  # Pytest testing
from ._utils._show_versions import show_versions
from .api import *  # noqa: F403
from .config._config import config_context, get_config, set_config
from .utils.utils_path import remove_paths
from .utils.utils_plot_mpl import stack_mpl_figures

test = PytestTester(__name__)
del PytestTester

######################################################################
## Public Interface define explicitly `__all__`
######################################################################

# Don't pollute namespace. Imported for internal use.
# del os, sys, pathlib, warnings

# public submodules are imported lazily, therefore are accessible from
# __getattr__.
_submodules = {
    ## A package is a directory with an __init__.py file that can define what attributes it exposes.
    "_api",
    "_astropy",
    "_build_utils",
    # '_clv',
    "_compat",
    "_datasets",
    "_decorates",
    "_docstrings",
    "_externals",
    "_factory_api",
    "_lib",
    ## Experimental, we keep transform api module to compatibility seaborn core.
    "_seaborn",
    "_testing",
    "_tweedie",
    "_utils",
    "api",
    "config",
    "experimental",
    "kds",
    "misc",
    "modelplotpy",
    "probscale",
    "snsx",
    "sphinxext",
    "stats",
    "typing",
    "utils",
    "visualkeras",
    ## A module is a .py file that is itself a module object when imported.
    "_globals",
    "_min_dependencies",
    "_preprocess",
    "environment_variables",
    "ml_package_versions",
    "sp_logging",
    "version",
}
## Define __all__ to control what gets imported with 'from module import *'.
## If __all__ is not defined, Python falls back to using the module's global namespace
## (as returned by dir() or globals().keys()) for 'from module import *'.
## This means that all names in the global namespace, except those starting with '_',
## will be imported by default.
## Reference: https://docs.python.org/3/tutorial/modules.html#importing-from-a-package
_discard = {"_discard", "_submodules"}
__all__ = tuple(
    sorted(
        [
            name
            for name in (
                _builtins.set(globals()).union(_submodules).difference(_discard)
            )
            ## Exclude private/internal names (those starting with '_') placeholder
            if not (name.startswith("...") and not name.endswith("..."))
        ]
    )
)


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
    Return a sorted list of custom attributes for the module.

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
    from . import api

    return sorted(
        _builtins.set(globals()).union(_submodules).union(dir(api)).difference(_discard)
    )


## Dynamically import submodules only when they are accessed (lazy-loading mechanism).
## Avoid loading unnecessary submodules, reducing initial import overhead.
## Provide clear error handling and suggestions for unresolved attributes.
def __getattr__(
    name: str,
    package: "str | None" = None,
    suggestion: bool = False,
) -> "any":
    """
    Dynamically handle undefined attribute access in a module.

    This function is triggered when an attribute is not found in the module.
    It attempts to dynamically import a submodule or resolve the attribute.
    Optionally, suggestions for close matches are provided.

    Parameters
    ----------
    name : str
        The name of the missing attribute being accessed.

    package : str, optional
        The package name for relative import resolution.
        Defaults to the current module's name.

    suggestion : bool, optional
        If True, suggests close attribute names when not found.

    Returns
    -------
    Any
        The resolved attribute or submodule if found.

    Raises
    ------
    AttributeError
        If the attribute cannot be resolved.
    """
    package = package or __name__
    try:
        # Check if it's already in the module's global scope
        if name in globals():
            return globals()[name]
        # Try importing as a submodule
        from ._compat.optional_deps import nested_import

        return nested_import(name, package)
    except (AttributeError, ImportError, ModuleNotFoundError, Exception) as e:
        suggestion_msg = ""
        if suggestion:
            from difflib import get_close_matches
            from sys import modules

            # Everything in namespace dir()
            available = list(globals().keys()) + list(
                getattr(modules[package], "__all__", [])
            )
            # Generate suggestions for mistyped names.
            matches = get_close_matches(name, available)
            if matches:
                suggestion_msg = f" Did you mean: {', '.join(matches)}?"
        # Raise an error indicating the attribute could not be found,
        # with suggestions if any.
        raise AttributeError(
            f"Module '{package}' has no attribute '{name}'.{suggestion_msg}\n{e}"
        ) from e


######################################################################
## online search helper scikit-plots
######################################################################


def online_help(
    query: str = "",
    docs_root_url: str = "https://scikit-plots.github.io/",
    search_page: str = "search.html",
    new_window: int = 0,
) -> bool:
    """
    Open the online documentation search page.

    Use a given search query in the default web browser.
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
    >>> import scikitplot
    >>> scikitplot.online_help("installation")

    """
    try:
        import os
        import webbrowser
        from urllib.parse import urlencode, urlparse

        # from scikitplot import __version__

        # Determine if the current version is in development or stable
        version_type = "dev" if "dev" in __version__ else "stable"

        # Construct the base documentation URL, appending the version type
        docs_root_url = os.getenv("DOCS_ROOT_URL", docs_root_url).strip().strip("/")
        docs_root_url = f"{docs_root_url}/{version_type}"

        # Build the search URL with query parameters
        search_url = f"{docs_root_url}/{search_page}"
        params = {"q": query}
        full_url = f"{search_url}{('&' if urlparse(search_url).query else '?')}{urlencode(params)}"

        ## This launches the URL in the browser
        return webbrowser.open(full_url, new=new_window)
    except Exception as e:
        print(f"Error opening documentation: {e}")  # noqa: T201
        return False


######################################################################
## globally seeding
######################################################################


# def setup_module(module) -> None:
#     """
#     Fixture to seed random number generators for reproducible tests.

#     This function ensures a globally controllable random seed for Python's built-in `random`,
#     and NumPy's RNG, optionally configurable via the `SKPLT_SEED` environment variable.

#     Parameters
#     ----------
#     module : Any
#         The test module passed by the testing framework (e.g., pytest). This parameter
#         is required by the `setup_module` hook but is not directly used in this function.

#     Notes
#     -----
#     - If `SKPLT_SEED` is not set in the environment, a random seed is generated.
#     - This function supports both legacy and newer NumPy random APIs.
#     """
#     try:
#         import os
#         import random
#         import numpy as np

#         # Get seed from environment variable or generate one
#         seed_env = os.environ.get("SKPLT_SEED")
#         if seed_env is not None:
#             seed = int(seed_env)
#         else:
#             seed = int(np.random.uniform() * np.iinfo(np.int32).max)  # noqa: NPY002

#         print(f"I: Seeding RNGs with {seed}")  # noqa: T201, UP031

#         # Seed both NumPy and Python RNG
#         # np.random.Generator
#         # Legacy NumPy seeding (safe across versions)
#         np.random.seed(seed)  # noqa: NPY002
#         random.seed(seed)

#     except Exception as e:
#         print(f"Warning: RNG seeding failed: {e}")  # noqa: T201


######################################################################
##
######################################################################
