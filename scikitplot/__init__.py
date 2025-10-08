# mypy: disallow-any-generics
# ruff: noqa: D205,F401,F403
# Flake8: noqa: F403
# type: ignore[]
# pylint: disable=import-error,unused-import,unused-variable,no-name-in-module,line-too-long,import-outside-toplevel

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
An intuitive library that seamlessly adds plotting capabilities and functionality
to any model objects or outputs, compatible with tools like scikit-learn,
XGBoost, TensorFlow, and more.

Documentation is available in the docstrings and online at
https://scikit-plots.github.io.
"""

from numpy import __version__ as __numpy_version__

######################################################################
## scikit-plots modules and objects
######################################################################

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y.0   # For first release after an increment in Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.Y.ZaN   # Alpha release
#   X.Y.ZbN   # Beta release
#   X.Y.ZrcN  # Release Candidate
#   X.Y.Z     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
# scikit-plots Releases (Next) (Docs)
# https://libraries.io/pypi/scikit-plots/versions
# Format: MAJOR.MINOR.PATCH.devN
# Format: MAJOR.MINOR.PATCHrcN
# Format: MAJOR.MINOR.PATCH
# Format: MAJOR.MINOR.PATCH.postN
# __version__ = "0.3.7.post0"  # 0.3.7
# __version__ = "0.3.8.post0"  # 0.3.8
# __version__ = "0.3.9.post0"  # 0.3.9
__version__ = "0.4.0.post5"  # 0.4.0  # 👈 (tagged) last release
# __version__ = "0.4.1rc0"  # 0.4.1
# __version__ = "0.4.2rc0"  # 0.4.2
# __version__ = "0.5.1rc0"  # 0.5.1
# __version__ = "0.5.2rc0"  # 0.5.2
# __version__ = "0.5.dev0"  # dev

# import logging as _logging
# logger.setLevel(logger.DEBUG)  # for debugging
from . import sp_logging as logger  # not a module or namespace, global attr
from .sp_logging import get_logger

try:
    # Trt to import meson built files, modules (etc. *.in)
    # This is the first import of an extension module within SciPy. If there's
    # a general issue with the install, such that extension modules are missing
    # or cannot be imported, this is where we'll get a failure - so give an
    # informative error message.
    from ._lib._ccallback import LowLevelCallable
    from ._reset import reset
    from ._utils._show_versions import show_versions
    from .config import *
    from .version import (  # type: ignore[reportMissingModuleSource]
        # If a version with git hash was stored,
        # use that instead so override version if any.
        __git_hash__,
        __version__,
    )
except (ImportError, ModuleNotFoundError):
    _BUILT_WITH_MESON = None
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


######################################################################
## Public Interface define explicitly `__all__`
######################################################################


# public submodules are imported lazily, therefore are accessible from
# __getattr__.
_submodules = sorted(
    # Set-like <class 'dict_keys'>
    globals().keys()
    | {
        ## A package is a directory with an __init__.py
        "_api",
        "_build_utils",
        "_compat",
        "_datasets",
        "_decorates",
        "_docstrings",
        "_factory_api",
        "_lib",
        "_testing",
        "_typing",
        "_utils",
        "api",
        "cexperimental",
        "cexternals",
        "config",
        "experimental",
        "externals",
        "kds",
        "misc",
        "modelplotpy",
        "preprocessing",
        "snsx",  # Seaborn-style Plotting.
        "stats",
        "utils",
        "visualkeras",
        ## A module is a .py file that is itself a module object when imported.
        "_globals",
        "_min_dependencies",
        "_preprocess",
        "_reset",
        "cli",
        "environment_variables",
        "exceptions",
        "ml_package_versions",
        "sp_logging",
        "version",
    }
    | {
        # attrs
        "LowLevelCallable",
        "__git_hash__",
        "__numpy_version__",
        "__version__",
        "get_logger",
        "logger",
        "reset",
        "online_help",
        "show_versions",
        "test",
    }
    | {
        "__bibtex__",
        "__citation__",
        "config_context",
        "get_config",
        "set_config",
        "show_config",
    }
)

## Define __all__ to control what gets imported with 'from module import *'.
## If __all__ is not defined, Python falls back to using the module's global namespace
## (as returned by dir() or globals().keys()) for 'from module import *'.
## This means that all names in the global namespace, except those starting with '_',
## will be imported by default.
## Reference: https://docs.python.org/3/tutorial/modules.html#importing-from-a-package
# Scope: dir() with no arguments returns the all defined names in the current local scope.
# Behavior: Includes functions, variables, classes, modules, __builtins__ that are defined so far.
# Scope: globals() returns a dict of all symbols defined at the module level.
# Behavior: Almost equivalent to dir() for a module context. No builtins included
# More explicit about scope—only includes actual global symbols.
# Keeps out internal modules, helpers, and unwanted globals
# def build_all():
#     import types
#     return [
#         name for name, val in globals().items()
#         if not name.startswith("_")
#         and not isinstance(val, types.ModuleType)
#     ]
# __all__ = [s for s in dir() if not s.startswith("_")]  # Remove dunders.
__all__ = tuple(_submodules)


######################################################################
## Customize lazy-loading mechanism `__dir__` with `__getattr__`
######################################################################


## Customize dir(object) behavior to control what attributes are displayed.
## By default, dir() lists all names in the module's global namespace,
## including functions, classes, variables, and special attributes like '__doc__'.
## Reference: https://docs.python.org/3/library/functions.html#dir
## dir() default behavior: Similar to globals().keys().
## To customize what dir() returns, define a custom __dir__() function within the module.
def __dir__() -> "list[str]":
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
    >>> import scikitplot as sp
    >>> dir(sp)  # sp.__dir__()
    ['attribute1', 'attribute2', 'attribute3']  # Example output

    >>> from scikitplot import __dir__
    >>> __dir__()
    ['attribute1', 'attribute2', 'attribute3']  # Example output

    """
    # logger.info("List of scikitplot flavors...")
    # return __all__
    # Remove, filters out private/dunder names.
    return [
        s
        for s in sorted(
            # Set-like <class 'dict_keys'>
            (globals().keys() | {})
            # .union(__all__)
            .union(_submodules)
            # submodule directly
            .union(dir(__import__(__name__ + ".api", fromlist=[""])))
            # diff
            .difference(
                {
                    "_submodules",
                }
            )
        )
        if not s.startswith("_")
    ] + [
        "__git_hash__",
        "__numpy_version__",
        "__version__",
        "__bibtex__",
        "__citation__",
    ]


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
    # logger.info(f"Try to load {name!r}")
    package = package or __name__
    try:
        # Check if it's already in the module's global scope
        if name in globals():  # ?frozenset
            return globals()[name]

        # Avoid importing things that aren't needed for building
        # which might import the main scikitplot module
        if name == "test":
            from ._testing._pytesttester import PytestTester  # Pytest testing

            test = PytestTester(__name__)
            del PytestTester
            return test

        # Try importing as a submodule
        # import_module(f"{package}.{name}")              # high-level function, Return submodule directly
        # import_module(f".{name}", package=package)      # high-level function, Return submodule directly
        # __import__(f"{package}.{name}", fromlist=[""])  # Return submodule directly (module_name, fromlist=["" or class_name])
        # __import__(f"{__name__}.{name}")                # low-level function, not return submodule directly
        from importlib import import_module

        if name in dir(__import__(__name__ + ".api", fromlist=[""])):
            from ._compat.optional_deps import nested_import

            # return any object, If any
            return nested_import(f"{package}.api.{name}")

        # Lazily load scikitplot flavors to avoid excessive dependencies.
        if name in ("visualkeras",):
            # Avoiding heavy imports top level module unless actually used
            from ._compat.optional_deps import LazyImport

            # return any object, If any
            return LazyImport(f"{package}.{name}")
        return import_module(f"{package}.{name}")  # submodule directly
    except (
        AttributeError,
        ImportError,
        ModuleNotFoundError,
        RecursionError,
    ) as e:
        suggestion_msg = ""
        if suggestion:
            # pylint: disable=import-outside-toplevel
            from difflib import get_close_matches
            from sys import modules

            # Everything in namespace dir()
            available = list(globals().keys()) + list(
                getattr(modules[package], "__all__", [])
            )
            # Generate suggestions for mistyped names.
            matches = get_close_matches(name, available)
            if matches:
                suggestion_msg += f"Did you mean: {', '.join(matches)}?\n\n"
        # Raise an error indicating the attribute could not be found,
        # with suggestions if any.
        raise AttributeError(
            f"Module {package!r} has no attribute {name!r}. {suggestion_msg}{e}"
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
    https://scikit-plots.github.io/dev/search.html?q=installation
    """
    try:
        # pylint: disable=import-outside-toplevel
        import os
        import sys
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
        # logger.error(f"{full_url}")
        sys.stderr.write(f"{full_url}")
        return webbrowser.open(full_url, new=new_window)
    except ModuleNotFoundError as e:
        logger.exception(f"Error opening documentation: {e}")
        return False


######################################################################
##
######################################################################
