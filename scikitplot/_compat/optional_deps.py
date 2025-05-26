# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Checks for optional dependencies using lazy import.

From `PEP 562 <https://www.python.org/dev/peps/pep-0562/>`_.
"""

# pylint: disable=broad-exception-caught
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-branches

from importlib import import_module
from importlib.util import find_spec

from .. import logger

# logger.setLevel(logger.DEBUG)
# from functools import lru_cache
from .._compat.python import lru_cache

# ---------------------- Safe Import ----------------------


@lru_cache(maxsize=128)
def safe_import(module_name: str):
    """
    Dynamically import a module by name with error handling and caching.

    Parameters
    ----------
    module_name : str
        Name of the module to import.

    Returns
    -------
    module
        Imported module object.

    Raises
    ------
    ImportError
        If the specified module cannot be imported.

    Examples
    --------
    >>> os_mod = safe_import("os")
    >>> math_mod = safe_import("math")
    """
    try:
        # Attempt to dynamically import the module by name
        return import_module(module_name)
    except ImportError as e:
        # Raise ImportError with additional context if import fails
        raise ImportError(
            f"Required module '{module_name}' is not installed or could not be imported: {e}"
        ) from e


# Try importing as a submodule
@lru_cache(maxsize=128)
def nested_import(  # noqa: PLR0912
    name: str,
    package: str = "scikitplot",
    silent: bool = True,
    verbose: bool = False,
    default: "any | None" = None,
    validate_callable: bool = False,
) -> "any":
    """
    Dynamically import a nested module or attribute.

    Attempts to import a module or resolve a deeply nested attribute from either
    a relative or absolute dotted path. Provides fallback behavior if import fails.

    Parameters
    ----------
    name : str
        Dotted path to the target object. Can be relative (e.g., '.sub.attr') or
        absolute (e.g., 'scikitplot.sub.attr').
    package : str, optional
        Root package used to resolve relative imports. Default is 'scikitplot'.
    silent : bool, optional
        If True, suppresses exceptions and returns `default` on failure.
        If False, raises the original exception. Default is True.
    verbose : bool, optional
        If True, logs debug info about import attempts and errors. Default is False.
    default : Any, optional
        Value to return on failure. If not provided, defaults to a no-op function:
        None. Sample: `lambda *a, **kw: None`.
    validate_callable : bool, optional
        If True, raises ValueError if the final imported object is not callable.

    Returns
    -------
    Any
        The imported module, submodule, or attribute if successful;
        otherwise the fallback `default`.

    Raises
    ------
    ModuleNotFoundError
        If no valid module is found and `silent=False`.
    AttributeError
        If attribute resolution fails and `silent=False`.
    ValueError
        If `validate_callable` is True and the imported object is not callable.

    Examples
    --------
    Import a function from a nested module using relative import:

    >>> plot_roc = nested_import("._preprocess._preprocess_data", package="scikitplot")

    Import a function using absolute import:

    >>> plot_roc = nested_import("scikitplot._preprocess._preprocess_data")

    Import a module:

    >>> preproc_module = nested_import("scikitplot._preprocess")

    Use a safe import that falls back to a no-op function on failure:

    >>> plot_roc_safe = nested_import(
    ...     "bad.path.missing", default=lambda *a, **kw: print("Skipped")
    ... )

    Use a safe import that falls back to an arbitrary object on failure:

    >>> fallback = nested_import("non.existing.module", default=object())
    """
    # Provide a safe no-op fallback if none was specified
    # if default is None:
    #     default = lambda *a, **kw: None

    try:
        # Normalize path by removing leading dot (for relative imports)
        dotted_path = name.lstrip(".")
        parts = dotted_path.split(".")
        # __import__(module_name, fromlist=[class_name])
        # __import__(f'{__name__}.{name}')  # low-level function
        # import_module(f".{name}", package=package)

        if verbose:
            logger.debug(f"Attempting to import '{name}' with base package '{package}'")

        # Attempt to import the longest resolvable module prefix
        for i in reversed(range(1, len(parts) + 1)):
            module_path = ".".join(parts[:i])
            try:
                # Use relative import if input was relative
                module = (
                    import_module(f".{module_path}", package=package)
                    # if name.startswith(".")
                    # else import_module(module_path)
                )
                if verbose:
                    logger.debug(f"Successfully imported module '{module_path}'")
                break  # Successfully imported a module
            except ModuleNotFoundError as e:
                if verbose:
                    logger.debug(f"ModuleNotFoundError for '{module_path}': {e}")
                continue
        else:
            # No module could be resolved
            raise ModuleNotFoundError(f"Could not import any module from '{name}'")

        # Resolve remaining dotted parts as attributes
        for attr in parts[i:]:
            try:
                module = getattr(module, attr)
                if verbose:
                    logger.debug(f"Resolved attribute '{attr}'")
            except AttributeError as e:
                raise AttributeError(
                    f"Failed to access attribute '{attr}' in '{name}'"
                ) from e

        if validate_callable and not callable(module):
            raise ValueError(f"Imported object '{name}' is not callable")

        return module

    except Exception as e:
        if verbose:
            logger.error(f"Import error for '{name}': {e}")
        if not silent:
            raise e
        return default


# First, the top-level packages:
# TODO: This list is a duplicate of the dependencies in pyproject.toml "all", but
# some of the package names are different from the pip-install name (e.g.,
# beautifulsoup4 -> bs4).
# Some optional parts of the standard library also find a place here,
# but don't appear in pyproject.toml
_optional_deps = [  # noqa: RUF005
    "asdf_astropy",
    "bleach",
    "bottleneck",
    "bs4",
    "bz2",  # stdlib
    "certifi",
    "dask",
    "fsspec",
    "h5py",
    "html5lib",
    "ipykernel",
    "IPython",
    "ipywidgets",
    "ipydatagrid",
    "jplephem",
    "lxml",
    "matplotlib",
    "mpmath",
    "pandas",
    "PIL",
    "pytz",
    "s3fs",
    "scipy",
    "skyfield",
    "sortedcontainers",
    "uncompresspy",
    "lzma",  # stdlib
    "pyarrow",
    "pytest_mpl",
    "array_api_strict",
] + [
    "tensorflow",
    "keras",
    "streamlit",
]
_deps = {k.upper(): k for k in _optional_deps}

# Any subpackages that have different import behavior:
_deps["PLT"] = "matplotlib"

__all__ = [f"HAS_{pkg}" for pkg in _deps]


@lru_cache(maxsize=128)
def __getattr__(name: str) -> bool:
    """
    Lazy attribute loader for feature flags indicating presence of dependencies.

    Checks if the specified dependency is installed by attempting to find its module spec.

    Parameters
    ----------
    name : str
        Name of the attribute being accessed, e.g., 'HAS_NUMPY'.

    Returns
    -------
    bool
        True if the dependency is installed, False otherwise.

    Raises
    ------
    AttributeError
        If the requested attribute is not in the allowed `__all__` list.

    Examples
    --------
    >>> HAS_NUMPY
    True  # if numpy is installed
    >>> HAS_PANDAS
    False  # if pandas is not installed
    """
    if name in __all__:
        # Extract the dependency key by removing "HAS_" prefix
        dep_key = name.removeprefix("HAS_")
        # Check if the dependency module can be found
        # Use importlib.util.find_spec to check existence
        return find_spec(_deps[dep_key]) is not None

    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}.")
