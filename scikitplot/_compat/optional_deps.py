# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Checks for optional dependencies using lazy import.

From `PEP 562 <https://www.python.org/dev/peps/pep-0562/>`_.
"""

from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec

# ---------------------- Safe Import ----------------------


@lru_cache(maxsize=128)
def safe_import(module_name: str):
    """
    Dynamically import a module by name with error handling.

    Parameters
    ----------
    module_name : str
        Name of the module to import.

    Returns
    -------
    module
        Imported module object.
    """
    # Dynamically import a module by name with error handling
    try:
        return import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Required module '{module_name}' is not installed: {e}"
        ) from e


# Try importing as a submodule
@lru_cache(maxsize=128)
def nested_import(
    name,
    package="scikitplot",
    silent: bool = True,
) -> "callable":
    """
    Import a nested module or attribute dynamically.

    Defaulting to package='scikitplot' for relative imports.

    Parameters
    ----------
    name : str
        Full dotted path to the function, e.g., '.api.metrics.plot_roc'.
        Can be absolute ('a.b.c.d') or relative ('.b.c.d').
    package : str, optional
        The package name to use as a base for relative imports.
        Defaults to 'scikitplot'.
    silent : str
        Not raise error.

    Returns
    -------
    module or attribute
        The imported module or attribute specified by the full dotted path.

    Raises
    ------
    AttributeError, ImportError, ModuleNotFoundError
    """
    try:
        # __import__(f'{__name__}.{name}')  # low-level function
        # import_module(f".{name}", package=package)
        if name.startswith("."):
            return import_module(name, package=package)
        mod = import_module(package)
        for attr in name.split("."):
            # Dynamically import module attr
            mod = getattr(mod, attr)
        return mod
    except (AttributeError, ImportError, ModuleNotFoundError, Exception) as e:
        if not silent:
            raise e


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
def __getattr__(name):
    if name in __all__:
        # Use importlib.util.find_spec to check existence
        return find_spec(_deps[name.removeprefix("HAS_")]) is not None

    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}.")
