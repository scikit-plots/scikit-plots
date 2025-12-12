# pylint: disable=broad-exception-caught
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-branches

# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Checks for optional dependencies using lazy import.

From `PEP 562 <https://www.python.org/dev/peps/pep-0562/>`_.
"""

from __future__ import annotations

import contextlib
import inspect
import sys

# from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType  # Inherited
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable

from .. import logger
from .._compat.python import lru_cache
from .._globals import _NoValue
from ..exceptions import ScikitplotException

# import_module(f".{name}", package=package)  # high-level function
# __import__(f'{__name__}.{name}')  # low-level function, not, not the submodule
# __import__(module_name, fromlist=[class_name])  # Return submodule directly


# --- nested module or attribute Import ---
@lru_cache(maxsize=128)
def nested_import(  # noqa: PLR0912
    name: str,
    package: str | None = "scikitplot",
    default: any | None = None,
    validate_callable: bool = False,
    error: str = "raise",  # "raise" or "ignore"
    verbose: bool = False,
) -> ModuleType | Callable[..., any] | any | None:
    """
    Dynamically import a nested module or attribute from a dotted path (e.g. ``"a.b.c"``).

    This helper supports:
    - Absolute imports (e.g. ``"scikitplot.metrics.plot_roc"``),
    - Explicit relative imports (e.g. ``".metrics.plot_roc"`` with ``package="scikitplot"``),
    - Attribute resolution on top of an imported module (e.g. functions, classes).

    Parameters
    ----------
    name : str
        Dotted path to the target object. Can be:
        - absolute: e.g. ``"scikitplot.metrics.plot_roc"``,
        - relative: starts with ``"."`` and uses ``package`` as the anchor.
    package : str, optional
        Anchor package for resolving *relative* imports (when ``name`` starts with ``"."``).
        Ignored for absolute imports. Default is ``"scikitplot"``.
    default : any, optional
        Value to return if the import fails and ``error="ignore"``. Default is ``None``.
    validate_callable : bool, optional
        If True, raises ``ValueError`` when the final imported object is not callable.
    error : {"raise", "ignore"}, optional
        Error handling strategy when import or attribute resolution fails.
        - ``"raise"``  : propagate ImportError / AttributeError.
        - ``"ignore"`` : return ``default`` instead of raising.
    verbose : bool, optional
        If True, logs additional debug information.

    Returns
    -------
    module_or_obj : types.ModuleType or callable or any or None
        The imported object, or ``default`` if import fails and ``error="ignore"`` is used.

    Raises
    ------
    TypeError
        If a relative import is requested (``name`` starts with ``"."``) but ``package`` is empty.
    ValueError
        If ``validate_callable`` is True and the imported object is not callable.
        Or if ``error`` is not one of ``{"raise", "ignore"}``.
    ImportError
        If no module prefix can be imported and ``error="raise"``.
    AttributeError
        If attribute resolution fails and ``error="raise"``.

    Notes
    -----
    - Uses ``importlib.import_module`` for both absolute and explicit relative imports.
    - Does *not* guess or auto-prefix external module names with ``package``: there is no
      heuristic rewriting of e.g. ``"numpy"`` to ``"scikitplot.numpy"``.
    - Caches successful calls with ``functools.lru_cache`` for speed.

    Examples
    --------
    Import a function using absolute import:

    >>> nested_import("scikitplot.metrics.plot_roc")
    <function plot_roc at ...>

    Import a function from a nested module using relative import:

    >>> nested_import(".metrics.plot_roc", package="scikitplot")
    <function plot_roc at ...>

    Use a safe import that falls back to an arbitrary object on failure:

    >>> fallback = object()
    >>> nested_import("non.existing.module", default=fallback) is fallback
    True
    """
    if error not in {"raise", "ignore"}:
        raise ValueError(
            f"Invalid error value {error!r}; expected 'raise' or 'ignore'."
        )

    # 1. Basic validation and normalization
    is_relative = name.startswith(".")

    if is_relative and not package:
        raise TypeError(
            f"the 'package' argument is required to perform a relative import for {name!r}"
        )

    stripped_name = name.lstrip(".")
    parts = stripped_name.split(".")

    # Short-circuit: plain module name without dots
    if len(parts) == 1:
        module_name = parts[0]
        try:
            if is_relative:
                module = import_module(f".{module_name}", package=package)
            else:
                module = import_module(module_name)
        except ModuleNotFoundError as exc:
            msg = f"Could not import module {name!r} (resolved as {module_name!r})"
            logger.info(msg)
            if error == "ignore":
                return default
            raise ImportError(msg) from exc

        if validate_callable and not callable(module):
            msg = f"Imported object {name!r} is not callable"
            logger.info(msg)
            raise ValueError(msg)

        if verbose:
            logger.info(f"Successfully imported {name!r}: {module!r}")
        return module

    # 2. Import the longest possible module prefix, without heuristics.
    module: ModuleType | any | None = None
    imported_prefix_index = 0  # how many parts belong to the module name

    for i in reversed(range(1, len(parts) + 1)):
        module_path = ".".join(parts[:i])
        try:
            if is_relative:
                module = import_module(f".{module_path}", package=package)
            else:
                module = import_module(module_path)
            imported_prefix_index = i
            break
        except ModuleNotFoundError:
            continue
        except ScikitplotException as e:  # project-specific error
            logger.exception(f"Unexpected error while importing {module_path!r}")
            if error == "ignore":
                return default
            raise e

    # 3. Handle case where no module prefix could be imported
    if module is None:
        msg = f"Could not import any module prefix from {name!r}"
        if is_relative:
            msg += f" using package {package!r}"
        logger.info(msg)
        if error == "ignore":
            return default
        raise ImportError(msg)

    # 4. Resolve remaining parts as attributes on the imported module
    obj: any = module
    try:
        for attr in parts[imported_prefix_index:]:
            obj = getattr(obj, attr)
    except (AttributeError, RecursionError) as e:
        attr_name = (
            parts[imported_prefix_index]
            if imported_prefix_index < len(parts)
            else "<?>"
        )
        logger.exception(
            f"Attribute {attr_name!r} not found while resolving {name!r} "
            f"starting from module {module!r} ({type(e).__name__!r}: {e})"
        )
        if error == "ignore":
            return default
        raise

    # 5. Optional callable validation
    if validate_callable and not callable(obj):
        msg = f"Imported object {name!r} is not callable"
        logger.info(msg)
        raise ValueError(msg)

    if verbose:
        logger.info(f"Successfully imported {name!r}: {obj!r}")
    return obj


# --- Lazily Import ---
def _get_source_path(obj: object):
    """
    Return the source file path of a module or callable.

    This function attempts to retrieve the source file path of a given object,
    which may be a module, function, class, or other callable. It gracefully
    handles multiple layers of decorators, user-defined proxies, and fallbacks
    for built-ins or C-extension modules.

    Parameters
    ----------
    obj : object
        The object to inspect. Can be a module, function, method, class, or
        any callable.

    Returns
    -------
    str
        The file path to the source code, or '<?>' if it cannot be determined.

    Notes
    -----
    - For modules, `__file__` is returned.
    - For callables, `inspect.unwrap` is used to get past multiple decorators.
    - If the object is a built-in or C extension, the path may point to a
      `.so`, `.pyd`, or `.dll` file.
    - If the object has a `__code__` attribute, it is used as a fallback.
    - If none of these methods succeed, `'<?>'` is returned.

    Examples
    --------
    >>> import math
    >>> _get_source_path(math)
    '/usr/lib/python3.10/lib-dynload/math.cpython-310-x86_64-linux-gnu.so'

    >>> def f():
    ...     pass
    >>> _get_source_path(f)
    '<current_file>.py'

    >>> import os
    >>> _get_source_path(os.path)
    '/usr/lib/python3.10/posixpath.py'
    """
    # Check if the object is a module (e.g., math, os)
    if isinstance(obj, ModuleType):
        # Use __file__ if available, fallback to '<?>'
        return getattr(obj, "__file__", "<?>")  # returns '<?>' not raise

    # Check if the object is callable (function, method, class, etc.)
    if callable(obj):
        # Handle custom proxy types like LazyImport that define get_real_func()
        if hasattr(obj, "get_real_func") and callable(
            getattr(obj, "get_real_func", None)
        ):
            # Ignore errors in custom proxies
            with contextlib.suppress(Exception):
                obj = obj.get_real_func()

        # Handle wrapped decorators using functools.wraps or __wrapped__
        # Continue with the outer object if unwrap fails
        with contextlib.suppress(Exception):
            # obj = obj.__wrapped__
            obj = inspect.unwrap(obj)

        # Attempt to retrieve the source file directly
        try:
            # _, lineno = inspect.getsourcelines(obj)
            return inspect.getfile(obj)
        except Exception:
            pass  # May be a built-in or a C-extension

        # Fallback via code object
        # try using the code object (for Python-defined functions)
        code = getattr(obj, "__code__", None)
        if code is not None:
            # lineno = obj.__code__.co_firstlineno
            filename = getattr(code, "co_filename", None)
            # __code__ exists but may not point to a valid file
            if filename:
                return filename
    # If all methods fail, return unknown
    return "<?>"


class LazyImport(ModuleType):
    """
    Lazily import a module, function, or attribute by dotted path.

    The target object is only imported the first time it is actually needed
    (accessed, called, or explicitly resolved). This helps avoid importing
    heavy dependencies at import time.

    Import availability can be checked cheaply (without a full import) using
    boolean context::

        lazy_np = LazyImport("numpy")
        if lazy_np:
            # numpy appears importable (based on importlib.util.find_spec)
            ...

    Parameters
    ----------
    name : str
        The dotted import path of the target module or attribute to load
        (e.g., "pandas.DataFrame").
    package : str, optional
        Root package used to resolve relative imports.
        Only relevant if `name` is a relative import (e.g., ".utils").
    parent_module_globals : str | dict | None, optional
        Default package `globals()` by `vars(sys.modules[package])`
    default : any, optional
        A fallback value to return if the import fails.
        If None, errors will be raised unless ``error="ignore"``.
    validate_callable : bool, optional
        If True, ensures the final resolved object is callable.
        Raises TypeError if not.
    error : {"raise", "ignore"}, optional
        Error handling strategy when import or resolution fails.
        - ``"raise"``  : propagate ImportError / AttributeError.
        - ``"ignore"`` : return ``default`` instead of raising.
    verbose : bool, optional
        If True, logs debug information during import resolution.

    Attributes
    ----------
    _name : str
        Full dotted import path of the object to be imported.
    _package : str | None
        Optional base package for relative imports.
    _default : any
        Fallback object if resolution fails and ``silent=True``.
    _validate_callable : bool
        If True, ensures that the resolved object is callable.
    _silent : bool
        If True, suppresses import errors and returns ``_default``.
    _verbose : bool
        If True, logs debug messages during resolution.
    _resolved : any
        Cached resolved object (after import), or ``_NoValue`` if not yet attempted.
    _loaded : bool
        Whether resolution has been attempted.

    Notes
    -----
    - Use ``lazy_obj.resolved`` to force resolution.
    - Use ``lazy_obj.is_loaded`` to know if resolution has happened.
    - Use ``if lazy_obj:`` to check if the import *appears* available
      without triggering a heavy import.

    Examples
    --------
    Lazy load a function:

    >>> lazy_fn = LazyImport("math.sqrt")
    >>> result = lazy_fn(
    ...     16,
    ... )  # Triggers actual import of `math` and resolution of `sqrt`

    Lazy load a module:

    >>> lazy_np = LazyImport("numpy")
    >>> lazy_np.array([1, 2, 3])  # Triggers import of `numpy`

    Lazy load with fallback on error:

    >>> fallback = LazyImport("nonexistent.module", default=lambda: "safe", silent=True)
    >>> fallback()  # Returns "safe" without raising ImportError
    """  # noqa: D205

    # 🚀 Less memory per instance (no __dict__ overhead).
    __slots__ = (
        "_default",
        "_error",
        "_loaded",
        "_name",
        "_package",
        "_parent_module_globals",
        "_resolved",
        "_validate_callable",
        "_verbose",
        # "__dict__",  # ← optional
    )

    def __init__(
        self,
        name: str,
        package: str | None = "scikitplot",
        parent_module_globals: str | dict | None = None,
        default: any | None = _NoValue,
        validate_callable: bool = False,
        error: str = "raise",
        verbose: bool = False,
    ) -> None:
        super().__init__(str(name))
        if error not in {"raise", "ignore"}:
            raise ValueError(
                f"Invalid error value {error!r}; expected 'raise' or 'ignore'."
            )

        self._name = name
        self._package = package
        self._default = default
        self._validate_callable = validate_callable
        self._error = error
        self._verbose = verbose

        if parent_module_globals is None or isinstance(parent_module_globals, str):
            if self._package and self._package in sys.modules:
                self._parent_module_globals = vars(sys.modules[self._package])
            else:
                self._parent_module_globals = {}
        elif isinstance(parent_module_globals, dict):
            self._parent_module_globals = parent_module_globals
        else:
            raise TypeError("parent_module_globals must be a dict, str, or None")

        self._loaded = False
        self._resolved: any = _NoValue

    # ------------------------------------------------------------------
    # Core resolution
    # ------------------------------------------------------------------
    def clear_cache(self) -> None:
        """Clear the cached resolved object and reset the loaded flag."""
        self._loaded = False
        self._resolved = _NoValue

    def _resolve(self) -> any:
        """
        Perform the actual import and cache the result.

        Returns
        -------
        any
            The resolved module or attribute, or ``_default`` if resolution fails
            and ``error='ignore'`` was set.
        """
        if self._resolved is not _NoValue:
            return self._resolved

        obj = nested_import(
            name=self._name,
            package=self._package,
            default=self._default,
            validate_callable=self._validate_callable,
            error=self._error,
            verbose=self._verbose,
        )

        # Insert into parent module's globals under the last path component
        try:
            simple_name = self._name.lstrip(".").split(".")[-1]
            self._parent_module_globals[simple_name] = obj
        except Exception:
            pass

        # Only register real modules in sys.modules
        if isinstance(obj, ModuleType):
            sys.modules[self._name] = obj

        self._resolved = obj
        self._loaded = True

        if self._verbose:
            logger.info(f"[LazyImport] Resolved {self._name!r} -> {obj!r}")
        return obj

    @property
    def resolved(self) -> any:
        """
        Force resolution and return the resolved object.

        Returns
        -------
        any
            The resolved module or attribute, or ``_default`` if resolution fails
            and ``error='ignore'`` was set.
        """
        return self._resolve()

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        """
        Return whether resolution has been attempted (regardless of success).

        Returns
        -------
        bool
        """
        return self._loaded

    @property
    def is_resolved(self) -> bool:
        """
        Return True if the target object was successfully resolved.

        This triggers resolution if it has not occurred yet.

        Returns
        -------
        bool
        """
        try:
            return self.resolved is not _NoValue
        except Exception:
            return False

    @property
    def resolved_type(self) -> type:
        """
        Return the type of the resolved object, or ``NoneType`` if unresolved.

        Returns
        -------
        type
        """
        try:
            return type(self.resolved)
        except Exception:
            return type(None)

    # ------------------------------------------------------------------
    # Representation & calling
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - representation only
        """
        Developer-friendly representation of the LazyImport instance.

        Returns
        -------
        str
            A string showing the import path and resolution state.
        """
        try:
            if self._resolved is _NoValue:
                resolved_type = "Unresolved"
                source = "?"
                status = "(Not loaded yet)"
            else:
                resolved_type = type(self._resolved).__name__
                source = _get_source_path(self._resolved)
                status = "(Loaded)"
            return f"<LazyImport → {resolved_type!r} {self._name!r} from {source!r} {status}>"
        except Exception as e:
            return f"<LazyImport (repr error: {e}) {self._name!r}>"

    def __call__(self, *args: any, **kwargs: any) -> any:
        """
        Call the resolved object if it is callable.

        Parameters
        ----------
        *args : tuple
            Positional arguments to pass to the callable.
        **kwargs : dict
            Keyword arguments to pass to the callable.

        Returns
        -------
        any
            The result of calling the imported object.

        Raises
        ------
        TypeError
            If the resolved object is not callable.
        """
        obj = self.resolved
        if callable(obj):
            return obj(*args, **kwargs)
        return obj

    # ------------------------------------------------------------------
    # Boolean / attribute behavior
    # ------------------------------------------------------------------
    def __bool__(self) -> bool:
        """
        Evaluate the truthiness of the lazy import *without* forcing a heavy import.

        Behavior
        --------
        - If the object has already been resolved, return ``bool(resolved)``.
        - If not yet resolved, return True if the top-level module appears
          importable according to ``importlib.util.find_spec``, otherwise False.

        This allows patterns like:

        >>> lazy_np = LazyImport("numpy")
        >>> if lazy_np:
        ...     # numpy appears to be available, now you may import/use it
        ...     arr = lazy_np.resolved.array([1, 2, 3])

        Returns
        -------
        bool
        """
        if self._loaded:
            try:
                return bool(self._resolved)
            except Exception:
                return False

        # Not loaded yet: cheap availability check via find_spec
        import importlib.util  # noqa: PLC0415

        dotted = self._name.lstrip(".")
        root = dotted.split(".", 1)[0]

        # For relative imports, check "<package>.<root>" if package is set
        candidate = root
        if self._name.startswith(".") and self._package:
            candidate = f"{self._package}.{root}"

        try:
            spec = importlib.util.find_spec(candidate)
        except Exception:
            return False

        return spec is not None

    def __dir__(self) -> list[str]:
        """
        Return attributes of the resolved object (forces resolution).

        This is primarily for interactive exploration and tab completion.
        """
        try:
            return sorted(set(dir(self.resolved)))
        except Exception:
            return sorted(super().__dir__())

    def __getattr__(self, attr: str) -> any:
        """
        Delegate attribute access to the resolved object.

        Raises
        ------
        AttributeError
            If the attribute is not found on the resolved object.
        """
        resolved = self._resolved
        if resolved is _NoValue:
            resolved = self._resolve()
        return getattr(resolved, attr)

    # ------------------------------------------------------------------
    # Hash / equality
    # ------------------------------------------------------------------
    def __hash__(self) -> int:
        """Hash based on the import path."""
        return hash(self._name)

    def __eq__(self, other: any) -> bool:
        """
        Compare this LazyImport to another object.

        - If ``other`` is another LazyImport, compare their resolved values.
        - Otherwise, compare the resolved object to ``other``.
        """
        try:
            if isinstance(other, LazyImport):
                return self.resolved == other.resolved
            return self.resolved == other
        except Exception:
            return False


# --- Safe Import ---
@lru_cache(maxsize=128)
def safe_import(
    module_name: str,
    *,
    error: str = "raise",
) -> ModuleType | None:
    """
    Dynamically import a module by name with error handling and caching.

    Parameters
    ----------
    module_name : str
        Name of the module to import.
    error : {"raise", "ignore"}, default "raise"
        Error handling strategy when import fails.
        - "raise"  : propagate ImportError with additional context.
        - "ignore" : return None instead of raising.

    Returns
    -------
    module : types.ModuleType or None
        Imported module object, or None if the module cannot be imported and
        ``error="ignore"`` is used.

    Raises
    ------
    ValueError
        If ``error`` is not "raise" or "ignore".
    ImportError
        If the specified module cannot be imported and ``error="raise"``.

    Examples
    --------
    >>> os_mod = safe_import("os")
    >>> math_mod = safe_import("math")
    >>> maybe_mod = safe_import("nonexistent_mod", error="ignore")
    >>> maybe_mod is None
    True
    """
    if error not in {"raise", "ignore"}:
        raise ValueError(
            f"Invalid error value {error!r}; expected 'raise' or 'ignore'."
        )

    try:
        return import_module(module_name)
    except ImportError as exc:
        if error == "ignore":
            return None
        raise ImportError(
            f"Required module {module_name!r} is not installed or could not be imported: {exc}"
        ) from exc


# First, the top-level packages:
# TODO: This list is a duplicate of the dependencies in pyproject.toml "all", but
# some of the package names are different from the pip-install name (e.g.,
# beautifulsoup4 -> bs4).
# Some optional parts of the standard library also find a place here,
# but don't appear in pyproject.toml
# ----------------------------------------------------------------------
# Optional dependency groups (usage-based, for developers & extras)
# ----------------------------------------------------------------------
# Core numeric / array APIs used by algorithms and computations
_CORE_NUMERIC_DEPS = [
    "numpy",
    "scipy",
    "mpmath",
    "bottleneck",
    "array_api_strict",
]

# Tabular / dataframe-like structures
_DATAFRAME_DEPS = [
    "pandas",
]

# Plotting and rendering backends
_PLOTTING_DEPS = [
    "matplotlib",
    "pytest_mpl",  # plotting tests / image comparison
    "aggdraw",  # Anti-Grain Geometry (AGG) graphics library
    "pillow",  # PIL / image IO for plots, thumbnails, etc.
]

# IO, storage, filesystem / cloud access (beyond pure stdlib)
_IO_DEPS = [
    "pyarrow",  # columnar / parquet, arrow arrays
    "pyyaml",  # config / metadata
    "h5py",  # HDF5 storage
    "s3fs",  # S3-backed filesystems
    "fsspec",  # general filesystem abstraction
    "bz2",  # stdlib compression (but used like an optional feature)
    "lzma",  # stdlib compression
    "uncompresspy",
]

# HTML / markup parsing and sanitization
_HTML_DEPS = [
    "bs4",  # beautifulsoup4
    "html5lib",
    "lxml",
    "bleach",
]

# Astronomy / time / ephemeris related
_ASTRONOMY_DEPS = [
    "asdf_astropy",
    "jplephem",
    "skyfield",
]

# Notebook / interactive frontends
_NOTEBOOK_DEPS = [
    "ipykernel",
    "IPython",
    "ipywidgets",
    "ipydatagrid",
]

# UI / app frontends for interactive demos or tools
_UI_DEPS = [
    "gradio",
    "streamlit",
]

# Machine learning / ANN backends
_ML_DEPS = [
    "tensorflow",
    "keras",
    "annoy",  # ANN index (Spotify Annoy)
    "voyager",  # ANN index / vector search backend
]

# Miscellaneous / general-purpose optional deps that don't fit above
_MISC_DEPS = [
    "certifi",
    "dask",
    "pytz",
    "sortedcontainers",
]

# Build the canonical list of optional deps (preserve order, remove dups)
_optional_deps: list[str] = []
for group in (
    _CORE_NUMERIC_DEPS,
    _DATAFRAME_DEPS,
    _PLOTTING_DEPS,
    _IO_DEPS,
    _HTML_DEPS,
    _ASTRONOMY_DEPS,
    _NOTEBOOK_DEPS,
    _UI_DEPS,
    _ML_DEPS,
    _MISC_DEPS,
):
    for dep in group:
        if dep not in _optional_deps:
            _optional_deps.append(dep)

_deps: dict[str, str] = {name.upper(): name for name in _optional_deps}
_deps["PLT"] = "matplotlib"  # short alias; gives HAS_PLT

__all__ = [f"HAS_{key}" for key in _deps]


@lru_cache(maxsize=128)
def __getattr__(name: str) -> bool:
    """
    Lazy attribute loader for feature flags indicating presence of dependencies.

    Accessing attributes like ``HAS_PANDAS`` or ``HAS_MATPLOTLIB`` returns
    a boolean indicating whether the corresponding optional dependency can
    be imported.

    Parameters
    ----------
    name : str
        Name of the attribute being accessed, e.g. ``"HAS_NUMPY"``.

    Returns
    -------
    bool
        True if the dependency is importable, False otherwise.

    Raises
    ------
    AttributeError
        If the requested attribute is not a recognized feature flag.

    Examples
    --------
    >>> HAS_NUMPY
    True  # if numpy is installed
    >>> HAS_PANDAS
    False  # if pandas is not installed
    >>> HAS_PLT
    True  # alias for matplotlib
    """
    if name in __all__:
        # Extract the dependency key by removing "HAS_" prefix
        dep_key = name.removeprefix("HAS_")
        target = _deps[dep_key]

        # Use importlib.util.find_spec to check existence without importing
        spec = find_spec(target)
        return spec is not None

    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}.")
