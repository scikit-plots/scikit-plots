# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Checks for optional dependencies using lazy import.

From `PEP 562 <https://www.python.org/dev/peps/pep-0562/>`_.
"""

# pylint: disable=broad-exception-caught
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-branches

import contextlib
import inspect
import types
from importlib import import_module
from importlib.util import find_spec
from typing import Optional

from .. import logger

# from functools import lru_cache
from .._compat.python import lru_cache
from ..exceptions import ScikitplotException

# Returns the top-level package/module, not the submodule.
# __import__(f'{__name__}.{name}')  # low-level function
# Returns the submodule directly.
# __import__(module_name, fromlist=[class_name])
# import_module(f".{name}", package=package)  # high-level function

# --- nested module or attribute Import ---


def _get_source_path(obj):
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
        The file path to the source code, or '?' if it cannot be determined.

    Notes
    -----
    - For modules, `__file__` is returned.
    - For callables, `inspect.unwrap` is used to get past multiple decorators.
    - If the object is a built-in or C extension, the path may point to a
      `.so`, `.pyd`, or `.dll` file.
    - If the object has a `__code__` attribute, it is used as a fallback.
    - If none of these methods succeed, `'?'` is returned.

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
    if isinstance(obj, types.ModuleType):
        # Use __file__ if available, fallback to '?'
        return getattr(obj, "__file__", "?")  # returns '?' not raise

    # Check if the object is callable (function, method, class, etc.)
    if callable(obj):
        # Handle custom proxy types like LazyImport that define get_real_func()
        if hasattr(obj, "get_real_func") and callable(obj.get_real_func):
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

        # Fallback: try using the code object (for Python-defined functions)
        if hasattr(obj, "__code__"):
            try:
                # lineno = obj.__code__.co_firstlineno
                return inspect.getfile(obj.__code__)
            except Exception:
                pass  # __code__ exists but may not point to a valid file

    # If all methods fail, return unknown
    return "?"


# Try importing as a submodule
@lru_cache(maxsize=128)
def nested_import(  # noqa: PLR0912
    name: "str | None" = ".",
    package: "str | None" = "scikitplot",
    default: "any | None" = None,
    validate_callable: bool = False,
    silent: bool = True,
    verbose: bool = False,
) -> "types.ModuleType | callable | None":
    """
    Dynamically import a nested module or attribute from a dotted path like `a.b.c.d`.

    Attempts to import a deeply nested module or object using either a relative
    or absolute dotted string path. Supports fallbacks and optional validation.

    Parameters
    ----------
    name : str
        Dotted path to the target object. Can be relative (e.g., '.sub.attr') or
        absolute (e.g., 'scikitplot.sub.attr') name of the module to import.
        Default is '.' (current package) relative to 'scikitplot' refers to 'scikitplot' itself.
    package : str, optional
        Base package for resolving relative imports (e.g., 'scikitplot').
        Required because Python uses it as the anchor point only for relative imports
        (i.e., when name starts with a dot '.').
        Default is 'scikitplot'.
    default : Any, optional
        Value to return on failure. Defaults to None.
        Sample: `lambda *a, **kw: None`.
    validate_callable : bool, optional
        If True, raises ValueError if the final imported object is not callable.
    silent : bool, optional
        If True, suppresses exceptions and returns `default` on failure.
    verbose : bool, optional
        If True, logs info/debug messages.
        Default is False.

    Returns
    -------
    module_or_obj : types.ModuleType, callable or Any
        The imported object or `default` if not found and `silent=True`.

    Raises
    ------
    ModuleNotFoundError
        If the module could not be imported and `silent=False`.
    AttributeError
        If an attribute in the path doesn't exist and `silent=False`.
    ValueError
        If `validate_callable=True` and the object is not callable.

    Notes
    -----
    - Relative imports (starting with '.') require a non-empty `package`.
    - If import fails and `silent=True`, the `default` is returned.
    - Objects returned can be modules, functions, classes, etc.
    - Relative import: import_module(name='.', package='scikitplot') support load submodule
    - Relative import: import_module(name='.', package='scikitplot.api') support load submodule
    - Absolute import: import_module(name='scikitplot', package=None) only load top-level

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
    try:
        # Determine whether import is relative (starts with '.')
        is_relative = name.startswith(".")
        # Require `package` for relative imports
        if is_relative and not package:
            raise TypeError(
                f"the 'package' argument is required to perform a relative import for {name!r}"
            )
        # Normalize the dotted path and split into parts (for relative imports)
        dotted_path = name.lstrip(".")  # Remove leading dot for relative paths
        parts = dotted_path.split(".")  # Split into module/attribute parts

        # Handle fully-qualified paths redundantly prefixed with package name
        if not is_relative and package and parts[0] == package:
            parts = parts[1:]

        module = None  # Will hold the successfully imported module
        i = 0  # Will track how many parts were used for module import

        # Attempt to import the longest valid resolvable module prefix
        for i in reversed(range(1, len(parts) + 1)):
            # Try importing the longest valid prefix as a module
            module_path = ".".join(parts[:i])
            try:
                # Aleways use relative import to handle for "internal relatives"
                module = import_module(
                    name=f".{module_path}",  # if is_relative module_path,
                    package=package,
                )
                # a module, If successfully imported
                break
            except ModuleNotFoundError:
                continue  # Try a shorter prefix
        else:
            # No module could be resolved
            # developer-friendly string (with quotes, escapes, etc.) "Format this using repr()"
            msg = f"Could not import any module {name!r} from {package!r}"
            logger.exception(msg)
            if silent:
                return default
            raise ModuleNotFoundError(msg)

        # Resolve remaining dotted parts as attributes on the module (if any)
        for attr in parts[i:]:
            # if hasattr(module, attr):
            # raise AttributeError
            module = getattr(module, attr, default) if silent else getattr(module, attr)

        # If required, ensure the final object is callable
        if validate_callable and not callable(module):
            msg = f"Imported object {name!r} is not callable"
            logger.error(msg)
            raise ValueError(msg)

        if verbose:
            logger.info(f"Successfully loaded {name!r}")
        return module

    except ScikitplotException as e:
        logger.exception(f"Import failed for {name!r}: {e}")
        if silent:
            return default
        raise e


# --- Lazily Import ---


class LazyImport(types.ModuleType):
    """
    Lazily import (No load, fully lazy) a module, or function or attribute object
    upon first time access or invocation by its dotted path.

    This class delays the import of a dotted module or attribute path (e.g., `a.b.c`)
    until it is first accessed or called. This is useful to reduce initial load time
    or avoid importing heavy dependencies unless needed.

    Parameters
    ----------
    name : str
        The dotted import path of the target module or attribute to load
        (e.g., "pandas.DataFrame").
    package : str, optional
        Root package used to resolve relative imports.
        Only relevant if `name` is a relative import (e.g., ".utils").
    default : Any, optional
        A fallback value to return if the import fails.
        If None, errors will be raised unless `silent=True`.
    validate_callable : bool, optional
        If True, ensures the final resolved object is callable.
        Raises TypeError if not.
    silent : bool, optional
        If True, suppresses import errors and returns `default` instead.
        Default is True.
    verbose : bool, optional
        If True, logs debug information during import resolution.

    Attributes
    ----------
    _name : str
        Dotted import path of the object to be imported.
    _package : str
        Optional base package for relative imports.
    _default : Any
        Fallback object if import fails and silent=True.
    _validate_callable : bool
        If True, ensures that the resolved object is callable.
    _silent : bool
        If True, suppresses import errors and returns default.
    _verbose : bool
        If True, logs debug messages during resolution.
    _resolved : Any
        Cached resolved object (after import).
    _loaded : bool
        Whether the object has been resolved and cached.
    is_loaded : bool
        True if the import has been performed and cached, False otherwise.

    Notes
    -----
    - Caches the result after first resolution.
    - Supports equality, hashing, boolean context.

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

    Notes
    -----
    - Use `lazy_obj.is_loaded` to check if the object has been loaded.
    - Use `repr()` or `dir()` for interactive exploration.
    """  # noqa: D205

    # 🚀 Less memory per instance (no __dict__ overhead).
    # __slots__ = (
    #     "_name",
    #     "_package",
    #     "_default",
    #     "_validate_callable",
    #     "_silent",
    #     "_verbose",
    #     "_resolved",
    #     "_loaded",
    #     # "__dict__",  # ← optional
    # )

    def __init__(
        self,
        name: "str | None" = ".",
        package: "str | None" = "scikitplot",
        default: "any | None" = None,
        validate_callable: bool = False,
        silent: bool = True,
        verbose: bool = False,
    ):
        super().__init__(name)
        self._name = name  # Full dotted import path
        self._package = package  # Optional package for relative imports
        self._default = default  # Fallback value if import fails
        self._validate_callable = validate_callable  # Enforce callable type
        self._silent = silent  # Suppress errors
        self._verbose = verbose  # Log debug info

        self._resolved: Optional[any] = None  # Cached resolved object
        self._loaded = (
            False,  # Import status flag, object is called or accessed triggers import
        )

    def _resolve(self) -> "types.ModuleType | callable | None":
        """
        Perform the actual import and cache the result.

        Returns
        -------
        types.ModuleType or Callable or None
            The resolved module or attribute, or None if resolution fails.
        """
        if self._loaded:
            return self._resolved
        # Resolve module/attr
        module = nested_import(
            self._name,
            self._package,
            self._default,
            self._validate_callable,
            self._silent,
            self._verbose,
        )
        # Cache and return the resolved object
        self._resolved = module
        if hasattr(self, "__dict__"):
            # Ignore errors in custom proxies
            # Cache module attributes to speed up repeated lookups
            # e.g., numpy.ufunc or other builtins don't have __dict__
            with contextlib.suppress(Exception):
                self.__dict__.update(module.__dict__)  # ← Requested enhancement
        self._loaded = True
        return module

    @property
    def resolved(self) -> any:
        """
        Public accessor for the resolved object.

        Returns
        -------
        Any
            The resolved object or fallback if `silent=True`, or None if resolution fails.

        Examples
        --------
        >>> lazy = LazyImport("math.sqrt")

        Before resolution:
        >>> lazy.is_loaded
        False

        Trigger resolution:
        >>> print(lazy.resolved_type)
        <class 'builtin_function_or_method'>

        Accessing resolved value:
        >>> print(lazy.resolved(25))
        5.0

        After resolution:
        >>> lazy.is_loaded
        True
        >>> lazy.is_resolved
        True
        >>> lazy.is_missing
        False

        Use in boolean context:
        >>> bool(LazyImport("math"))  # True, once loaded
        True

        Handle missing modules safely:
        >>> bool(LazyImport("missing.module", default=False, silent=True))
        False
        """
        try:
            return self._resolve()
        except Exception:
            return None

    def clear_cache(self):
        """
        Clear the cached resolved object.

        Use this to force re-importing on the next access. Helpful for testing,
        debugging, or reloading updated modules.
        """
        self._resolved = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """
        Return whether the import attempt has been made.

        Returns
        -------
        bool
            True if the object has been attempted to load (regardless of success), False otherwise.

        Examples
        --------
        >>> lazy = LazyImport("math.sqrt")

        Before resolution:
        >>> lazy.is_loaded
        False

        Trigger resolution:
        >>> print(lazy.resolved_type)
        <class 'builtin_function_or_method'>

        Accessing resolved value:
        >>> print(lazy.resolved(25))
        5.0

        After resolution:
        >>> lazy.is_loaded
        True
        >>> lazy.is_resolved
        True
        >>> lazy.is_missing
        False

        Use in boolean context:
        >>> bool(LazyImport("math"))  # True, once loaded
        True

        Handle missing modules safely:
        >>> bool(LazyImport("missing.module", default=False, silent=True))
        False
        """
        return self._loaded

    @property
    def is_resolved(self) -> bool:
        """
        Return True if the target object was successfully resolved.

        Returns
        -------
        bool
            True if resolved to a non-None object, False otherwise.

        Examples
        --------
        >>> lazy = LazyImport("math.sqrt")

        Before resolution:
        >>> lazy.is_loaded
        False

        Trigger resolution:
        >>> print(lazy.resolved_type)
        <class 'builtin_function_or_method'>

        Accessing resolved value:
        >>> print(lazy.resolved(25))
        5.0

        After resolution:
        >>> lazy.is_loaded
        True
        >>> lazy.is_resolved
        True
        >>> lazy.is_missing
        False

        Use in boolean context:
        >>> bool(LazyImport("math"))  # True, once loaded
        True

        Handle missing modules safely:
        >>> bool(LazyImport("missing.module", default=False, silent=True))
        False
        """
        try:
            return self.resolved is not None
        except Exception:
            return False

    @property
    def is_missing(self) -> bool:
        """
        Return True if the object is missing (import failed or resolved to None).

        Returns
        -------
        bool
            True if import failed or resolved to None, False otherwise.

        Examples
        --------
        >>> lazy = LazyImport("math.sqrt")

        Before resolution:
        >>> lazy.is_loaded
        False

        Trigger resolution:
        >>> print(lazy.resolved_type)
        <class 'builtin_function_or_method'>

        Accessing resolved value:
        >>> print(lazy.resolved(25))
        5.0

        After resolution:
        >>> lazy.is_loaded
        True
        >>> lazy.is_resolved
        True
        >>> lazy.is_missing
        False

        Use in boolean context:
        >>> bool(LazyImport("math"))  # True, once loaded
        True

        Handle missing modules safely:
        >>> bool(LazyImport("missing.module", default=False, silent=True))
        False
        """
        return not self.is_resolved

    @property
    def resolved_type(self) -> type:
        """
        Return the type of the resolved object, or type(None) `NoneType` if unresolved.

        Returns
        -------
        type
            The type of the resolved object, or `NoneType` if unresolved.

        Examples
        --------
        >>> lazy = LazyImport("math.sqrt")

        Before resolution:
        >>> lazy.is_loaded
        False

        Trigger resolution:
        >>> print(lazy.resolved_type)
        <class 'builtin_function_or_method'>

        Accessing resolved value:
        >>> print(lazy.resolved(25))
        5.0

        After resolution:
        >>> lazy.is_loaded
        True
        >>> lazy.is_resolved
        True
        >>> lazy.is_missing
        False

        Use in boolean context:
        >>> bool(LazyImport("math"))  # True, once loaded
        True

        Handle missing modules safely:
        >>> bool(LazyImport("missing.module", default=False, silent=True))
        False
        """
        try:
            return type(self.resolved)
        except Exception:
            return type(None)

    def __bool__(self) -> bool:
        """
        Evaluate the truthiness of the resolved object.

        Allows the LazyImport instance to be used in boolean contexts like
        `if`, `while`, or logical operations. Resolution is triggered on
        first use. If resolution fails and a `default` is provided, its
        truthiness will be used instead.

        Returns
        -------
        bool
            True if the resolved object is truthy. Falls back to `default`
            if resolution fails and `silent=True`.

        Examples
        --------
        >>> lazy = LazyImport("math.sqrt")

        Before resolution:
        >>> lazy.is_loaded
        False

        Trigger resolution:
        >>> print(lazy.resolved_type)
        <class 'builtin_function_or_method'>

        Accessing resolved value:
        >>> print(lazy.resolved(25))
        5.0

        After resolution:
        >>> lazy.is_loaded
        True
        >>> lazy.is_resolved
        True
        >>> lazy.is_missing
        False

        Use in boolean context:
        >>> bool(LazyImport("math"))  # True, once loaded
        True

        Handle missing modules safely:
        >>> bool(LazyImport("missing.module", default=False, silent=True))
        False
        """
        try:
            return bool(self.resolved)
        except ScikitplotException:
            return bool(self._default)

    def __call__(self, *args, **kwargs) -> any:
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
        Any
            The result of calling the imported object.

        Raises
        ------
        TypeError
            If the resolved object is not callable.
        """
        obj = self.resolved
        if callable(obj):
            return obj(*args, **kwargs)
        # or ?
        return obj
        # raise TypeError(
        #     f"Lazy-imported object '{self._name}' is not callable"
        # )

    def __repr__(self) -> str:
        """
        Developer-friendly representation of the LazyImport instance.

        Returns
        -------
        str
            A string showing the import path and resolution state.
        """
        # resolved_type = type(self._resolved).__name__ if self._loaded else "unresolved"
        # getattr(self._resolved, '__module__', '?')
        return (
            f"<LazyImport → {type(self._resolved).__name__} "
            f"'{self._name}' from '{_get_source_path(self._resolved)}'>"
        )
        # return f"<LazyImport '{self._name}' (not loaded)>"

    def __getattr__(self, attr: str) -> any:
        """
        Delegate attribute access to the real resolved object.

        Parameters
        ----------
        attr : str
            The attribute name to access.

        Returns
        -------
        Any
            The attribute value from the resolved object.

        Raises
        ------
        AttributeError
            If the attribute is not found on the resolved object.
        """
        return getattr(self.resolved, attr, self)  # raise AttributeError

    @property
    def __doc__(self):
        # Adding __doc__ property = lazy until docstring requested, then eager on demand.
        # No way to have real docstring on proxy without importing the target.
        # Force load the real object and return its docstring
        # Docstring access (? or help()):
        try:
            help(self._resolved)
            # doc = help(self._resolved)
            # doc = inspect.getdoc(self._resolved)
            return getattr(self._resolved, "__doc__", "")  # returns not raise
        except Exception:
            return "No docstring."

    @property
    def __class__(self):
        """Make isinstance and others work properly."""
        # Forward to resolved object's class
        try:
            return self._resolved.__class__
        except Exception:
            return super().__class__

    def __dir__(self) -> list[str]:
        """
        Provide intelligent tab autocompletion support.

        Returns
        -------
        list of str
            A sorted list of available attributes from the resolved object.
        """
        try:
            return sorted(set(dir(self._resolved)))
        except Exception:
            return sorted(set(super().__dir__()) | {"is_loaded"})

    def _ipython_key_completions_(self):
        """IPython tab completion support."""
        return self.__dir__()

    @property
    def __path__(self):
        """Required if this is a package, for relative imports to work."""
        return getattr(self._resolved, "__path__", [])

    def __eq__(self, other: any) -> bool:
        """
        Compare this LazyImport to another object or LazyImport.

        If the object is another LazyImport, compares their resolved values.
        Otherwise, compares the resolved object to `other`.

        Parameters
        ----------
        other : Any
            Another object or LazyImport instance to compare against.

        Returns
        -------
        bool
            True if the resolved objects are equal, False otherwise.

        Examples
        --------
        >>> LazyImport("math") == importlib.import_module("math")  # True
        >>> LazyImport("math") == LazyImport("math")  # True
        >>> LazyImport("math") == "math"  # False
        """
        try:
            return self.resolved == (
                other.resolved if isinstance(other, LazyImport) else other
            )
        except ScikitplotException:
            return False

    def __hash__(self) -> int:
        """
        Return a hash based on the import path for the LazyImport instance.

        Makes the LazyImport object usable as a key in dictionaries or elements in sets.

        Returns
        -------
        int
            Hash of the module's dotted import name.

        Examples
        --------
        >>> lazy = LazyImport("math")
        >>> hash(lazy) == hash("math")  # True
        >>> {lazy: "cached"}  # Works in dictionaries
        """
        return hash(self._name)

    def __getstate__(self) -> dict:
        """
        Prepare the LazyImport instance for pickling.

        Returns
        -------
        dict
            A dictionary representing the serializable state,
            including attributes from __slots__ and __dict__ if present.
        """
        state = {}

        # Collect attributes from __slots__, if defined
        slots = getattr(self, "__slots__", [])  # returns [] not raise
        for slot in slots:
            # __slots__ can be a single string or iterable of strings (in case of inheritance)
            if isinstance(slot, str):
                # Ignore errors in custom
                with contextlib.suppress(AttributeError):
                    state[slot] = getattr(self, slot)  # raises AttributeError
            else:  # if slot is iterable (multiple inheritance)
                for s in slot:
                    # Ignore errors in custom
                    with contextlib.suppress(AttributeError):
                        state[s] = getattr(self, s)  # raises AttributeError

        # Include attributes from __dict__ (dynamic attributes) if available
        if hasattr(self, "__dict__"):
            state.update(self.__dict__)

        return state

    def __setstate__(self, state: dict) -> None:
        """
        Restore the LazyImport instance from the pickled state.

        Parameters
        ----------
        state : dict
            Dictionary of saved attributes to restore.
            The dictionary representing the serialized state.
        """
        for key, value in state.items():
            setattr(self, key, value)

    # def __del__(self):
    #     """
    #     Clean up cached state when the object is garbage collected.
    #     """
    #     self.clear_cache()

    # def _log(self, message: str, level: int = logger.DEBUG):
    #     if self._verbose:
    #         logger.log(level, f"[LazyImport] {message}")


# --- Safe Import ---


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
    # sample UI app
    "gradio",
    "streamlit",
    "pyyaml",
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
