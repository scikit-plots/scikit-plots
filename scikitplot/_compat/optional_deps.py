# pylint: disable=broad-exception-caught
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-branches

# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Checks for optional dependencies using lazy import.

From `PEP 562 <https://www.python.org/dev/peps/pep-0562/>`_.
"""

import contextlib
import inspect
import sys
import types

# from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from typing import Optional

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
    package: "str | None" = "scikitplot",
    default: "any | None" = None,
    validate_callable: bool = False,
    silent: bool = True,
    verbose: bool = False,
) -> "types.ModuleType | callable | any | None":
    """
    Dynamically import a nested module or attribute from a dotted path (e.g., 'a.b.c').

    Supports both absolute and relative imports, fallback behavior, and validation of callables.

    Parameters
    ----------
    name : str
        Dotted path to the target object. Can be relative (e.g., '.sub.attr') or
        absolute (e.g., 'scikitplot.sub.attr').
    package : str, optional
        Anchor package for resolving relative imports (when `name` starts with '.').
        Default is 'scikitplot'.
    default : Any, optional
        Value to return if the import fails and `silent=True`.
        Default is None.
    validate_callable : bool, optional
        If True, raises ValueError if the final imported object is not callable.
    silent : bool, optional
        If True, suppresses errors and returns `default` on failure.
    verbose : bool, optional
        If True, logs additional debug information.
        Default is False.

    Returns
    -------
    module_or_obj : types.ModuleType, callable, Any or None
        Imported object or `default` if import fails and `silent` is True.

    Raises
    ------
    ValueError
        If `validate_callable` is True and the imported object is not callable.
    ImportError
        If import fails and `silent` is False.

    Notes
    -----
    - Uses `importlib.import_module` for both absolute and relative imports.
    - Supports importing deeply nested attributes (e.g., functions or classes).
    - Relative imports require a non-empty `package` argument.
    - Does not cache imported modules; uses standard import system behavior.
    - Absolute import: import_module(name='scikitplot', package=None) only load top-level
    - Relative import: import_module(name='.', package='scikitplot') support load submodule
    - Relative import: import_module(name='.', package='scikitplot.api') support load submodule

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
        # Normalize dotted path by removing leading dot (if relative)
        dotted_path = name.lstrip(".")  # Remove leading dot for relative paths
        parts = dotted_path.split(".")  # Split into module/attribute parts

        # For relative imports or internal package shortcuts, prepend package name
        if package and parts[0] != package:  # or is_relative
            # parts.insert(0, 5)  # Insert 5 at index 0
            parts = [package, *parts]  # handle for module "internal relatives"

        module = default  # Will hold the successfully imported module
        i = 0  # module_prefix_len : Will track how many parts were used for module import

        # Try importing the longest possible module path (from full to partial)
        for i in reversed(range(1, len(parts) + 1)):
            # Try importing the longest valid prefix as a module
            module_path = ".".join(parts[:i])
            try:
                # module = import_module(
                #     name=f".{module_path}" if is_relative else module_path,
                #     package=package,
                # )
                module = import_module(
                    name=module_path,
                )
                # if isinstance(module, LazyImport):
                #     module = module.resolved
                # a module, If successfully imported
                break
            except ModuleNotFoundError:
                continue  # Try a shorter prefix
            except ScikitplotException as e:
                logger.exception(f"Unexpected error while importing {module_path!r}")
                raise e

        if module is default:
            # No module could be resolved
            # developer-friendly string (with quotes, escapes, etc.) "Format this using repr()"
            msg = f"Could not import any module prefix from {name!r} using package {package!r}"
            logger.info(msg)
            if not silent:
                raise ImportError(msg)

        # Resolve remaining dotted parts as attributes on the module (if any)
        try:
            # Resolve the rest of the dotted path as attributes
            for attr in parts[i:]:
                # if not hasattr(module, attr):
                # raise AttributeError
                module = getattr(module, attr)
        except (AttributeError, RecursionError) as e:
            logger.exception(
                f"Attribute {attr!r} not found in {module!r} ({type(e).__name__!r}: {e})"
            )
            module = default  # Will hold the successfully getattr
            if not silent:
                raise e
        except ScikitplotException as e:
            logger.exception(f"Unexpected error while importing {module_path!r}")
            raise e

        # If required, ensure the final object is callable
        if validate_callable and not callable(module):
            msg = f"Imported object {name!r} is not callable"
            logger.info(msg)
            raise ValueError(msg)

        if verbose:
            logger.info(f"Successfully imported {name!r}: {module!r}")
        return module

    except ScikitplotException as e:
        logger.exception(f"Import failed for {name!r}: {e}")
        if not silent:
            raise e


# --- Lazily Import ---


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
    return "<?>"


class LazyImport(types.ModuleType):
    """
    Lazily import (No load, fully lazy) a module, or function or attribute object
    upon first time access or invocation by its dotted path.

    Delay the import of a dotted module or attribute path (e.g., `a.b.c`)
    until it is first accessed or called.

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
    - Caches the result after first resolution.
    - Supports equality, hashing, boolean context.
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
        name: str,
        package: "str | None" = "scikitplot",
        parent_module_globals: "str | dict | None" = None,
        default: "any | None" = _NoValue,
        validate_callable: bool = False,
        silent: bool = True,
        verbose: bool = False,
    ):
        """
        Reduce initial load time or avoid importing heavy dependencies unless needed.

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
        """  # noqa: D205

        super().__init__(str(name))  # self.__name__,  Full dotted import path
        self._name = name  # Full dotted import path
        self._package = package  # Optional package for relative imports
        self._default = default  # Fallback value if import fails
        self._validate_callable = validate_callable  # Enforce callable type
        self._silent = silent  # Suppress errors
        self._verbose = verbose  # Log debug info

        if parent_module_globals is None or isinstance(parent_module_globals, str):
            self._parent_module_globals = (
                vars(sys.modules[self._package]) if self._package in sys.modules else {}
            )
        elif isinstance(parent_module_globals, dict):
            self._parent_module_globals = parent_module_globals
        else:
            raise TypeError

        # Import status flag, object is called or accessed triggers import
        self._loaded = False
        self._resolved: Optional[any] = _NoValue  # Cached resolved object

    def clear_cache(self):
        """
        Clear the cached resolved object.

        Use this to force re-importing on the next access. Helpful for testing,
        debugging, or reloading updated modules.
        """
        self._loaded = False
        self._resolved = _NoValue

    def _resolve(self) -> "types.ModuleType | callable | None":
        """
        Perform the actual import and cache the result.

        Returns
        -------
        types.ModuleType or Callable or None
            The resolved module or attribute, or None if resolution fails.
        """
        if self._resolved is not _NoValue:
            # If already loaded, return the loaded module.
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
        logger.info(f"Loaded '{self._package}.{self._name}' as Module: {module!r}")
        # prevent RecursionError
        # if isinstance(module, LazyImport):
        #     module = module.resolved

        # Import the target module and insert it into the parent's namespace
        # module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._name.strip(".")] = module
        sys.modules[self._name] = module

        # Ignore errors in custom proxies
        # Cache module attributes to speed up repeated lookups
        # e.g., numpy.ufunc or other builtins don't have __dict__
        with contextlib.suppress(Exception):
            if hasattr(self, "__dict__"):
                # Update this object's dict so that if someone keeps a reference to the `LazyLoader`
                # lookups are efficient (`__getattr__` is only called on lookups that fail).
                self.__dict__.update(module.__dict__)  # ← Requested enhancement

        # Cache and return the resolved object to reuse.
        self._resolved = module
        self._loaded = True
        return self._resolved

    @property
    def resolved(self) -> "types.ModuleType | callable | None":
        """
        Public accessor for the resolved object.

        Returns
        -------
        types.ModuleType or Callable or None
            The resolved module or attribute, or None if resolution fails.

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

        Use in boolean context:
        >>> bool(LazyImport("math"))  # True, once loaded
        True

        Handle missing modules safely:
        >>> bool(LazyImport("missing.module", default=False, silent=True))
        False
        """
        try:
            return self._resolve()
        except (AttributeError, ImportError, ModuleNotFoundError) as e:
            logger.exception(f"{e}")
            raise e

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

        Use in boolean context:
        >>> bool(LazyImport("math"))  # True, once loaded
        True

        Handle missing modules safely:
        >>> bool(LazyImport("missing.module", default=False, silent=True))
        False
        """
        try:
            return self.resolved is not _NoValue
        except Exception:
            return False

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

    # @property
    # def __path__(self):
    #     """
    #     Required if this is a package, for relative imports to work.
    #
    #      __path__ is ONLY for Packages:
    #      If a module is just a .py file, like module1.py, it does not have __path__.
    #      If a module is a directory with an __init__.py file, it is a package,
    #      and gets __path__ automatically.
    #     """
    #     return getattr(self.resolved, "__path__", [])

    # @property
    # def __doc__(self):
    #     # Adding __doc__ property = lazy until docstring requested, then eager on demand.
    #     # No way to have real docstring on proxy without importing the target.
    #     # Force load the real object and return its docstring
    #     # Docstring access (? or help()):
    #     try:
    #         help(self.resolved)
    #         # doc = help(self.resolved)
    #         # doc = inspect.getdoc(self.resolved)
    #         return getattr(self.resolved, "__doc__", "")  # returns not raise
    #     except Exception:
    #         return "No docstring."

    # @property
    # def __class__(self):
    #     """Make isinstance and others work properly."""
    #     # Forward to resolved object's class
    #     try:
    #         return self.resolved.__class__
    #     except Exception:
    #         return super().__class__

    def __repr__(self) -> str:
        """
        Developer-friendly representation of the LazyImport instance.

        Returns
        -------
        str
            A string showing the import path and resolution state.
        """
        try:
            # Avoid calling self.resolved if it's not ready, just show type if safely available
            resolved = self.__dict__.get("_resolved", _NoValue)
            if resolved is None or resolved is _NoValue:
                resolved_type = "Unresolved"
                source = "?"
                status = "(Not loaded yet)"
            else:
                resolved_type = type(resolved).__name__
                source = _get_source_path(resolved)
                status = "(Not loaded yet)"

            return f"<LazyImport → {resolved_type!r} {self._name!r} from {source!r} {status}>"

        except Exception as e:
            return f"<LazyImport (repr error: {e}) {self._name!r}>"

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
        return obj
        # or ?
        # raise TypeError(
        #     f"Lazy-imported object '{self._name}' is not callable"
        # )

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

        Use in boolean context:
        >>> bool(LazyImport("math"))  # True, once loaded
        True

        Handle missing modules safely:
        >>> bool(LazyImport("missing.module", default=False, silent=True))
        False
        """
        try:
            # Prevent recursive loop by checking for sentinel and resolving safely
            # Don't call bool() on another LazyImport
            # if isinstance(self.resolved, LazyImport):
            #     return True  # or False, depending on design
            return self.resolved not in [None, _NoValue]
        except ScikitplotException:
            return False

    def __dir__(self) -> "list[str]":
        """
        Provide intelligent tab autocompletion support.

        Returns
        -------
        list of str
            A sorted list of available attributes from the resolved object.
        """
        # return sorted(set(super().__dir__()) | {"is_loaded", "resolved"})
        return sorted(set(dir(self.resolved)))

    # def _ipython_key_completions_(self):
    #     """IPython tab completion support."""
    #     return dir(self)

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
        resolved = self.__dict__.get("_resolved", _NoValue)
        if resolved is _NoValue:
            resolved = self._resolve()
        return getattr(resolved, attr)  # raise AttributeError

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

    # def __getstate__(self) -> dict:
    #     """
    #     Prepare the LazyImport instance for pickling.

    #     Returns
    #     -------
    #     dict
    #         A dictionary representing the serializable state,
    #         including attributes from __slots__ and __dict__ if present.
    #     """
    #     state = {}

    #     # Collect attributes from __slots__, if defined
    #     slots = getattr(self, "__slots__", [])  # returns [] not raise
    #     for slot in slots:
    #         # __slots__ can be a single string or iterable of strings (in case of inheritance)
    #         if isinstance(slot, str):
    #             # Ignore errors in custom
    #             with contextlib.suppress(AttributeError):
    #                 state[slot] = getattr(self, slot)  # raises AttributeError
    #         else:  # if slot is iterable (multiple inheritance)
    #             for s in slot:
    #                 # Ignore errors in custom
    #                 with contextlib.suppress(AttributeError):
    #                     state[s] = getattr(self, s)  # raises AttributeError

    #     # Include attributes from __dict__ (dynamic attributes) if available
    #     if hasattr(self, "__dict__"):
    #         state.update(self.__dict__)

    #     return state

    # def __setstate__(self, state: dict) -> None:
    #     """
    #     Restore the LazyImport instance from the pickled state.

    #     Parameters
    #     ----------
    #     state : dict
    #         Dictionary of saved attributes to restore.
    #         The dictionary representing the serialized state.
    #     """
    #     for key, value in state.items():
    #         setattr(self, key, value)

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
    "scikitplot",
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
        logger.info(f"name {name}")
        # Extract the dependency key by removing "HAS_" prefix
        dep_key = name.removeprefix("HAS_")
        # Check if the dependency module can be found
        # Use importlib.util.find_spec to check existence
        return find_spec(_deps[dep_key]) is not None

    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}.")
