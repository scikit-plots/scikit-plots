# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=unused-argument
# pylint: disable=broad-exception-caught
# pylint: disable=logging-fstring-interpolation

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
BaseDecorator Framework.

This module defines `BaseDecorator`, a flexible, feature-rich Python decorator
that wraps any function with:

- Execution logging (console and optional file-based logs)
- Performance profiling (execution time and memory usage)
- Async and multithreaded execution capabilities
- Singleton pattern for global configuration and state reuse
- Serialization support for saving/loading decorator configuration

Example usage:
---------------
>>> @BaseDecorator(verbose=True)
... def greet(name):
...     print(f"Hello, {name}!")
>>> greet("Alice")
"""  # pylint: disable=too-many-lines

## Standard library imports
import asyncio
import contextlib  # noqa: F401
import functools
import inspect
import io  # noqa: F401

## Standard library imports
# import logging
import os  # noqa: F401
import re  # noqa: F401
import sys  # noqa: F401
import threading
import time as _time
import tracemalloc
import warnings  # noqa: F401
import weakref
from abc import ABCMeta, abstractmethod  # noqa: F401
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path  # noqa: F401
from typing import TYPE_CHECKING

## Third-party imports
import joblib  # type: ignore[reportMissingModuleSource]

# import psutil  # type: ignore[reportMissingModuleSource]
## Local application/library imports
from .. import logger
from ..utils.utils_params import _resolve_args_and_kwargs

# Runtime-safe imports for type hints (avoids runtime overhead)
if TYPE_CHECKING:
    from collections.abc import Coroutine
    from typing import (
        ClassVar,
        Optional,
        TypeVar,
        Union,
    )

    # built-in convention type like T, U, and V
    # A generic function
    # A generic type representing a callable function
    # that can take any arguments and return any result.
    # callable[ArgType, ReturnType]
    # F = "callable[..., any]"
    F = TypeVar("F", bound="callable[..., any]")
    # A decorator that wraps a function
    # A type representing a function decorator
    # (a callable that takes a function as input and returns a function).
    # Decorator = 'callable[[F], F]'
    FuncDec = TypeVar("FuncDec", bound="callable[[F], F]")
    # A generic type variable for class decorator
    # A type representing a subclass of BaseDecorator
    # (used for singleton or decorator patterns).
    ClsDec = TypeVar("ClsDec", bound="BaseDecorator")
    # Later, use Union for functions that accept either
    DecoratorLike = TypeVar("DecoratorLike", bound="Union[F, FuncDec, ClsDec]")

## Set up basic logger configuration
# logging.basicConfig(level=_logging.INFO)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

######################################################################
## decorator_dispatcher
######################################################################


# Dispatch: A flexible decorator that can be used both with and without arguments.
# Case 1: Without Arguments (standard usage, simple):
#     - @decorate applied directly to a function (e.g., @decorate)
#     - The decorator is applied immediately, with `func` passed automatically.
# Case 2: With Arguments (parameterized usage, factory):
#     - @decorate(verbose=True) for customized behavior.
#     - Returns an `inner decorator` that can accept the function to decorate.
def decorator_dispatcher(
    # A direct decorator when callable is passed
    simple: "FuncDec",
    # A factory decorator (parameterized or class) when no callable is passed
    factory: "Union[callable[..., FuncDec], callable[..., ClsDec]]",
    *args: "tuple[any, ...]",  # Positional arguments
    **kwargs: "dict[str, any]",  # Keyword arguments
) -> "DecoratorLike":
    """
    Dispatch a decorator to be used with or without arguments (parentheses).

    This utility supports both forms of decorators:
    ::

        - @decorator
            When used without parentheses, the `simple` decorator is applied directly
            to the target function or class.

        - @decorator() or @decorator(...)
            When used with parentheses (even empty), the `factory` is invoked to produce
            and return a decorator that will then be applied to the target.

    Parameters
    ----------
    simple : callable
        A non-parameterized decorator applied directly to the function/class when
        used as `@decorator` (i.e., without parentheses).
        This is invoked immediately with the target (function or class).
    factory : Union[callable[..., FuncDec], callable[..., ClsDec]]
        A decorator factory that receives any additional arguments and returns
        a decorator (e.g., when using `@decorator()` or `@decorator(...)`).
    *args : tuple
        Positional arguments — may include the target function/class
        in non-parameterized usage.
    **kwargs : dict
        Keyword arguments — used in parameterized form and may include
        additional configuration.

    Returns
    -------
    callable
        A function that dispatches to either `simple` or a `factory`-generated decorator
        depending on how the decorator is invoked.

    Examples
    --------
    >>> def simple(func):
    ...     def wrapper(*a, **k):
    ...         print("Simple")
    ...         return func(*a, **k)
    ...
    ...     return wrapper

    >>> def factory(prefix):
    ...     def decorator(func):
    ...         def wrapper(*a, **k):
    ...             print(f"Factory: {prefix}")
    ...             return func(*a, **k)
    ...
    ...         return wrapper
    ...
    ...     return decorator

    >>> def my_decorator(*args, **kwargs):
    ...     return decorator_dispatcher(simple, factory("Hi"), *args, **kwargs)

    >>> @my_decorator
    ... def greet():
    ...     print("Hello")

    >>> greet()
    Simple
    Hello

    >>> @my_decorator()
    ... def shout():
    ...     print("HEY")

    >>> shout()
    Factory: Hi
    HEY
    """
    # Check if 'func' is passed explicitly in kwargs or if the first positional argument
    # is a callable
    # (i.e., decorator without parameters `()` or with `(func)` or with `(func=func)` )
    # If used directly as a decorator (e.g., @decorator), apply immediately
    func = kwargs.pop("func", None) or (args[0] if args and callable(args[0]) else None)
    # If a callable is passed (i.e., target is a function),
    # apply the direct decorator
    if callable(func):
        return simple(
            func,
        )
    # Otherwise, return the factory decorator (for when no callable is passed)
    # (i.e., decorator with parameters `()` or `(...)`)
    # If not used directly as a decorator factory (e.g., @decorator(), @decorator(...))
    return factory


######################################################################
## @decorate functional
######################################################################


# Hint: ← Without parentheses — refers to func as first arg
# Hint: ← With parentheses — can not refers to func need inner decorator func implementation.
# Hint: from functools import partial to decorator = partial(decorator, verbose=True)
def decorate(
    # The target function to be decorated (passed when no parameters are used)
    # func: "callable[..., any]",
    # Not needed as a placeholder, but kept for parameterized usage
    # *,  # indicates that all following parameters must be passed as keyword
    *args: tuple,  # not need placeholder
    # The target function to be decorated (passed when parameters are used, if any)
    # func: "Optional[callable[..., any]]" = None,
    **kwargs: dict,  # Keyword arguments passed to the decorator for customization (e.g., verbose)
) -> "Union[F, FuncDec]":
    """
    Create a generic decorator that supports both parameterized and non-parameterized usage.

    This decorator can be used directly (`@decorate`) or with parameters
    (`@decorate(param=value)`).
    It wraps the target function, optionally modifying its behavior based on
    decorator-specific arguments.

    This supports both:
    - @decorate
    - @decorate(verbose=True)

    Parameters
    ----------
    *args : tuple
        Positional arguments passed to the decorator (ignored by default).
    **kwargs : dict
        Keyword arguments passed to the decorator for configuration. These can be used
        to customize the behavior of the wrapper (ignored by default).
        - func (callable) : The target function to be wrapped, if any.
          This is automatically set when the decorator is used without parentheses
          (e.g., `@decorate`).

    Returns
    -------
    callable
        A function that either directly decorates or returns another decorator function.

    Examples
    --------
    >>> @_decorator
    ... def greet():
    ...     print("Hello")

    >>> @_decorator(verbose=True)
    ... def greet():
    ...     print("Hello")

    Notes
    -----
    - This decorator can be used both with and without parameters.
    - This structure enables reusability across decorators with shared patterns.
    """

    # The case where the decorator is called with parameters (returns a decorator)
    def decorator(func: "callable[..., any]") -> "callable[..., any]":
        """
        Decorate the actual decorator function that wraps the target function.

        Parameters
        ----------
        func : callable[..., any]
            The function to be decorated.

        Returns
        -------
        callable[..., any]
            The wrapped function.
        """

        @functools.wraps(func)
        def wrapper(*args_f, **kwargs_f) -> "any":
            # c = {**a, **b}  # Non-destructive merge (3.5+), Safe, non-mutating
            # c = a | b       # Non-destructive merge (3.9+)
            # a.update(b)     # All Versions but In-place update
            # Call the actual function
            print("Decorated: ", *args, **kwargs)  # noqa: T201
            return func(*args_f, **kwargs_f)

        return wrapper

    # Dispatch: A flexible decorator that can be used both with and without arguments.
    # Case 1: Without Arguments (standard usage, simple):
    #     - @decorate applied directly to a function (e.g., @decorate)
    #     - The decorator is applied immediately, with `func` passed automatically.
    # Case 2: With Arguments (parameterized usage, factory):
    #     - @decorate(verbose=True) for customized behavior.
    #     - Returns an `inner decorator` that can accept the function to decorate.
    return decorator_dispatcher(
        decorator,
        decorator,
        *args,
        **kwargs,
    )


######################################################################
## (make class is callable) cls.__new__.__init__.__call__(...)
## @BaseDecorator  # ← Note no parentheses decorated function directly
## @BaseDecorator()     # ← Note the parentheses to create an instance
## @BaseDecorator(...)  # ← Note the parentheses to create an instance
######################################################################


class DecorateMeta(ABCMeta):
    """
    Metaclass for enabling advanced decorator behaviors.

    This metaclass is designed to provide extensibility for decorator classes.
    Developers can extend this metaclass to modify the class creation logic,
    allowing for custom behaviors such as registration or tracking subclasses.

    Currently, this metaclass does not implement any specific behavior but serves
    as a foundation for more advanced decorator functionalities in custom subclasses.

    Inheriting from `DecorateMeta` allows custom decorator classes to have additional
    control over their instantiation and behavior, making the framework more flexible
    and adaptable.

    Attributes
    ----------
    None (Currently)

    Methods
    -------
    No additional methods are provided in this base class.
    """


class BaseDecorator(metaclass=DecorateMeta):
    """
    Decorate a thread-safe, singleton-based Python decorator class.

    That adds advanced features to any function:
    - Accepts arbitrary parameters (*args, **kwargs)
    - Logging (to console and optionally to file)
    - Execution time and memory usage profiling
    - Async and threaded execution support
    - Serialization and deserialization of decorator state.

    Parameters
    ----------
    *args : tuple
        Optional positional arguments
        (not directly used in base class).
        Optional positional arguments. Supports:
        - args[0] (F) : The function to be wrapped.
    **kwargs : dict
        Optional keyword arguments. Supports:
        - func (F) : The target function to be wrapped.
        - verbose (bool, optional, default=False) :
            Enables debug-level logging.
        - log_to_file (bool, optional, default=False) :
            If True, function logs are written to a file.

    Attributes
    ----------
    strict : bool
        If `True`, disallows extra keyword arguments during argument resolution.
        Default is `False`.

    See Also
    --------
    BaseDecorator
        Provides the default monitoring functionality.
        Acts as the foundation and defined to entry point `BaseDecorator.decorate`.
    BaseDecorator.decorate
        The main entry point for monitoring logic.
        Overriding this class method customizes behavior for all `Type[BaseDecorator]`
        instances.
    BaseDecorator.decorate_monitor
        A backup implementation of the monitoring functionality.
        Especially used when `BaseDecorator.decorate` is overridden or customized.

    Notes
    -----
    This decorator can be used both with and without parameters. It supports both:
    - Bare usage: `@BaseDecorator`
    - Parameterized usage: `@BaseDecorator(arg=value)`

    Examples
    --------
    >>> @BaseDecorator(verbose=True)  # ← Note the parentheses to create an instance
    ... def greet(name):
    ...     print(f"Hello, {name}!")

    >>> greet("Alice")

    >>> BaseDecorator.execute_in_thread(greet, "Bob")
    >>> BaseDecorator.execute_async(async_greet, "Charlie")
    """

    ## Singleton instance, thread lock, and thread pool for execution
    ## Store singleton instances in a class-level dictionary
    ## dictionary on the base class that tracks a Singleton instance per subclass
    _instance: "ClassVar[dict[type[BaseDecorator], BaseDecorator]]" = {}  # noqa: RUF012
    ## Used for thread-safe operations
    ## Reentrant: The same thread can acquire the lock multiple times without blocking.
    _lock: "threading.RLock" = threading.RLock()
    ## Executor to manage threaded execution for async tasks
    _executor: "ThreadPoolExecutor" = ThreadPoolExecutor(max_workers=10)

    # --------------------- make sure class is callable ---------------------

    def __new__(
        # cls refers to the class itself, and is used in class methods (@classmethod).
        cls: "type[BaseDecorator]",  # cls is of type 'Type[BaseDecorator]'
        *args: "tuple[any, ...]",
        **kwargs: "dict[str, any]",
    ) -> "DecoratorLike":
        """
        Ensure that only a single instance of the decorator is created.

        If the decorator is used directly on a function, the function is immediately decorated.

        Parameters
        ----------
        *args : tuple
            Optional positional arguments
            (not directly used in base class).
            Optional positional arguments. Supports:
            - args[0] (F) : The function to be wrapped.
        **kwargs : dict
            Optional keyword arguments. Supports:
            - func (F) : The target function to be wrapped.
            - verbose (bool, optional, default=False) :
                Enables debug-level logging.
            - log_to_file (bool, optional, default=False) :
                If True, function logs are written to a file.

        Returns
        -------
        Union[F, BaseDecorator]
            Either the decorated function or the singleton instance.
        """
        # Class.method() or obj.method()
        # Ensure only one instance exists (singleton pattern)
        with cls._lock:
            # Create a new instance if none exists:
            if cls._instance.get(cls) is None:
                logger.debug(f"Creating new instance for {cls.__name__}.")
                cls._instance[cls] = (
                    # follows the method resolution order (MRO).
                    super().__new__(cls)  # ✅ Allocate instance safely
                )
                # cls._instance[cls].__init__(*args, **kwargs)  # Ensure __init__ is called
            else:
                logger.debug(f"Using existing singleton instance for {cls.__name__}.")
        # Enable use as a decorator with/without arguments:
        # Case 1: used as @BaseDecorator, apply immediately  (no parentheses)
        # ← Without parentheses — refers to the class itself, not an instance
        # Case 2: used as @BaseDecorator() or @BaseDecorator(...)
        # ← With parentheses — creates an instance of the class
        return decorator_dispatcher(
            cls._instance[cls],
            cls._instance[cls],
            *args,
            **kwargs,
        )

    def __init__(
        # self refers to the instance of the class.
        self,  # self is an instance of 'BaseDecorator'
        *args: "tuple[any, ...]",
        **kwargs: "dict[str, any]",
    ) -> None:
        """
        Initialize the decorator instance with any provided arguments.

        The decorator can be configured with `verbose` for debug-level logging
        and `log_to_file` for file-based logging.

        Parameters
        ----------
        *args : tuple
            Optional positional arguments
            (not directly used in base class).
            Optional positional arguments. Supports:
            - args[0] (F) : The function to be wrapped.
        **kwargs : dict
            Optional keyword arguments. Supports:
            - func (F) : The target function to be wrapped.
            - verbose (bool, optional, default=False) :
                Enables debug-level logging.
            - log_to_file (bool, optional, default=False) :
                If True, function logs are written to a file.
        """
        # Needs instance (obj.method())
        # Initialize any instance-specific data here
        # Avoid reinitialization if already initialized
        # pylint: disable=access-member-before-definition
        if hasattr(self, "_initialized") and self._initialized:
            # Prevent re-initialization of the singleton
            # Only initialize once for singleton
            return

        self._args = args
        self._kwargs = kwargs
        self._initialized = True

        self.verbose: bool = kwargs.get("verbose", False)  # Default to False
        self.log_to_file: bool = kwargs.get("log_to_file", False)  # Default to False

        # Set logging level if verbose is enabled
        if self.verbose:
            # logger.setLevel(logging.DEBUG)
            logger.setLevel(logger.INFO)
            logger.debug(f"{self.__class__.__name__} initialized with verbose logging.")

    def __call__(
        # self refers to the instance of the class.
        self,  # self is an instance of 'BaseDecorator'
        *args: "tuple[any, ...]",
        **kwargs: "dict[str, any]",
    ) -> "F":
        """
        Allow the decorator to be used with arguments (e.g., @BaseDecorator(verbose=True)).

        The decorator wraps the function with logging and profiling capabilities.

        Parameters
        ----------
        args :
            args.
        kwargs :
            kwargs.

        Returns
        -------
        F
            The decorated function with additional functionality.
        """
        # Needs instance (obj.method())
        # Dispatch: A flexible decorator that can be used both with and without arguments.
        # Case 1: Without Arguments (standard usage, simple):
        #     - @decorate applied directly to a function (e.g., @decorate)
        #     - The decorator is applied immediately, with `func` passed automatically.
        # Case 2: With Arguments (parameterized usage, factory):
        #     - @decorate(verbose=True) for customized behavior.
        #     - Returns an `inner decorator` that can accept the function to decorate.
        return decorator_dispatcher(
            # cls = type(self)  # or self.__class__
            # self.decorate       # <-- BAD: using self instead of cls (cls = type(self))
            type(self).decorate,  # Calls the correct overridden classmethod
            type(self).decorate,  # Calls the correct overridden classmethod
            *args,
            **kwargs,
        )

    # --------------------- decorator parameters updater ---------------------

    # --- Properties for args ---
    @property
    def args(self):
        # Needs instance (obj.method())
        # Return the actual data inside _args
        return self._args

    @args.setter
    def args(self, value):
        # Needs instance (obj.method())
        if not isinstance(value, tuple):
            raise TypeError("args must be a tuple")
        self._args = value

    # --- Properties for kwargs ---
    @property
    def kwargs(self):
        # Needs instance (obj.method())
        # Return the actual data inside _kwargs
        return self._kwargs

    @kwargs.setter
    def kwargs(self, value):
        if not isinstance(value, dict):
            raise TypeError("kwargs must be a dictionary")
        self._kwargs = value

    def show_config(self) -> None:
        """
        Display the current configuration of the decorator, including arguments and settings.

        Notes
        -----
        This method can be used to check how the decorator is configured.
        """
        # Needs instance (obj.method())
        # Logs the actual args and kwargs without using @property
        # cls = type(self)  # or self.__class__
        # logger.info(f"{self.__class__.__name__} Configuration:")
        logger.info(
            f"\n{type(self).__name__} Configuration:"
            f"\nArgs: {self._args}"  # Logs actual args directly
            f"\nKwargs: {self._kwargs}"  # Logs actual kwargs directly
        )

    def decorator_args_kwargs_normalizer(self) -> None:
        """
        Normalize decorator keyword arguments by ensuring required defaults are set.

        This method ensures that expected configuration keys are present in
        `self.kwargs`, assigning default values where needed. This helps avoid
        key errors and centralizes default configuration behavior.

        Currently ensures:
        - `"verbose"` is present and defaults to `False` if not explicitly provided.

        Returns
        -------
        None
        """
        # Needs instance (obj.method())
        # Equivalent: self.kwargs.setdefault("verbose", self.kwargs.get("verbose", False))
        # self.kwargs.setdefault("verbose", False)

    def decorator_args_kwargs_sanitizer(
        # self refers to the instance of the class.
        self,  # self is an instance of 'BaseDecorator'
        *args: "tuple[any, ...]",
        **kwargs: "dict[str, any]",
    ) -> None:
        """
        Validate and sanitize keyword arguments passed to the decorator.

        This method checks for the presence and correctness of known decorator keyword
        arguments to ensure proper configuration. If any known argument has an unexpected
        type or value, a ValueError is raised.

        Parameters
        ----------
        *args : tuple
            Optional positional arguments
            (not directly used in base class).
            Optional positional arguments. Supports:
            - args[0] (F) : The function to be wrapped.
        **kwargs : dict
            Optional keyword arguments. Supports:
            - func (F) : The target function to be wrapped.
            - verbose (bool, optional, default=False) :
                Enables debug-level logging.
            - log_to_file (bool, optional, default=False) :
                If True, function logs are written to a file.

        Returns
        -------
        None
            This method does not return anything.
            It raises an exception if validation fails.

        Raises
        ------
        ValueError
            If the `verbose` keyword is present but is not of type `bool`.
        """
        # Needs instance (obj.method())
        # if "verbose" in kwargs and not isinstance(kwargs["verbose"], bool):
        #     raise ValueError("verbose must be a boolean.")

    # --------------------- func parameters updater ---------------------

    @staticmethod
    def get_default_parameters(
        func: "callable[..., any]",
    ) -> "dict[str, any]":
        """
        Extract default parameter values from a function's signature.

        This method uses Python's inspect module to retrieve a mapping of parameter
        names to their default values for any callable object. Parameters without
        default values are excluded.

        Parameters
        ----------
        func : callable[..., any]
            The target function or callable whose default parameter values should be extracted.

        Returns
        -------
        Dict[str, any]
            A dictionary mapping parameter names to their default values.

        Examples
        --------
        >>> def foo(a, b=2, c=3):
        ...     pass
        >>> get_default_parameters(foo)
        {'b': 2, 'c': 3}
        """
        # Class.method() or obj.method()
        sig = inspect.signature(func)
        return {
            p.name: p.default
            for p in sig.parameters.values()
            if p.default is not inspect.Parameter.empty
        }

    @staticmethod
    def func_default_args_kwargs_broadcaster(
        func: "callable[..., any]",
        *args: "tuple[any, ...]",
        **kwargs: "dict[str, any]",
    ) -> "tuple[tuple[any, ...], dict[str, any]]":
        """
        Resolve arguments to a function, applying default values using introspection.

        This utility attempts to bind the provided arguments (`args`, `kwargs`) to the
        signature of the given `func`, filling in any missing parameters with their
        default values. If the binding fails (e.g., due to missing required arguments),
        a warning is logged and the original `args` and `kwargs` are returned unchanged.

        Parameters
        ----------
        func : callable
            The target function whose signature will be used to resolve default values.
        *args : tuple
            Positional arguments intended for the function call.
        **kwargs : dict
            Keyword arguments intended for the function call.

        Returns
        -------
        Tuple[Tuple[any, ...], Dict[str, any]]
            A tuple containing the resolved positional and keyword arguments. If resolution
            fails, the original `args` and `kwargs` are returned.

        Notes
        -----
        - This method is useful when decorator logic wants to ensure that all parameters
        (including defaults) are accessible before the function is invoked.
        - To broadcast decorator `kwargs` into the function, ensure `**kwargs` is accepted
        by the target function.
        """
        # Class.method() or obj.method()
        try:
            # _kwargs = {**self.kwargs, **_kwargs} if self.use_decorator_param else kwargs
            _args, _kwargs = _resolve_args_and_kwargs(
                *args,
                **{**kwargs, "func": func},
            )
            return _args, _kwargs
        except TypeError as e:
            logger.warning(
                f"If you want to broadcast default parameters via decorator `kwargs`, "
                f"make sure {func.__name__} accepts `**kwargs` or explicitly include "
                f"all expected arguments. Default broadcasting failed for {func.__name__}: "
                f"{e}"
            )
            # fallback if not broadcast
            return args, kwargs

    # --------------------- handle cls instance ---------------------

    # --- Explicit Cleanup --- (optional)
    # def __del__(self):
    #     """Clean up resources if needed (e.g., close connections)"""
    #     print(f"Cleaning up resources for {self.__class__.__name__}")
    #     # Example cleanup tasks (e.g., closing network connections, freeing memory)
    #     # self._close_connections()  # Example method to close connections
    #     pass

    @classmethod
    def release_instance(
        # cls refers to the class itself, and is used in class methods (@classmethod).
        cls: "type[BaseDecorator]",  # cls is of type 'Type[BaseDecorator]'
    ) -> None:
        """
        Destroys the singleton instance, releasing any resources.

        Notes
        -----
        This method can be called when the decorator is no longer needed.
        """
        # Class.method() or obj.method()
        # Destroy the singleton instance
        with cls._lock:
            if cls._instance.get(cls) is not None:
                logger.info("Releasing singleton instance")  # Log release
                cls._instance[cls] = None

    @classmethod
    def get_instance(
        # cls refers to the class itself, and is used in class methods (@classmethod).
        cls: "type[BaseDecorator]",  # cls is of type 'Type[BaseDecorator]'
    ) -> "BaseDecorator":
        """
        Retrieve or creates the singleton instance of the decorator.

        Returns
        -------
        BaseDecorator
            The singleton instance of the decorator.
        """
        # Class.method() or obj.method()
        # Retrieve or create the singleton instance
        with cls._lock:
            if cls._instance.get(cls) is None:
                logger.debug("Creating singleton instance")  # Debug log
                cls._instance[cls] = cls()
        return cls._instance[cls]

    # --- Use Weak References for Singleton Instance ---
    @classmethod
    def get_weak_instance(
        # cls refers to the class itself, and is used in class methods (@classmethod).
        cls: "type[BaseDecorator]",  # cls is of type 'Type[BaseDecorator]'
    ):
        """Return a weak reference to the Singleton instance."""
        # Class.method() or obj.method()
        with cls._lock:
            if cls._instance.get(cls) is None:
                logger.info("No Singleton instance. Creating one now...")
                cls._instance[cls] = cls()  # Create an instance if it doesn't exist
            return weakref.ref(cls._instance[cls])

    @classmethod
    def entry_point(
        # cls refers to the class itself, and is used in class methods (@classmethod).
        cls: "type[BaseDecorator]",  # cls is of type 'Type[BaseDecorator]'
        *args: "tuple[any, ...]",
        **kwargs: "dict[str, any]",
    ) -> None:
        """
        Entry point to initialize or retrieve the singleton instance of the decorator.

        The instance is created if it does not exist.

        Parameters
        ----------
        *args : any
            Arguments passed to the constructor.
        **kwargs : dict
            Keyword arguments passed to the constructor.
        """
        # Class.method() or obj.method()
        logger.info(f"Entry point invoked: {cls.__name__}")
        # Initialize or retrieve the singleton and print its configuration
        with cls._lock:
            # Ensure initialization of the instance here or reuse
            instance = cls(*args, **kwargs)
            # Call show_config on the instance to display the configuration
            instance.show_config()

    # --------------------- decorator (__call__) entry_point ---------------------

    # --- decorator main entry point method monitoring & logging ---
    @staticmethod
    def decorate(
        # cls refers to the class itself, and is used in class methods (@classmethod).
        # cls: "Type[BaseDecorator]",  # cls is of type 'Type[BaseDecorator]'
        *args: "tuple[any, ...]",
        **kwargs: "dict[str, any]",
    ) -> "F":
        """
        Apply logging, profiling, and exception handling to the wrapped function.

        Additionally, if `log_to_file` is True, logs are written to a file.

        Parameters
        ----------
        *args : tuple
            Optional positional arguments
            (not directly used in base class).
            Optional positional arguments. Supports:
            - args[0] (F) : The function to be wrapped.
        **kwargs : dict
            Optional keyword arguments. Supports:
            - func (F) : The target function to be wrapped.
            - verbose (bool, optional, default=False) :
                Enables debug-level logging.
            - log_to_file (bool, optional, default=False) :
                If True, function logs are written to a file.

        Returns
        -------
        F
            The wrapped function.

        See Also
        --------
        BaseDecorator
            Provides the default monitoring functionality.
            Acts as the foundation and defined to entry point `BaseDecorator.decorate`.
        BaseDecorator.decorate
            The main entry point for monitoring logic.
            Overriding this class method customizes behavior for all `Type[BaseDecorator]`
            instances.
        BaseDecorator.decorate_monitor
            A backup implementation of the monitoring functionality.
            Especially used when `BaseDecorator.decorate` is overridden or customized.
        """

        # Class.method() or obj.method()
        ## The case where the decorator is called with parameters (returns a decorator)
        def decorator(
            func: "callable[..., any]",
        ) -> "callable[..., any]":
            ## Set up file-based logging if required
            log_file_handler = None
            log_to_file = kwargs.get("log_to_file", False)
            if log_to_file:
                filename = kwargs.get("log_filename", f"log_{func.__name__}.txt")
                ## Log file named after the function
                log_file = filename
                log_file_handler = logger.FileHandler(log_file)
                # log_file_handler.setLevel(logger.DEBUG)
                log_file_handler.setLevel(logger.INFO)
                formatter = logger.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s"
                )
                log_file_handler.setFormatter(formatter)
                logger.addHandler(log_file_handler)

            ## Retain original function attributes (name, docstring)
            @functools.wraps(func)
            def wrapper(
                *args_f: "any",
                **kwargs_f: "any",
            ) -> "any":
                """
                Wrap function.

                That handles logging, profiling, and exception handling
                for the decorated function.

                Parameters
                ----------
                *args_f : any
                    Positional arguments for the wrapped function.
                **kwargs_f : any
                    Keyword arguments for the wrapped function.

                Returns
                -------
                any
                    The result of the wrapped function execution.
                """
                ## If verbose is not passed, check its default value from function signature
                verbose = (
                    kwargs_f.get("verbose")  # passed func kwargs
                    or BaseDecorator.get_default_parameters(func).get(
                        "verbose"
                    )  # default kwargs
                    or kwargs.get("verbose", False)  # instance kwargs
                )
                logger.debug(
                    f"'{func.__name__}' function wrapped  to "
                    f"{BaseDecorator.__name__}.decorate: "
                    f"{verbose}'"
                )  # Debug log of function call
                start_time = _time.perf_counter()  # Start time for execution profiling
                tracemalloc.start()  # Start memory tracking
                try:
                    # Call the original function
                    result = func(*args_f, **kwargs_f)
                    # BaseDecorator do noting
                    return result  # noqa: RET504
                except Exception as e:
                    # Log any exceptions
                    logger.exception(f"'{func.__name__}' function error in : {e}")
                    raise
                    # BaseDecorator.wrapper_exception_handler(e, func)
                finally:
                    end_time = _time.perf_counter()  # End time for execution profiling
                    current, peak = (
                        tracemalloc.get_traced_memory()
                    )  # Get memory usage stats
                    tracemalloc.stop()  # Stop memory tracking
                    duration = end_time - start_time  # Execution duration
                    # Log execution time and memory usage
                    logger.info(
                        f"'{func.__name__}' "
                        f"Execution-Time={duration:.4f} seconds, "
                        f"Memory Usage: Current={current / 1024:.2f} KB, Peak={peak / 1024:.2f} KB"
                    )
                    if log_file_handler:
                        # Clean up file logging handler
                        logger.removeHandler(log_file_handler)
                        log_file_handler.close()

            return wrapper  # Return the decorated function

        # Dispatch: A flexible decorator that can be used both with and without arguments.
        # Case 1: Without Arguments (standard usage, simple):
        #     - @decorate applied directly to a function (e.g., @decorate)
        #     - The decorator is applied immediately, with `func` passed automatically.
        # Case 2: With Arguments (parameterized usage, factory):
        #     - @decorate(verbose=True) for customized behavior.
        #     - Returns an `inner decorator` that can accept the function to decorate.
        return decorator_dispatcher(
            decorator,
            decorator,
            *args,
            **kwargs,
        )

    # --- decorator backup method monitoring & logging ---
    # Alias the method instead of redefining
    decorate_monitor = decorate
    # Alias the method instead of redefining
    decorator_dispatcher = decorator_dispatcher

    # --- wrapper_exception_handler ---
    @staticmethod
    def wrapper_exception_handler(
        e: "Exception",
        func: "Optional[callable]" = None,
    ) -> "Optional[any]":
        """
        Centralized exception handling method.

        This hook is invoked when an exception is raised during the execution
        of the wrapped logic. Override this to implement custom error handling.

        Parameters
        ----------
        e : Exception
            The exception instance that was raised.
        func : F | None
            The function to be get name.

        Returns
        -------
        any | None
            A fallback value or error response. Can be re-raised or suppressed
            depending on the implementation.
        """
        # Class.method() or obj.method()
        try:
            thread_id = threading.get_ident()
            logger.error(
                f"Error occurred in thread {thread_id}: {e}",
                exc_info=True,
            )
        except Exception:
            # Log any exceptions
            logger.exception(
                f"'{getattr(func, __name__, 'no_name')}' function error in : {e}"
            )
        # raise e

    # --------------------- optimization & syncranization ---------------------

    @classmethod
    def execute_in_thread(
        # cls refers to the class itself, and is used in class methods (@classmethod).
        cls: "type[BaseDecorator]",  # cls is of type 'Type[BaseDecorator]'
        func: "F",
        *args: "tuple[any, ...]",
        **kwargs: "dict[str, any]",
    ) -> None:
        """
        Execute the given function in a separate thread.

        Parameters
        ----------
        func : F
            The function to be executed in a separate thread.
        *args : any
            Positional arguments passed to the function.
        **kwargs : any
            Keyword arguments passed to the function.
        """
        # ## TODO: Launch app in a separate thread (non-blocking)
        # thread = threading.Thread(target=run_app, daemon=True)
        # thread.start()
        # ## Run after slight delay
        # run_delayed()
        # ## Keep the main thread alive while Streamlit runs
        # thread.join()

        # Class.method() or obj.method()
        # Run a function asynchronously in a new thread
        def _thread_wrapper() -> None:
            try:
                # Log execution start
                logger.info(
                    f"'{func.__name__}' to running in {cls.__name__} "
                    f"classmethod in thread..."
                )
                func(*args, **kwargs)  # Execute the function
            except Exception as e:
                # Log errors
                logger.exception(f"Threaded execution error in '{func.__name__}': {e}")

        cls._executor.submit(_thread_wrapper)  # Submit to the thread pool for execution

    @classmethod
    def execute_async(
        # cls refers to the class itself, and is used in class methods (@classmethod).
        cls: "type[BaseDecorator]",  # cls is of type 'Type[BaseDecorator]'
        func: "callable[..., Coroutine[any, any, any]]",
        *args: "tuple[any, ...]",
        **kwargs: "dict[str, any]",
    ) -> None:
        """
        Execute the given async function within an event loop.

        Parameters
        ----------
        func : callable[..., Coroutine]
            The async function to execute.
        *args : any
            Positional arguments passed to the function.
        **kwargs : any
            Keyword arguments passed to the function.
        """

        # Class.method() or obj.method()
        # Run an async function in an event loop
        async def _async_wrapper():
            try:
                # Log async execution start
                logger.info(
                    f"'{func.__name__}' to running in {cls.__name__} "
                    f"classmethod in async..."
                )
                await func(*args, **kwargs)  # Execute the async function
            except Exception as e:
                logger.exception(
                    f"Async execution error in '{func.__name__}': {e}"
                )  # Log errors

        try:
            if asyncio.get_event_loop().is_running():
                # Run async task in existing loop
                task = asyncio.create_task(_async_wrapper())
                # This forces any exceptions to surface
                task.add_done_callback(lambda t: t.exception())
            else:
                asyncio.run(_async_wrapper())  # Run async task in new loop
        except Exception as e:
            logger.exception(
                f"Failed to run async task: {e}"
            )  # Log if running async fails

    # --- Context Manager Support for Resource Management --- (optional)
    # from contextlib import contextmanager

    # @contextmanager
    # def manage_resources(self):
    #     """Context manager for managing resources (e.g., opening/closing connections)"""
    #     try:
    #         print(f"Resources for {self.__class__.__name__} are now being used.")
    #         yield self
    #     finally:
    #         print(f"Resources for {self.__class__.__name__} are now being cleaned up.")
    #         # Perform cleanup if necessary
    #         pass

    # --------------------- Serialization Methods (joblib) ---------------------

    @classmethod
    def serialize(
        # cls refers to the class itself, and is used in class methods (@classmethod).
        cls: "type[BaseDecorator]",  # cls is of type 'Type[BaseDecorator]'
        filename: str,
        *args: "tuple[any, ...]",
        **kwargs: "dict[str, any]",
    ) -> None:
        """
        Save the singleton instance to a file using joblib.

        Parameters
        ----------
        filename : str
            The filename to store the serialized object.
        *args : any
            Additional arguments to pass to `joblib.dump`.
        **kwargs : any
            Additional keyword arguments to pass to `joblib.dump`.
        """
        # Class.method() or obj.method()
        # Save the singleton instance to a file
        with cls._lock:
            if cls._instance.get(cls) is not None:
                try:
                    # Serialize the instance
                    joblib.dump(cls._instance[cls], filename, *args, **kwargs)
                    # Log success
                    logger.info(f"Serialized {cls.__name__} instance to '{filename}'")
                except Exception as e:
                    logger.exception(f"Serialization failed: {e}")  # Log failure
            else:
                logger.warning(
                    "No instance to serialize."
                )  # Log if no instance is found

    @classmethod
    def deserialize(
        # cls refers to the class itself, and is used in class methods (@classmethod).
        cls: "type[BaseDecorator]",  # cls is of type 'Type[BaseDecorator]'
        filename: str,
        *args: "tuple[any, ...]",
        **kwargs: "dict[str, any]",
    ) -> "Optional[type[BaseDecorator]]":
        """
        Load the singleton instance from a file using joblib.

        Parameters
        ----------
        filename : str
            The filename to load the serialized object from.
        *args : any
            Additional arguments to pass to `joblib.load`.
        **kwargs : any
            Additional keyword arguments to pass to `joblib.load`.

        Returns
        -------
        Optional[BaseDecorator]
            The deserialized singleton instance, or None if an error occurred.
        """
        # Class.method() or obj.method()
        # Load the singleton instance from a file
        with cls._lock:
            try:
                # Deserialize the instance
                cls._instance[cls] = joblib.load(filename, *args, **kwargs)
                # Log success
                logger.info(f"Deserialized {cls.__name__} instance from '{filename}'")
                return cls._instance[cls]
            except FileNotFoundError:
                logger.warning(
                    f"File not found: {filename}"
                )  # Log if file doesn't exist
            except Exception as e:
                logger.exception(
                    f"Deserialization error: {e}"
                )  # Log if deserialization fails
        return None


######################################################################
## @BaseDecorator inheritances
######################################################################


class DummyDecorator(BaseDecorator):
    """
    Dummy implementation of BaseDecorator that overrides the decorate method.

    To add simple print statements before and after the target function call.

    This class is useful for demonstration or testing purposes and does not
    utilize any dynamic arguments passed to the decorator.

    Inherits
    --------
    BaseDecorator
        Provides the base interface and optional configuration support.
    """

    # Dispatch: A flexible decorator that can be used both with and without arguments.
    # Case 1: Without Arguments (standard usage, simple):
    #     - @decorate applied directly to a function (e.g., @decorate)
    #     - The decorator is applied immediately, with `func` passed automatically.
    # Case 2: With Arguments (parameterized usage, factory):
    #     - @decorate(verbose=True) for customized behavior.
    #     - Returns an `inner decorator` that can accept the function to decorate.
    # Without Arguments Only Case 1 if direct call class method
    @staticmethod
    def decorate(func):  # pylint: disable=arguments-differ
        """
        Override the decorate method.

        To wrap a function with custom pre- and post-execution print statements.

        Simple sample Without Arguments the decorator is applied immediately,
        if direct call to class method

        Parameters
        ----------
        func : callable
            The function to be decorated.

        Returns
        -------
        callable
            The wrapped function with additional behavior.
        """

        def wrapper(*args, **kwargs):
            print("Custom before function")  # noqa: T201
            result = func(*args, **kwargs)
            print("Custom after function")  # noqa: T201
            return result

        return wrapper


######################################################################
##
######################################################################
