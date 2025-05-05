"""
decorate.py
This module provides a base class `DecoratorMixin` for defining decorators.
"""

import functools
import inspect

# import re
# import warnings
import logging
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import (  # noqa: F401
        Any,
        Callable,
        Optional,
        Type,
        Union,
    )

logger = logging.getLogger(__name__)


class DecoratorMeta(ABCMeta):
    """
    Optional custom metaclass for decorator mixin system.

    Currently acts as a placeholder for future extension or customization
    related to metaclass behavior.
    """


class DecoratorMixin(metaclass=DecoratorMeta):
    """
    Base class for handle decorators and wrappers.

    Parameters
    ----------
    *args : tuple
        Positional arguments passed to the decorator.

    **kwargs : dict
        Keyword arguments passed to the decorator.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        if args and kwargs:
            raise AssertionError("Only positional or keyword args are allowed")

    # The target object to be decorated
    def __call__(
        self,
        obj: "Callable",
        *args: tuple,
        **kwargs: dict,
    ) -> "Callable":
        if not callable(obj):
            raise TypeError(f"Can only decorate callable objects, got {type(obj)}")

    @abstractmethod
    @functools.wraps
    def wrapper(
        self,
        *args: tuple,
        **kwargs: dict,
    ) -> "Callable":
        """Subclasses must implement the logic for wrap the object."""
        raise NotImplementedError("Subclasses must override `decorate`")
        ...

    @abstractmethod
    def decorate(
        self,
        *args: tuple,
        **kwargs: dict,
    ) -> "Callable":
        """
        Instance-Level Method (decorator)
        Subclasses must implement the logic for decorating the object.
        """
        raise NotImplementedError("Subclasses must override `decorate`")
        ...

    @classmethod
    def decorator(
        cls: "DecoratorMixin",
        *args: tuple,
        **kwargs: dict,
    ) -> "Callable[[Callable[..., Any]], Callable[..., Any]]":  # Callable[[F], F]
        """
        Class-Level Method (decorator)
        Class-level shortcut to create a new one and apply a Substitution decorator.

        Parameters
        ----------
        *args : tuple
            Positional arguments for substitution.
            Not recommended due to lack of variable names.
        **kwargs : dict
            Keyword (named) arguments for substitution.
            Recommended for clarity and compatibility with both styles.

        Returns
        -------
        Callable

        Notes
        -----
        This is a class method, which is bound to the class itself and not to
        any instance. When you call `DecoratorMixin.decorator`, it refers to this method.


        Examples
        --------
        >>> @Substitution.decorator(param="Alan")
        ... def greet(): ...
        """
        return cls(*args, **kwargs)


def _get_param_w_index(
    *args, func=None, params=None, **kwargs
) -> "tuple[str, Any, int]":
    """
    Retrieve the parameter and its param_index from the function signature.

    .. versionadded:: 0.3.9

    Parameters
    ----------
    *args : tuple
        Positional arguments passed to the function.

    func : callable
        The original function to inspect.

    params : list of str
        List of possible parameter names to search for.

    **kwargs : dict
        Keyword arguments passed to the function.

    Returns
    -------
    tuple
        A tuple containing respectively its name, and its param_index, and the parameter.
        If the parameter is not found, returns (None, None, None).

    Raises
    ------
    ValueError
        If none of the specified parameters are found.

    """
    # Retrieve the signature of the wrapped function
    signature = inspect.signature(func)

    # Initialize variable to hold parameter information
    param_key = None
    default = None
    param_index = None

    # Determine the parameter and its param_index
    for param_index, (name, parameter) in enumerate(  # noqa: B007
        signature.parameters.items()
    ):  # noqa: B007
        if name in params:
            param_key = name
            # If the parameter has a default value, store it
            if parameter.default is not inspect.Parameter.empty:
                default = parameter.default
            break  # Stop once we find the first match

    # If no matching parameter is found, return None values
    if param_key is None:
        return None, None, None

    # Step 3: Extract the parameter value from args or kwargs
    param_value = (
        kwargs.get(param_key, default)  # Prefer kwargs if present
        if param_key in kwargs
        else (
            args[param_index]  # Otherwise use args by param_index, if available
            if param_index < len(args)
            else default
        )  # Fallback to default if neither args nor kwargs contain it
    )

    return param_key, param_index, param_value


def _get_args_kwargs(
    *args,
    param_key=None,
    param_index=None,
    param_value=None,
    **kwargs,
):
    """
    Create new args and kwargs with the new parameter value.

    .. versionadded:: 0.3.9

    Parameters
    ----------
    *args : tuple
        Positional arguments passed to the function.

    param_key : str
        The name of the parameter being replaced.

    param_index : int
        The index of the parameter in the positional arguments.

    param_value : any
        The new value to replace the original parameter.

    **kwargs : dict
        Keyword arguments passed to the function.

    Returns
    -------
    tuple
        A tuple containing the new args and kwargs.

    Raises
    ------
    ValueError
        If the parameter cannot be found in args or kwargs.

    """
    new_args = list(args)

    # Only replace if the parameter exists in args or kwargs
    if param_key in kwargs:
        kwargs[param_key] = param_value
    elif param_index is not None and param_index < len(new_args):
        new_args[param_index] = param_value
    else:
        raise ValueError(
            f"The specified parameter {param_key} was not found in the function's arguments."
        )

    return new_args, kwargs
