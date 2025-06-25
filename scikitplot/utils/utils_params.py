"""utils_params.py."""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=broad-exception-caught
# pylint: disable=logging-fstring-interpolation

# import os
# import re
# import shutil
# # from pathlib import Path
# from datetime import datetime
# from collections.abc import Sequence
# from contextlib import nullcontext

# import functools
# import importlib
import inspect as _inspect
from typing import TYPE_CHECKING

# import logging
from .. import logger as _logger

if TYPE_CHECKING:
    from typing import TypeVar

    F = TypeVar("F", bound=callable[..., any])

__all__ = [
    "_get_args_kwargs",
    "_get_param_w_index",
    "_resolve_args_and_kwargs",
]

######################################################################
## resolve args and_kwargs
######################################################################


def _get_args_kwargs(
    *args,
    **kwargs,
):
    """
    Create new args and kwargs with the new parameter value.

    .. versionadded:: 0.3.9

    Parameters
    ----------
    *args : tuple
        Positional arguments passed to the function.
    **kwargs : dict
        Keyword arguments passed to the function.
        param_key : str
            The name of the parameter being replaced.
        param_index : int
            The index of the parameter in the positional arguments.
        param_value : any
            The new value to replace the original parameter.

    Returns
    -------
    tuple
        A tuple containing the new args and kwargs.

    Raises
    ------
    ValueError
        If the parameter cannot be found in args or kwargs.
    """
    param_key = kwargs.pop("param_key", None)
    param_index = kwargs.pop("param_index", None)
    param_value = kwargs.pop("param_value", None)

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


def _get_param_w_index(
    *args,
    **kwargs,
) -> "tuple[str, int, any]":
    """
    Retrieve the parameter and its param_index from the function signature.

    .. versionadded:: 0.3.9

    Parameters
    ----------
    *args : tuple
        Positional arguments passed to the function.
    **kwargs : dict
        Keyword arguments passed to the function.
        func : callable
            The original function to _inspect.
        params : list of str
            List of possible parameter names to search for.

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
    func = kwargs.pop("func", None)
    params = kwargs.pop("params", [])

    # Retrieve the signature of the wrapped function
    sig = _inspect.signature(func)

    # Initialize variable to hold parameter information
    param_key = None
    param_default = None
    param_index = None

    # Determine the parameter and its param_index
    for param_index, (name, parameter) in enumerate(  # noqa: B007
        sig.parameters.items()
    ):
        if name in params:
            param_key = name
            # If the parameter has a default value, store it
            if parameter.default is not _inspect.Parameter.empty:
                param_default = parameter.default
            break  # Stop once we find the first match

    # If no matching parameter is found, return None values
    if param_key is None:
        return None, None, None

    # Step 3: Extract the parameter value from args or kwargs
    param_value = (
        kwargs.get(param_key, param_default)  # Prefer kwargs if present
        if param_key in kwargs
        else (
            args[param_index]  # Otherwise use args by param_index, if available
            if param_index < len(args)
            else param_default
        )  # Fallback to default if neither args nor kwargs contain it
    )

    return param_key, param_index, param_value


def _resolve_args_and_kwargs(
    *args,
    **kwargs,
):
    """
    Resolve and separate positional and keyword arguments for a function.

    Applying default values. Can log or raise errors on unexpected kwargs.

    Parameters
    ----------
    *args : tuple
        Positional arguments to resolve.
    **kwargs : dict
        Keyword arguments to resolve. Can include 'verbose=True' for logging extras.
        func : callable
            The target function whose signature is used for resolution.
        strict : bool, optional
            If True, enforce that all required parameters are provided and no extras are allowed.
            If False (default), allow partial binding and ignore extra keys (loggable).

    Returns
    -------
    tuple
        A tuple (resolved_args, resolved_kwargs) where:
        - resolved_args is a tuple of bound positional arguments.
        - resolved_kwargs is a dict of keyword arguments, including defaults.

    Raises
    ------
    TypeError
        If required arguments are missing or unexpected keys are found (in strict mode).

    Notes
    -----
    - Uses `_inspect.signature()` and argument binding utilities.
    - Useful for function wrappers, config validation, deferred execution, etc.
    """
    func = kwargs.pop("func", None)
    strict = kwargs.pop("strict", False)

    # Get the signature of the function
    sig = _inspect.signature(func)
    try:
        # Attempt to bind the provided args and kwargs
        # strict=True: enforce full binding with all required args
        # strict=False: allow partial binding (some args may be missing)
        if strict:
            bound_args = sig.bind(*args, **kwargs)
        else:
            bound_args = sig.bind_partial(*args, **kwargs)

        # After binding, apply default values to missing parameters
        bound_args.apply_defaults()

    except TypeError as e:
        # If binding fails (e.g., missing required arg), raise with context
        raise TypeError(f"Argument resolution failed: {e}") from e

    # Identify all valid parameter names from the function signature
    # Determine if any kwargs were passed that do not match the signature
    extra_kwargs = set(kwargs) - set(sig.parameters)

    # In non-strict mode, optionally log unknown kwargs if 'verbose' was passed
    if not strict and kwargs.get("verbose", False) and extra_kwargs:
        _logger.debug(f"Unexpected kwargs logged: {extra_kwargs}")
    # In strict mode, raise an error for any unknown extra kwargs
    if strict and extra_kwargs:
        raise TypeError(f"⚠️ Unexpected keyword arguments: {extra_kwargs}")
    # Return the resolved args and kwargs (fully applied with defaults)
    return bound_args.args, bound_args.kwargs


######################################################################
##
######################################################################
