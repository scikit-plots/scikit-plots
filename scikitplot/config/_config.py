# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# This module was copied from the scikit-learn project.
# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/_config.py

"""Global configuration state and functions for management."""

import os
import threading
from contextlib import contextmanager

__all__ = [
    "get_config",
    "set_config",
    "config_context",
]

def _parse_env_bool(name, default=False):
    """
    Parse a boolean from an environment variable.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : bool, default False
        Value returned when the variable is absent.

    Returns
    -------
    bool
        True only when the variable is set to '1', 'true', 'yes', or 'on'
        (case-insensitive).

    Notes
    -----
    Developer note
        Bug fix: ``bool(os.environ.get(name, 'False'))`` evaluates to ``True``
        because ``bool('False') == True`` — any non-empty string is truthy.
        This helper uses an explicit allow-list for truthy string values.
    """
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _parse_env_int(name, default):
    """
    Parse an integer from an environment variable, falling back on error.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : int
        Value returned when the variable is absent or not a valid integer.

    Returns
    -------
    int
        Parsed integer, or *default* when parsing fails.

    Notes
    -----
    Developer note
        Bug fix: ``int(os.environ.get(name, str(default)))`` raises
        ``ValueError`` with no recovery when the variable holds a
        non-integer string.  This helper warns and falls back instead.
    """
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        import warnings
        warnings.warn(
            f"Environment variable {name}={val!r} is not a valid integer; "
            f"using default {default}.",
            UserWarning,
            stacklevel=0,
        )
        return default


_global_config = {
    "assume_finite": _parse_env_bool("SKPLT_ASSUME_FINITE", default=False),
    "working_memory": _parse_env_int("SKPLT_WORKING_MEMORY", default=1024),
    "display": "diagram",
    "array_api_dispatch": False,
    "transform_output": "default",
    "skip_parameter_validation": False,
}
_threadlocal = threading.local()


def _get_threadlocal_config():
    """
    Get a threadlocal **mutable** configuration.

    If the configuration does not exist, copy the default global configuration.
    """
    if not hasattr(_threadlocal, "global_config"):
        _threadlocal.global_config = _global_config.copy()
    return _threadlocal.global_config


def get_config():
    """
    Retrieve current values for configuration set by :func:`set_config`.

    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.

    See Also
    --------
    config_context : Context manager for global scikit-plots configuration.
    set_config : Set global scikit-plots configuration.

    Examples
    --------
    >>> import scikitplot
    >>> config = scikitplot.get_config()
    >>> config.keys()
    dict_keys([...])
    """
    # Return a copy of the threadlocal configuration so that users will
    # not be able to modify the configuration with the returned dict.
    return _get_threadlocal_config().copy()


def set_config(
    assume_finite=None,
    working_memory=None,
    display=None,
    array_api_dispatch=None,
    transform_output=None,
    skip_parameter_validation=None,
):
    """
    Set global scikit-plots configuration.

    .. versionadded:: 0.19

    Parameters
    ----------
    assume_finite : bool, default=None
        If True, validation for finiteness will be skipped,
        saving time, but leading to potential crashes. If
        False, validation for finiteness will be performed,
        avoiding error.  Global default: False.

        .. versionadded:: 0.4

    working_memory : int, default=None
        If set, scikit-plots will attempt to limit the size of temporary arrays
        to this number of MiB (per job when parallelised), often saving both
        computation time and memory on expensive operations that can be
        performed in chunks. Global default: 1024.

        .. versionadded:: 0.4

    display : {'text', 'diagram'}, default=None
        If 'diagram', estimators will be displayed as a diagram in a Jupyter
        lab or notebook context. If 'text', estimators will be displayed as
        text. Default is 'diagram'.

        .. versionadded:: 0.4

    array_api_dispatch : bool, default=None
        Use Array API dispatching when inputs follow the Array API standard.
        Default is False.

        See the :ref:`User Guide <array_api>` for more details.

        .. versionadded:: 0.4

    transform_output : str, default=None
        Configure output of `transform` and `fit_transform`.

        See :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py`
        for an example on how to use the API.

        - `"default"`: Default output format of a transformer
        - `"pandas"`: DataFrame output
        - `"polars"`: Polars output
        - `None`: Transform configuration is unchanged

        .. versionadded:: 0.4

    skip_parameter_validation : bool, default=None
        If `True`, disable the validation of the hyper-parameters' types and values in
        the fit method of estimators and for arguments passed to public helper
        functions. It can save time in some situations but can lead to low level
        crashes and exceptions with confusing error messages.

        Note that for data parameters, such as `X` and `y`, only type validation is
        skipped but validation with `check_array` will continue to run.

        .. versionadded:: 0.4

    See Also
    --------
    config_context : Context manager for global scikit-plots configuration.
    get_config : Retrieve current values of the global configuration.

    Examples
    --------
    >>> from scikitplot import set_config
    >>> set_config(display="diagram")  # doctest: +SKIP
    """
    local_config = _get_threadlocal_config()

    if assume_finite is not None:
        local_config["assume_finite"] = assume_finite
    if working_memory is not None:
        local_config["working_memory"] = working_memory
    if display is not None:
        local_config["display"] = display
    if array_api_dispatch is not None:
        from ..utils._array_api import _check_array_api_dispatch

        _check_array_api_dispatch(array_api_dispatch)
        local_config["array_api_dispatch"] = array_api_dispatch
    if transform_output is not None:
        local_config["transform_output"] = transform_output
    if skip_parameter_validation is not None:
        local_config["skip_parameter_validation"] = skip_parameter_validation


@contextmanager
def config_context(
    *,
    assume_finite=None,
    working_memory=None,
    display=None,
    array_api_dispatch=None,
    transform_output=None,
    skip_parameter_validation=None,
):
    """
    Context manager for global scikit-plots configuration.

    Parameters
    ----------
    assume_finite : bool, default=None
        If True, validation for finiteness will be skipped,
        saving time, but leading to potential crashes. If
        False, validation for finiteness will be performed,
        avoiding error. If None, the existing value won't change.
        The default value is False.

    working_memory : int, default=None
        If set, scikit-plots will attempt to limit the size of temporary arrays
        to this number of MiB (per job when parallelised), often saving both
        computation time and memory on expensive operations that can be
        performed in chunks. If None, the existing value won't change.
        The default value is 1024.

    display : {'text', 'diagram'}, default=None
        If 'diagram', estimators will be displayed as a diagram in a Jupyter
        lab or notebook context. If 'text', estimators will be displayed as
        text. If None, the existing value won't change.
        The default value is 'diagram'.

        .. versionadded:: 0.4

    array_api_dispatch : bool, default=None
        Use Array API dispatching when inputs follow the Array API standard.
        Default is False.

        See the :ref:`User Guide <array_api>` for more details.

        .. versionadded:: 0.4

    transform_output : str, default=None
        Configure output of `transform` and `fit_transform`.

        See :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py`
        for an example on how to use the API.

        - `"default"`: Default output format of a transformer
        - `"pandas"`: DataFrame output
        - `"polars"`: Polars output
        - `None`: Transform configuration is unchanged

        .. versionadded:: 0.4

    skip_parameter_validation : bool, default=None
        If `True`, disable the validation of the hyper-parameters' types and values in
        the fit method of estimators and for arguments passed to public helper
        functions. It can save time in some situations but can lead to low level
        crashes and exceptions with confusing error messages.

        Note that for data parameters, such as `X` and `y`, only type validation is
        skipped but validation with `check_array` will continue to run.

        .. versionadded:: 0.4

    Yields
    ------
    None.

    See Also
    --------
    set_config : Set global scikit-plots configuration.
    get_config : Retrieve current values of the global configuration.

    Notes
    -----
    All settings, not just those presently modified, will be returned to
    their previous values when the context manager is exited.

    Examples
    --------
    >>> import scikitplot
    >>> from scikitplot.utils.validation import assert_all_finite
    >>> with scikitplot.config_context(assume_finite=True):
    ...     assert_all_finite([float("nan")])
    >>> with scikitplot.config_context(assume_finite=True):
    ...     with scikitplot.config_context(assume_finite=False):
    ...         assert_all_finite([float("nan")])
    Traceback (most recent call last):
    ...
    ValueError: Input contains NaN...
    """
    old_config = get_config()
    set_config(
        assume_finite=assume_finite,
        working_memory=working_memory,
        display=display,
        array_api_dispatch=array_api_dispatch,
        transform_output=transform_output,
        skip_parameter_validation=skip_parameter_validation,
    )

    try:
        yield
    finally:
        set_config(**old_config)
