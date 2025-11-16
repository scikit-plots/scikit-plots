# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# This module was copied from the numpy project.
# https://github.com/numpy/numpy/blob/main/numpy/exceptions.py

# This module was copied from the astropy project.
# https://github.com/astropy/astropy/blob/main/astropy/utils/exceptions.py

# This module was copied from the mlflow project.
# https://github.com/mlflow/mlflow/blob/master/mlflow/exceptions.py

"""
Custom warnings and errors used across scikit-plots.

This module contains errors/exceptions and warnings of general use for
scikit-plots. Exceptions that are specific to a given subpackage should *not* be
here, but rather in the particular subpackage.
"""

import json as _json

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

__all__ = [
    # from astropy
    "ScikitplotBackwardsIncompatibleChangeWarning",
    "ScikitplotDeprecationWarning",
    "ScikitplotPendingDeprecationWarning",
    "ScikitplotUserWarning",
    "ScikitplotWarning",
    "DuplicateRepresentationWarning",
    # "NoValue",  # see: globals
    # from mlflow
    "ScikitplotException",
    "ExecutionException",
    "MissingConfigException",
    "InvalidUrlException",
    "ScikitplotTracingException",
    "ScikitplotTraceDataException",
    "ScikitplotTraceDataNotFound",
    "ScikitplotTraceDataCorrupted",
    # from numpy
    "ComplexWarning",
    "VisibleDeprecationWarning",
    "ModuleDeprecationWarning",
    "TooHardError",
    "AxisError",
    "DTypePromotionError",
    # from pip
    "CommandError",
]

# Disallow reloading this module so as to preserve the identities of the
# classes defined here.
if '_is_loaded' in globals():
    raise RuntimeError('Reloading scikitplot._globals is not allowed')
_is_loaded = True

######################################################################
## Numpy exceptions
######################################################################


# Exception used in shares_memory()
class TooHardError(RuntimeError):
    """``max_work`` was exceeded.

    This is raised whenever the maximum number of candidate solutions
    to consider specified by the ``max_work`` parameter is exceeded.
    Assigning a finite number to ``max_work`` may have caused the operation
    to fail.

    """
    pass


class ComplexWarning(RuntimeWarning):
    """
    The warning raised when casting a complex dtype to a real dtype.

    As implemented, casting a complex number to a real discards its imaginary
    part, but this behavior may not be what the user actually wants.

    """
    pass


class RankWarning(RuntimeWarning):
    """Matrix rank warning.

    Issued by polynomial functions when the design matrix is rank deficient.

    """
    pass


class DTypePromotionError(TypeError):
    """Multiple DTypes could not be converted to a common one.

    This exception derives from ``TypeError`` and is raised whenever dtypes
    cannot be converted to a single common one.  This can be because they
    are of a different category/class or incompatible instances of the same
    one (see Examples).

    Notes
    -----
    Many functions will use promotion to find the correct result and
    implementation.  For these functions the error will typically be chained
    with a more specific error indicating that no implementation was found
    for the input dtypes.

    Typically promotion should be considered "invalid" between the dtypes of
    two arrays when `arr1 == arr2` can safely return all ``False`` because the
    dtypes are fundamentally different.

    Examples
    --------
    Datetimes and complex numbers are incompatible classes and cannot be
    promoted:

    >>> import numpy as np
    >>> np.result_type(np.dtype("M8[s]"), np.complex128)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
     ...
    DTypePromotionError: The DType <class 'numpy.dtype[datetime64]'> could not
    be promoted by <class 'numpy.dtype[complex128]'>. This means that no common
    DType exists for the given inputs. For example they cannot be stored in a
    single array unless the dtype is `object`. The full list of DTypes is:
    (<class 'numpy.dtype[datetime64]'>, <class 'numpy.dtype[complex128]'>)

    For example for structured dtypes, the structure can mismatch and the
    same ``DTypePromotionError`` is given when two structured dtypes with
    a mismatch in their number of fields is given:

    >>> dtype1 = np.dtype([("field1", np.float64), ("field2", np.int64)])
    >>> dtype2 = np.dtype([("field1", np.float64)])
    >>> np.promote_types(dtype1, dtype2)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
     ...
    DTypePromotionError: field names `('field1', 'field2')` and `('field1',)`
    mismatch.

    """  # noqa: E501
    pass


class AxisError(ValueError, IndexError):
    """Axis supplied was invalid.

    This is raised whenever an ``axis`` parameter is specified that is larger
    than the number of array dimensions.
    For compatibility with code written against older numpy versions, which
    raised a mixture of :exc:`ValueError` and :exc:`IndexError` for this
    situation, this exception subclasses both to ensure that
    ``except ValueError`` and ``except IndexError`` statements continue
    to catch ``AxisError``.

    Parameters
    ----------
    axis : int or str
        The out of bounds axis or a custom exception message.
        If an axis is provided, then `ndim` should be specified as well.
    ndim : int, optional
        The number of array dimensions.
    msg_prefix : str, optional
        A prefix for the exception message.

    Attributes
    ----------
    axis : int, optional
        The out of bounds axis or ``None`` if a custom exception
        message was provided. This should be the axis as passed by
        the user, before any normalization to resolve negative indices.

        .. versionadded:: 1.22
    ndim : int, optional
        The number of array dimensions or ``None`` if a custom exception
        message was provided.

        .. versionadded:: 1.22


    Examples
    --------
    >>> import numpy as np
    >>> array_1d = np.arange(10)
    >>> np.cumsum(array_1d, axis=1)
    Traceback (most recent call last):
      ...
    numpy.exceptions.AxisError: axis 1 is out of bounds for array of dimension 1

    Negative axes are preserved:

    >>> np.cumsum(array_1d, axis=-2)
    Traceback (most recent call last):
      ...
    numpy.exceptions.AxisError: axis -2 is out of bounds for array of dimension 1

    The class constructor generally takes the axis and arrays'
    dimensionality as arguments:

    >>> print(np.exceptions.AxisError(2, 1, msg_prefix='error'))
    error: axis 2 is out of bounds for array of dimension 1

    Alternatively, a custom exception message can be passed:

    >>> print(np.exceptions.AxisError('Custom error message'))
    Custom error message

    """

    __slots__ = ("_msg", "axis", "ndim")

    def __init__(self, axis, ndim=None, msg_prefix=None):
        if ndim is msg_prefix is None:
            # single-argument form: directly set the error message
            self._msg = axis
            self.axis = None
            self.ndim = None
        else:
            self._msg = msg_prefix
            self.axis = axis
            self.ndim = ndim

    def __str__(self):
        axis = self.axis
        ndim = self.ndim

        if axis is ndim is None:
            return self._msg
        else:
            msg = f"axis {axis} is out of bounds for array of dimension {ndim}"
            if self._msg is not None:
                msg = f"{self._msg}: {msg}"
            return msg


######################################################################
## ModuleDeprecationWarning class
######################################################################


class ModuleDeprecationWarning(DeprecationWarning):
    """Module deprecation warning.

    .. warning::

        This warning should not be used, since nose testing is not relevant
        anymore.

    The nose tester turns ordinary Deprecation warnings into test failures.
    That makes it hard to deprecate whole modules, because they get
    imported by default. So this is a special Deprecation warning that the
    nose tester will let pass without making tests fail.

    """
    pass


# class ModuleDeprecationWarning(DeprecationWarning):
#     """
#     Module deprecation warning class.

#     This custom warning class is used to signal the deprecation of an entire module.
#     The `nose` testing framework treats ordinary `DeprecationWarning` as test failures,
#     which makes it challenging to deprecate whole modules. To address this, this special
#     `ModuleDeprecationWarning` is defined, which the `nose` tester will allow without
#     causing test failures.

#     This is especially useful when deprecating entire modules or submodules without
#     breaking existing tests.

#     Attributes
#     ----------
#     __module__ : str
#         The module in which this warning is defined, set to 'scikitplot'.

#     Methods
#     -------
#     __module__
#         A string representing the module that contains this warning.

#     """

#     # Set the module for the warning to 'scikitplot'
#     __module__: str = "scikitplot"


# ModuleDeprecationWarning.__module__ = 'scikitplot'

######################################################################
## VisibleDeprecationWarning class
######################################################################


class VisibleDeprecationWarning(UserWarning):
    """Visible deprecation warning.

    By default, python will not show deprecation warnings, so this class
    can be used when a very visible warning is helpful, for example because
    the usage is most likely a user bug.

    """
    pass


# class VisibleDeprecationWarning(UserWarning):
#     """
#     Visible deprecation warning class.

#     In Python, deprecation warnings are usually suppressed by default. This custom warning
#     class is designed to make deprecation warnings more visible, which is useful when
#     the usage is likely a user mistake or bug. This class ensures that the warning is shown
#     to the user more prominently, alerting them about deprecated functionality.

#     It is useful in situations where deprecation indicates potential issues with the
#     user's code and immediate attention is required.

#     Attributes
#     ----------
#     __module__ : str
#         The module in which this warning is defined, set to 'scikitplot'.

#     Methods
#     -------
#     __module__
#         A string representing the module that contains this warning.

#     """

#     # Set the module for the warning to 'scikitplot'
#     __module__: str = "scikitplot"


# VisibleDeprecationWarning.__module__ = 'scikitplot'

######################################################################
## Astropy exceptions
######################################################################

class ScikitplotWarning(Warning):
    """
    The base warning class from which all scikit-plots warnings should inherit.

    Any warning inheriting from this class is handled by the scikit-plots logger.
    """


class ScikitplotUserWarning(UserWarning, ScikitplotWarning):
    """
    The primary warning class for scikit-plots.

    Use this if you do not need a specific sub-class.
    """


class ScikitplotDeprecationWarning(ScikitplotWarning):
    """
    A warning class to indicate a deprecated feature.
    """


class ScikitplotPendingDeprecationWarning(PendingDeprecationWarning, ScikitplotWarning):
    """
    A warning class to indicate a soon-to-be deprecated feature.
    """


class ScikitplotBackwardsIncompatibleChangeWarning(ScikitplotWarning):
    """
    A warning class indicating a change in astropy that is incompatible
    with previous versions.

    The suggested procedure is to issue this warning for the version in
    which the change occurs, and remove it for all following versions.
    """


class DuplicateRepresentationWarning(ScikitplotWarning):
    """
    A warning class indicating a representation name was already registered.
    """


# class _NoValue:
#     """Special keyword value.

#     This class may be used as the default value assigned to a
#     deprecated keyword in order to check if it has been given a user
#     defined value.
#     """

#     def __repr__(self):
#         return "astropy.utils.exceptions.NoValue"


# NoValue = _NoValue()

######################################################################
## Mlflow exceptions
######################################################################


class ScikitplotException(Exception):  # noqa: N818
    """
    Generic exception thrown to surface failure information about external-facing operations.

    The error message associated with this exception may be exposed to clients in HTTP responses
    for debugging purposes. If the error text is sensitive, raise a generic `Exception` object
    instead.
    """

    def __init__(self, message, error_code=0, **kwargs):
        """
        Thrown to surface failure information about external-facing operations.

        Parameters
        ----------
        message:
            The message or exception describing the error that occurred. This will be
            included in the exception's serialized JSON representation.
        error_code:
            An appropriate error code for the error that occurred; it will be
            included in the exception's serialized JSON representation. This should
            be one of the codes listed in the `scikitplot.protos.databricks_pb2` proto.
        kwargs:
            Additional key-value pairs to include in the serialized JSON representation
            of the ScikitplotException.
        """
        try:
            self.error_code = error_code
        except (ValueError, TypeError):
            self.error_code = 0
        message = str(message)
        self.message = message
        self.json_kwargs = kwargs
        super().__init__(message)

    def serialize_as_json(self):  # noqa: D102
        exception_dict = {"error_code": self.error_code, "message": self.message}
        exception_dict.update(self.json_kwargs)
        return _json.dumps(exception_dict)

    def get_http_status_code(self):  # noqa: D102
        return 500

    @classmethod
    def invalid_parameter_value(cls, message, **kwargs):
        """
        Construct an `ScikitplotException` object with the `INVALID_PARAMETER_VALUE` error code.

        Parameters
        ----------
        message:
            The message describing the error that occurred. This will be included in the
            exception's serialized JSON representation.
        kwargs:
            Additional key-value pairs to include in the serialized JSON representation
            of the ScikitplotException.
        """
        return cls(message, error_code=0, **kwargs)


class CommandError(ScikitplotException):
    """Raised when there is an error in command-line arguments"""


class ExecutionException(ScikitplotException):
    """Exception thrown when executing a project fails."""


class MissingConfigException(ScikitplotException):
    """Exception thrown when expected configuration file/directory not found."""


class InvalidUrlException(ScikitplotException):
    """Exception thrown when a http request fails to send due to an invalid URL."""


class _UnsupportedMultipartUploadException(ScikitplotException):
    """Exception thrown when multipart upload is unsupported by an artifact repository."""

    MESSAGE = "Multipart upload is not supported for the current artifact repository"

    def __init__(self):
        super().__init__(self.MESSAGE, error_code=0)


class ScikitplotTracingException(ScikitplotException):
    """
    Exception thrown from tracing logic.

    Tracing logic should not block the main execution flow in general, hence this exception
    is used to distinguish tracing related errors and handle them properly.
    """

    def __init__(self, message, error_code=0):
        super().__init__(message, error_code=error_code)


class ScikitplotTraceDataException(ScikitplotTracingException):
    """Exception thrown for trace data related error."""

    def __init__(
        self,
        error_code: str,
        request_id: "str | None" = None,
        artifact_path: "str | None" = None,
    ):
        if request_id:
            self.ctx = f"request_id={request_id}"
        elif artifact_path:
            self.ctx = f"path={artifact_path}"

        if error_code == -1:
            super().__init__(
                f"Trace data not found for {self.ctx}", error_code=error_code
            )
        elif error_code == -1:
            super().__init__(
                f"Trace data is corrupted for {self.ctx}", error_code=error_code
            )


class ScikitplotTraceDataNotFound(ScikitplotTraceDataException):
    """Exception thrown when trace data is not found."""

    def __init__(
        self, request_id: "str | None" = None, artifact_path: "str | None" = None
    ):
        super().__init__("NOT_FOUND", request_id, artifact_path)


class ScikitplotTraceDataCorrupted(ScikitplotTraceDataException):
    """Exception thrown when trace data is corrupted."""

    def __init__(
        self, request_id: "str | None" = None, artifact_path: "str | None" = None
    ):
        super().__init__("INVALID_STATE", request_id, artifact_path)


def __getattr__(name: str):
    if name in ("ErfaError", "ErfaWarning"):
        import warnings

        warnings.warn(
            f"Importing `from scikitplot.exceptions import {name}` was deprecated "
            "in version 0.4 and will stop working in a future version. "
            f"Instead, please use\n`from erfa import {name}`\n\n",
            category=ScikitplotDeprecationWarning,
            stacklevel=1,
        )

        # import erfa
        return getattr(__import__('erfa'), name, None)

    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}.")
