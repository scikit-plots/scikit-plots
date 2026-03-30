# scikitplot/exceptions.py
#
# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
#
# This module was copied from the numpy project.
# https://github.com/numpy/numpy/blob/main/numpy/exceptions.py
#
# This module was copied from the astropy project.
# https://github.com/astropy/astropy/blob/main/astropy/utils/exceptions.py
#
# This module was copied from the mlflow project.
# https://github.com/mlflow/mlflow/blob/master/mlflow/exceptions.py

"""
Custom warnings and errors used across scikit-plots.

This module contains errors/exceptions and warnings of general use for
scikit-plots. Exceptions that are specific to a given subpackage should *not* be
here, but rather in the particular subpackage.

The builtin warning hierarchy is::

    Warning
    ├── DeprecationWarning          ← base of ScikitplotDeprecationWarning
    ├── PendingDeprecationWarning   ← base of ScikitplotPendingDeprecationWarning
    └── UserWarning                 ← base of ScikitplotUserWarning

Notes
-----
**Developer notes**

* Do *not* import this module from ``__init__.py`` before it has been fully
  bootstrapped.  The ``_is_loaded`` guard prevents accidental re-import, which
  would create duplicate class identities and break ``isinstance`` checks across
  reload boundaries.

* ``error_code`` in :class:`ScikitplotException` and its subclasses may be
  either an ``int`` (legacy convention) or a ``str`` sentinel (e.g.
  ``"NOT_FOUND"``, ``"INVALID_STATE"``).  Both forms are preserved verbatim in
  the JSON payload; callers must not assume a particular type.

* :func:`__getattr__` at module scope handles the deprecated ``ErfaError`` /
  ``ErfaWarning`` aliases.  ``stacklevel=2`` ensures the warning frame points
  at the *caller* rather than at the ``__getattr__`` body itself.
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
    # from mlflow
    "ScikitplotException",
    "ExecutionException",
    "MissingConfigException",
    "InvalidUrlException",
    "ScikitplotTracingException",
    "ScikitplotTraceDataException",
    "ScikitplotTraceDataNotFound",
    "ScikitplotTraceDataCorrupted",
    "CommandError",
    # from numpy
    "ComplexWarning",
    "RankWarning",
    "VisibleDeprecationWarning",
    "ModuleDeprecationWarning",
    "TooHardError",
    "AxisError",
    "DTypePromotionError",
]

# ---------------------------------------------------------------------------
# Reload guard
# Disallow reloading this module so as to preserve the identities of the
# classes defined here.  A reload would create new class objects, breaking
# ``isinstance`` checks for any exception/warning instance that was raised
# before the reload.
# ---------------------------------------------------------------------------
if '_is_loaded' in globals():
    raise RuntimeError('Reloading scikitplot.exceptions is not allowed')
_is_loaded = True


######################################################################
## Numpy exceptions
######################################################################


class TooHardError(RuntimeError):
    """``max_work`` was exceeded.

    This is raised whenever the maximum number of candidate solutions
    to consider specified by the ``max_work`` parameter is exceeded.
    Assigning a finite number to ``max_work`` may have caused the operation
    to fail.
    """
    pass


class ComplexWarning(RuntimeWarning):
    """The warning raised when casting a complex dtype to a real dtype.

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
    two arrays when ``arr1 == arr2`` can safely return all ``False`` because the
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
        The out-of-bounds axis, or a custom exception message string.
        When a custom message is supplied, ``ndim`` must be omitted (or
        ``None``) and ``msg_prefix`` must be ``None``.
    ndim : int, optional
        The number of array dimensions.  Required when ``axis`` is an int.
    msg_prefix : str, optional
        A prefix prepended to the auto-generated message, separated by
        ``": "``.

    Attributes
    ----------
    axis : int or None
        The out-of-bounds axis, or ``None`` for the single-argument form.
    ndim : int or None
        The number of array dimensions, or ``None`` for the single-argument
        form.

    Notes
    -----
    .. versionadded:: 1.22
        The ``axis`` and ``ndim`` attributes.

    The ``__slots__`` declaration prevents creation of a ``__dict__``,
    keeping instances lightweight.

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
            # Single-argument form: ``axis`` is a plain string message.
            self._msg = axis
            self.axis = None
            self.ndim = None
        else:
            # Structured form: ``axis`` is an int, ``ndim`` is an int.
            # ``msg_prefix`` is an optional string prefix (may be None).
            self._msg = msg_prefix
            self.axis = axis
            self.ndim = ndim

    def __str__(self):
        axis = self.axis
        ndim = self.ndim

        if axis is ndim is None:
            # Single-argument form – return the verbatim message.
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


######################################################################
## Astropy exceptions
######################################################################


class ScikitplotWarning(Warning):
    """The base warning class from which all scikit-plots warnings should inherit.

    Any warning inheriting from this class is handled by the scikit-plots logger.
    """


class ScikitplotUserWarning(UserWarning):
    """The primary warning class for scikit-plots.

    Use this if you do not need a specific sub-class.
    """


class ScikitplotDeprecationWarning(DeprecationWarning):
    """A warning class to indicate a deprecated feature."""


class ScikitplotPendingDeprecationWarning(PendingDeprecationWarning):
    """A warning class to indicate a soon-to-be deprecated feature."""


class ScikitplotBackwardsIncompatibleChangeWarning(ScikitplotWarning):
    """A warning class indicating a change in scikit-plots that is incompatible
    with previous versions.

    The suggested procedure is to issue this warning for the version in
    which the change occurs, and remove it for all following versions.
    """


class DuplicateRepresentationWarning(ScikitplotWarning):
    """A warning class indicating a representation name was already registered."""


######################################################################
## Mlflow exceptions
######################################################################


class ScikitplotException(Exception):  # noqa: N818
    """Generic exception thrown to surface failure information about external-facing operations.

    The error message associated with this exception may be exposed to clients
    in HTTP responses for debugging purposes.  If the error text is sensitive,
    raise a generic :exc:`Exception` object instead.

    Parameters
    ----------
    message : str or Exception
        The message or exception describing the error that occurred.  Converted
        to ``str`` unconditionally.  Included in the serialised JSON payload.
    error_code : int or str, optional
        An error code for the error that occurred.  May be an integer
        (legacy convention) or a string sentinel such as ``"NOT_FOUND"``.
        Defaults to ``0``.  Included in the serialised JSON payload.
    **kwargs
        Additional key-value pairs included in the serialised JSON payload.
        Values must be JSON-serialisable; non-serialisable values are
        coerced to their ``str`` representation during serialisation.

    Attributes
    ----------
    error_code : int or str
        The error code as supplied.
    message : str
        The string form of the supplied message.
    json_kwargs : dict
        The extra keyword arguments to include in the JSON payload.

    Notes
    -----
    **User note** – catch this class (or its subclasses) when you need to
    distinguish scikit-plots operational errors from other :exc:`Exception`
    types.

    **Developer note** – ``error_code`` is stored verbatim; do not apply
    integer-only logic elsewhere in the codebase without first checking its
    type.  The ``try/except`` that previously guarded the attribute assignment
    was dead code (attribute assignment never raises :exc:`ValueError` or
    :exc:`TypeError`) and has been removed.
    """

    def __init__(self, message, error_code=0, **kwargs):
        self.error_code = error_code
        message = str(message)
        self.message = message
        self.json_kwargs = kwargs
        super().__init__(message)

    def serialize_as_json(self):
        """Serialise the exception as a JSON string.

        Returns
        -------
        str
            A JSON-encoded object containing at minimum ``error_code`` and
            ``message``, plus any extra keyword arguments supplied at
            construction time.

        Notes
        -----
        If any value in ``json_kwargs`` is not JSON-serialisable, the entire
        payload falls back to a ``str``-coerced representation so that
        serialisation never raises.
        """
        exception_dict = {"error_code": self.error_code, "message": self.message}
        exception_dict.update(self.json_kwargs)
        try:
            return _json.dumps(exception_dict)
        except (TypeError, ValueError):
            # Coerce every value to str to guarantee a valid JSON response.
            safe_dict = {k: str(v) for k, v in exception_dict.items()}
            return _json.dumps(safe_dict)

    def get_http_status_code(self):
        """Return the HTTP status code associated with this exception.

        Returns
        -------
        int
            Always ``500`` for the base class.
        """
        return 500

    @classmethod
    def invalid_parameter_value(cls, message, **kwargs):
        """Construct a :class:`ScikitplotException` with the ``INVALID_PARAMETER_VALUE`` code.

        Parameters
        ----------
        message : str
            The message describing the error that occurred.  Included in the
            exception's serialised JSON representation.
        **kwargs
            Additional key-value pairs to include in the serialised JSON
            representation.

        Returns
        -------
        ScikitplotException
            A new instance with ``error_code=0``.
        """
        return cls(message, error_code=0, **kwargs)


class CommandError(ScikitplotException):
    """Raised when there is an error in command-line arguments."""


class ExecutionException(ScikitplotException):
    """Exception thrown when executing a project fails."""


class MissingConfigException(ScikitplotException):
    """Exception thrown when expected configuration file/directory not found."""


class InvalidUrlException(ScikitplotException):
    """Exception thrown when a http request fails to send due to an invalid URL."""


class _UnsupportedMultipartUploadException(ScikitplotException):
    """Exception thrown when multipart upload is unsupported by an artifact repository.

    Notes
    -----
    This class is intentionally private (underscore prefix) and excluded from
    ``__all__``; it is raised only by internal upload machinery.
    """

    MESSAGE = "Multipart upload is not supported for the current artifact repository"

    def __init__(self):
        super().__init__(self.MESSAGE, error_code=0)


class ScikitplotTracingException(ScikitplotException):
    """Exception thrown from tracing logic.

    Tracing logic should not block the main execution flow in general; hence
    this exception is used to distinguish tracing-related errors and handle
    them gracefully without surfacing them to end users.

    Parameters
    ----------
    message : str or Exception
        Description of the tracing error.
    error_code : int or str, optional
        Error code; defaults to ``0``.
    """

    def __init__(self, message, error_code=0):
        super().__init__(message, error_code=error_code)


class ScikitplotTraceDataException(ScikitplotTracingException):
    """Exception thrown for trace data related errors.

    Parameters
    ----------
    error_code : str
        A string identifier for the error kind.  Recognised values:

        ``"NOT_FOUND"``
            Trace data could not be located.
        ``"INVALID_STATE"``
            Trace data was found but is corrupted or in an unexpected state.

        Any other value produces a generic error message.
    request_id : str or None, optional
        The trace request identifier to include in the error message.
        Takes priority over *artifact_path* when both are supplied.
    artifact_path : str or None, optional
        The artifact path to include in the error message.  Used only when
        *request_id* is ``None``.

    Attributes
    ----------
    ctx : str
        The context string embedded in the error message.  One of
        ``"request_id=<value>"``, ``"path=<value>"``, or ``"unknown"``.

    Notes
    -----
    **Bug fixes applied (three defects present in the original upstream):**

    1. ``self.ctx`` was never initialised when both *request_id* and
       *artifact_path* were ``None``, causing ``AttributeError`` inside
       ``super().__init__()``.  Now defaults to ``"unknown"``.

    2. The second branch was ``elif error_code == -1:`` — identical to the
       ``if`` above it, making the corrupted-state path dead code.  Replaced
       with the correct string sentinel ``"INVALID_STATE"``, matching what
       :class:`ScikitplotTraceDataCorrupted` actually passes.

    3. :class:`ScikitplotTraceDataNotFound` passes ``"NOT_FOUND"`` and
       :class:`ScikitplotTraceDataCorrupted` passes ``"INVALID_STATE"`` as
       *error_code*, but the old comparisons used integer literals (``-1``),
       so no branch ever matched and ``super().__init__()`` was never called,
       leaving the exception object in an uninitialised state.  Comparisons
       now use the correct string sentinels.
    """

    def __init__(
        self,
        error_code: str,
        request_id: "str | None" = None,
        artifact_path: "str | None" = None,
    ):
        # Fix 1: always initialise self.ctx so super().__init__() never
        # raises AttributeError regardless of which arguments are supplied.
        if request_id is not None:
            self.ctx = f"request_id={request_id}"
        elif artifact_path is not None:
            self.ctx = f"path={artifact_path}"
        else:
            self.ctx = "unknown"

        # Fix 2 & 3: compare against the string sentinels that the
        # subclasses actually pass, not against integer literals.
        if error_code == "NOT_FOUND":
            super().__init__(
                f"Trace data not found for {self.ctx}", error_code=error_code
            )
        elif error_code == "INVALID_STATE":
            super().__init__(
                f"Trace data is corrupted for {self.ctx}", error_code=error_code
            )
        else:
            super().__init__(
                f"Trace data error ({error_code!r}) for {self.ctx}",
                error_code=error_code,
            )


class ScikitplotTraceDataNotFound(ScikitplotTraceDataException):
    """Exception thrown when trace data is not found.

    Parameters
    ----------
    request_id : str or None, optional
        Trace request identifier to embed in the error message.
    artifact_path : str or None, optional
        Artifact path to embed in the error message when *request_id* is
        ``None``.
    """

    def __init__(
        self, request_id: "str | None" = None, artifact_path: "str | None" = None
    ):
        super().__init__("NOT_FOUND", request_id, artifact_path)


class ScikitplotTraceDataCorrupted(ScikitplotTraceDataException):
    """Exception thrown when trace data is corrupted.

    Parameters
    ----------
    request_id : str or None, optional
        Trace request identifier to embed in the error message.
    artifact_path : str or None, optional
        Artifact path to embed in the error message when *request_id* is
        ``None``.
    """

    def __init__(
        self, request_id: "str | None" = None, artifact_path: "str | None" = None
    ):
        super().__init__("INVALID_STATE", request_id, artifact_path)


######################################################################
## Module-level __getattr__ for deprecated aliases
######################################################################


def __getattr__(name: str):
    """Handle deprecated attribute access at module level.

    Parameters
    ----------
    name : str
        The attribute name being accessed.

    Returns
    -------
    type
        The requested class from the ``erfa`` package.

    Raises
    ------
    AttributeError
        If *name* is not a known deprecated alias.

    Notes
    -----
    ``stacklevel=2`` ensures the :exc:`DeprecationWarning` is reported at
    the *caller* frame (the ``import`` or ``getattr`` statement in user
    code), not inside this function body.
    """
    if name in ("ErfaError", "ErfaWarning"):
        import warnings

        warnings.warn(
            f"Importing `from scikitplot.exceptions import {name}` was deprecated "
            "in version 0.4 and will stop working in a future version. "
            f"Instead, please use\n`from erfa import {name}`\n\n",
            category=ScikitplotDeprecationWarning,
            # Fix: stacklevel=2 → warning frame is the caller, not __getattr__
            stacklevel=2,
        )

        return getattr(__import__('erfa'), name)

    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}.")
