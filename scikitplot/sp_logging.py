# pylint: disable=import-error
# pylint: disable=unused-argument
# pylint: disable=broad-exception-caught
# pylint: disable=logging-fstring-interpolation
# pylint: disable=invalid-name
# pylint: disable=import-outside-toplevel
# pylint: disable=reimported
# pylint: disable=too-many-lines

# ruff: noqa: UP037

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
:py:mod:`~.sp_logging` (alias, :py:obj:`~.logger`) module provide logging utilities.

Inspired by `"Tensorflow's logging system"
<https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/tf_logging.py#L94>`_.

This module provides advanced logging utilities for Python applications,
including support for singleton-based logging with customizable formatters,
handlers, and thread-safety.

It extends Python's standard logging library to enhance usability
and flexibility for large-scale projects.

Scikit-plots logging helpers, supports vendoring.

Module Dependencies:
- Python standard library: :py:mod:`logging`


References
----------
.. [1] `Tensorflow contributors. (2025).
   "Tensorflow's logging system"
   Tensorflow. https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/tf_logging.py#L94
   <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/tf_logging.py#L94>`_
"""

from __future__ import annotations

# import inspect
import logging as _logging
import os
import sys
import threading  # Python 2 to thread.get_ident
import traceback
from logging import (
    CRITICAL,
    DEBUG,
    ERROR,
    FATAL,
    INFO,
    NOTSET,
    WARNING,
)
from logging import (
    WARNING as WARN,  # logging WARN deprecated
)
from typing import TYPE_CHECKING

# Runtime-safe imports for type hints (avoids runtime overhead)
if TYPE_CHECKING:
    from typing import (  # noqa: F401
        IO,
        Callable,
        Optional,
        TypeVar,
    )

    # Define a generic callable type for decorator functions
    F = TypeVar("F", bound="Callable[..., any]")

__all__ = [
    "CRITICAL",
    "DEBUG",
    "ERROR",
    "FATAL",
    "INFO",
    "NOTSET",
    "WARN",
    "WARNING",
    "AlwaysStdErrHandler",
    "GoogleLogFormatter",
    "_default_log_level",
    "_get_thread_id",
    "_is_jupyter_notebook",
    "critical",
    "debug",
    "error",
    "error_log",
    "exception",
    "fatal",
    "getEffectiveLevel",
    "get_logger",  # func based
    "get_verbosity",
    "info",
    "log",
    "log_every_n",
    "log_first_n",
    "log_if",
    "setLevel",
    "set_verbosity",
    "vlog",
    "warn",
    "warning",
    # "SpLogger",  # class based
]

######################################################################
## module level variables
######################################################################

# Don't use this directly. Use get_logger() instead.
_logger = None
# Reentrant: The same thread can acquire the lock multiple times without blocking.
# _logger_lock = threading.Lock()
_logger_lock = threading.RLock()

_level_names = {
    CRITICAL: "CRITICAL",
    FATAL: "CRITICAL",
    ERROR: "ERROR",
    WARNING: "WARNING",
    WARN: "WARNING",
    INFO: "INFO",
    DEBUG: "DEBUG",
    NOTSET: "NOTSET",
}

# Mask to convert integer thread ids to unsigned quantities for logging
# purposes
_THREAD_ID_MASK = 2 * sys.maxsize + 1

_log_prefix = None  # later set to google2_log_prefix

# Counter to keep track of number of log entries per token.
_log_counter_per_token = {}

######################################################################
## sp_logging falling back to python logging
######################################################################


def __getattr__(name: str):
    """
    Dynamic attribute resolver for the sp_logging module.

    If an attribute is not found in this module, this function attempts to
    retrieve it from the standard Python 'logging' module as a fallback.
    This is useful for making a custom logging wrapper behave like the
    built-in 'logging' module for common usage.

    Parameters
    ----------
    name : str
        The attribute name being accessed.

    Returns
    -------
    any
        The corresponding attribute from the built-in 'logging' module if found.

    Raises
    ------
    AttributeError
        If the attribute is not found in the logging module either.

    Notes
    -----
    This function makes it possible to do things like:

    >>> from sp_logging import DEBUG, warning
    >>> warning("This will behave like logging.warning")

    Examples
    --------
    >>> hasattr(sp_logging, "INFO")
    True  # Delegated to logging.INFO

    >>> sp_logging.NonexistentAttribute
    AttributeError: Module 'sp_logging' has no attribute 'NonexistentAttribute'...
    """
    try:
        # Attempt to retrieve attribute from the logging module
        attr = getattr(_logging, name, None)  # or getattr(get_logger(), name, None)
        get_logger().debug(f"Falling back to logging.{name}")
        return attr
    except AttributeError as e:
        # Raise a clear error if not found in both sp_logging and logging
        raise AttributeError(
            f"Module 'sp_logging' has no attribute '{name}', "
            f"and it was not found in the standard 'logging' module either."
        ) from e


######################################################################
## Determine default log level
######################################################################


def _is_jupyter_notebook() -> bool:
    """
    Detect whether the current Python environment is a Jupyter notebook.

    This function attempts to determine whether the code is executing inside
    a Jupyter notebook. It uses a combination of lightweight heuristics
    (environment variables and loaded modules) and, as a fallback, attempts to
    inspect the IPython shell configuration.

    Returns
    -------
    bool
        True if the code is running in a Jupyter notebook environment,
        False otherwise.

    Notes
    -----
    - This detection is based on common patterns in Jupyter environments.
    - It checks for known environment variables (like 'JPY_PARENT_PID') and
      loaded modules (e.g., 'ipykernel'), which are typically present in Jupyter sessions.
    - If those checks fail, it attempts to import IPython and inspect the shell
      configuration for signs of an active IPython kernel.
    - The function avoids importing heavy dependencies unless necessary.
    - This method may not distinguish between classic Jupyter, JupyterLab,
      or VSCode notebooks, but it covers most interactive notebook cases.

    Examples
    --------
    >>> _is_jupyter_notebook()
    True  # if running inside a notebook

    >>> _is_jupyter_notebook()
    False  # if running from a standard script or terminal
    """
    ## First, check environment clues — these are fast and commonly reliable
    if "JPY_PARENT_PID" in os.environ:
        return True  # Jupyter sets this to identify the parent notebook process
    if "ipykernel" in sys.modules:
        return True  # Jupyter notebooks always use ipykernel
    ## Fallback: try importing and inspecting the IPython shell
    try:
        try:
            from IPython import get_ipython  # type: ignore[reportMissingModuleSource]
        except ImportError:
            get_ipython = None
        ## If no IPython shell is active, this is likely not a notebook
        if (
            get_ipython is None  # type: ignore[reportMissingModuleSource]
            or not callable(get_ipython)  # type: ignore[reportMissingModuleSource]
        ):
            return False
        ## Check if the IPython shell is configured as a kernel app (notebook backend)
        if "IPKernelApp" in get_ipython().config:  # type: ignore[reportMissingModuleSource]
            # shell = get_ipython().__class__.__name__
            return True
    except (ImportError, AttributeError, NameError, Exception):
        # any error during inspection implies not running in Jupyter
        return False
    # If none of the above checks confirm Jupyter, return False
    return False


def _default_log_level(verbose: bool = False) -> int:
    """
    Determine the default log level based on environment and verbosity.

    This function checks the SKPLT_VERBOSE environment variable to decide
    whether to enable verbose (DEBUG-level) logging. If the environment
    variable is not set, it uses the value of the verbose argument.

    Parameters
    ----------
    verbose : bool, optional
        Whether to enable verbose logging. Defaults to False. Ignored if
        SKPLT_VERBOSE is set.

    Returns
    -------
    int
        The logging level (e.g., logging.DEBUG or logging.WARNING).

    Notes
    -----
    - If SKPLT_VERBOSE is set (to any non-empty string), DEBUG logging is enabled.
    - If SKPLT_VERBOSE is unset, the function uses the `verbose` parameter instead.
    - Useful for setting a default log level in CLI tools or libraries.

    Examples
    --------
    >>> _default_log_level()
    30  # logging.WARNING

    >>> _default_log_level(verbose=True)
    10  # logging.DEBUG

    >>> os.environ["SKPLT_VERBOSE"] = "1"
    >>> _default_log_level()
    10  # logging.DEBUG
    """
    env_value = os.getenv("SKPLT_VERBOSE")
    is_verbose = bool(env_value) if env_value is not None else verbose
    return _logging.DEBUG if is_verbose else _logging.WARNING


######################################################################
## THREAD ID helper
######################################################################


def _get_thread_id(thread_id_mask: int = _THREAD_ID_MASK) -> int:
    """
    Get the ID of the current thread, masked as an unsigned integer.

    This is useful for logging and debugging where thread identifiers
    must be displayed or compared as positive integers, even if the system
    may return signed values.

    Parameters
    ----------
    thread_id_mask : int, optional
        A bitmask to apply to the thread ID to convert it to an unsigned quantity.
        Defaults to a value that ensures correct masking across platforms.

    Returns
    -------
    int
        The current thread's ID, bitmasked as an unsigned integer.

    Notes
    -----
    - This function uses threading.get_ident() to retrieve the native thread ID.
    - The default bitmask (_THREAD_ID_MASK) converts signed thread IDs to unsigned
      representations for consistent logging and storage.
    - Useful in systems where thread IDs might appear negative on certain architectures.

    Examples
    --------
    >>> _get_thread_id()
    140712536721152

    >>> _get_thread_id(0xFFFFFFFF)
    32518  # Lower 32 bits of the thread ID
    """
    # Fall back to default mask if a falsy value is passed
    mask = thread_id_mask or _THREAD_ID_MASK

    # Get the native thread ID
    thread_id = threading.get_ident()

    # Mask the ID to ensure it's treated as an unsigned integer
    return thread_id & mask


######################################################################
## _logger_find_caller, _GetFileAndLine
######################################################################


def _get_caller(offset=3):
    """Return a code and frame object for the lowest non-logging stack frame."""
    # Use sys._getframe().  This avoids creating a traceback object.
    # pylint: disable=protected-access
    f = sys._getframe(offset)
    # pylint: enable=protected-access
    our_file = f.f_code.co_filename
    f = f.f_back
    while f:
        code = f.f_code
        if code.co_filename != our_file:
            return code, f
        f = f.f_back
    return None, None


# The definition of `findCaller` changed in Python 3.2,
# and further changed in Python 3.8
if sys.version_info.major >= 3 and sys.version_info.minor >= 8:  # noqa: PLR2004, YTT204

    def _logger_find_caller(stack_info=False, stacklevel=1):
        code, frame = _get_caller(4)
        sinfo = None
        if stack_info:
            sinfo = "\n".join(traceback.format_stack())
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name, sinfo)
        else:  # noqa: RET505
            return "(unknown file)", 0, "(unknown function)", sinfo

elif (
    sys.version_info.major >= 3 and sys.version_info.minor >= 2  # noqa: PLR2004, YTT204
):  # noqa: PLR2004, YTT204

    def _logger_find_caller(stack_info=False):
        code, frame = _get_caller(4)
        sinfo = None
        if stack_info:
            sinfo = "\n".join(traceback.format_stack())
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name, sinfo)
        else:  # noqa: RET505
            return "(unknown file)", 0, "(unknown function)", sinfo

else:

    def _logger_find_caller():
        code, frame = _get_caller(4)
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name)
        else:  # noqa: RET505
            return "(unknown file)", 0, "(unknown function)"


def _GetFileAndLine():  # noqa: N802
    """Return (filename, linenumber) for the stack frame."""
    code, f = _get_caller()
    if not code:
        return ("<unknown>", 0)
    return (code.co_filename, f.f_lineno)


######################################################################
## google2 log_prefix
######################################################################


def _get_default_log_level_name() -> str:
    _level = _logging.getLogger().getEffectiveLevel()
    return _logging.getLevelName(_level)


def google2_log_prefix(level=None, timestamp=None, file_and_line=None):
    """Assemble a logline prefix using the google2 format."""
    # pylint: disable=global-variable-not-assigned
    # Remove the global statement if you're only reading
    # global _level_names  # noqa: PLW0602
    # pylint: enable=global-variable-not-assigned

    # Record current time
    import time

    now = timestamp or time.time()
    now_tuple = time.localtime(now)
    now_microsecond = int(1e6 * (now % 1.0))

    (filename, line) = file_and_line or _GetFileAndLine()
    basename = os.path.basename(filename)  # noqa: PTH119

    # Severity string
    severity = "I"
    level = level or _get_default_log_level_name()
    if level in _level_names:
        severity = _level_names[level][0]

    s = "%c %04d%02d%02d %02d:%02d:%02d.%06d %5d %s:%d] " % (  # noqa: UP031
        severity,  # level letter
        now_tuple[0],  # year
        now_tuple[1],  # month
        now_tuple[2],  # day
        now_tuple[3],  # hour
        now_tuple[4],  # min
        now_tuple[5],  # sec
        now_microsecond,  # microsec
        _get_thread_id(),
        basename,
        line,
    )

    return s  # noqa: RET504


_log_prefix = google2_log_prefix

######################################################################
## LogRecordFactory alternate or combined _logger_find_caller
## https://docs.python.org/3/library/logging.html#logrecord-objects
######################################################################

# _old_factory = _logging.getLogRecordFactory()

# def record_factory(*args, **kwargs):
#     """record_factory"""
#     record = _old_factory(*args, **kwargs)
#     try:
#         # Start from frame 0 and walk up
#         co_filename, f_lineno = _GetFileAndLine()
#         if co_filename and f_lineno:
#             record.caller_filename = os.path.basename(co_filename)  # noqa: PTH119
#             record.caller_lineno = f_lineno
#         else:
#             # Fallback: use standard info if walk failed
#             record.caller_filename = os.path.basename(record.pathname)  # noqa: PTH119
#             record.caller_lineno = record.lineno
#     except Exception:
#         record.caller_filename = record.filename
#         record.caller_lineno = record.lineno
#     return record

# # Register it globally
# _logging.setLogRecordFactory(record_factory)

######################################################################
## logging Formatter
######################################################################


class GoogleLogFormatter(_logging.Formatter):
    """
    A custom logging formatter inherited from :py:class:`~logging.Formatter`.

    That formats log messages in a Google-style format::

      >>> # Google-style format
      >>> `YYYY-MM-DD HH:MM:SS.mmmmmm logger_name log_level message`

    Parameters
    ----------
    datefmt : str, optional
        Date format for `asctime`. Default is '%Y-%m-%d %H:%M:%S'.
    default_time_format : str, optional
        Default time format. Default is '%Y-%m-%d %H:%M:%S'.
    default_msec_format : str, optional
        Default millisecond format. Default is '%s,%03d'.
    backend : {'json', 'pprint'}, optional
        Backend to use for formatting the log output. Default is None.
    use_datetime : bool, optional
        Whether to include microseconds in the timestamp using datetime. Default is True.

    Notes
    -----
    This formatter outputs logs in a structured format with the following fields if any:

    * `asctime`: The timestamp of the log entry.
    * `levelname`: The log level (e.g., DEBUG, INFO, WARNING).
    * `name`: The name of the logger.
    * `thread`: The thread ID.
    * `filename`: The name of the file generating the log entry.
    * `lineno`: The line number where the log entry was generated.
    * `message`: The log message.

    See Also
    --------
    logging.Formatter :
        logging Formatter.
    """

    def __init__(
        self,
        datefmt: str = "%Y-%m-%d %H:%M:%S",
        default_time_format: str = "%Y-%m-%d %H:%M:%S",
        default_msec_format: str = "%s,%03d",
        backend: "Optional[str]" = None,
        use_datetime: "Optional[bool]" = True,
    ) -> None:
        """Initialize the GoogleLogFormatter with the desired Formatter."""
        # formatTime time module's strftime() function does not support microseconds
        # Need to manual extract seconds and microseconds,time.time() with microseconds
        # sec, mics  = int(now), int(1e6 * (now % 1.0)) or int(1e6 * (now - sec))
        # datetime module's strftime() function support microseconds
        # %m: Two-digit month (01-12)
        # %d: Two-digit day of the month (01-31)
        # %H: Two-digit hour in 24-hour format (00-23)
        # %M: Two-digit minutes (00-59)
        # %S: Two-digit seconds (00-59)
        # %f: Microseconds (000000-999999)
        # Format as 'MM-DD HH:MM:SS.microsecond'
        # formatted_time = datetime.datetime.now().strftime('%m-%d %H:%M:%S.%f')
        # print(formatted_time)  # e.g., "12-25 14:30:45.123456"
        # parsed_date = datetime.strptime(formatted_time, '%m-%d %H:%M:%S.%f')
        # print(parsed_date)  # e.g., "12-25 14:30:45.123456"
        # Include microseconds in the timestamp
        if use_datetime:
            datefmt += ".%f"  # datetime Get the timestamp with microseconds
        super().__init__()
        self.datefmt = datefmt
        self.default_time_format = default_time_format
        self.default_msec_format = (
            default_msec_format  # for time does not support microseconds
        )
        self.backend = backend

    def format(self, record: "_logging.LogRecord") -> str:
        """
        Format the log record into a JSON string or a pretty-printed dictionary.

        Parameters
        ----------
        record : logging.LogRecord
            The log record containing log information (message, level, etc.).

        Returns
        -------
        str
            The formatted log message (either in literal str, JSON or pretty-print format).
        """
        from datetime import datetime

        log_obj = {
            # "asctime": f"{self.formatTime(record, datefmt=self.datefmt)} ",
            "asctime": f"{datetime.now().strftime(self.datefmt)}: ",
            "levelname": f"{record.levelname[:1]} ",
            "name": f"{record.name} ",
            "thread": f"{record.thread} ",
            # "pathname" : f"pathname {record.pathname}",
            "filename": f"{getattr(record, 'caller_filename', record.filename)}",
            "lineno": f":{getattr(record, 'caller_lineno', record.lineno)}",
            "funcName": f":{record.funcName}] ",
            "message": f"{record.getMessage()}",
        }
        try:
            if self.backend == "json":
                import json

                # Format JSON with custom options
                return json.dumps(
                    log_obj,
                    indent=1,  # Pretty-print with 1 space
                    sort_keys=False,  # Do not sort keys
                    separators=(",", ": "),  # Maintain default spacing
                    ensure_ascii=False,
                )
            if self.backend == "pprint":
                import pprint

                # Pretty printing format
                return pprint.pformat(log_obj)
        except Exception:
            pass
        # Fallback to a literal string format
        return "".join(log_obj.values())


def _make_default_formatter(
    formatter: "Optional[_logging.Formatter | str]" = "GOOGLE_FORMAT",
    time_format: "Optional[str]" = None,
    use_datetime: "Optional[bool]" = True,
) -> "_logging.Formatter":
    """
    Create and return a default logging Formatter instance based on the provided formatter type.

    Parameters
    ----------
    formatter : Union[str, logging.Formatter], optional
        The formatter type or instance to create. Options are:

        * 'BASIC_FORMAT': A basic formatter (default format).
        * 'CUSTOM_FORMAT': A custom formatter with detailed log fields.
        * 'GOOGLE_FORMAT': Google-style formatter.
        * An instance of `logging.Formatter`. Default is 'GOOGLE_FORMAT'.

    time_format : str, optional
        Time format for the log's timestamp (`asctime`). Default is '%Y-%m-%d %H:%M:%S'.

    use_datetime : bool, optional
        Whether to include microseconds in the timestamp using datetime. Default is True.

    Returns
    -------
    logging.Formatter
        The Formatter instance based on the specified `formatter` type.
    """
    # Configure time format (default if none is provided)
    time_format = time_format or "%Y-%m-%d %H:%M:%S"
    try:
        if isinstance(formatter, _logging.Formatter):
            return formatter
        if formatter == "BASIC_FORMAT":
            return _logging.Formatter(
                fmt=_logging.BASIC_FORMAT,
                datefmt=None,
            )
        if formatter == "GOOGLE_FORMAT":
            return GoogleLogFormatter(datefmt=time_format, use_datetime=use_datetime)
        if formatter == "CUSTOM_FORMAT":
            # d=(
            #   '{'
            #   '"asctime": "%(asctime)s", '
            #   '"levelname": "%(levelname)s", '
            #   '"name": "%(name)s", '
            #   '"thread": "%(thread)s", '
            #   '"filename": "%(filename)s", '
            #   '"lineno": "%(lineno)d", '
            #   # '"pathname": "%(pathname)s", '
            #   '"message": "%(message)s", '
            #   '}'
            # )
            custom = (
                "%(asctime)s "
                "%(levelname)s "
                "%(name)s "
                "%(thread)s "
                "%(filename)s "
                "%(lineno)d "
                # '%(pathname)s '
                "%(message)s "
            )
            return _logging.Formatter(
                fmt=custom,
                datefmt=time_format,
            )
    except Exception:
        # sys.stderr.write(e)
        # Fallback to basic formatter if other formatters are not available
        return _logging.Formatter(
            fmt=_logging.BASIC_FORMAT,
            datefmt=None,
        )


######################################################################
## logging Handler
######################################################################


class AlwaysStdErrHandler(_logging.StreamHandler):  # type: ignore[type-arg]
    """
    A custom logging handler inherited from :py:class:`~logging.StreamHandler`.

    That enforces the use of a specific output stream: either standard error
    (`sys.stderr`) or standard output (`sys.stdout`).

    This handler is particularly useful for environments where log streams must
    be explicitly directed, such as Jupyter notebooks or specialized logging setups.

    Parameters
    ----------
    use_stderr : bool, default= not _is_jupyter_notebook()
        If True, the handler will use standard error `sys.stderr` as the stream.
        If False, the handler will use standard output `sys.stdout` as the stream.

    See Also
    --------
    logging.StreamHandler :
        Writes logging records, appropriately formatted, to a stream.
        This class does not close the stream, as `sys.stdout` or `sys.stderr`
        may be used.
    _is_jupyter_notebook :
        Determines if the environment is a Jupyter notebook. For define `use_stderr`.
    """

    def __init__(self, use_stderr: "bool | None" = None) -> None:
        """
        Initialize the AlwaysStdErrHandler with the desired stream.

        Attributes
        ----------
        _use_stderr : bool or None
            Stores the value of the `use_stderr` parameter.
            If None use not _is_jupyter_notebook()
        _stream : IO[str]
            Points to the chosen stream (`sys.stderr` or `sys.stdout`).

        """
        self._use_stderr = use_stderr or not _is_jupyter_notebook()
        self._stream = sys.stderr if use_stderr else sys.stdout
        super().__init__(stream=self._stream)

    # def get_name(self):
    #     """Getter for the name property."""
    #     return super().name  # Use the parent class's name property behavior.
    # def set_name(self, value):
    #     """Setter for the name property."""
    #     super(
    #         _logging.StreamHandler,
    #         self.__class__,
    #     ).name.fset(self, value)  # Set the name in the parent class.
    # ## Define the property
    # name = property(
    #     get_name,
    #     set_name,
    #     doc="This is the name property for AlwaysStdErrHandler.",
    # )
    # Add a docstring to the inherited 'name' property if it doesn't already have one
    if not _logging.StreamHandler.name.__doc__:
        import textwrap  # textwrap.dedent

        _logging.StreamHandler.name.__doc__ = textwrap.dedent(
            """
            This is the name property for StreamHandler.

            Returns
            -------
            Optional[str]
                The current handler object name if provided, otherwise None.
        """
        )

    @property  # type: ignore [override]
    def stream(self) -> "IO[str]":
        """
        Get the current logging stream.

        Returns
        -------
        IO[str]
            The current stream object (`sys.stderr` or `sys.stdout`).
        """
        return self._stream

    @stream.setter
    def stream(self, value: "IO[str]") -> None:
        """
        Set the stream for logging. This method ensures the stream is either stderr or stdout.

        Parameters
        ----------
        value : IO[str]
            The new stream to set. Must be either `sys.stderr` or `sys.stdout`.

        Raises
        ------
        AssertionError
            If the provided stream is not `sys.stderr` or `sys.stdout`.
        ValueError
            If the provided stream is not `sys.stderr` or `sys.stdout`.
        """
        # if condition returns True, then nothing happens:
        # assert (
        #   (value is sys.stderr) or (value is sys.stdout),
        #   "The stream must be either sys.stderr or sys.stdout."
        # )
        if value not in (sys.stderr, sys.stdout):
            # Fallback to default stream not raise error
            super().setStream(stream=self._stream)
            # raise ValueError("The stream must be either sys.stderr or sys.stdout.")


def _make_default_handler(
    handler: "Optional[_logging.Handler]" = None,
    formatter: "Optional[_logging.Formatter]" = None,
) -> "_logging.Handler":
    """
    Create and return a default logging Handler.

    Instance based on the provided `handler` type.

    Parameters
    ----------
    handler : Optional[logging.Handler], default=None
        The handler type or instance to create. Can be one of:

        * A custom logging handler (e.g., StreamHandler, RotatingFileHandler).
        * 'RotatingFileHandler': A rotating file handler.
        * 'RichHandler': A rich handler for formatted output.
        * None: Falls back to AlwaysStdErrHandler.

    formatter : Optional[logging.Formatter], default=None
        The formatter to use for the handler. If not provided, the default formatter is used.

    Returns
    -------
    logging.Handler
        The Handler instance that matches the specified `handler` type.
    """
    # Configure formatter (default if none is provided)
    formatter = formatter or _make_default_formatter()
    try:
        if isinstance(handler, _logging.Handler):
            pass
        elif handler is None:
            handler = AlwaysStdErrHandler()
        elif handler == "RotatingFileHandler":
            handler = _logging.handlers.RotatingFileHandler(
                "skplt.log",
                maxBytes=5000,
                backupCount=2,  # 5KB per file, 2 backups
            )
        elif handler == "RichHandler":
            # Use RichHandler for better console output
            from rich.console import Console  # type: ignore[reportMissingImports]
            from rich.logging import RichHandler  # type: ignore[reportMissingImports]

            handler = RichHandler(console=Console(stderr=True))
    except Exception:
        # Fallback to AlwaysStdErrHandler if other handlers are not available
        handler = AlwaysStdErrHandler()
    # finally:
    handler.setFormatter(formatter)
    return handler


######################################################################
## Function to expose the initialized logging logger based on singleton logic
######################################################################


# Expose the logger to other modules.
def get_logger() -> "_logging.Logger":
    """
    Return SP (scikitplot) logger instance.

    Returns
    -------
    logging.Logger
        An instance of the Python logging library Logger.

    See Also
    --------
    scikitplot.logger :
        An alias of :py:mod:`~.sp_logging` module, providing logging functionality.
    logging.getLogger :
        Standard library function to retrieve :py:class:`logging.Logger` instance.
        For more: https://docs.python.org/3/library/logging.html

    Notes
    -----
    See Python documentation (https://docs.python.org/3/library/logging.html)
    for detailed API. Below is only a summary.

    The logger has 5 levels of logging from the most serious to the least:

    1. `CRITICAL` or `FATAL`
    2. `ERROR`
    3. `WARNING`
    4. `INFO`
    5. `DEBUG`
    6. `NOTSET`

    The logger has the following methods, based on these logging levels:

    1. `critical(msg, *args, **kwargs)` or `fatal(msg, *args, **kwargs)`
    2. `error(msg, *args, **kwargs)`
    3. `warning(msg, *args, **kwargs)` or `warn(msg, *args, **kwargs)`
    4. `info(msg, *args, **kwargs)`
    5. `debug(msg, *args, **kwargs)`

    The `msg` can contain string formatting.  An example of logging at the `ERROR`
    level
    using string formatting is:

    >>> sp.get_logger().error("The value %d is invalid.", 3)

    You can also specify the logging verbosity.  In this case, the
    WARN level log will not be emitted:

    >>> sp.get_logger().setLevel(sp.sp_logging.WARNING)
    >>> sp.get_logger().debug(
    ...     "This is a debug."
    ... )  # This will not be shown, as level is WARNING.
    >>> sp.get_logger().info(
    ...     "This is a info."
    ... )  # This will not be shown, as level is WARNING.
    >>> sp.get_logger().warning("This is a warning.")

    Examples
    --------
    Get the root ``logger`` from ``module attr``:

    .. jupyter-execute::

        >>> from scikitplot import logger
        >>> logger.setLevel(logger.INFO)  # default WARNING
        >>> logger.info("This is a info message from the sp logger.")

        >>> import scikitplot as sp
        >>> sp.logger.setLevel(sp.logger.INFO)  # default WARNING
        >>> sp.logger.info("This is a info message from the sp logger.")

    Get the root ``logger`` from ``func``:

    .. jupyter-execute::

        >>> import scikitplot as sp
        >>> sp.get_logger().setLevel(sp.sp_logging.INFO)  # default WARNING
        >>> sp.get_logger().info("This is a info message from the sp logger.")
    """
    # Ensure the root logger is initialized
    # pylint: disable=global-statement
    # pylint: disable=global-variable-not-assigned
    global _logger  # noqa: PLW0603
    # pylint: enable=global-variable-not-assigned
    # pylint: enable=global-statement

    # Use double-checked locking to avoid taking lock unnecessarily.
    if _logger:
        return _logger

    _logger_lock.acquire()

    try:
        if _logger:
            return _logger
        ######################################################################
        ## Define scikit-plots the top-level logger
        ######################################################################
        # _logging.config.fileConfig('logging.conf')
        # Scope the scikit-plots logger
        name = "scikitplot"
        # Configure the logger here (only once for the entire project).
        # Scope the scikitplot logger to not conflict with users' loggers.
        # logger: Main application logger for diagnostics and debugging output.
        logger = _logging.getLogger(name)

        # Override findCaller on the logger to skip internal helper functions
        logger.findCaller = _logger_find_caller

        # Disable propagation to prevent double logging
        # Set propagate to False to prevent messages from propagating to the root logger
        logger.propagate = False

        # Don't further configure the TensorFlow logger if the root logger is
        # already configured. This prevents double logging in those cases.
        if not _logging.getLogger().handlers:
            # Determine whether we are in an interactive environment
            _interactive = False
        try:
            # This is only defined in interactive shells.
            if sys.ps1:
                _interactive = True
        except AttributeError:
            # Even now, we may be in an interactive shell with `python -i`.
            _interactive = sys.flags.interactive

        # If we are in an interactive environment (like Jupyter), set loglevel
        # to INFO and pipe the output to stdout.
        # pylint: disable=possibly-used-before-assignment
        if _interactive:
            logger.setLevel(INFO)
            _logging_target = sys.stdout
        else:
            _logging_target = sys.stderr

        ## This uses the default logging configuration
        # _logging.basicConfig()  # Using default settings
        # _logging.basicConfig(
        #     format   = _logging.BASIC_FORMAT,
        #     datefmt  = None,
        #     # level    = _logging.WARNING,  # _default_log_level()
        #     handlers = [_logging.StreamHandler(_logging_target)],
        # )

        # Add a handler if needed (since propagation is disabled)
        ## Configure handler (default is StreamHandler)
        _handler = _make_default_handler()
        # _handler = _logging.StreamHandler(_logging_target)
        # _handler.setFormatter(_logging.Formatter(_logging.BASIC_FORMAT, None))
        logger.addHandler(_handler)  # Add the handler to the logger

        _logger = logger
        return _logger
    finally:
        _logger_lock.release()
        _logger.debug(
            f"logger initialized as "
            f"{_logging.getLevelName(get_logger().getEffectiveLevel())}",
            stacklevel=4,
        )


######################################################################
## Exposed logging helpers
######################################################################


# logging.TaskLevelStatusMessage
def TaskLevelStatusMessage(msg):  # noqa: N802
    """Log logging.TaskLevelStatusMessage."""
    error(msg)


def getEffectiveLevel():  # noqa: N802
    """Return how much logging output will be produced."""
    return get_logger().getEffectiveLevel()


def get_verbosity():
    """Return how much logging output will be produced."""
    return get_logger().getEffectiveLevel()


def setLevel(v):  # noqa: N802
    """Set the threshold for what messages will be logged."""
    get_logger().setLevel(v)


def set_verbosity(v):
    """Set the threshold for what messages will be logged."""
    get_logger().setLevel(v)


def _GetNextLogCountPerToken(token):  # noqa: N802
    """
    Wrap for _log_counter_per_token.

    Parameters
    ----------
    token :
        The token for which to look up the count.

    Returns
    -------
    int : The number of times this function has been called with
          *token* as an argument (starting at 0)
    """
    # pylint: disable=global-variable-not-assigned
    # Remove the global statement if you're only reading
    # global _log_counter_per_token  # noqa: PLW0602
    # pylint: enable=global-variable-not-assigned
    _log_counter_per_token[token] = 1 + _log_counter_per_token.get(token, -1)
    return _log_counter_per_token[token]


def log_every_n(level, msg, n, *args):
    """
    Log 'msg % args' at level 'level' once per 'n' times.

    Logs the 1st call, (N+1)st call, (2N+1)st call,  etc.
    Not threadsafe.

    Parameters
    ----------
    level :
        The level at which to log.
    msg :
        The message to be logged.
    n :
        The number of times this should be called before it is logged.
    *args :
        The args to be substituted into the msg.
    """
    count = _GetNextLogCountPerToken(_GetFileAndLine())
    log_if(level, msg, not (count % n), *args)


def log_first_n(level, msg, n, *args):  # pylint: disable=invalid-name
    """
    Log 'msg % args' at level 'level' only first 'n' times.

    Not threadsafe.

    Parameters
    ----------
    level :
        The level at which to log.
    msg :
        The message to be logged.
    n :
        The number of times this should be called before it is logged.
    *args :
        The args to be substituted into the msg.
    """
    count = _GetNextLogCountPerToken(_GetFileAndLine())
    log_if(level, msg, count < n, *args)


######################################################################
## Exposed logging helper funcs
######################################################################


def critical(msg, *args, **kwargs):
    """
    Log a message at the CRITICAL log level.

    Parameters
    ----------
    msg : str
        The log message to be logged.
    args : any
        Arguments for string formatting in the message.
    kwargs : any
        Additional keyword arguments for logging.
    """
    get_logger().critical(msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    """
    Log a message at the DEBUG log level.

    Parameters
    ----------
    msg : str
        The log message to be logged.
    args : any
        Arguments for string formatting in the message.
    kwargs : any
        Additional keyword arguments for logging.
    """
    get_logger().debug(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    """
    Log a message at the ERROR log level.

    Parameters
    ----------
    msg : str
        The log message to be logged.
    args : any
        Arguments for string formatting in the message.
    kwargs : any
        Additional keyword arguments for logging.
    """
    get_logger().error(msg, *args, **kwargs)


def error_log(error_msg, *args, level=ERROR, **kwargs):
    """Empty helper method."""
    del error_msg, args, level, kwargs


def exception(msg, *args, exc_info=True, **kwargs):
    """
    Log a message with severity 'ERROR' on the root logger.

    With exception information. If the logger has no handlers,
    basicConfig() is called to add a console handler
    with a pre-defined format.
    """
    error(msg, *args, exc_info=exc_info, **kwargs)


def fatal(msg, *args, **kwargs):
    """
    Log a message at the FATAL - CRITICAL log level.

    Parameters
    ----------
    msg : str
        The log message to be logged.
    args : any
        Arguments for string formatting in the message.
    kwargs : any
        Additional keyword arguments for logging.
    """
    get_logger().fatal(msg, *args, **kwargs)


# logging.flush
def flush():
    """Log logging.flush."""
    raise NotImplementedError()


def info(msg, *args, **kwargs):
    """
    Log a message at the INFO log level.

    Parameters
    ----------
    msg : str
        The log message to be logged.
    args : any
        Arguments for string formatting in the message.
    kwargs : any
        Additional keyword arguments for logging.
    """
    get_logger().info(msg, *args, **kwargs)


def log(level, msg, *args, **kwargs):
    """
    Log a message at the specified log level.

    Parameters
    ----------
    level : int
        The logging level (e.g., DEBUG, INFO, WARNING, etc.).
    msg : str
        The log message to be logged.
    args : any
        Arguments for string formatting in the message.
    kwargs : any
        Additional keyword arguments for logging.
    """
    get_logger().log(level, msg, *args, **kwargs)


def log_if(level, msg, condition, *args, **kwargs):
    """Log 'msg % args' at level 'level' only if condition is fulfilled."""
    if condition:
        log(level, msg, *args, **kwargs)


# Code below is taken from pyglib/logging
def vlog(level, msg, *args, **kwargs):
    """Log a message at the specified log level."""
    get_logger().log(level, msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    """
    Log a message at the WARN - WARNING log level.

    Parameters
    ----------
    msg : str
        The log message to be logged.
    args : any
        Arguments for string formatting in the message.
    kwargs : any
        Additional keyword arguments for logging.
    """
    get_logger().warning(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    """
    Log a message at the WARNING log level.

    Parameters
    ----------
    msg : str
        The log message to be logged.
    args : any
        Arguments for string formatting in the message.
    kwargs : any
        Additional keyword arguments for logging.
    """
    get_logger().warning(msg, *args, **kwargs)


def sanitize_log_message(msg: str) -> str:
    """
    Sanitize a log message by redacting sensitive data.

    Parameters
    ----------
    msg : str
        The log message to sanitize.

    Returns
    -------
    str
        The sanitized log message with sensitive data redacted.
    """
    sensitive_keywords = ["secret", "password", "token", "key"]
    for keyword in sensitive_keywords:
        if keyword in msg.lower():
            return "[REDACTED] Sensitive information detected in log message."
    return msg


######################################################################
## Class (SingletonBase) like SpLogger to expose the initialized logger
######################################################################


class SpLogger:
    """
    A singleton logger class.

    That provides a shared logger instance with customizable
    name, formatter, handler, logging level, and thread-safety.

    This class implements the Singleton pattern, ensuring only a single instance of
    the logger exists throughout the application. It supports different log levels
    (e.g., DEBUG, INFO, WARNING) and thread-safe logging.

    .. important::

        If the attribute is not defined within the :py:class:`~.SpLogger` class, it will be
        dynamically fetched from the `logging` module. This is particularly useful
        for logging constants like `DEBUG`, `INFO`, `ERROR`, etc., which are often
        used in logging configurations but are not necessarily attributes of the
        logger class itself.

    .. attention::

        Be cautious when using dynamically retrieved attributes, as it may
        lead to unexpected behavior if the module's constants change.

    Parameters
    ----------
    formatter : Optional[logging.Formatter], default=None
        The formatter to use for logging.
        If None, a default customized formatter is applied.
    handler : Optional[logging.Handler], default=None
        The handler to use for logging.
        If None, a default customized handler is applied.
    *args : any, optional
        Additional positional arguments for customization.
    **kwargs : any, optional
        Additional keyword arguments for further customization.

    See Also
    --------
    get_logger :
        Function that provides a shared :py:class:`logging.Logger` instance.
    logger :
        An alias of :py:mod:`sp_logging` module, providing logging functionality.
    logging.getLogger :
        Standard library function to retrieve :py:class:`logging.Logger` instance,
        for more https://docs.python.org/3/library/logging.html.
    """

    # cls attr
    ## Store singleton instances in a class-level
    _instance: "_logging.logger" = None  # noqa: UP037

    # Thread-safe logging lock
    _lock = _logger_lock or threading.Lock()

    # Directly or Dynamically reference the module-level variable
    # CRITICAL  = FATAL = getattr(__import__(__name__), 'CRITICAL', 50)
    # ERROR     = getattr(__import__(__name__), 'ERROR', 40)
    # WARNING   = WARN = getattr(__import__(__name__), 'WARNING', 30)
    # INFO      = getattr(__import__(__name__), 'INFO', 20)
    # DEBUG     = getattr(__import__(__name__), 'DEBUG', 10)
    # NOTSET    = NOTSET  # direct use
    def __getattr__(self, name):
        """
        Resolve the attribute name from the instance or fall back to the logging module.

        If the attribute is not found within the class, this method checks for it
        in the `logging` module (e.g., `DEBUG`, `INFO`, etc.).

        Parameters
        ----------
        name : str
            The name of the attribute to retrieve.

        Returns
        -------
        any :
            The value of the attribute `name`, either from the class or the `logging` module.

        Raises
        ------
        AttributeError :
            If the attribute is not found in the class or the `logging` module.
        """
        # First, check if the attribute is in the instance's dictionary
        if name in self.__dict__:
            return self.__dict__[name]

        # Now, dynamically import the sp_logging module and check for the attribute
        if hasattr(__import__(__name__), name):
            return getattr(__import__(__name__), name)

        # Now, dynamically import the logging module and check for the attribute
        if hasattr(_logging, name):  # logging module explicitly
            return getattr(_logging, name)

        # If not found, raise an AttributeError
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    # def __new__(
    #     # cls refers to the class itself, and is used in class methods (@classmethod).
    #     cls,
    #     *args: tuple[any, ...], **kwargs: dict[str, any],  # noqa: ARG004
    # ) -> _logging.logger:
    #     """
    #     Create a single instance of logging.Logger.

    #     Parameters
    #     ----------
    #     *args : tuple
    #         Optional positional arguments.

    #     **kwargs : dict
    #         Optional keyword arguments.

    #     Returns
    #     -------
    #     logging.logger
    #         Either the decorated function or the singleton instance.
    #     """
    #     # Class.method() or obj.method()
    #     # Ensure only one instance exists (singleton pattern)
    #     with cls._lock:
    #         # Create a new instance if none exists:
    #         if cls._instance is None:
    #             cls._instance = get_logger()
    #     return cls._instance


# Instantiate the class so its methods can be accessed directly via the module
# sp_logger = SpLogger()

######################################################################
##
######################################################################
