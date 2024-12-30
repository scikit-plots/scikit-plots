"""
Scikit-plots Logging Module

This module provides advanced logging utilities for Python applications, 
including support for singleton-based logging with customizable formatters, 
handlers, and thread-safety. It extends Python's standard logging library 
to enhance usability and flexibility for large-scale projects.

Module Dependencies:
  - Python standard library: :py:mod:`logging`

logging helpers, supports vendoring
"""
# """
# Classes
# -------
# SpLogger
#     Singleton-based logger with support for thread-safety, customizable name, 
#     formatter, handler, and logging level.

# PrettyJSONFormatter
#     Custom formatter to pretty-print JSON logs with customizable options.

# AlwaysStdErrHandler
#     A custom logging handler that ensures logs are output to stderr or stdout 
#     based on user preference.

# Functions
# ---------
# get_logger(use_stderr: bool = True) -> logging.Logger
#     Retrieve the singleton logger instance, initializing it if necessary.

# _make_default_formatter(formatter=None, time_format=None) -> logging.Formatter
#     Create a default logging formatter with support for JSON and custom formats.

# _make_default_handler(handler=None, formatter=None) -> logging.Handler
#     Create a default logging handler with customizable formatter and stream.

# Notes
# -----
# This module builds on Python's standard `logging` library. For more information 
# on Python's logging API, refer to the official documentation:
# https://docs.python.org/3/library/logging.html

# The `SpLogger` class ensures a single shared logger instance across the project, 
# simplifying configuration and usage. The `get_logger` function provides easy 
# access to this shared logger or or vice versa.

# Logging Levels:
#   - NOTSET (0)  : NOTSET
#   - DEBUG (10)  : Detailed information useful during development,
#     typically of interest only when diagnosing problems.
#   - INFO (20)   : Confirmation that things are working as expected.
#   - WARNING (30): An indication that something unexpected happened,
#     or indicative of some problem in the near future.
#   - ERROR (40)  : Due to a more serious problem,
#     the software has not been able to perform some function.
#   - CRITICAL = FATAL (50): A very serious error, indicating that
#     the program itself may be unable to continue running.
    
# Examples
# --------
# Basic Usage by func
#     >>> from scikitplot import get_logger
#     >>> get_logger().warning("Warning information.")

# Basic Usage by class
#     >>> from scikitplot import SpLogger
#     >>> SpLogger().warning("Warning information.")
# """
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=unused-import
# pylint: disable=g-bad-import-order
# pylint: disable=invalid-name
from __future__ import annotations
import os
import sys
import json
import pprint
import threading  # Python 2 to thread.get_ident
import contextlib
import time as time
from datetime import datetime
from typing import (
  Any,
  Optional,
  Union,
  Iterator,
  Mapping,
  IO,
)
import logging
from logging import (
  NOTSET,
  DEBUG,
  INFO,
  WARNING,
  WARNING as WARN,    # logger WARN deprecated
  ERROR,
  CRITICAL, FATAL,
  StreamHandler,
)

from ._globals import SingletonBase

__all__ = [
  '_default_log_level',
  '_get_thread_id',
  '_is_jupyter_notebook',
  'NOTSET',
  'DEBUG',
  'INFO',
  'WARNING', 'WARN',
  'ERROR',
  'FATAL', 'CRITICAL',
  'get_logger',         # func based
  'getEffectiveLevel', 'setLevel',
  'log',
  'log_if',
  'error_log',
  'vlog',
  'critical', 'fatal',
  'error',
  'warning', 'warn',
  'info',
  'debug',
  'SpLogger',           # class based
  'sp_logger',          # class instance
]

# Don't use this directly. Use get_logger() instead.
_logger = None
# Reentrant: The same thread can acquire the lock multiple times without blocking.
# _logger_lock = threading.RLock()
_logger_lock = threading.Lock()

######################################################################
## THREAD ID helper
######################################################################

# Mask to convert integer thread ids to unsigned quantities for logging
# purposes
_THREAD_ID_MASK = 2 * sys.maxsize + 1
def _get_thread_id() -> int:
    """
    Get the id of the current thread, suitable for logging as an unsigned quantity.

    Returns
    -------
    int
        The thread ID masked and converted to an unsigned quantity.
    """
    # Get id of current thread, suitable for logging as an unsigned quantity.
    thread_id = threading.get_ident()
    return thread_id & _THREAD_ID_MASK

######################################################################
## Define and Set log level
######################################################################

# Define and Set log level
def _default_log_level(debug_mode: bool = False) -> int:
    """ Define and Set log level """
    val: bool | str | None = debug_mode or os.getenv("SKPLT_DEBUG")
    return logging.WARNING if val is None else logging.DEBUG

def _is_jupyter_notebook() -> bool:
    """ Define running enviroment is jupyter notebook. """
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            return False
        # Check if IPKernelApp is running
        if 'IPKernelApp' in get_ipython().config:
            return True
    except ImportError:
        pass
    # Check environment variables
    return 'JPY_PARENT_PID' in os.environ or 'ipykernel' in sys.modules
  
######################################################################
## google2 log_prefix
######################################################################

_log_prefix = None  # later set to google2_log_prefix

# Counter to keep track of number of log entries per token.
_log_counter_per_token = {}
  
_level_names = {
  CRITICAL: 'CRITICAL', FATAL : 'CRITICAL',
  ERROR   : 'ERROR',
  WARNING : 'WARNING',  WARN  : 'WARNING',
  INFO    : 'INFO',
  DEBUG   : 'DEBUG',
}

# def _GetFileAndLine():
#   """Returns (filename, linenumber) for the stack frame."""
#   code, f = _get_caller()
#   if not code:
#     return ('<unknown>', 0)
#   return (code.co_filename, f.f_lineno)

# def google2_log_prefix(level, timestamp=None, file_and_line=None):
#   """Assemble a logline prefix using the google2 format."""
#   # pylint: disable=global-variable-not-assigned
#   global _level_names
#   # pylint: enable=global-variable-not-assigned

#   # Record current time
#   now             = timestamp or time.time()
#   now_tuple       = time.localtime(now)
#   now_microsecond = int(1e6 * (now % 1.0))

#   (filename, line) = file_and_line #or _GetFileAndLine()
#   basename = os.path.basename(filename)

#   # Severity string
#   severity = 'I'
#   if level in _level_names:
#     severity = _level_names[level][0]

#   s = '%c%02d%02d %02d:%02d:%02d.%06d %5d %s:%d] ' % (
#       severity,
#       now_tuple[1],  # month
#       now_tuple[2],  # day
#       now_tuple[3],  # hour
#       now_tuple[4],  # min
#       now_tuple[5],  # sec
#       now_microsecond,
#       _get_thread_id(),
#       basename,
#       line)

#   return s
  
# _log_prefix = google2_log_prefix

######################################################################
## logging Formatter
######################################################################

class GoogleLogFormatter(logging.Formatter):
    """
    Custom logging formatter that formats log messages in a Google-style format:
    YYYY-MM-DD HH:MM:SS.mmmmmm logger_name log_level message

    This formatter outputs logs in a structured format with the following fields:
    - `asctime`: The timestamp of the log entry.
    - `levelname`: The log level (e.g., DEBUG, INFO, WARNING).
    - `name`: The name of the logger.
    - `thread`: The thread ID.
    - `filename`: The name of the file generating the log entry.
    - `lineno`: The line number where the log entry was generated.
    - `message`: The log message.
    """
    def __init__(
        self,
        datefmt: str = '%Y-%m-%d %H:%M:%S',
        default_time_format: str = '%Y-%m-%d %H:%M:%S',
        default_msec_format: str = '%s,%03d',
        backend: Optional[str] = None,
        use_datetime: Optional[bool] = True,
    ) -> None:
        """
        Initialize the GoogleLogFormatter with the desired Formatter.
    
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
        """
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
        if use_datetime: datefmt += '.%f'  # datetime Get the timestamp with microseconds
        super().__init__()
        self.datefmt = datefmt
        self.default_time_format = default_time_format
        self.default_msec_format = default_msec_format # for time does not support microseconds
        self.backend = backend

    def format(self, record: logging.LogRecord) -> str:
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
        log_obj = {
          # "asctime0"  : f"{self.formatTime(record, datefmt=None)} ",
          "asctime"    : f"{datetime.now().strftime(self.datefmt)} ",
          "levelname"  : f"{record.levelname[:1]} ",
          "name"       : f"{record.name} ",
          "thread"     : f"{record.thread} ",
          "filename"   : f"{record.filename}",
          "lineno"     : f":{record.lineno}] ",
          "message"    : f"{record.getMessage()}",
        }
        try:
          if self.backend == 'json':
            # Format JSON with custom options
            return json.dumps(
              log_obj,
              indent=1,  # Pretty-print with 1 space
              sort_keys=False,  # Do not sort keys
              separators=(',', ': '),  # Maintain default spacing
              ensure_ascii=False,
            )
          elif self.backend == 'pprint':
            # Pretty printing format
            return pprint.pformat(log_obj)
        except Exception as e:
          pass
        # Fallback to a literal string format
        return ''.join(log_obj.values())

def _make_default_formatter(
    formatter: Optional[Union[logging.Formatter, str]] = 'CUSTOM_FORMAT',
    time_format: Optional[str] = None,
    use_datetime: Optional[bool] = True,
) -> logging.Formatter:
    """
    Create and return a default logging Formatter instance based on the provided formatter type.

    Parameters
    ----------
    formatter : Union[str, logging.Formatter], optional
        The formatter type or instance to create. Options are:
        - 'BASIC_FORMAT': A basic formatter (default format).
        - 'CUSTOM_FORMAT': A custom formatter with detailed log fields.
        - 'GOOGLE_FORMAT': Google-style formatter.
        - An instance of `logging.Formatter`. Default is 'GOOGLE_FORMAT'.

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
    time_format = time_format or '%Y-%m-%d %H:%M:%S'
    try:
      if isinstance(formatter, logging.Formatter):
        return formatter
      elif formatter == 'BASIC_FORMAT':
        return logging.Formatter(logging.BASIC_FORMAT, None)
      elif formatter == 'CUSTOM_FORMAT':
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
        short=(
          '%(asctime)s '
          '%(levelname)s '
          '%(name)s '
          '%(thread)s '
          '%(filename)s '
          '%(lineno)d '
          # '%(pathname)s '
          '%(message)s '
        )
        return logging.Formatter(
          fmt=short,
          datefmt=time_format
        )
      elif formatter == 'GOOGLE_FORMAT':
        return GoogleLogFormatter(datefmt=time_format, use_datetime=use_datetime)
    except Exception as e:
      # sys.stderr.write(e)
      pass
    # Fallback to basic formatter if other formatters are not available
    return logging.Formatter(logging.BASIC_FORMAT, None)

######################################################################
## logging Handler
######################################################################

class AlwaysStdErrHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """
    A custom logging handler that allows selecting between standard error (stderr)
    and standard output (stdout) with enforced rules for stream assignment.

    StreamHandler class
    A handler class which writes logging records, appropriately formatted,
    to a stream. Note that this class does not close the stream, as
    sys.stdout or sys.stderr may be used.
    """
    def __init__(self, use_stderr: bool = not _is_jupyter_notebook()) -> None:
        """
        Initialize the AlwaysStdErrHandler with the desired stream.

        Parameters
        ----------
        use_stderr : bool, default= not _is_jupyter_notebook()
            If True, the handler will use standard error (sys.stderr) as the stream.
            If False, the handler will use standard output (sys.stdout) as the stream.
    
        See Also
        --------
        _is_jupyter_notebook : 
            Define running enviroment is jupyter notebook.
        """
        self._use_stderr = use_stderr
        self._stream = sys.stderr if use_stderr else sys.stdout
        super().__init__(stream=self._stream)

    @property  # type: ignore [override]
    def stream(self) -> IO[str]:
        """
        Get the current logging stream.

        Returns
        -------
        IO[str]
            The current stream object (`sys.stderr` or `sys.stdout`).
        """
        return self._stream

    @stream.setter
    def stream(self, value: IO[str]) -> None:
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
    handler: Optional[logging.Handler] = None,
    formatter: Optional[logging.Formatter] = None
) -> logging.Handler:
    """
    Create and return a default logging Handler instance based on the provided
    `handler` type.

    Parameters
    ----------
    handler : Optional[logging.Handler], default=None
        The handler type or instance to create. Can be one of:
        - A custom logging handler (e.g., StreamHandler, RotatingFileHandler).
        - 'RotatingFileHandler': A rotating file handler.
        - 'RichHandler': A rich handler for formatted output.
        - None: Falls back to AlwaysStdErrHandler.

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
      if isinstance(handler, logging.Handler):
        handler.setFormatter(formatter)
        return handler
      elif handler == 'RotatingFileHandler':
        from logging.handlers import RotatingFileHandler        
        handler = RotatingFileHandler(
            "app.log", maxBytes=5000, backupCount=2  # 5KB per file, 2 backups
        )
        handler.setFormatter(formatter)
        return handler
      elif handler == 'RichHandler':
        # Use RichHandler for better console output
        from rich.console import Console
        from rich.logging import RichHandler
        console = Console(stderr=True)
        handler = RichHandler(console=console)
        handler.setFormatter(formatter)
        return handler
    except:
      pass
    # Fallback to AlwaysStdErrHandler if other handlers are not available
    handler = AlwaysStdErrHandler()
    handler.setFormatter(formatter)
    return handler

######################################################################
## Function based to expose the initialized logger
######################################################################

# Expose the logger to other modules.
def get_logger(use_stderr: bool = True) -> logging.Logger:
    """
    Return SP (scikitpot) logger instance.
    
    See Also
    --------
    SpLogger :
        A singleton logger class that provides a shared :py:class:`logging.Logger` instance
        with customizable name, formatter, handler, logging level, and thread-safety.
    sp_logger :
        An instance of :py:class:`SpLogger` class, providing logging functionality.
    logging.getLogger :
        Standard library function to retrieve :py:class:`logging.Logger` instance.
        For more: https://docs.python.org/3/library/logging.html
    
    Returns
    -------
    logging.Logger
        An instance of the Python logging library Logger.
    
    Notes
    -----
    See Python documentation (https://docs.python.org/3/library/logging.html)
    for detailed API. Below is only a summary.

    The logger has 5 levels of logging from the most serious to the least:

    1. `FATAL`
    2. `ERROR`
    3. `WARNING`
    4. `INFO`
    5. `DEBUG`

    The logger has the following methods, based on these logging levels:

    1. `fatal(msg, *args, **kwargs)`
    2. `error(msg, *args, **kwargs)`
    3. `warn(msg, *args, **kwargs)`
    4. `info(msg, *args, **kwargs)`
    5. `debug(msg, *args, **kwargs)`

    The `msg` can contain string formatting.  An example of logging at the `ERROR`
    level
    using string formating is:

    >>> sp.get_logger().error("The value %d is invalid.", 3)

    You can also specify the logging verbosity.  In this case, the
    WARN level log will not be emitted:
    
    >>> sp.get_logger().setLevel(WARNING)
    >>> sp.get_logger().debug("This is a debug.")  # This will not be shown, as level is WARNING.
    >>> sp.get_logger().info("This is a info.")    # This will not be shown, as level is WARNING.
    >>> sp.get_logger().warning("This is a warning.")
    
    Examples
    --------
    Get a root logger by module:
    
    .. jupyter-execute::
    
        >>> import scikitplot.sp_logging as logging  # module logger
        >>> logging.setLevel(logging.INFO)  # default WARNING
        >>> logging.info("This is a info message from the sp logger.")
    
    Get a root logger by func:
    
    .. jupyter-execute::
    
        >>> from scikitplot import sp_logging, get_logger; logging=get_logger()  # pure logger, not have level
        >>> logging.setLevel(sp_logging.INFO)  # default WARNING
        >>> logging.info("This is a info message from the sp logger.")
    """
    # Ensure the root logger is initialized
    global _logger
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
      # This uses the default logging configuration
      # logging.basicConfig()  # Using default settings
      # logging.basicConfig(
      #     # level=logging.WARN,
      #     # handlers=[logging.StreamHandler()],
      #     format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
      #     datefmt="%Y-%m-%d %H:%M:%S"
      # )
      # Scope the scikit-plots logger
      name = 'scikitplot'
      # Configure the logger here (only once for the entire project).
      logger = logging.getLogger(name)
      # Set propagate to False to prevent messages from propagating to the root logger
      logger.propagate = False
      # Configure handler (default is StreamHandler)
      handler = _make_default_handler() #or logging.StreamHandler()
      # handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
      # Add the handler to the logger
      logger.addHandler(handler)
      # Set the logging level (default is WARNING)
      logger.setLevel(_default_log_level())

      _logger = logger
      return _logger
    finally:
      _logger_lock.release()
      _logger.debug(f"{_logger.name} logger initialized.")

######################################################################
## Exposed loggers helper funcs
######################################################################
  
def getEffectiveLevel():
  """Return how much logging output will be produced."""
  return get_logger().getEffectiveLevel()

def setLevel(v):
  """Sets the threshold for what messages will be logged."""
  get_logger().setLevel(v)
  
def log(level, msg, *args, **kwargs):
  """
  Logs a message at the specified log level.

  Parameters
  ----------
  level : int
      The logging level (e.g., DEBUG, INFO, WARNING, etc.).
  msg : str
      The log message to be logged.
  args : Any
      Arguments for string formatting in the message.
  kwargs : Any
      Additional keyword arguments for logging.
  """
  get_logger().log(level, msg, *args, **kwargs)
  
def log_if(level, msg, condition, *args, **kwargs):
  """Log 'msg % args' at level 'level' only if condition is fulfilled."""
  if condition:
    log(level, msg, *args, **kwargs)
  
def error_log(error_msg, level=ERROR, *args, **kwargs):
  """Empty helper method."""
  del error_msg, level
  
# Code below is taken from pyglib/logging
def vlog(level, msg, *args, **kwargs):
  """Logs a message at the specified log level."""
  get_logger().log(level, msg, *args, **kwargs)

def critical(msg, *args, **kwargs):
  """
  Logs a message at the CRITICAL log level.

  Parameters
  ----------
  msg : str
      The log message to be logged.
  args : Any
      Arguments for string formatting in the message.
  kwargs : Any
      Additional keyword arguments for logging.
  """
  get_logger().critical(msg, *args, **kwargs)

def fatal(msg, *args, **kwargs):
  """
  Logs a message at the FATAL - CRITICAL log level.

  Parameters
  ----------
  msg : str
      The log message to be logged.
  args : Any
      Arguments for string formatting in the message.
  kwargs : Any
      Additional keyword arguments for logging.
  """
  get_logger().fatal(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
  """
  Logs a message at the ERROR log level.

  Parameters
  ----------
  msg : str
      The log message to be logged.
  args : Any
      Arguments for string formatting in the message.
  kwargs : Any
      Additional keyword arguments for logging.
  """
  get_logger().error(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
  """
  Logs a message at the WARNING log level.

  Parameters
  ----------
  msg : str
      The log message to be logged.
  args : Any
      Arguments for string formatting in the message.
  kwargs : Any
      Additional keyword arguments for logging.
  """
  get_logger().warning(msg, *args, **kwargs)

def warn(msg, *args, **kwargs):
  """
  Logs a message at the WARN - WARNING log level.

  Parameters
  ----------
  msg : str
      The log message to be logged.
  args : Any
      Arguments for string formatting in the message.
  kwargs : Any
      Additional keyword arguments for logging.
  """
  get_logger().warning(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
  """
  Logs a message at the INFO log level.

  Parameters
  ----------
  msg : str
      The log message to be logged.
  args : Any
      Arguments for string formatting in the message.
  kwargs : Any
      Additional keyword arguments for logging.
  """
  get_logger().info(msg, *args, **kwargs)

def debug(msg, *args, **kwargs):
  """
  Logs a message at the DEBUG log level.

  Parameters
  ----------
  msg : str
      The log message to be logged.
  args : Any
      Arguments for string formatting in the message.
  kwargs : Any
      Additional keyword arguments for logging.
  """
  get_logger().debug(msg, *args, **kwargs)
  
######################################################################
## Class (SingletonBase) based SpLogger to expose the initialized logger
######################################################################

class SpLogger(SingletonBase):
    """
    A singleton logger class that provides a shared logger instance with customizable
    name, formatter, handler, logging level, and thread-safety.

    This class implements the Singleton pattern, ensuring only a single instance of
    the logger exists throughout the application. It supports different log levels
    (e.g., DEBUG, INFO, WARNING) and thread-safe logging.

    See Also
    --------
    sp_logger :
        An instance of :py:class:`SpLogger` class, providing logging functionality.
    get_logger :
        Function that provides a shared :py:class:`logging.Logger` instance.
    logging.getLogger :
        Standard library function to retrieve :py:class:`logging.Logger` instance,
        for more https://docs.python.org/3/library/logging.html.

    Notes
    -----
    See Python documentation (https://docs.python.org/3/library/logging.html)
    for detailed API. Below is only a summary.

    The logger has 5 levels of logging from the most serious to the least.
    
    Here supported logging levels:
    
    - `CRITICAL`, `FATAL`
    - `ERROR`
    - `WARNING`
    - `INFO`
    - `DEBUG`

    The logger methods support string formatting. For example:
    
    >>> sp.SpLogger().error("The value %d is invalid.", 3)

    You can change the verbosity of the logger as follows:

    >>> sp.SpLogger().setLevel(WARNING)
    >>> sp.SpLogger().debug("This is a debug.")  # This will not be shown, as level is ERROR.
    >>> sp.SpLogger().info("This is a info.")    # This will not be shown, as level is ERROR.
    >>> sp.SpLogger().warning("This is a warning.")
    
    Examples
    --------
    Example of logging an INFO message by module:

    .. jupyter-execute::
    
        >>> import scikitplot.sp_logging as logging  # module logger
        >>> logging.setLevel(logging.INFO)  # default WARNING
        >>> logging.info("This is a info message from the sp logger.")
    
    Example of logging an INFO message by class:

    .. jupyter-execute::
    
        >>> from scikitplot import SpLogger; logging=SpLogger()  # class logger
        >>> logging.setLevel(logging.INFO)  # default WARNING
        >>> logging.info("This is a info message from the sp logger.")
    """
    # cls attr
    CRITICAL  = FATAL = CRITICAL
    ERROR     = ERROR
    WARNING   = WARN = WARNING
    INFO      = INFO
    DEBUG     = DEBUG
    NOTSET    = NOTSET
  
    def __init__(
        self,
        name: Optional[str] = None,
        formatter: Optional['Formatter'] = None,
        handler: Optional['StreamHandler'] = None,
        log_level: Optional[int] = None,
        thread_safe: bool = True,
        *args: Any,
        **kwargs: Any
    ):
        """
        Initializes the logger instance with custom configuration.

        Parameters
        ----------
        name : Optional[str], default=None
            The name of the logger. If None, defaults to 'scikitplot'.
        formatter : Optional[logging.Formatter], default=None
            The formatter to use for logging. If None, a default formatter is applied.
        handler : Optional[logging.Handler], default=None
            The handler to use for logging. If None, a StreamHandler is applied.
        log_level : Optional[int], default=None
            The logging level to set for the logger (e.g., DEBUG, INFO, WARNING). Defaults to WARNING.
        thread_safe : bool, default=True
            Indicates whether thread-safety is enabled for the logger. If True, a threading lock is used.
        args : Any, optional
            Additional positional arguments for customization.
        kwargs : Any, optional
            Additional keyword arguments for further customization.

        Attributes
        ----------
        logger : logging.Logger
            The logger instance used for logging messages.
        lock : Optional[threading.Lock]
            A lock used for thread-safe logging operations.
        """
        # instance attr
        if not hasattr(self, "logger"):  # Ensure initialization happens only once
            self._name = name or "scikitplot"
            self.logger = get_logger() or logging.getLogger(self._name)
          
            # Thread-safe logging lock
            self.lock = _logger_lock or threading.Lock() if thread_safe else None

    @property
    def name(self) -> str:
        """
        Gets the name of the logger.

        Returns
        -------
        str
            The name of the logger instance.
        """
        return self._name

    @name.setter
    def name(self, value: str):
        """
        Sets the name of the logger.

        Parameters
        ----------
        value : str
            The new name for the logger.
        """
        if not isinstance(value, str):
            raise ValueError("Logger name must be a string.")
        self._name = value
        self.logger.name = value

    @property
    def level(self) -> int:
        """
        Gets the current logging level of the logger.

        Returns
        -------
        int
            The current logging level (e.g., DEBUG, INFO, WARNING) of the logger.
        """
        return self.logger.level

    @level.setter
    def level(self, value: int):
        """
        Sets the logging level of the logger.

        Parameters
        ----------
        value : int
            The new logging level (e.g., DEBUG, INFO, WARNING).
        """
        if not isinstance(value, int):
            raise ValueError("Logging level must be an integer.")
        self.logger.setLevel(value)

    # @staticmethod
    def getEffectiveLevel(self) -> int:
        """
        Gets the current logging level of the logger.

        Returns
        -------
        int
            The current logging level (e.g., DEBUG, INFO, WARNING) of the logger.
        """
        return self.logger.getEffectiveLevel()

    # @staticmethod
    def setLevel(self, level: int):
        """
        Set the logger's logging level.

        Parameters
        ----------
        level : int
            The logging level to set (e.g., DEBUG, INFO, WARNING).
        """
        if not isinstance(level, int):
            raise ValueError("Logging level must be an integer.")
        self.logger.setLevel(level)

    def _format_msg_with_thread(self, msg: str) -> str:
        """
        Adds the current thread ID to the log message.

        Parameters
        ----------
        msg : str
            The message to be logged.

        Returns
        -------
        str
            The formatted message with the thread ID.

        Notes
        -----
        This method ensures thread-safety by using a lock if thread-safe logging is enabled.
        """
        # if self.lock:
        #   with self.lock:
        #     thread_id = _get_thread_id()
        #     return f"[Thread-{thread_id}] {msg}"
        return f"{msg}"

    def log(self, level: int, msg: str, *args: Any, **kwargs: Any):
        """
        Logs a message at the specified log level.

        Parameters
        ----------
        level : int
            The logging level (e.g., DEBUG, INFO, WARNING, etc.).
        msg : str
            The log message to be logged.
        args : Any
            Arguments for string formatting in the message.
        kwargs : Any
            Additional keyword arguments for logging.
        """
        self.logger.log(level, self._format_msg_with_thread(msg), *args, **kwargs)
  
    def log_if(self, level, msg, condition, *args, **kwargs):
      """Log 'msg % args' at level 'level' only if condition is fulfilled."""
      if condition:
          self.logger.log(level, self._format_msg_with_thread(msg), *args, **kwargs)
  
    def error_log(self, error_msg, level=ERROR, *args, **kwargs):
      """Empty helper method."""
      del error_msg, level

    def vlog(self, level: int, msg: str, *args: Any, **kwargs: Any):
        """Logs a message at the specified log level."""
        self.logger.log(level, self._format_msg_with_thread(msg), *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any):
        """Logs a message at the CRITICAL level."""
        self.log(CRITICAL, msg, *args, **kwargs)

    def fatal(self, msg: str, *args: Any, **kwargs: Any):
        """Logs a message at the CRITICAL level."""
        self.log(FATAL, msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any):
        """Logs a message at the ERROR level."""
        self.log(ERROR, msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any):
        """Logs a message at the WARNING level."""
        self.log(WARNING, msg, *args, **kwargs)

    def warn(self, msg: str, *args: Any, **kwargs: Any):
        """Logs a message at the WARNING level."""
        self.log(WARNING, msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any):
        """Logs a message at the INFO level."""
        self.log(INFO, msg, *args, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any):
        """Logs a message at the DEBUG level."""
        self.log(DEBUG, msg, *args, **kwargs)

# Instantiate the class so its methods can be accessed directly via the module
sp_logger = SpLogger()
sp_logger.__doc__ = """\
An instance of :py:class:`SpLogger`, providing logging functionality.
        
See Also
--------
SpLogger :
    A singleton logger class that provides a shared :py:class:`logging.Logger` instance
    with customizable name, formatter, handler, logging level, and thread-safety.
get_logger :
    Function that provides a shared :py:class:`logging.Logger` instance.
logging.getLogger :
    Standard library function to retrieve :py:class:`logging.Logger` instance,
    for more https://docs.python.org/3/library/logging.html.

Examples
--------
Example of logging an INFO message by module:

.. jupyter-execute::

    >>> import scikitplot.sp_logging as logging  # module logger
    >>> logging.setLevel(logging.INFO)  # default WARNING
    >>> logging.info("This is a info message from the sp logger.")

Example of logging an INFO message by class:

.. jupyter-execute::

    >>> from scikitplot import SpLogger; logging=SpLogger()  # class logger
    >>> logging.setLevel(logging.INFO)  # default WARNING
    >>> logging.info("This is a info message from the sp logger.")

Example of logging an INFO message by class instance:

.. jupyter-execute::

    >>> from scikitplot import sp_logger as logging  # class instance logger
    >>> logging.setLevel(logging.INFO)  # default WARNING
    >>> logging.info("This is a info message from the sp logger.")
"""

######################################################################
## 
######################################################################