"""
sp_logging_utils.py.

Copied from mlflow.
https://github.com/mlflow/mlflow/blob/master/mlflow/utils/logging_utils.py
"""

# pylint: disable=line-too-long
# pylint: disable=redefined-builtin
# pylint: disable=missing-function-docstring

# ruff: noqa: D102

import contextlib as _contextlib
import re as _re
import sys as _sys

from .. import sp_logging as _logging
from ..environment_variables import SKPLT_LOGGING_LEVEL

# Logging format example:
# 2018/11/20 12:36:37 INFO scikitplot.sagemaker: Creating new SageMaker endpoint
LOGGING_LINE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOGGING_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"


class ScikitplotLoggingStream:
    """
    A Python stream.

    For use with event logging APIs throughout scikitplot (`eprint()`,
    `logger.info()`, etc.). This stream wraps `sys.stderr`, forwarding `write()` and
    `flush()` calls to the stream referred to by `sys.stderr` at the time of the call.
    It also provides capabilities for disabling the stream to silence event logs.
    """

    def __init__(self):
        self._enabled = True

    def write(self, text):
        if self._enabled:
            _sys.stderr.write(text)

    def flush(self):
        if self._enabled:
            _sys.stderr.flush()

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value


MLFLOW_LOGGING_STREAM = ScikitplotLoggingStream()


def disable_logging():
    """
    Disables the `ScikitplotLoggingStream`.

    Used by event logging APIs throughout scikitplot
    (`eprint()`, `logger.info()`, etc), silencing all subsequent event logs.
    """
    MLFLOW_LOGGING_STREAM.enabled = False


def enable_logging():
    """
    Enable the `ScikitplotLoggingStream`.

    Used by event logging APIs throughout scikitplot
    (`eprint()`, `logger.info()`, etc), emitting all subsequent event logs. This
    reverses the effects of `disable_logging()`.
    """
    MLFLOW_LOGGING_STREAM.enabled = True


class ScikitplotFormatter(_logging.Formatter):
    """
    Custom Formatter Class to support colored log.

    ANSI characters might not work natively on older Windows, so disabling the feature for win32.
    See https://github.com/borntyping/python-colorlog/blob/dfa10f59186d3d716aec4165ee79e58f2265c0eb/colorlog/escape_codes.py#L16C8-L16C31
    """

    # Copied from color log package https://github.com/borntyping/python-colorlog/blob/dfa10f59186d3d716aec4165ee79e58f2265c0eb/colorlog/escape_codes.py#L33-L50
    COLORS = {  # noqa: RUF012
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "purple": 35,
        "cyan": 36,
        "white": 37,
        "light_black": 90,
        "light_red": 91,
        "light_green": 92,
        "light_yellow": 93,
        "light_blue": 94,
        "light_purple": 95,
        "light_cyan": 96,
        "light_white": 97,
    }
    RESET = "\033[0m"

    def format(self, record):
        if color := getattr(record, "color", None):  # noqa: SIM102
            if color in self.COLORS and _sys.platform != "win32":
                color_code = self._escape(self.COLORS[color])
                return f"{color_code}{super().format(record)}{self.RESET}"
        return super().format(record)

    def _escape(self, code: int) -> str:
        return f"\033[{code}m"


def _configure_scikitplot_loggers(root_module_name):
    _logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "scikitplot_formatter": {
                    "()": ScikitplotFormatter,
                    "format": LOGGING_LINE_FORMAT,
                    "datefmt": LOGGING_DATETIME_FORMAT,
                },
            },
            "handlers": {
                "scikitplot_handler": {
                    "formatter": "scikitplot_formatter",
                    "class": "_logging.StreamHandler",
                    "stream": MLFLOW_LOGGING_STREAM,
                },
            },
            "loggers": {
                root_module_name: {
                    "handlers": ["scikitplot_handler"],
                    "level": (SKPLT_LOGGING_LEVEL.get() or "INFO").upper(),
                    "propagate": False,
                },
            },
        }
    )


def eprint(*args, **kwargs):
    """Eprint."""
    print(*args, file=MLFLOW_LOGGING_STREAM, **kwargs)


class LoggerMessageFilter(_logging.Filter):
    """LoggerMessageFilter."""

    def __init__(self, module: str, filter_regex: _re.Pattern):
        super().__init__()
        self._pattern = filter_regex
        self._module = module

    def filter(self, record):
        if record.name == self._module and self._pattern.search(  # noqa: SIM103
            record.msg
        ):  # noqa: SIM103
            return False
        return True


@_contextlib.contextmanager
def suppress_logs(module: str, filter_regex: _re.Pattern):
    """
    Context manager that suppresses log messages.

    From the specified module that match the specified
    regular expression. This is useful for suppressing
    expected log messages from third-party
    libraries that are not relevant to the current test.
    """
    logger = _logging.getLogger(module)
    filter = LoggerMessageFilter(module=module, filter_regex=filter_regex)
    logger.addFilter(filter)
    try:
        yield
    finally:
        logger.removeFilter(filter)


def _debug(s: str) -> None:
    """Debug function to test logging level."""
    _logging.getLogger(__name__).debug(s)
