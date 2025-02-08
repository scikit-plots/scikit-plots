.. _sp_logging-index:

======================================================================
Logging System
======================================================================

This module contains functions related to :py:mod:`~.sp_logging` (Scikit-plots Logging).

.. note::

    The Scikit-plots logging system is meant for internal scikit-plots usage.
    For use in other packages, we recommend implementing your own logger instead.

.. tip::

    The Scikit-plots logging system compatible with python logging API system.
    This module defines a logging class based on the built-in logging module.

Configuring the logging system
----------------------------------------------------------------------

First, import the logger, Get a root logger by module:

.. jupyter-execute::

    >>> import scikitplot.sp_logging as logging  # module logger
    >>> logging.setLevel(logging.INFO)           # default WARNING
    >>> logging.info("This is a info message from the sp logger.")

Get a root logger by func:

.. jupyter-execute::

    >>> from scikitplot import sp_logging, get_logger; logging=get_logger()  # pure python logger, not have direct log level
    >>> logging.setLevel(sp_logging.INFO)                                    # default WARNING
    >>> logging.info("This is a info message from the sp logger.")

Get a root logger by class:

.. jupyter-execute::

    >>> from scikitplot import SpLogger; logging=SpLogger()  # class logger
    >>> logging.setLevel(logging.INFO)                       # default WARNING
    >>> logging.info("This is a info message from the sp logger.")

Get a root logger by class instance:

.. jupyter-execute::

    >>> from scikitplot import sp_logger as logging  # class instance logger
    >>> logging.setLevel(logging.INFO)               # default WARNING
    >>> logging.info("This is a info message from the sp logger.")

.. note::

    This module builds on Python's standard :py:mod:`logging` library.
    For more information on Python's logging API, refer to the official
    documentation: https://docs.python.org/3/library/logging.html
    ::

      >>> import logging
      >>> logger = logging.getLogger(__name__)
      >>> logger.setLevel(logging.DEBUG)

    Logging Levels:

    * `NOTSET` (0)  : NOTSET
    * `DEBUG` (10)  : Detailed information useful during development,
      typically of interest only when diagnosing problems.
    * `INFO` (20)   : Confirmation that things are working as expected.
    * `WARNING` (30): An indication that something unexpected happened,
      or indicative of some problem in the near future.
    * `ERROR` (40)  : Due to a more serious problem,
      the software has not been able to perform some function.
    * `CRITICAL` = `FATAL` (50): A very serious error, indicating that
      the program itself may be unable to continue running.
