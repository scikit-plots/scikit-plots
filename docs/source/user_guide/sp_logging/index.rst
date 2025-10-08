.. _sp_logging-index:

..
  https://devguide.python.org/documentation/markup/#sections
  https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#sections
  # with overline, for parts    : ######################################################################
  * with overline, for chapters : **********************************************************************
  = for sections                : ======================================================================
  - for subsections             : ----------------------------------------------------------------------
  ^ for subsubsections          : ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  " for paragraphs              : """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. # https://rsted.info.ucl.ac.be/
.. # https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#paragraph-level-markup
.. # https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#footnotes
.. # attention, caution, danger, error, hint, important, note, tip, warning, admonition, seealso
.. # versionadded, versionchanged, deprecated, versionremoved, rubric, centered, hlist

======================================================================
Logging System
======================================================================

This module contains functions related to :py:mod:`~.sp_logging` (Scikit-plots Logging).

.. note::

    The Scikit-plots logging system is meant for internal scikit-plots usage.
    For use in other packages, we recommend implementing your own ``logger`` instead.

.. tip::

    The Scikit-plots logging system compatible with python logging API system.
    This module defines a logging class based on the built-in logging module.

Configuring the logging system
----------------------------------------------------------------------

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
