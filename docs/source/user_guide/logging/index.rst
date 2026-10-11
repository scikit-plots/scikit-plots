.. currentmodule:: scikitplot.logging

.. _logging-index:

Logging
=======

``scikitplot.logging`` wraps Python's standard logging model with a Scikit-Plots package logger, Google-style formatting, verbosity helpers and rate-limited convenience functions.

.. important::

   The current repository snapshot has several open compatibility findings in this submodule. Prefer the basic ``get_logger`` / ``setLevel`` / standard ``debug``-``critical`` path and read **Known limits** before depending on helper or environment behavior.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   levels_and_loggers
   formatting_and_helpers
   python_api
   known_limits_and_troubleshooting

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Getting started
      :link: logging-getting-started
      :link-type: ref

      Use the package logger and set verbosity without configuring duplicate handlers.

   .. grid-item-card:: Levels and loggers
      :link: logging-levels-and-loggers
      :link-type: ref

      Understand package-level helpers, effective level, verbosity, and stdlib compatibility.

   .. grid-item-card:: Formatting and helpers
      :link: logging-formatting-and-helpers
      :link-type: ref

      Google-style formatting, sanitization, rate-limited helpers, and stderr behavior.

   .. grid-item-card:: Python API
      :link: logging-python-api
      :link-type: ref

      Public constants, stdlib-compatible objects, and Scikit-Plots helpers.

   .. grid-item-card:: Known limits and troubleshooting
      :link: logging-known-limits-and-troubleshooting
      :link-type: ref

      Current compatibility findings and safe workarounds.

Which page should I use?
------------------------

Start with **Getting started**. Then choose the page that owns the decision
you are making: architecture and lifecycle, configuration, persistence,
performance, public API, or limits. The guide intentionally separates those
concerns so examples do not accidentally become policy.

Further reading
---------------

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
