.. _logging-getting-started:

Getting started
===============

.. code-block:: python

   from scikitplot import logging as splog

   logger = splog.get_logger()
   splog.setLevel(splog.INFO)
   logger.info("training started")

The package logger is built on Python's standard :mod:`logging` types. Normal
stdlib concepts—levels, handlers, formatters and records—still apply.

Avoid repeatedly adding handlers from library code. Handler ownership belongs
at the application/configuration boundary; libraries should emit records rather
than reconfigure the process-wide logging graph on every call.
