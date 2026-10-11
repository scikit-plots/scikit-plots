.. _logging-formatting-and-helpers:

Formatting and helpers
======================

:class:`GoogleLogFormatter` produces a compact prefix with severity, logger,
thread/process and call-site information. :class:`AlwaysStdErrHandler` is the
package's stderr-oriented handler abstraction.

Convenience helpers include ``log_if``, ``log_every_n``, ``log_first_n`` and
``vlog``. Use them to reduce repetitive application logic, but do not use a
rate-limited helper for events that must be recorded exactly once for audit or
security purposes.

Call-site metadata is most reliable when records flow through the intended
package helpers; direct low-level Logger calls can have different stack/caller
semantics.
