.. _logging-known-limits-and-troubleshooting:

Known limits and troubleshooting
================================

The maintenance review for this repository snapshot records open findings that
are user-visible. In particular:

* package-level forwarding for some stdlib compatibility attributes is not yet
  fully restored after the package move;
* ``error_log`` is currently a no-op;
* documented environment-driven logging policy is not yet consistently applied;
* ``AlwaysStdErrHandler`` stream switching has an open defect;
* repeated/shared initialization can stack handlers; and
* unknown formatter names can return ``None`` instead of failing clearly.

Until those findings are closed, prefer explicit ``get_logger`` + ``setLevel``
and the standard severity helpers. If output is duplicated, inspect the logger's
handler list before adding another handler. Do not paper over duplicate emission
by filtering records downstream.
