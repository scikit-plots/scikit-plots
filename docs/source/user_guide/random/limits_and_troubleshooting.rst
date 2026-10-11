.. _random-limits-and-troubleshooting:

Limits and troubleshooting
==========================

``scikitplot.random`` is a compiled KISS RNG implementation. Import failures on
a platform without a compatible extension are packaging/build failures rather
than a cue to substitute Python's ``random`` module silently.

The implementation consumes ``kissrandom.h`` from the independent C-external
Annoy ownership domain. Applications should not depend on that header path.

If a restored state is rejected, check state schema/version compatibility before
modifying values manually. If parallel runs differ despite fixed seeds, audit
where streams are created and how work is scheduled before assuming the RNG
itself is non-deterministic.
