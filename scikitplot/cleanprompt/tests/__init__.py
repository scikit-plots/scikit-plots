"""
Focused tests for :mod:`scikitplot.cleanprompt`.

Notes
-----
**Developer notes.** One test module per source module, named
``test_<source>.py`` for ``<source>.py``, so ownership of every assertion is
readable from the file tree. ``test_regressions.py`` is the exception: it holds
one named test per defect reproduced against the upstream project, so that
those cannot be silently lost when a source module is reorganised.

Tests import through the package facade where they exercise public behaviour,
and reach into a private module only where the assertion is about that module's
internals. The distinction is deliberate: a test that reaches inside to check
something the public surface guarantees will keep passing after the public
surface breaks.
"""
