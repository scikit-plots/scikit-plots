.. currentmodule:: scikitplot._externals._sphinx_ext._ansi_sanitizer

.. _externals-sphinx-ext-ansi-sanitizer-index:

======================================================================
ANSI sanitizer
======================================================================

Use ``_ansi_sanitizer`` when captured terminal output can reach a LaTeX Sphinx
build.  It removes terminal escape/control sequences from text nodes before
LaTeX renders them.  HTML and other builders are left unchanged.

Enable it
----------------------------------------------------------------------

Add the extension to ``conf.py``::

   extensions += [
       "scikitplot._externals._sphinx_ext._ansi_sanitizer",
   ]

There are no extension configuration values.  The sanitizer connects to
``doctree-resolved`` and runs only when ``app.builder.format == "latex"``.

Plain-string use
----------------------------------------------------------------------

The same parser is available without running a Sphinx build::

   from scikitplot._externals._sphinx_ext._ansi_sanitizer import (
       strip_terminal_controls,
   )

   clean = strip_terminal_controls("\x1b[31merror\x1b[0m")
   assert clean == "error"

``strip_terminal_controls`` requires a string and raises ``TypeError`` for
other input types.  It keeps printable text, tabs, line feeds and carriage
returns while removing supported ECMA-48 style control sequences and remaining
C0/C1 control characters.

Why this is narrower than a generic regex
----------------------------------------------------------------------

The implementation distinguishes control-sequence forms instead of using one
catch-all expression.  That matters for truncated/two-byte escapes and
operating-system command sequences: over-greedy removal can consume ordinary
visible text.  Unterminated string sequences are handled conservatively so
printable text is not discarded merely because captured output was truncated.

When not to use it
----------------------------------------------------------------------

This extension is not an HTML terminal renderer and does not translate ANSI
colors into CSS.  Use it when the desired contract is **remove terminal
controls from LaTeX text**, not when preserving terminal styling is required.
