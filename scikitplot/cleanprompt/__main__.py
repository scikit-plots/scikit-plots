"""
Entry point for ``python -m scikitplot.cleanprompt``.

Notes
-----
**Developer notes.** This module does nothing but delegate. Keeping it empty of
logic means the command-line surface is exercised by calling
:func:`scikitplot.cleanprompt._cli.main` with an argument list, with no
subprocess and no stream capture, so every branch is reachable from a unit test.

It imports only :mod:`scikitplot.cleanprompt._cli`, which imports only the base
tier. ``python -m scikitplot.cleanprompt --help`` therefore works on an
installation with no optional dependency present, and reports the missing tier
through ``capabilities`` rather than through a traceback.
"""

from __future__ import annotations

import sys

from ._cli import main

if __name__ == "__main__":
    sys.exit(main())
