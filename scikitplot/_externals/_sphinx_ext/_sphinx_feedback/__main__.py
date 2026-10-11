"""``python -m ..._sphinx_feedback``: see :mod:`._cli`."""

from __future__ import annotations

import sys

from ._cli import main

if __name__ == "__main__":
    sys.exit(main())
