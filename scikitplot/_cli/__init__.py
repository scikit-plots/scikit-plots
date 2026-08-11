# scikitplot/_cli/__init__.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""scikit-plots command-line interface (framework-neutral runtime).

``argparse`` is always available; ``click`` is an optional adapter selected via
``SCIKITPLOT_CLI_FRONTEND``. Importing this package stays stdlib-only.
"""

from __future__ import annotations

from .app import main

__all__ = ["main"]
