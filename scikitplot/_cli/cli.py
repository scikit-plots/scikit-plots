# scikitplot/_cli/cli.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Back-compat shim (deprecated).

The CLI moved to a framework-neutral runtime. The console entry point is now
``scikitplot._cli.app:main``. This module remains only so that the historical
target ``scikitplot._cli.cli:cli`` keeps resolving during the transition.

See ``_maintenance/DECISIONS.md`` (ADR-CLI-100) and ``INTEGRATION.md``.
"""

from __future__ import annotations

from .app import main

# Historical name: `scikitplot._cli.cli:cli`. `cli()` == `main()`; `cli.main(...)`
# also works because both resolve to the same callable.
cli = main

__all__ = ["cli", "main"]
