# scikitplot/__main__.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""CLI Interface: ``python -m scikitplot``.

Routes to the framework-neutral CLI app (argparse default, click optional).
See ``scikitplot/_cli/_maintenance/DECISIONS.md`` (ADR-CLI-100/101, ADR-CLI-105).
"""

from __future__ import annotations

from ._cli.app import main

if __name__ == "__main__":
    raise SystemExit(main())
