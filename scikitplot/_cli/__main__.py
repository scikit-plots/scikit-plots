# scikitplot/_cli/__main__.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Enable ``python -m scikitplot._cli``."""

from __future__ import annotations

from .app import main

if __name__ == "__main__":
    raise SystemExit(main())
