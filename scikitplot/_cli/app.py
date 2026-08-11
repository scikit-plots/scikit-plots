# scikitplot/_cli/app.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""CLI application root: frontend selection and top-level error handling."""

from __future__ import annotations

import os
import sys
from typing import Mapping, Sequence

from . import exit_codes
from .errors import CliError


def _version_string() -> str:
    try:
        from .. import __version__  # ruff: ignore[import-outside-top-level]
    except Exception:  # pragma: no cover - defensive  # ruff: ignore[blind-except]
        __version__ = "unknown"
    return f"scikitplot {__version__}"


def _select_frontend(env: Mapping[str, str] | None = None) -> str:
    """Choose the frontend. argparse is the deterministic default (ADR-CLI-101).

    Honors ``SCIKITPLOT_CLI_FRONTEND`` = ``argparse`` | ``click``.
    """
    env = os.environ if env is None else env
    choice = env.get("SCIKITPLOT_CLI_FRONTEND", "").strip().lower()
    if choice == "click":
        from ._frontends import (  # ruff: ignore[import-outside-top-level]
            is_click_available,
        )

        if is_click_available():
            return "click"
        sys.stderr.write("scikitplot: click not installed; falling back to argparse.\n")
    return "argparse"


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``scikitplot`` console script and ``python -m``.

    Returns
    -------
    int
        Process exit code.
    """
    frontend = _select_frontend()
    try:
        if frontend == "click":
            from ._frontends import _click  # ruff: ignore[import-outside-top-level]

            return _click.run(argv)
        from ._frontends import _argparse  # ruff: ignore[import-outside-top-level]

        return _argparse.run(argv)
    except CliError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        if exc.hint:
            sys.stderr.write(f"Hint: {exc.hint}\n")
        return exc.exit_code
    except KeyboardInterrupt:  # pragma: no cover - interactive
        sys.stderr.write("Interrupted.\n")
        return exit_codes.INTERRUPTED


__all__ = ["main"]
