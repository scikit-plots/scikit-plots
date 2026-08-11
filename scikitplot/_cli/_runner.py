# scikitplot/_cli/_runner.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone module runner: ``python -m <module>`` for a single function.

Minimal, stdlib-only argparse over a library function's *native* parameters.
No centralized registry, no click, no subcommands (ADR-CLI-105). Reuses the
neutral ``Param`` -> argparse mapping so a module runner's flags behave exactly
like the centralized CLI's.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable, Sequence

from ._frontends._argparse import add_param
from ._spec import Param


def run_module(
    func: Callable[..., Any],
    params: Sequence[Param] = (),
    argv: Sequence[str] | None = None,
    *,
    prog: str | None = None,
) -> int:
    """Run ``func`` as a standalone ``python -m`` entry point.

    Parameters
    ----------
    func : callable
        Library function to invoke. An :class:`int` return is used as the exit
        code; anything else yields ``0``.
    params : sequence of Param
        Neutral specs describing ``func``'s CLI-exposed parameters.
    argv : sequence of str, optional
        Argument vector; defaults to ``sys.argv[1:]``.
    prog : str, optional
        Program name for help; defaults to the real dotted module path.

    Returns
    -------
    int
        Process exit code.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if prog is None:
        # Under `python -m pkg.mod` the module loads as __main__; recover its
        # real dotted name from its spec so help text reads correctly.
        main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
        mod_name = getattr(main_spec, "name", None) or func.__module__
        prog = f"python -m {mod_name}"
    doc = (func.__doc__ or "").strip()
    parser = argparse.ArgumentParser(
        prog=prog,
        add_help=True,
        description=doc.splitlines()[0] if doc else None,
    )
    for prm in params:
        add_param(parser, prm)
    ns = parser.parse_args(argv)
    kwargs = {prm.dest: getattr(ns, prm.dest) for prm in params}
    result = func(**kwargs)
    return result if isinstance(result, int) else 0


__all__ = ["run_module"]
