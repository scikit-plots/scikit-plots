# scikitplot/_cli/_frontends/_argparse.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""argparse frontend: always available, the deterministic default.

Projects the neutral registry specs onto :mod:`argparse`. This module is
stdlib-only.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .._spec import CommandSpec, Param
from ..context import Context
from ..loader import dispatch
from ..registry import BUILTIN_COMMANDS


def add_param(parser: argparse.ArgumentParser, prm: Param) -> None:
    """Add one neutral :class:`Param` to an argparse parser.

    Handles arguments, count options, flags (with a Python 3.8-safe negation),
    multiple/append, choices, and typed options.
    """
    if prm.kind == "argument":
        parser.add_argument(
            prm.dest,
            nargs="*" if prm.multiple else "?",
            default=prm.default,
            help=prm.help,
            metavar=prm.metavar or prm.dest.upper(),
        )
        return
    if prm.count:
        parser.add_argument(
            *prm.flags,
            action="count",
            default=prm.default or 0,
            dest=prm.dest,
            help=prm.help,
        )
        return
    if prm.kind == "flag":
        if prm.negatable and hasattr(argparse, "BooleanOptionalAction"):
            parser.add_argument(
                *prm.flags,
                action=argparse.BooleanOptionalAction,
                default=bool(prm.default),
                dest=prm.dest,
                help=prm.help,
            )
        else:
            parser.add_argument(
                *prm.flags,
                action="store_true",
                default=bool(prm.default),
                dest=prm.dest,
                help=prm.help,
            )
            if prm.negatable:  # explicit --no-x keeps 3.8 parity with click
                negations = tuple(
                    "--no-" + f.lstrip("-") for f in prm.flags if f.startswith("--")
                )
                parser.add_argument(
                    *negations,
                    action="store_false",
                    dest=prm.dest,
                    help=argparse.SUPPRESS,
                )
        return
    parser.add_argument(
        *prm.flags,
        dest=prm.dest,
        help=prm.help,
        type=prm.type or str,
        default=prm.default,
        required=prm.required,
        action="append" if prm.multiple else "store",
        choices=list(prm.choices) if prm.choices else None,
        metavar=prm.metavar,
    )


def _add_verbosity(parser: argparse.ArgumentParser, suffix: str) -> None:
    """Add repeatable ``-v``/``-q`` to a parser under suffixed dests.

    Distinct dests (``_verbose{suffix}``/``_quiet{suffix}``) let the root parser
    and each subparser record their counts independently, so verbosity given
    before AND after the command combines instead of one overriding the other.
    """
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        dest=f"_verbose{suffix}",
        help="Increase verbosity (repeatable: -v, -vv, -vvv).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=0,
        dest=f"_quiet{suffix}",
        help="Decrease verbosity (repeatable: -q, -qq).",
    )


def build_parser(
    specs: Sequence[CommandSpec] = BUILTIN_COMMANDS,
) -> argparse.ArgumentParser:
    """Build the root argparse parser from command metadata only."""
    root = argparse.ArgumentParser(
        prog="scikitplot",
        add_help=True,
        description="scikit-plots command-line interface (argparse frontend).",
    )
    root.add_argument(
        "-V",
        "--version",
        action="store_true",
        dest="_version",
        help="Show the scikit-plots version and exit.",
    )
    _add_verbosity(root, "_root")
    sub = root.add_subparsers(dest="_command", metavar="COMMAND")
    for spec in specs:
        if spec.hidden:
            continue
        cp = sub.add_parser(
            spec.name,
            help=spec.summary,
            aliases=list(spec.aliases),
            description=spec.summary,
            add_help=spec.delegate is None,
        )
        if spec.delegate is not None:
            # Cosmetic only: delegated commands are intercepted before argparse
            # parsing (see run/_split_delegated). This entry makes them appear in
            # top-level `--help` and captures stray args if ever reached directly.
            cp.add_argument("_delegate_argv", nargs=argparse.REMAINDER)
            cp.set_defaults(_spec=spec)
            continue
        for prm in spec.params:
            add_param(cp, prm)
        _add_verbosity(cp, "_sub")  # accept -v/-q after the command too
        cp.set_defaults(_spec=spec)
    return root


def _net_verbosity(ns: argparse.Namespace) -> int:
    from ..logging import resolve  # ruff: ignore[import-outside-top-level]

    verbose = getattr(ns, "_verbose_root", 0) + getattr(ns, "_verbose_sub", 0)
    quiet = getattr(ns, "_quiet_root", 0) + getattr(ns, "_quiet_sub", 0)
    return resolve(verbose, quiet)


def _globals_parser() -> argparse.ArgumentParser:
    """
    Run a parser with only the global options, for the prefix before a delegated
    command (whose own args must not be parsed here).
    """  # ruff: ignore[missing-blank-line-after-summary]
    gp = argparse.ArgumentParser(prog="scikitplot", add_help=True)
    gp.add_argument("-V", "--version", action="store_true", dest="_version")
    _add_verbosity(gp, "_root")
    return gp


def _split_delegated(argv: Sequence[str]):
    """If argv targets a delegated command, return (prefix, spec, rest).

    Scans for the first token that names a registered command. If that command
    is delegated, everything after it is returned verbatim as ``rest`` (so the
    submodule owns parsing, including ``--help``). Native commands and the
    no-command case return ``None`` (handled by the normal argparse path).
    """
    from ..registry import resolve  # ruff: ignore[import-outside-top-level]

    for index, token in enumerate(argv):
        spec = resolve(token)
        if spec is None:
            continue
        if spec.delegate is not None:
            return list(argv[:index]), spec, list(argv[index + 1 :])
        return None  # first recognized command is native -> normal path
    return None


def run(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch. Returns a process exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)

    delegated = _split_delegated(argv)
    if delegated is not None:
        prefix, spec, rest = delegated
        gns = _globals_parser().parse_args(prefix)  # -v/-q/-V/-h before the command
        verbosity = _net_verbosity(gns)
        from ..logging import configure  # ruff: ignore[import-outside-top-level]

        configure(verbosity)
        if getattr(gns, "_version", False):
            from ..app import _version_string  # ruff: ignore[import-outside-top-level]

            sys.stdout.write(_version_string() + "\n")
            return 0
        from ..loader import run_delegate  # ruff: ignore[import-outside-top-level]

        return run_delegate(spec.delegate, rest, install_hint=spec.install_hint)

    parser = build_parser()
    ns = parser.parse_args(argv)
    verbosity = _net_verbosity(ns)
    from ..logging import configure  # ruff: ignore[import-outside-top-level]

    configure(verbosity)  # route diagnostics to stderr at the resolved level
    if getattr(ns, "_version", False):
        from ..app import _version_string  # ruff: ignore[import-outside-top-level]

        sys.stdout.write(_version_string() + "\n")
        return 0
    spec: CommandSpec | None = getattr(ns, "_spec", None)
    if spec is None:
        parser.print_help(sys.stderr)  # diagnostics -> stderr
        return 0
    params = {prm.dest: getattr(ns, prm.dest) for prm in spec.params}
    ctx = Context(fmt=params.get("fmt", "text"), verbosity=verbosity)
    return dispatch(spec, params, ctx)


__all__ = ["add_param", "build_parser", "run"]
