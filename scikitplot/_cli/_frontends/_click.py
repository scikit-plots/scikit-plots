# scikitplot/_cli/_frontends/_click.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""click frontend: optional adapter, imported only when selected.

Projects the same neutral registry specs onto :mod:`click`. Command params come
from the registry, not from importing handlers, so rendering help never imports
a handler; the handler is imported lazily in :func:`loader.dispatch` on
invocation (strict-lazy help boundary, guide s9/s96).
"""

from __future__ import annotations

import sys
from typing import Any, Sequence

import click

from .._spec import CommandSpec, Param
from ..context import Context
from ..errors import CliError
from ..loader import dispatch
from ..registry import BUILTIN_COMMANDS


def _decorate(fn, prm: Param):
    """Apply one neutral :class:`Param` as a click decorator."""
    if prm.kind == "argument":
        return click.argument(
            prm.dest,
            nargs=-1 if prm.multiple else 1,
            required=prm.required,
            default=None if prm.multiple else prm.default,
        )(fn)
    if prm.count:
        return click.option(
            *prm.flags,
            prm.dest,
            count=True,
            default=prm.default or 0,
            help=prm.help,
        )(fn)
    if prm.kind == "flag":
        if prm.negatable:
            decl = prm.flags[0] + "/--no-" + prm.flags[0].lstrip("-")
            return click.option(
                decl,
                prm.dest,
                default=bool(prm.default),
                help=prm.help,
            )(fn)
        return click.option(
            *prm.flags,
            prm.dest,
            is_flag=True,
            default=bool(prm.default),
            help=prm.help,
        )(fn)
    return click.option(
        *prm.flags,
        prm.dest,
        type=click.Choice(list(prm.choices)) if prm.choices else (prm.type or str),
        default=prm.default,
        required=prm.required,
        multiple=prm.multiple,
        metavar=prm.metavar,
        help=prm.help,
    )(fn)


def _add_verbosity_options(fn):
    """Attach repeatable ``-v``/``-q`` count options to a click callback."""
    fn = click.option(
        "-q",
        "--quiet",
        "_quiet",
        count=True,
        help="Decrease verbosity (repeatable: -q, -qq).",
    )(fn)
    fn = click.option(
        "-v",
        "--verbose",
        "_verbose",
        count=True,
        help="Increase verbosity (repeatable: -v, -vv, -vvv).",
    )(fn)
    return fn  # ruff: ignore[unnecessary-assign]


def _make_command(spec: CommandSpec) -> click.Command:
    def callback(**params: Any) -> None:
        from ..logging import (  # ruff: ignore[import-outside-top-level]
            configure,
            resolve,
        )

        cctx = click.get_current_context()
        root_v = (cctx.obj or {}).get("verbosity", 0)
        cmd_v = resolve(params.pop("_verbose", 0), params.pop("_quiet", 0))
        verbosity = root_v + cmd_v
        configure(verbosity)
        ctx = Context(fmt=params.get("fmt", "text"), verbosity=verbosity)
        code = dispatch(spec, params, ctx)
        if code:
            raise SystemExit(code)

    callback.__name__ = spec.name.replace("-", "_")
    cmd = callback
    for prm in reversed(spec.params):  # reversed -> declared order preserved
        cmd = _decorate(cmd, prm)
    cmd = _add_verbosity_options(cmd)  # accept -v/-q after the command too
    return click.command(
        name=spec.name,
        help=spec.summary,
        hidden=spec.hidden,
    )(cmd)


def build_group(specs: Sequence[CommandSpec] = BUILTIN_COMMANDS) -> click.Group:
    """Build the root click group from command metadata only."""

    @click.group(
        name="scikitplot",
        help="scikit-plots command-line interface (click frontend).",
    )
    @click.version_option(package_name="scikitplot", prog_name="scikitplot")
    @click.option(
        "-v",
        "--verbose",
        "_verbose",
        count=True,
        help="Increase verbosity (repeatable: -v, -vv, -vvv).",
    )
    @click.option(
        "-q",
        "--quiet",
        "_quiet",
        count=True,
        help="Decrease verbosity (repeatable: -q, -qq).",
    )
    @click.pass_context
    def root(cctx: click.Context, _verbose: int, _quiet: int) -> None:
        from ..logging import (  # ruff: ignore[import-outside-top-level]
            configure,
            resolve,
        )

        cctx.ensure_object(dict)
        # Root-level verbosity; commands add their own and sum the two so that
        # `-v <cmd> -v` combines identically to the argparse frontend.
        cctx.obj["verbosity"] = resolve(_verbose, _quiet)
        configure(cctx.obj["verbosity"])

    for spec in specs:
        command = _make_command(spec)
        root.add_command(command)
        for alias in spec.aliases:
            root.add_command(command, name=alias)
    return root


def run(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` with click and dispatch. Returns a process exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    group = build_group()
    try:
        return (
            group.main(
                args=argv,
                standalone_mode=False,
                prog_name="scikitplot",
            )
            or 0
        )
    except click.ClickException as exc:  # usage errors -> stderr, exit 2
        exc.show()
        return exc.exit_code
    except click.exceptions.Abort:
        return 130
    except CliError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        if exc.hint:
            sys.stderr.write(f"Hint: {exc.hint}\n")
        return exc.exit_code


__all__ = ["build_group", "run"]
