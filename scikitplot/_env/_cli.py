# scikitplot/_env/_cli.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Command line: ``scikitplot env`` (also ``python -m scikitplot._env``).

=============  ==============================================================
Command        What it does
=============  ==============================================================
``doctor``     tools found, usable backends, targets
``validate``   parse and check a request (no files read, nothing planned)
``plan``       print the plan of ``ensure`` / ``recreate`` / ``remove``
``create``     build or update; ``--recreate`` rebuilds from scratch
``remove``     delete the environment
``run``        run a command inside it (``-- CMD ARGS``)
``shell-init`` print, or ``--write``, the shell start-up block
``batch``      plan / run every request of a JSON Lines file
=============  ==============================================================

Exit status: 0 success; 1 a step failed; 2 invalid request or dependency
file; 3 cannot be built here (backend, target, network, conversion); 4 an
approval was refused; 5 another run holds the environment; ``run`` returns
the command's own status; 130 interrupted.

Notes
-----
**User notes.** ``--format json`` prints one JSON document on stdout (errors
included, with ``code`` and ``hint``), for CI and agents; ``--events FILE``
appends JSON Lines events; ``-v``/``-vv`` show INFO/DEBUG logs (command
output) on stderr. ``--yes`` approves everything the plan shows, ``--approve
KIND`` one kind. ``plan`` never needs an approval.

**Developer notes.** The command line is a thin layer over :py:mod:`._api`;
every option maps to one request field or API argument.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Sequence, TextIO

__all__ = ["main"]

#: Exit status by error class (most specific first).
EXIT = (
    ("LockBusyError", 5),
    ("ApprovalRequiredError", 4),
    ("PlanningError", 3),
    ("ConfigError", 2),
    ("SourceError", 2),
    ("ExecutionError", 1),
)


def _request_options(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("request")
    g.add_argument(
        "-c", "--config", help="configuration file (TOML, JSON, YAML, pyproject)"
    )
    where = g.add_mutually_exclusive_group()
    where.add_argument(
        "-n", "--name", help="environment name (under the environments home)"
    )
    where.add_argument("-p", "--prefix", help="environment directory")
    g.add_argument("--python", help="Python version, e.g. 3.12")
    g.add_argument("--backend", help="auto, uv, venv, pipenv, micromamba, mamba, conda")
    g.add_argument(
        "--package",
        dest="packages",
        action="append",
        metavar="SPEC",
        help="PyPI requirement (repeatable)",
    )
    g.add_argument(
        "--conda-package",
        dest="conda_packages",
        action="append",
        metavar="SPEC",
        help="Conda match spec (repeatable)",
    )
    g.add_argument(
        "-s",
        "--source",
        dest="sources",
        action="append",
        metavar="FILE",
        help="dependency file: requirements*.txt, environment.yml,"
        " pyproject.toml, Pipfile (repeatable)",
    )
    g.add_argument(
        "--group",
        dest="groups",
        action="append",
        metavar="NAME",
        help="extra / dependency group of the pyproject or Pipfile sources",
    )
    g.add_argument("--channel", dest="channels", action="append", help="Conda channel")
    g.add_argument("--index-url")
    g.add_argument("--extra-index-url", dest="extra_index_urls", action="append")
    g.add_argument("--find-links", dest="find_links", action="append")
    g.add_argument("--target", help="host, a Conda subdir, emscripten-wasm32, pyodide")
    g.add_argument("--root-prefix", help="Conda root prefix (package cache)")
    g.add_argument("--profile", help="preset: ci, agent, airgap, interactive")
    g.add_argument(
        "--verify-import",
        dest="verify_imports",
        action="append",
        metavar="MODULE",
        help="import after building (repeatable)",
    )
    p = parser.add_argument_group("policy")
    p.add_argument("--interaction", choices=("never", "auto", "always"))
    p.add_argument("--network", choices=("online", "offline"))
    p.add_argument("--offline", dest="network", action="store_const", const="offline")
    p.add_argument("--bootstrap", choices=("never", "if-missing"))
    p.add_argument("--conversion", choices=("exact", "allow-approximate"))


def _approval_options(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("approvals (never read from files)")
    g.add_argument(
        "-y", "--yes", action="store_true", help="approve every approval the plan needs"
    )
    g.add_argument(
        "--approve",
        action="append",
        choices=("bootstrap", "shell_init", "remove"),
        help="approve one kind (repeatable)",
    )
    g.add_argument("--events", metavar="FILE", help="append JSON Lines events to FILE")


def _parser() -> argparse.ArgumentParser:
    # Accepted before and after the command. SUPPRESS keeps an absent option
    # from resetting a given one; the defaults are filled in by ``main`` (not
    # ``set_defaults``, which would change the actions shared through
    # ``parents`` and so the subcommands' defaults too).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--format",
        choices=("text", "json"),
        default=argparse.SUPPRESS,
        help="output format (default: text)",
    )
    common.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=argparse.SUPPRESS,
        help="INFO logs; -vv DEBUG logs (command output)",
    )
    parser = argparse.ArgumentParser(
        prog="scikitplot env",
        parents=[common],
        description="Build the same Python environment anywhere: plan, approve, execute.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    _add = sub.add_parser

    def add_parser(name: str, **kwargs: Any) -> argparse.ArgumentParser:
        return _add(name, parents=[common], **kwargs)

    sub.add_parser = add_parser  # type: ignore[method-assign]
    sub.add_parser("doctor", help="tools, backends and targets of this host")
    for name, text in (
        ("validate", "check a request"),
        ("plan", "show the plan"),
        ("create", "build or update"),
        ("remove", "delete"),
    ):
        cmd = sub.add_parser(name, help=text)
        _request_options(cmd)
        if name == "plan":
            cmd.add_argument(
                "--operation",
                choices=("ensure", "recreate", "remove"),
                default="ensure",
            )
        if name in ("create", "remove"):
            _approval_options(cmd)
            cmd.add_argument(
                "--lock-timeout", type=float, default=0.0, metavar="SECONDS"
            )
        if name == "create":
            cmd.add_argument(
                "--recreate",
                action="store_true",
                help="delete and build again (approval 'remove')",
            )
    run = sub.add_parser("run", help="run a command inside the environment")
    _request_options(run)
    run.add_argument("argv", nargs=argparse.REMAINDER, help="-- COMMAND [ARGS...]")
    shell = sub.add_parser("shell-init", help="shell start-up block")
    shell.add_argument("shell", choices=("bash", "zsh", "fish", "powershell"))
    shell.add_argument(
        "--write", action="store_true", help="add it to the start-up file"
    )
    shell.add_argument("--rc-file", help="start-up file (default: the shell's)")
    shell.add_argument("-y", "--yes", action="store_true")
    batch = sub.add_parser("batch", help="every request of a JSON Lines file")
    batch.add_argument("file")
    batch.add_argument(
        "--operation", choices=("ensure", "recreate", "remove"), default="ensure"
    )
    batch.add_argument("--dry-run", action="store_true", help="plan only")
    batch.add_argument("--keep-going", action="store_true")
    _approval_options(batch)
    return parser


def _fields(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    fields: dict[str, Any] = {}
    for key in (
        "name",
        "prefix",
        "python",
        "backend",
        "packages",
        "conda_packages",
        "channels",
        "index_url",
        "extra_index_urls",
        "find_links",
        "target",
        "root_prefix",
        "profile",
        "verify_imports",
    ):
        value = getattr(args, key, None)
        if value is not None:
            fields[key] = value
    if args.sources:
        groups = args.groups or []
        fields["sources"] = [
            (
                {"path": s, "groups": groups}
                if groups and s.endswith(("pyproject.toml", "Pipfile"))
                else s
            )
            for s in args.sources
        ]
    elif args.groups:
        from ._core._errors import ConfigError  # noqa: PLC0415

        raise ConfigError("--group needs a --source pyproject.toml or Pipfile")
    policy = {
        k: getattr(args, k)
        for k in ("interaction", "network", "bootstrap", "conversion")
        if getattr(args, k, None)
    }
    if policy:
        fields["policy"] = policy
    return args.config, fields


def _approve(args: argparse.Namespace) -> Any:
    return (
        True
        if getattr(args, "yes", False)
        else tuple(getattr(args, "approve", None) or ())
    )


def _emit(out: TextIO, fmt: str, data: Any, text: str) -> None:
    out.write(
        (
            json.dumps(data, indent=2, sort_keys=True, default=str)
            if fmt == "json"
            else text
        )
        + "\n"
    )


def _result_text(result: Any) -> str:
    if not result.changed:
        return f"{result.plan.prefix}: up to date (nothing to do)"
    lines = [f"{result.plan.prefix}: done with {result.plan.backend}"]
    lines += [f"  ok  {s.id} ({s.seconds:.1f}s)" for s in result.steps]
    if result.receipt:
        lines.append(f"  receipt {result.receipt}")
    return "\n".join(lines)


def _dispatch(args: argparse.Namespace, out: TextIO) -> int:  # noqa: PLR0911
    from . import _api  # noqa: PLC0415

    fmt = args.format
    if args.command == "doctor":
        data = _api.doctor()
        lines = [
            (
                f"host     {data['host']['system']} {data['host']['machine']}"
                f" ({data['host']['conda_subdir']}), Python {data['host']['python']}"
            ),
            f"home     {data['host']['home']}",
        ]
        for name, info in data["backends"].items():
            mark = (
                "yes"
                if info["available"]
                else ("bootstrap" if info["bootstrappable"] else "no")
            )
            lines.append(f"backend  {name:<10} {mark:<9} {info['path'] or ''}")
        _emit(out, fmt, data, "\n".join(lines))
        return 0
    if args.command == "shell-init":
        block = _api.shell_init(
            args.shell,
            write=args.write,
            rc_file=args.rc_file,
            approve=True if args.yes else (),
        )
        _emit(
            out,
            fmt,
            {"shell": args.shell, "block": block, "written": args.write},
            block.rstrip("\n"),
        )
        return 0
    if args.command == "batch":
        records = _api.batch(
            args.file,
            operation=args.operation,
            approve=_approve(args),
            events=args.events,
            keep_going=args.keep_going,
            dry_run=args.dry_run,
        )
        ok = all(r["ok"] for r in records)
        _emit(
            out,
            fmt,
            {"ok": ok, "results": records},
            "\n".join(
                f"{'ok ' if r['ok'] else 'ERR'} {r['id']}"
                + ("" if r["ok"] else f"  {r['error']['message']}")
                for r in records
            ),
        )
        return 0 if ok else 1
    config, fields = _fields(args)
    if args.command == "validate":
        request = _api.validate(config, **fields)
        _emit(
            out,
            fmt,
            request.to_dict(provenance=True),
            f"valid; digest {request.digest[:12]}",
        )
        return 0
    if args.command == "plan":
        plan = _api.plan(config=config, operation=args.operation, **fields)
        _emit(
            out,
            fmt,
            plan.to_dict(),
            plan.render() + "".join(f"\n--- {k}\n{v}" for k, v in plan.artifacts),
        )
        return 0
    if args.command == "run":
        argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
        return _api.run(argv, config=config, **fields)
    common = dict(
        config=config,
        approve=_approve(args),
        events=args.events,
        lock_timeout=args.lock_timeout,
        **fields,
    )
    if args.command == "create":
        result = _api.create(recreate=args.recreate, **common)
    else:
        result = _api.remove(**common)
    _emit(out, fmt, result.to_dict(), _result_text(result))
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """
    Run the ``scikitplot env`` command line.

    Parameters
    ----------
    argv : sequence of str or None, default=None
        Arguments; :py:data:`sys.argv` without the program when None.
    stdout, stderr : text stream or None, default=None
        Output streams (tests pass buffers).

    Returns
    -------
    int
        Exit status (see the module documentation).
    """
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    args = _parser().parse_args(None if argv is None else list(argv))
    args.format = getattr(args, "format", "text")
    args.verbose = getattr(args, "verbose", 0)
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG if args.verbose > 1 else logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
            stream=err,
        )
    from ._core._errors import EnvError  # noqa: PLC0415

    try:
        return _dispatch(args, out)
    except KeyboardInterrupt:
        err.write("interrupted\n")
        return 130
    except EnvError as exc:
        status = (
            exc.returncode
            if getattr(exc, "step", None) == "run" and exc.returncode
            else None
        ) or next(
            code
            for name, code in (*EXIT, (type(exc).__name__, 1))
            if any(c.__name__ == name for c in type(exc).__mro__)
        )
        if args.format == "json":
            _emit(out, "json", {"ok": False, "error": exc.to_dict()}, "")
        else:
            err.write(
                f"error[{exc.code}]: {exc}\n"
                + (f"  hint: {exc.hint}\n" if exc.hint else "")
            )
        return status
