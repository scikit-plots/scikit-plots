# libs/_tools/__main__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Command line of the partial-distribution tooling.

Run from the repository root::

    python -m libs._tools <command> [options]

See ``python -m libs._tools --help`` for the commands.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import generate, registry, staging


def _out(line: str) -> None:
    """Write one line of results to standard output."""
    sys.stdout.write(line + "\n")


def _err(line: str) -> None:
    """Write one line of diagnostics to standard error."""
    sys.stderr.write(line + "\n")


def _lib_dir(name: str) -> Path:
    """Return the lib directory of a distribution given in any spelling."""
    distributions = staging.load_distributions()
    dist = distributions.get(name)
    return staging.repo_root() / "libs" / registry.directory_of(dist.name)


def _cmd_list(_args: argparse.Namespace) -> int:
    root = staging.repo_root()
    distributions = staging.load_distributions(root)
    owned = staging.check_ownership(
        distributions.DISTRIBUTIONS, root / staging.PACKAGE_NAME
    )
    for dist in distributions.DISTRIBUTIONS:
        files = owned[dist.name]
        size = sum(
            (root / staging.PACKAGE_NAME / path).stat().st_size for path in files
        )
        _out(
            f"{dist.name:<28} libs/{registry.directory_of(dist.name):<12} "
            f"{len(files):>4} files {size / 1024:>8.0f} KiB  "
            f"{', '.join(dist.trees)}"
        )
    return 0


def _cmd_generate(_args: argparse.Namespace) -> int:
    written = generate.generate()
    for path in written:
        _out(f"wrote {path}")
    _out(f"{len(written)} file(s) written." if written else "Up to date.")
    return 0


def _cmd_check(_args: argparse.Namespace) -> int:
    stale = generate.check()
    if not stale:
        _out("Generated files are up to date.")
        return 0
    for path in stale:
        _err(f"stale: {path}")
    _err(
        f"{len(stale)} generated file(s) are out of date. "
        "Run: python -m libs._tools generate"
    )
    return 1


def _cmd_stage(args: argparse.Namespace) -> int:
    lib_dir = _lib_dir(args.name)
    staged = staging.stage(args.name, lib_dir)
    _out(f"staged {len(staged)} file(s) into {lib_dir}")
    return 0


def _cmd_unstage(args: argparse.Namespace) -> int:
    distributions = staging.load_distributions()
    names = [args.name] if args.name else [d.name for d in distributions.DISTRIBUTIONS]
    for name in names:
        staging.unstage(_lib_dir(name))
    _out(f"unstaged {len(names)} lib director{'y' if len(names) == 1 else 'ies'}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    from . import verify  # noqa: PLC0415 - not needed by the commands above

    artefacts = verify.build(args.names or None, Path(args.outdir))
    for path in artefacts:
        _out(str(path))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from . import verify  # noqa: PLC0415 - not needed by the commands above

    return verify.main(args)


def build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="python -m libs._tools",
        description="Tooling for the partial distributions of scikit-plots.",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    commands.required = True

    sub = commands.add_parser("list", help="Show every distribution and what it owns.")
    sub.set_defaults(run=_cmd_list)

    sub = commands.add_parser("generate", help="Rewrite the generated packaging files.")
    sub.set_defaults(run=_cmd_generate)

    sub = commands.add_parser(
        "check", help="Exit 1 if a generated packaging file is missing or stale."
    )
    sub.set_defaults(run=_cmd_check)

    sub = commands.add_parser(
        "stage", help="Copy a distribution's files into its lib directory (debugging)."
    )
    sub.add_argument("name", help="Distribution name, in any spelling.")
    sub.set_defaults(run=_cmd_stage)

    sub = commands.add_parser(
        "unstage", help="Remove staged files and build residue from lib directories."
    )
    sub.add_argument("name", nargs="?", help="Distribution name; all when omitted.")
    sub.set_defaults(run=_cmd_unstage)

    sub = commands.add_parser("build", help="Build sdists and wheels.")
    sub.add_argument("names", nargs="*", help="Distribution names; all when omitted.")
    sub.add_argument(
        "--outdir", default="dist/libs", help="Output directory (default: dist/libs)."
    )
    sub.set_defaults(run=_cmd_build)

    sub = commands.add_parser(
        "verify",
        help="Build, then install and test the distributions in clean environments.",
    )
    sub.add_argument(
        "names", nargs="*", help="Distributions to install and test; all when omitted."
    )
    sub.add_argument(
        "--outdir",
        default="dist/libs",
        help="Where artefacts are built (default: dist/libs).",
    )
    sub.add_argument(
        "--python",
        action="append",
        default=None,
        metavar="VERSION",
        help="Python version to test, e.g. 3.12. Repeatable. Default: the running one.",
    )
    sub.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse the artefacts already in --outdir.",
    )
    sub.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running each part's own test suite (imports and commands still run).",
    )
    sub.add_argument(
        "--report", default=None, metavar="FILE", help="Also write the results as JSON."
    )
    sub.set_defaults(run=_cmd_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the tooling command line.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code: 0 on success, 1 when a check fails, 2 for a usage
        error or inconsistent inputs.
    """
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        return int(args.run(args))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        _err(f"error: {message}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__: list[str] = ["build_parser", "main"]
