"""Workflow CLI template.

This file is shipped as **template data**. Use
``scikitplot.cython.copy_workflow(...)`` to copy a workflow to a working
folder, then run:

    python cli.py train --help
    python cli.py hpo --help
    python cli.py predict --help

Design
------
- Executable-friendly: defines ``parse_args`` and ``main``.
- No import-time side effects.
- Loads sibling scripts by filesystem path (does not require a package).

Security
--------
Do not run untrusted workflow folders. These scripts can execute arbitrary code
from the copied folder.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Sequence


def _load_module_from_file(path: Path, module_name: str) -> ModuleType:
    """Load a Python module from an explicit file path.

    Parameters
    ----------
    path : pathlib.Path
        Path to a ``.py`` file.
    module_name : str
        Module name to assign.

    Returns
    -------
    types.ModuleType
        Loaded module.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ImportError
        If loading fails.
    """
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(str(path))

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {path}")

    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as e:
        raise ImportError(f"Failed to import {path}: {e}") from e
    return mod


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Parameters
    ----------
    argv : Sequence[str] or None, default=None
        CLI argv. If None, uses ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="workflow-cli",
        description="Run workflow templates (train / hpo / predict).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # We keep subcommand args minimal here; each script exposes its own --help.
    for cmd in ("train", "hpo", "predict"):
        sp = sub.add_parser(cmd, help=f"Run {cmd}.py")
        sp.add_argument(
            "args",
            nargs=argparse.REMAINDER,
            help="Arguments passed through to the script",
        )

    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : Sequence[str] or None, default=None
        CLI argv. If None, uses ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status.
    """
    ns = parse_args(argv)

    here = Path(__file__).resolve().parent
    script = here / f"{ns.command}.py"
    mod = _load_module_from_file(script, module_name=f"workflow_{ns.command}")

    if not hasattr(mod, "main"):
        raise AttributeError(f"{script.name} does not define main(argv=None)")

    # Pass-through: each script handles its own argparse.
    cmd_argv = list(ns.args)
    try:
        rc = mod.main(cmd_argv)  # type: ignore[attr-defined]
    except SystemExit as e:
        # If the script calls SystemExit, normalize to an int.
        return int(getattr(e, "code", 1) or 0)

    return int(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
