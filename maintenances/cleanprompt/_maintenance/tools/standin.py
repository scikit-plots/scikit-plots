"""
Build a minimal parent package to test cleanprompt without the full library.

Usage::

    python -B maintenances/cleanprompt/_maintenance/tools/standin.py DEST [--run PYTHON]

Notes
-----
**User notes.** ``scikitplot`` has compiled extensions; an interpreter
without them built (an old CPython, a fresh CI image) cannot import the
whole library. cleanprompt depends on nothing else in it but
``scikitplot._testing``, so this copies those two into ``DEST`` under an
empty ``scikitplot/__init__.py``, with a ``pytest.ini`` that turns warnings
into errors, and with ``--run PYTHON`` runs the package suite there with that
interpreter (for example a Python 3.8 environment with pytest installed).

**Developer notes.** Idempotent: ``DEST`` is rebuilt from scratch on every
run, so a stale copy never survives. To make that safe, the tool only
deletes a directory it created itself, recognised by the marker file
``.cleanprompt-standin``; an existing directory without the marker is
refused. Caches are not copied.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MARKER = ".cleanprompt-standin"
PYTEST_INI = "[pytest]\nfilterwarnings = error\n"
PACKAGES = ("cleanprompt", "_testing")


def _root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "scikitplot" / "cleanprompt" / "_engine.py").is_file():
            return candidate
    msg = "could not locate the repository root"
    raise SystemExit(msg)


def build(dest: Path, root: Path | None = None) -> Path:
    """
    Rebuild the stand-in parent at ``dest``.

    Parameters
    ----------
    dest : Path
        Target directory: absent, empty, or an earlier stand-in.
    root : Path, optional
        Checkout root; found from this file when omitted.

    Returns
    -------
    Path
        ``dest``.

    Raises
    ------
    SystemExit
        If ``dest`` exists, is not empty and is not an earlier stand-in.
    """
    root = root or _root()
    if dest.exists():
        if not (dest / MARKER).is_file() and any(dest.iterdir()):
            msg = f"{dest} exists and is not a stand-in (no {MARKER}); refusing to delete it"
            raise SystemExit(msg)
        shutil.rmtree(dest)
    package = dest / "scikitplot"
    package.mkdir(parents=True)
    (dest / MARKER).write_text("built by standin.py\n", encoding="utf-8")
    (dest / "pytest.ini").write_text(PYTEST_INI, encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc")
    for name in PACKAGES:
        shutil.copytree(root / "scikitplot" / name, package / name, ignore=ignore)
    return dest


def main(argv: list[str] | None = None) -> int:
    """Build the stand-in, and run the suite when ``--run`` is given."""
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("dest", type=Path)
    parser.add_argument(
        "--run", metavar="PYTHON", help="interpreter to run pytest with"
    )
    args = parser.parse_args(argv)
    dest = build(args.dest.resolve())
    sys.stdout.write(f"stand-in ready: {dest}\n")
    if not args.run:
        return 0
    command = [args.run, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "scikitplot/cleanprompt/tests"]  # fmt: skip
    # The interpreter is the caller's own choice on its own machine.
    return subprocess.run(command, cwd=dest, check=False).returncode  # noqa: S603


if __name__ == "__main__":
    raise SystemExit(main())
