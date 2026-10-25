# scikitplot/_env/_xeus/_pip.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Pure-Python PyPI packages for a browser (``emscripten-wasm32``) prefix.

Notes
-----
**User notes.** pip cannot install into a prefix whose Python runs in the
browser, so the host's pip downloads the wheels for the prefix's Python
version into a temporary directory (``--no-deps``: dependencies must come
from emscripten-forge / conda-forge) and the files are copied into the
prefix. Only pure-Python wheels are accepted; anything compiled is refused
before a single file is copied.

**Developer notes.** Rewritten from the Xeus add-on's ``_pip.py``. Fixed on
the way: the binary-suffix check (``".exe.dll"`` never matched a suffix), the
unclosed ``conda-meta`` file, ``RECORD`` paths that could escape the prefix
(``../../../etc``), and copying before validating (a refused package left
half its files behind).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable, Sequence

from .._core._errors import ExecutionError

__all__ = ["install_pure_python", "prefix_python_version"]

logger = logging.getLogger(__name__)

#: Suffixes of compiled files that cannot run in the browser prefix.
BINARY_SUFFIXES = frozenset({".so", ".a", ".dylib", ".lib", ".dll", ".exe", ".pyd"})

_DATA = "../../"  # RECORD prefix of files outside site-packages (pip --target)


def prefix_python_version(prefix: str | Path) -> str:
    """
    Return the ``major.minor`` Python version installed in a Conda prefix.

    Parameters
    ----------
    prefix : str or pathlib.Path
        Conda prefix.

    Returns
    -------
    str
        For example ``"3.13"``.

    Raises
    ------
    ExecutionError
        When the prefix has no Python package record, or several.
    """
    records = sorted(Path(prefix, "conda-meta").glob("python-3.*.json"))
    if len(records) != 1:
        raise ExecutionError(
            f"{prefix} has {len(records)} python package records; exactly one is needed"
            " to install PyPI packages",
            step="install-pypi",
            hint="add python to the Conda packages of the environment",
        )
    version = json.loads(records[0].read_text(encoding="utf-8"))["version"].split(".")
    return f"{version[0]}.{version[1]}"


def _plan_copy(dist_info: Path, staging: Path) -> list[tuple[str, bool]]:
    """
    Validate one distribution in the staging directory before anything is copied.

    Parameters
    ----------
    dist_info : pathlib.Path
        Its ``*.dist-info`` directory.
    staging : pathlib.Path
        The pip ``--target`` directory.

    Returns
    -------
    list of (str, bool)
        ``(RECORD path, inside site-packages)`` of every file.

    Raises
    ------
    ExecutionError
        For a wheel that is not pure Python, a compiled file, a path that
        leaves the prefix, or a listed file that is missing.
    """
    wheel = (dist_info / "WHEEL").read_text(encoding="utf-8")
    if "Root-Is-Purelib: true" not in wheel:
        raise ExecutionError(
            f"{dist_info.name} is not a pure-Python wheel",
            step="install-pypi",
            hint="take the package from emscripten-forge instead",
        )
    rows = list(
        csv.reader(io.StringIO((dist_info / "RECORD").read_text(encoding="utf-8")))
    )
    pairs = []
    for row in rows:
        if not row:
            continue
        path = row[0]
        in_site = not path.startswith(_DATA)
        rel = PurePosixPath(path if in_site else path[len(_DATA) :])
        # A backslash or a drive is a Windows path that PurePosixPath would not
        # split: "..\\..\\x" must not pass as one harmless name.
        if (
            rel.is_absolute()
            or ".." in rel.parts
            or "\\" in path
            or PureWindowsPath(path).drive
        ):
            raise ExecutionError(
                f"{dist_info.name}: RECORD path {path!r} leaves the prefix",
                step="install-pypi",
            )
        if rel.suffix.lower() in BINARY_SUFFIXES:
            raise ExecutionError(
                f"{dist_info.name} contains the compiled file {path!r}",
                step="install-pypi",
                hint="only pure-Python packages can be added this way",
            )
        if not (staging / rel).is_file() and not path.endswith("RECORD"):
            raise ExecutionError(
                f"{dist_info.name}: {path!r} is listed but missing", step="install-pypi"
            )
        pairs.append((path, in_site))
    return pairs


def install_pure_python(
    prefix: str | Path,
    requirements: Sequence[str],
    *,
    index_args: Sequence[str] = (),
    python: str | None = None,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> list[str]:
    """
    Install pure-Python PyPI packages into a browser Conda prefix.

    Parameters
    ----------
    prefix : str or pathlib.Path
        The ``emscripten-wasm32`` prefix (already created).
    requirements : sequence of str
        pip arguments: requirement strings, ``-r FILE``, ``-e PATH``.
    index_args : sequence of str, default=()
        Index arguments of the request (``--index-url``, ``--find-links``,
        ``--no-index`` under the offline policy).
    python : str or None, default=None
        Host Python that runs pip; :py:data:`sys.executable` when None.
    run : callable, default=subprocess.run
        Process runner (tests pass a fake).

    Returns
    -------
    list of str
        Names of the installed distributions (``*.dist-info`` directories).

    Raises
    ------
    ExecutionError
        When pip fails or a package is not pure Python; nothing is copied
        in that case.
    """
    prefix = Path(prefix)
    version = prefix_python_version(prefix)
    site = prefix / "lib" / f"python{version}" / "site-packages"
    args = [
        a if a != "-e" else "" for a in requirements
    ]  # editable is meaningless here
    args = [a for a in args if a]
    with tempfile.TemporaryDirectory(prefix="skplt-xeus-pip-") as tmp:
        staging = Path(tmp)
        argv = [
            python or sys.executable,
            "-m",
            "pip",
            "install",
            *args,
            *index_args,
            "--target",
            str(staging),
            "--python-version",
            version,
            "--no-deps",
            "--no-compile",
            "--no-input",
            "--disable-pip-version-check",
        ]
        done = run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            check=False,
        )
        for line in (done.stdout or "").splitlines():
            logger.debug("pip: %s", line)
        if done.returncode != 0:
            tail = "\n".join((done.stdout or "").splitlines()[-20:])
            raise ExecutionError(
                f"pip failed with exit status {done.returncode}:\n{tail}",
                step="install-pypi",
                returncode=done.returncode,
            )
        plans = {d: _plan_copy(d, staging) for d in sorted(staging.glob("*.dist-info"))}
        for dist_info, pairs in plans.items():
            for path, in_site in pairs:
                rel = path if in_site else path[len(_DATA) :]
                dest = (site if in_site else prefix) / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(staging / rel, dest)
            # site-packages is three levels below the prefix, not two (pip --target)
            record = site / dist_info.name / "RECORD"
            record.write_text(
                record.read_text(encoding="utf-8").replace(_DATA, "../../../"),
                encoding="utf-8",
            )
        return [d.name for d in plans]
