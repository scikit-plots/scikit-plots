#!/usr/bin/env python3
#
# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Install the local ``_meson_features`` package into the active meson's
module directory so that ``import('features')`` resolves correctly.

Called from meson.build via:

.. code-block:: meson

    run_command(
        py,
        meson.project_source_root() / 'scikitplot/_build_utils/install_meson_features.py',
        meson.project_source_root() / 'scikitplot/_build_utils/_meson_features',
        check: true,
    )

The destination is always::

    <mesonbuild.modules.__file__>/../features/

Notes
-----
**Why copy into mesonbuild.modules?**

Stock Meson resolves ``import('features')`` via::

    importlib.import_module('mesonbuild.modules.features')

``mesonbuild`` is a *regular* Python package (it has an ``__init__.py``),
not a namespace package (PEP 420).  Python's import system therefore
anchors ``mesonbuild.*`` entirely inside the installed ``mesonbuild/``
directory.  There is no ``MESON_MODULE_PATH``, no plugin path, and no
namespace-package merging that would let a local ``meson_ext/`` directory
shadow a sub-package of an installed regular package.

Consequence: the only way to make ``import('features')`` find our module
is to place it at ``<mesonbuild.modules location>/features/``.  That is
what this script does.

**Idempotency**

The up-to-date check is performed **before** any destructive operation.
``_needs_update`` compares modification timestamps of every source file
against the corresponding destination file.  Only when the check returns
``True`` is the destination removed and replaced.  Re-running
``meson setup`` is therefore cheap when nothing has changed.

**Atomic replacement**

Copy is performed to a ``<dst>.tmp`` directory first.  Only after the
copy succeeds is the previous destination removed and the temp directory
renamed into place.  A failed copy leaves the previous installation
intact.

References
----------
- https://github.com/numpy/meson/tree/main-numpymeson/mesonbuild/modules/features
- https://github.com/mesonbuild/meson/blob/1.5.0/mesonbuild/interpreter/interpreter.py

Examples
--------
Direct invocation (for testing outside meson):

.. code-block:: shell

    python install_meson_features.py path/to/_meson_features
"""

from __future__ import annotations

import os
import shutil
import sys


def _needs_update(src_dir: str, dst_dir: str) -> bool:
    """Return ``True`` when *dst_dir* is missing or any source file is newer.

    Parameters
    ----------
    src_dir : str
        Absolute path to the source ``_meson_features`` directory.
    dst_dir : str
        Absolute path to the destination inside ``mesonbuild/modules/``.

    Returns
    -------
    bool
        ``True``  → destination is absent or stale; copy is required.
        ``False`` → every source file is older than its destination counterpart;
        no copy needed.

    Notes
    -----
    Comparison is based on ``os.path.getmtime`` (float seconds since epoch).
    Sub-directories inside *src_dir* are not recursed into; the package
    layout is intentionally flat (``__init__.py``, ``module.py``,
    ``feature.py``, ``utils.py``).
    """
    if not os.path.isdir(dst_dir):
        return True
    for name in os.listdir(src_dir):
        src_file = os.path.join(src_dir, name)
        dst_file = os.path.join(dst_dir, name)
        if not os.path.exists(dst_file):
            return True
        if os.path.getmtime(src_file) > os.path.getmtime(dst_file):
            return True
    return False


def main() -> None:
    """Entry point: validate arguments, locate meson modules dir, install.

    Parameters
    ----------
    sys.argv[1] : str
        Path to the local ``_meson_features`` source directory.

    Raises
    ------
    SystemExit
        Exit code 1 on any validation or installation failure, with a
        diagnostic message on stderr.

    Notes
    -----
    The function is intentionally a single linear flow so that every error
    path is explicit and testable.  No intermediate state is left on disk
    after a failure: the ``<dst>.tmp`` staging directory is cleaned up in
    the exception handler before exit.
    """
    if len(sys.argv) != 2:
        print(
            f"Usage: {sys.argv[0]} <path/to/_meson_features>",
            file=sys.stderr,
        )
        sys.exit(1)

    src_dir = os.path.abspath(sys.argv[1])

    if not os.path.isdir(src_dir):
        print(
            f"ERROR: source directory not found: {src_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Locate the active mesonbuild modules directory.
    try:
        import mesonbuild.modules as _mods
    except ImportError:
        print(
            "ERROR: mesonbuild is not importable from this Python interpreter.\n"
            f"       interpreter: {sys.executable}",
            file=sys.stderr,
        )
        sys.exit(1)

    mods_dir = os.path.dirname(_mods.__file__)
    dst_dir = os.path.join(mods_dir, "features")

    # Always overwrite — meson setup is the authoritative trigger.
    # if os.path.isdir(dst_dir):
    #     shutil.rmtree(dst_dir)

    if not _needs_update(src_dir, dst_dir):
        print(f"features module up-to-date: {dst_dir}")
        return

    # Atomic-ish replacement: copy to a temp name then rename.
    dst_tmp = dst_dir + ".tmp"
    if os.path.exists(dst_tmp):
        shutil.rmtree(dst_tmp)

    try:
        shutil.copytree(src_dir, dst_tmp)
        if os.path.isdir(dst_dir):
            shutil.rmtree(dst_dir)
        os.rename(dst_tmp, dst_dir)
    except Exception as exc:
        if os.path.exists(dst_tmp):
            shutil.rmtree(dst_tmp, ignore_errors=True)
        print(
            f"ERROR: failed to install features module to {dst_dir!r}:\n"
            f"       {exc}\n"
            "       Ensure the Python interpreter has write access to the\n"
            f"       meson site-packages directory:\n"
            f"       {mods_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Installed features module: {src_dir!r} → {dst_dir!r}")


if __name__ == "__main__":
    main()
