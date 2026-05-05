#!/usr/bin/env python3
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
moved into place via ``shutil.move``.  A failed copy leaves the previous
installation intact.

**Windows compatibility**

Two Windows-specific problems are handled explicitly:

1. ``shutil.rmtree`` raises ``PermissionError`` on read-only files.
   ``shutil.copytree`` uses ``copy2`` by default, which preserves the
   source file's read-only attribute.  Files checked out of git are
   often read-only on Windows.  ``_rmtree`` registers an error handler
   that clears the read-only bit and retries, using ``onexc`` on
   Python >= 3.12 (where ``onerror`` is deprecated) and ``onerror``
   on older versions.

2. ``os.rename`` can fail on Windows immediately after ``shutil.rmtree``
   because Windows file-system handles may still be open asynchronously.
   ``shutil.move`` is used instead; it calls ``os.replace`` internally
   and handles the edge cases that ``os.rename`` does not.

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
import stat
import sys


def _rmtree_error_handler(*args: object) -> None:
    """Clear the read-only bit on *path* and retry *func*.

    Compatible with both the legacy ``onerror`` callback signature
    ``(func, path, exc_info)`` used by Python < 3.12 and the new
    ``onexc`` callback signature ``(func, path, exc)`` introduced in
    Python 3.12.  Both signatures pass ``func`` as the first positional
    argument and ``path`` as the second; only the third argument differs,
    and this handler does not need it.

    Parameters
    ----------
    args[0] : callable
        The ``shutil`` internal function that failed (e.g. ``os.unlink``).
    args[1] : str
        The filesystem path that triggered the error.
    args[2] : object
        Either a ``(type, value, traceback)`` tuple (``onerror``) or an
        exception instance (``onexc``).  Unused by this handler.

    Notes
    -----
    Silently ignores a second failure after chmod so that callers do not
    crash when a file is locked by another process.  The outer caller
    will detect the incomplete removal and report a clear error.
    """
    func, path = args[0], args[1]
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


def _rmtree(path: str) -> None:
    """Remove *path* recursively, handling read-only files on Windows.

    On POSIX, delegates to ``shutil.rmtree`` directly.  On Windows,
    registers ``_rmtree_error_handler`` so that read-only files are
    made writable before deletion.  Uses ``onexc`` on Python >= 3.12
    and ``onerror`` on older releases.

    Parameters
    ----------
    path : str
        Directory to remove.

    Raises
    ------
    OSError
        If removal still fails after the read-only workaround (e.g. the
        file is locked by another process).
    """
    if sys.platform != "win32":
        shutil.rmtree(path)
        return

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_rmtree_error_handler)
    else:
        shutil.rmtree(path, onerror=_rmtree_error_handler)


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

    if not _needs_update(src_dir, dst_dir):
        print(f"features module up-to-date: {dst_dir}")
        return

    # Atomic-ish replacement: copy to a temp name then move into place.
    #
    # shutil.move is used instead of os.rename because on Windows,
    # os.rename can fail immediately after shutil.rmtree while file-system
    # handles are still being released asynchronously.  shutil.move uses
    # os.replace internally and handles these edge cases.
    dst_tmp = dst_dir + ".tmp"
    if os.path.exists(dst_tmp):
        _rmtree(dst_tmp)

    try:
        shutil.copytree(src_dir, dst_tmp)
        if os.path.isdir(dst_dir):
            _rmtree(dst_dir)
        shutil.move(dst_tmp, dst_dir)
    except Exception as exc:
        if os.path.exists(dst_tmp):
            try:
                _rmtree(dst_tmp)
            except OSError:
                pass
        print(
            f"ERROR: failed to install features module to {dst_dir!r}:\n"
            f"       {exc}\n"
            "       Ensure the Python interpreter has write access to the\n"
            f"       meson site-packages directory:\n"
            f"       {mods_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # → this cause win an invisible byte $'\302\203', print the safe ASCII.
    print(f"Installed features module: {src_dir!r} -> {dst_dir!r}")


if __name__ == "__main__":
    main()
