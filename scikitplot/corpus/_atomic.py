# scikitplot/corpus/_atomic.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Atomic file publication primitive.

A single, well-tested way to publish a file so that concurrent writers never
observe or clobber each other's staging files, and readers never see a partial
file. Every cache/export/storage writer in :mod:`scikitplot.corpus` funnels
through here instead of rolling its own ``path + ".tmp"`` scheme (the
CORPUS-TMP-001 predictable-temp race).

Guarantees
----------
* **Unique staging file.** Each publish stages to a unique same-directory temp
  created with :func:`tempfile.mkstemp`, so two processes publishing the same
  target do not share (and cannot delete) each other's temp.
* **Durable.** The staging file is ``fsync``-ed before it is published, and the
  containing directory is ``fsync``-ed after (best-effort; a no-op where the
  platform cannot sync a directory).
* **Atomic.** Publication is a single :func:`os.replace`, which is atomic on
  POSIX and Windows for same-filesystem paths.
* **No orphans on failure.** If the writer or sync fails, the staging file is
  removed and the original error propagates.

Notes
-----
**Developer note:** Same-directory staging is required for atomic replace —
``os.replace`` across filesystems is not atomic and raises ``OSError``. Callers
therefore never pass a temp dir; the temp always lives beside the target.
"""

from __future__ import annotations

import os
import pathlib
import tempfile
from typing import Callable, Union

__all__ = ["atomic_write_bytes", "atomic_write_path"]

StrPath = Union[str, "os.PathLike[str]"]


def _fsync_file(path: pathlib.Path) -> None:
    """``fsync`` a file by path (best-effort; ignores platforms that can't)."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _fsync_dir(path: pathlib.Path) -> None:
    """``fsync`` a directory so the rename is durable (no-op on Windows)."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        # Directories cannot be fsync-ed on some platforms (e.g. Windows).
        pass
    finally:
        os.close(fd)


def atomic_write_path(
    target: StrPath,
    writer: Callable[[pathlib.Path], None],
    *,
    suffix: str = ".tmp",
) -> pathlib.Path:
    """Atomically publish a file produced by ``writer``.

    Creates a unique staging file beside *target*, invokes ``writer(tmp_path)``
    to populate it, ``fsync``s it, then atomically replaces *target*. On any
    error the staging file is removed and the error re-raised.

    Parameters
    ----------
    target : str or os.PathLike
        Final path to publish. Parent directories are created if needed.
    writer : callable
        ``writer(tmp_path)`` must write the complete file contents to the given
        staging path (e.g. ``lambda p: numpy.save(str(p), arr)``).
    suffix : str, optional
        Suffix for the staging file. Use one the writer expects — e.g.
        ``".npy"`` for :func:`numpy.save`, which otherwise appends it.

    Returns
    -------
    pathlib.Path
        The published *target* path.

    Raises
    ------
    Exception
        Whatever ``writer`` raises (after the staging file is cleaned up).
    """
    target = pathlib.Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.name + ".", suffix=suffix
    )
    os.close(fd)  # the writer opens the path itself
    tmp_path = pathlib.Path(tmp_name)
    try:
        writer(tmp_path)
        _fsync_file(tmp_path)
        os.replace(tmp_path, target)
    except BaseException:
        try:  # ruff: ignore[suppressible-exception]
            tmp_path.unlink()
        except OSError:
            pass
        raise
    _fsync_dir(target.parent)
    return target


def atomic_write_bytes(
    target: StrPath,
    data: bytes,
    *,
    suffix: str = ".tmp",
) -> pathlib.Path:
    """Atomically publish raw *data* bytes to *target*.

    Convenience wrapper over :func:`atomic_write_path` that writes and
    ``fsync``s the bytes itself.

    Parameters
    ----------
    target : str or os.PathLike
        Final path to publish.
    data : bytes
        Payload to write.
    suffix : str, optional
        Staging-file suffix.

    Returns
    -------
    pathlib.Path
        The published *target* path.
    """

    def _write(tmp_path: pathlib.Path) -> None:
        with open(tmp_path, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())

    return atomic_write_path(target, _write, suffix=suffix)
