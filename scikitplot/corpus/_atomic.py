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

import errno
import logging
import os
import pathlib
import shutil
import tempfile
from typing import Callable, Union

logger = logging.getLogger(__name__)

__all__ = ["atomic_write_bytes", "atomic_write_path"]

StrPath = Union[str, "os.PathLike[str]"]


#: ``errno`` values meaning "this platform cannot sync this object", as opposed
#: to "the sync was attempted and failed". Windows cannot sync a directory
#: handle at all, and some filesystems reject ``fsync`` on a directory with
#: ``EINVAL``. Anything outside this set is a real I/O failure and is reported.
_UNSUPPORTED_SYNC_ERRNOS: frozenset[int] = frozenset(
    code
    for code in (
        getattr(errno, name, None)
        for name in ("EINVAL", "ENOTSUP", "EOPNOTSUPP", "ENOSYS", "EBADF")
    )
    if code is not None
)

#: Additional ``errno`` values accepted when syncing a *directory*: opening a
#: directory for reading is refused outright on Windows.
_UNSUPPORTED_DIR_SYNC_ERRNOS: frozenset[int] = _UNSUPPORTED_SYNC_ERRNOS | frozenset(
    code
    for code in (getattr(errno, name, None) for name in ("EACCES", "EPERM", "EISDIR"))
    if code is not None
)


def _describe(exc: OSError) -> str:
    """Return the symbolic ``errno`` name for ``exc``, or its numeric value."""
    return errno.errorcode.get(exc.errno, str(exc.errno))


def _sync(path: pathlib.Path, *, tolerated: frozenset[int], kind: str) -> None:
    """``fsync`` *path*, propagating a real failure and reporting a downgrade.

    Parameters
    ----------
    path : pathlib.Path
        File or directory to sync.
    tolerated : frozenset of int
        ``errno`` values that mean the platform cannot perform this sync.
    kind : str
        ``"file"`` or ``"directory"``, used in the downgrade message.

    Raises
    ------
    OSError
        If the sync was attempted and failed for any reason outside
        ``tolerated`` — the payload may not be on stable storage, and a caller
        that was told the write succeeded would be wrong.

    Notes
    -----
    **Developer.** The previous form caught every :class:`OSError` and returned,
    so ``EIO`` from a failing disk and ``EINVAL`` from a platform that cannot
    sync a directory produced identical behaviour and publication still reported
    success. Only the second is a platform limitation; it is downgraded and
    logged with its ``errno`` named, because a weaker durability guarantee that
    nobody is told about is indistinguishable from the strong one.

    Carrying the downgrade in the return value rather than the log belongs to
    the publication-sequence slice, which changes this function's contract.
    """
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError as exc:
        if exc.errno in tolerated:
            logger.warning(
                "durability downgraded: cannot open %s %s to sync (%s); "
                "the write is published but not known to be on stable storage.",
                kind,
                path,
                _describe(exc),
            )
            return
        raise
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno in tolerated:
            logger.warning(
                "durability downgraded: this platform cannot sync %s %s (%s); "
                "the write is published but not known to be on stable storage.",
                kind,
                path,
                _describe(exc),
            )
            return
        raise
    finally:
        os.close(fd)


def _fsync_file(path: pathlib.Path) -> None:
    """``fsync`` a file by path; a real I/O failure is raised, not swallowed."""
    _sync(path, tolerated=_UNSUPPORTED_SYNC_ERRNOS, kind="file")


def _fsync_dir(path: pathlib.Path) -> None:
    """``fsync`` a directory so the rename is durable, where the platform can."""
    _sync(path, tolerated=_UNSUPPORTED_DIR_SYNC_ERRNOS, kind="directory")


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
        # The staging path may be a directory: a writer is free to replace the
        # empty staging file with a populated directory, which the artifact
        # writer does. unlink() cannot remove one, so the failure was swallowed
        # here and the staging directory was left beside the target after every
        # failed publication.
        try:  # ruff: ignore[suppressible-exception]
            if tmp_path.is_dir():
                shutil.rmtree(tmp_path, ignore_errors=True)
            else:
                tmp_path.unlink(missing_ok=True)
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
