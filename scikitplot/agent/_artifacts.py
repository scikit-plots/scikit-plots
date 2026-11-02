# scikitplot/agent/_artifacts.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Content-addressed storage for large tool results.

Notes
-----
**User.** Give an agent ``artifacts=`` and ``offload_results_over=N``: a tool
result whose canonical JSON is longer than ``N`` bytes is written to the store
and the step records only ``result_ref`` -- its digest and size -- instead of
the value. States stay small, so saving after every step stays cheap, and the
planner sees the reference; a planner that needs the value reads it with
``store.get(ref["digest"])``.

**Developer.** The key is :func:`~scikitplot.agent._canonical.digest` of the
result, which is the ``result_digest`` the audit trail already records for
every executed step: an audit record names its artifact by construction.
:meth:`ArtifactStore.get` recomputes the digest of what it read and refuses a
mismatch, so a changed artifact is detected, not returned. A put is
idempotent; a crash between a put and the state save leaves an unreferenced
artifact, never a reference without one.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ._canonical import canonical_json, digest, plain
from ._errors import StateError
from ._state import RUN_ID
from ._store import _sync_directory

__all__ = ["ArtifactStore", "DirectoryArtifactStore", "MemoryArtifactStore"]


def _check_digest(value: Any) -> str:
    if not isinstance(value, str) or not RUN_ID.fullmatch(value):
        raise ValueError(f"an artifact digest is 64 hexadecimal digits, got {value!r}")
    return value


class ArtifactStore:
    """
    Keeps JSON values under the digest of their canonical form.

    Subclasses implement :meth:`_write` and :meth:`_read`; the digest and its
    check are done here, once.
    """

    def put(self, value: Any) -> str:
        """
        Store ``value`` and return its digest.

        Parameters
        ----------
        value : object
            A JSON value (:func:`~scikitplot.agent._canonical.plain`).

        Returns
        -------
        str
            64 hexadecimal digits.
        """
        text = canonical_json(plain(value, what="an artifact"))
        key = digest(value)
        if not self.has(key):
            self._write(key, text)
        return key

    def get(self, key: str) -> Any:
        """
        Return the value stored under ``key``.

        Raises
        ------
        KeyError
            If nothing is stored under ``key``.
        StateError
            If what is stored does not have digest ``key``.
        """
        text = self._read(_check_digest(key))
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise StateError(f"artifact {key}: not JSON: {exc}") from None
        if digest(value) != key:
            raise StateError(f"artifact {key}: its content does not have this digest")
        return value

    def has(self, key: str) -> bool:
        """Whether a value is stored under ``key``."""
        try:
            self._read(_check_digest(key))
        except KeyError:
            return False
        return True

    def _write(self, key: str, text: str) -> None:
        raise NotImplementedError

    def _read(self, key: str) -> str:
        raise NotImplementedError


class MemoryArtifactStore(ArtifactStore):
    """
    Keep artifacts in memory, as their canonical JSON text.

    Examples
    --------
    >>> store = MemoryArtifactStore()
    >>> key = store.put({"rows": [1, 2, 3]})
    >>> store.get(key)
    {'rows': [1, 2, 3]}
    """

    def __init__(self) -> None:
        self._texts: dict[str, str] = {}

    def _write(self, key: str, text: str) -> None:
        self._texts[key] = text

    def _read(self, key: str) -> str:
        return self._texts[key]


class DirectoryArtifactStore(ArtifactStore):
    """
    Keep each artifact in ``<directory>/<first two digits>/<digest>.json``.

    Parameters
    ----------
    directory : str or os.PathLike
        Created if missing.

    Notes
    -----
    Written like :class:`~scikitplot.agent.JsonFileStore`: a temporary file
    in the same directory, synced, renamed into place and the directory
    synced, so a reader sees a whole artifact or none. The digest is checked
    to be 64 hexadecimal digits before it names a file.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        """Return the file that holds ``key``."""
        key = _check_digest(key)
        return self.directory / key[:2] / f"{key}.json"

    def _write(self, key: str, text: str) -> None:
        target = self.path(key)
        target.parent.mkdir(exist_ok=True)
        handle, temp = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{key}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as out:
                out.write(text)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temp, target)
            _sync_directory(target.parent)
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise

    def has(self, key: str) -> bool:
        """Whether a file is stored under ``key``."""
        return self.path(key).is_file()

    def _read(self, key: str) -> str:
        try:
            return self.path(key).read_text(encoding="utf-8")
        except FileNotFoundError:
            raise KeyError(key) from None
