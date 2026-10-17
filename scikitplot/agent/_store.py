# scikitplot/agent/_store.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Where run states are saved, so a run resumes after a pause or a crash.

A store keeps the latest state of each run under its id. The agent saves the
state after every step and *before* every tool call (with the call marked in
flight), so a state read back after a crash says whether a call may have run.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ._canonical import canonical_json
from ._errors import StateError
from ._state import RUN_ID, RunState

__all__ = ["JsonFileStore", "MemoryStore", "StateStore"]


def _check_id(run_id: Any) -> str:
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise ValueError(f"a run id is 64 hexadecimal digits, got {run_id!r}")
    return run_id


class StateStore:
    """Saves and loads run states. Subclasses implement both methods."""

    def save(self, state: RunState) -> None:
        """Keep ``state`` as the latest state of its run."""
        raise NotImplementedError

    def load(self, run_id: str) -> RunState:
        """Return the latest state of ``run_id``; raise ``KeyError`` if none."""
        raise NotImplementedError

    def has(self, run_id: str) -> bool:
        """Whether a state of ``run_id`` is kept."""
        try:
            self.load(run_id)
        except KeyError:
            return False
        return True


def _checked(state: RunState, run_id: str, where: Any) -> RunState:
    if state.run_id != run_id:
        raise StateError(f"{where} holds run {state.run_id}, not {run_id}")
    return state


class MemoryStore(StateStore):
    """
    Keep states in memory, as their JSON text (so a load is a real round trip).

    Examples
    --------
    >>> from scikitplot.agent import MemoryStore
    >>> MemoryStore().run_ids()
    ()
    """

    def __init__(self) -> None:
        self._texts: dict[str, str] = {}

    def save(self, state: RunState) -> None:
        """Keep ``state`` as the latest state of its run."""
        self._texts[_check_id(state.run_id)] = canonical_json(state.to_dict())

    def load(self, run_id: str) -> RunState:
        """Return the latest state of ``run_id``."""
        state = RunState.from_dict(json.loads(self._texts[_check_id(run_id)]))
        return _checked(state, run_id, "the store")

    def has(self, run_id: str) -> bool:
        """Whether a state of ``run_id`` is kept."""
        return _check_id(run_id) in self._texts

    def run_ids(self) -> tuple[str, ...]:
        """Return the ids of the runs kept, sorted."""
        return tuple(sorted(self._texts))


def _sync_directory(directory: Path) -> None:
    """Make a rename in ``directory`` durable, where the platform allows it."""
    if os.name != "posix":
        return  # Windows has no directory handles to sync
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class JsonFileStore(StateStore):
    """
    Keep each run's state in ``<directory>/<run_id>.json``.

    Parameters
    ----------
    directory : str or os.PathLike
        Created if missing.

    Notes
    -----
    A save writes a temporary file in the same directory, syncs it, renames
    it over the old one (:func:`os.replace`) and syncs the directory, so a
    reader sees the old state or the new one, never a partial file, and the
    new one survives a power loss once ``save`` returns. The run id is checked to be 64 hexadecimal
    digits before it names a file, so it cannot name one outside
    ``directory``.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, run_id: str) -> Path:
        """Return the file that holds ``run_id``'s state."""
        return self.directory / f"{_check_id(run_id)}.json"

    def save(self, state: RunState) -> None:
        """Write ``state`` atomically."""
        target = self.path(state.run_id)
        text = canonical_json(state.to_dict())
        handle, temp = tempfile.mkstemp(
            dir=self.directory, prefix=f".{state.run_id}.", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as out:
                out.write(text)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temp, target)
            _sync_directory(self.directory)
        except BaseException:
            Path(temp).unlink(missing_ok=True)  # the old state stays as it was
            raise

    def load(self, run_id: str) -> RunState:
        """Read ``run_id``'s state."""
        path = self.path(run_id)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise KeyError(run_id) from None
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise StateError(f"{path}: not JSON: {exc}") from None
        return _checked(RunState.from_dict(data), run_id, path)

    def has(self, run_id: str) -> bool:
        """Whether a state file of ``run_id`` exists."""
        return self.path(run_id).is_file()

    def run_ids(self) -> tuple[str, ...]:
        """Return the ids of the runs kept, sorted."""
        return tuple(
            sorted(
                p.stem
                for p in self.directory.glob("*.json")
                if RUN_ID.fullmatch(p.stem)
            )
        )
