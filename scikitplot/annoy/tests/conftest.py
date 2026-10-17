"""
Fixtures of the Annoy test suite.

A test writes its files to ``tmp_path`` and never into this directory
(Rule 30): every file under ``scikitplot/`` is shipped in the wheel, and
the vendored suite used to leave index files here.
"""

from __future__ import annotations

import contextlib
import os

import pytest

#: The directory the guard watches: the tests and their committed fixtures.
HERE = os.path.dirname(os.path.abspath(__file__))

#: Written by the interpreter when it imports a module, not by a test.
_INTERPRETER_CACHE = frozenset({"__pycache__"})


def snapshot(directory: str) -> dict[str, tuple[int, int] | None]:
    """
    Return each name in ``directory`` with its size and modification time.

    Parameters
    ----------
    directory : str
        The directory to list. Only its top level is read.

    Returns
    -------
    dict of str to (int, int) or None
        ``{name: (st_size, st_mtime_ns)}``, without the interpreter's
        ``__pycache__``. A Python source maps to ``None``: only its presence
        counts, so editing a test during a run is not reported as residue.
        A symbolic link is described by its own status.

    Notes
    -----
    The status comes from ``os.lstat`` on each path, not from the directory
    listing, which on Windows may show a stale size and time for a file that
    another handle holds open.
    """
    return {
        name: (
            None
            if name.endswith(".py")
            else _status(os.lstat(os.path.join(directory, name)))
        )
        for name in os.listdir(directory)
        if name not in _INTERPRETER_CACHE
    }


def _status(stat: os.stat_result) -> tuple[int, int]:
    return (stat.st_size, stat.st_mtime_ns)


def changes(directory: str, before: dict[str, tuple[int, int] | None]) -> list[str]:
    """
    Return the names in ``directory`` that are new or changed since ``before``.

    Parameters
    ----------
    directory : str
        The directory to compare.
    before : dict
        A :func:`snapshot` of ``directory``.

    Returns
    -------
    list of str
        Sorted; a removed name is not listed.
    """
    after = snapshot(directory)
    return sorted(
        name for name in after if name not in before or after[name] != before[name]
    )


def _refuse(directory: str, names: list[str], where: str) -> None:
    # Raised, not asserted: the guard holds under ``python -O``.
    raise AssertionError(
        f"wrote into {directory} {where}: {names}; write to tmp_path (Rule 30)"
    )


@contextlib.contextmanager
def writes_nothing_into(directory: str, reported: set[str] | None = None):
    """
    Fail when the managed block adds or changes a file in ``directory``.

    Parameters
    ----------
    directory : str
        The directory to watch. Only its top level is listed.
    reported : set of str or None, default=None
        When given, the names this guard reports are added to it, so that an
        outer guard does not report them a second time.

    Yields
    ------
    None

    Raises
    ------
    AssertionError
        On exit, naming every new or changed file (:func:`changes`).

    Notes
    -----
    **User notes.** A failure names the files; write them to ``tmp_path``
    instead.

    **Developer notes.** One listing before and one after, so the cost does
    not grow with the work inside. A file removed by the block is not
    reported: only additions and changes are residue. The check runs only
    when the block exits normally; an exception from the block propagates
    unchanged. A file left by an earlier run is in the first listing, so it
    is not reported again; a test that rewrites it is.
    """
    before = snapshot(directory)
    yield
    names = changes(directory, before)
    if names:
        if reported is not None:
            reported.update(names)
        _refuse(directory, names, "during this module")


#: The directory as it was when pytest imported this file, which it does
#: before it imports any test module here: a write at a module's import time
#: (the vendored suite used to download a fixture there) is after it.
_AT_IMPORT = snapshot(HERE)

#: Names a module guard already reported; the session guard skips them.
_REPORTED: set[str] = set()


@pytest.fixture(scope="session", autouse=True)
def _nothing_written_since_import():
    """
    Fail the session when the directory changed outside every module guard.

    Notes
    -----
    This catches a write at a module's import time, which no module guard
    sees. Its error is attached to the last test of the session, which may
    belong to another package; the message says where the write happened.
    """
    yield
    names = [name for name in changes(HERE, _AT_IMPORT) if name not in _REPORTED]
    if names:
        _refuse(HERE, names, "outside any module's tests (at import time)")


@pytest.fixture(scope="module", autouse=True)
def _writes_nothing_here():
    """
    Fail a module that adds or changes a file in the test directory.

    Notes
    -----
    The error is attached to the module's last test: the guard lists the
    directory once per module, not once per test, following the cost rule
    of ``scikitplot/conftest.py`` (no file access per test). Under parallel
    runs (``pytest-xdist``) a write is reported by whichever modules were
    running when it happened; it is a write into the package either way.
    """
    with writes_nothing_into(HERE, _REPORTED):
        yield
