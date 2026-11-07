"""
Build each stage of the release-security chain once per test process.

A Run N test needs the signed, verified state of the runs before it. Building
that state is most of the cost of a chain test, so the builder at each
module boundary (for example ``test_archive_native_status_evidence._preserved``
for the Run 160 state a Run 161 test starts from) is called through
:func:`shared`:

* the first call of a stage in a process builds it once, in a private
  directory of this process;
* every later call returns a *fresh deep copy* of the built value, so a test
  can change the dictionaries, lists and adapters it receives without
  affecting another test;
* the stage's files are fingerprinted when built and checked before every
  use and after every test (``conftest.py``), so a test that writes into a
  shared stage fails at once instead of leaking into the next test;
* a second call of the same stage *within one test* builds a new, independent
  chain in the caller's directory, exactly as before: a test that compares
  two chains still gets two.

A test that must change a stage's files takes :func:`private`, a copy of the
stage under its own ``tmp_path`` with every path in the value rewritten.

Notes
-----
**User notes.** Nothing a test asserts changes: the shared value is the one
the builder returns, built by the same code with the same arguments; only
*when* it is built changes. Outside pytest (no ``PYTEST_CURRENT_TEST``) every
call builds in the caller's directory, as the helpers always did.

**Developer notes.** Objects of the ``cryptography`` package (private keys,
public keys, certificates) are immutable from Python, so the deep copy keeps
the same objects instead of relying on how a given ``cryptography`` version
implements ``__deepcopy__``. The fingerprint covers every entry under the
stage directory: relative path, kind (file, directory, symlink), permission
bits, symlink target and file bytes. Stage directories are removed when the
process exits.
"""

from __future__ import annotations

import atexit
import copy
import hashlib
import os
import shutil
import stat
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")

__all__ = ["StageMutatedError", "private", "shared", "stage_roots", "verify_all"]


class StageMutatedError(AssertionError):
    """A shared stage's files changed after it was built."""


@dataclass
class _Stage:
    name: str
    root: Path
    value: Any
    fingerprint: str
    used_by: list[str] = field(default_factory=list)
    #: The stages this one was built from: its value points into them.
    depends_on: tuple[str, ...] = ()


_LOCK = threading.RLock()
_STAGES: dict[str, _Stage] = {}
_CALLS: dict[tuple[str, str], int] = {}
#: The process's stage directory, created on first use (one item, or none).
_BASE: list[Path] = []
#: The stages being built, innermost last; a shared() call made while one is
#: being built makes it depend on the stage it returns.
_BUILDING: list[list[str]] = []


def _base() -> Path:
    if not _BASE:
        base = Path(tempfile.mkdtemp(prefix="skplt-security-stages-")).resolve()
        atexit.register(shutil.rmtree, str(base), True)
        _BASE.append(base)
    return _BASE[0]


def _current_test() -> str | None:
    """Return the running test's node id, or None outside a pytest test."""
    current = os.environ.get("PYTEST_CURRENT_TEST")
    if not current:
        return None
    return current.rsplit(" ", 1)[0]


def fingerprint(root: Path) -> str:
    """
    Return the SHA-256 of everything under ``root``.

    Parameters
    ----------
    root : Path
        A directory.

    Returns
    -------
    str
        Hex digest over each entry's relative path, kind, permission bits,
        symlink target and file bytes, in sorted path order.
    """
    digest = hashlib.sha256()
    entries = [
        Path(directory) / name
        for directory, dirnames, filenames in os.walk(root, followlinks=False)
        for name in dirnames + filenames
    ]
    for path in sorted(entries, key=lambda p: p.relative_to(root).as_posix()):
        info = os.lstat(path)
        rel = path.relative_to(root).as_posix()
        if stat.S_ISLNK(info.st_mode):
            kind, payload = "l", os.readlink(path).encode()
        elif stat.S_ISDIR(info.st_mode):
            kind, payload = "d", b""
        else:
            kind, payload = "f", path.read_bytes()
        digest.update(f"{rel}\0{kind}\0{stat.S_IMODE(info.st_mode):o}\0".encode())
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def _immutable_memo(value: Any) -> dict[int, Any]:
    """Map every ``cryptography`` object reachable from ``value`` to itself."""
    memo: dict[int, Any] = {}
    seen = set()
    stack = [value]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if type(obj).__module__.startswith("cryptography."):
            memo[id(obj)] = obj
            continue
        if isinstance(obj, dict):
            stack.extend(obj.keys())
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        elif hasattr(obj, "__dict__") and not isinstance(obj, type):
            stack.extend(vars(obj).values())
    return memo


def _fresh(value: T) -> T:
    """Deep-copy ``value``, keeping ``cryptography`` objects shared."""
    return copy.deepcopy(value, _immutable_memo(value))


def _rebase(  # noqa: PLR0911, PLR0912 - one branch per kind of value
    value: Any, old: Path, new: Path, seen: dict[int, Any] | None = None
) -> Any:
    """Return ``value`` with every path under ``old`` moved under ``new``."""
    seen = {} if seen is None else seen
    if id(value) in seen:
        return seen[id(value)]
    if isinstance(value, Path):
        try:
            return new / value.relative_to(old)
        except ValueError:
            return value
    if isinstance(value, str):
        prefix = str(old)
        if value == prefix or value.startswith(prefix + os.sep):
            return str(new) + value[len(prefix) :]
        return value
    if type(value).__module__.startswith("cryptography."):
        return value
    if isinstance(value, dict):
        out: Any = {}
        seen[id(value)] = out
        for k, v in value.items():
            out[_rebase(k, old, new, seen)] = _rebase(v, old, new, seen)
        return out
    if isinstance(value, list):
        out = []
        seen[id(value)] = out
        out.extend(_rebase(v, old, new, seen) for v in value)
        return out
    if isinstance(value, tuple):
        items = [_rebase(v, old, new, seen) for v in value]
        if hasattr(value, "_fields"):  # a named tuple
            return type(value)(*items)
        return type(value)(items)
    if isinstance(value, (set, frozenset)):
        return type(value)(_rebase(v, old, new, seen) for v in value)
    if hasattr(value, "__dict__") and not isinstance(value, type):
        seen[id(value)] = value
        for k, v in list(vars(value).items()):
            setattr(value, k, _rebase(v, old, new, seen))
        return value
    return value


def _check(stage: _Stage, test: str | None = None) -> None:
    """Check ``stage`` and, first, every stage it was built from."""
    for name in stage.depends_on:
        _check(_STAGES[name], test)
    now = fingerprint(stage.root)
    if now != stage.fingerprint:
        users = ", ".join(stage.used_by[-5:]) or "nothing yet"
        after = f" after {test}" if test else ""
        raise StageMutatedError(
            f"shared stage {stage.name!r} ({stage.root}) changed{after}; "
            f"it was last used by: {users}. A test that writes into a stage "
            "must take _stages.private(...) instead of _stages.shared(...)."
        )


def verify_all(test: str | None = None) -> None:
    """
    Check that no shared stage has changed.

    Parameters
    ----------
    test : str or None, default=None
        The test that just ran: only the stages it used are checked, so it is
        the one named when a check fails. None checks every stage.

    Raises
    ------
    StageMutatedError
        If a stage's files differ from their fingerprint.
    """
    with _LOCK:
        for stage in _STAGES.values():
            if test is None or test in stage.used_by:
                _check(stage, test)


def _build(name: str, build: Callable[[Path], T], fallback: Path) -> _Stage:
    """Build ``name`` in its own directory, in the state ``fallback`` is in.

    A builder receives what its caller would have passed: an existing
    directory when ``fallback`` exists (a test's ``tmp_path``), a path to be
    created otherwise (``tmp_path / "run160"``), under the same final name.
    """
    root = _base() / name
    if root.exists():
        raise RuntimeError(f"stage directory {root} already exists")
    root.mkdir(parents=True)
    target = root / fallback.name
    if fallback.is_dir():
        target.mkdir()
    _BUILDING.append([])
    try:
        value = build(target)
    except BaseException:
        # No half-built stage is kept: the next call builds again and fails
        # for the same reason, not because the directory exists.
        shutil.rmtree(root, ignore_errors=True)
        raise
    finally:
        depends_on = tuple(dict.fromkeys(_BUILDING.pop()))
    return _Stage(name, root, value, fingerprint(root), depends_on=depends_on)


def shared(name: str, build: Callable[[Path], T], fallback: Path) -> T:
    """
    Return the stage ``name``, built once per process by ``build``.

    Parameters
    ----------
    name : str
        The stage's identity, for example ``"run160"``. One name, one builder.
    build : callable
        ``build(directory)`` builds the stage in ``directory`` and returns its
        value, as the predecessor helpers do.
    fallback : Path
        Where to build when the stage is not shared: outside pytest, or on a
        second call of ``name`` within one test (an independent chain).

    Returns
    -------
    object
        A fresh deep copy of the built value (shared case), or the value of
        ``build(fallback)``.

    Raises
    ------
    StageMutatedError
        If the stage's files changed since it was built.
    """
    test = _current_test()
    with _LOCK:
        if test is None:
            return build(fallback)
        calls = _CALLS.get((test, name), 0)
        _CALLS[(test, name)] = calls + 1
        if calls:
            return build(fallback)
        stage = _STAGES.get(name)
        if stage is None:
            stage = _STAGES[name] = _build(name, build, fallback)
        _check(stage)
        stage.used_by.append(test)
        if _BUILDING:
            _BUILDING[-1].append(name)
        # Building in ``fallback`` created its parent directories, and callers
        # write beside it (``tmp_path / "run161"`` holds both the Run 160 build
        # and the Run 161 files): keep that one side effect, nothing more.
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return _fresh(stage.value)


def private(name: str, build: Callable[[Path], T], directory: Path) -> T:
    """
    Return a copy of the stage ``name`` under ``directory``, safe to write.

    Parameters
    ----------
    name : str
        The stage, as passed to :func:`shared`.
    build : callable
        Its builder (used when the stage has not been built yet).
    directory : Path
        Where the copy goes; it must not exist yet.

    Returns
    -------
    object
        The stage's value with every path under the stage directory rewritten
        to the copy.
    """
    with _LOCK:
        stage = _STAGES.get(name)
        if stage is None:
            stage = _STAGES[name] = _build(name, build, Path(directory))
        _check(stage)
        shutil.copytree(stage.root, directory, symlinks=True)
        return _rebase(_fresh(stage.value), stage.root, Path(directory))


def stage_roots() -> tuple[Path, ...]:
    """Return the directories of the stages built so far (tuple of Path)."""
    with _LOCK:
        return tuple(stage.root for stage in _STAGES.values())
