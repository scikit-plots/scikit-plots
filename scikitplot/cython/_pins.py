# scikitplot/cython/_pins.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Pin/Alias registry for :mod:`scikitplot.cython`.

Pins map a human-friendly alias to an immutable cache key, enabling stable
re-imports across restarts:

- ``pin(key, alias="fast_fft")``
- ``import_pinned("fast_fft")``

Design goals:

- Per-cache-dir registry (pins are stored next to cache entries).
- Strict alias validation (Python identifier-like).
- Strict collision rules with opt-in overwrite.
- Lock-protected updates for concurrency safety.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from ._cache import is_valid_key, peek_cache_dir, resolve_cache_dir
from ._lock import build_lock

_ALIAS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

__all__ = [
    "PinRegistryError",
    "list_pins",
    "pin",
    "resolve_pinned_key",
    "unpin",
]


class PinRegistryError(ValueError):
    """Raised when the pin registry (``pins.json``) is corrupt or unreadable.

    Subclasses :class:`ValueError` so existing ``except ValueError`` handlers
    still catch it, while giving corruption an explicit, nameable type instead
    of silently degrading to an empty registry (CYTHON-PIN-001).
    """


def _pins_path(cache_root: Path) -> Path:
    return cache_root / "pins.json"


def _pins_lock_dir(cache_root: Path) -> Path:
    # A directory lock is portable and avoids partial write races.
    return cache_root / ".pins.lock"


def _atomic_write_json(path: Path, payload: dict[str, str]) -> None:
    """Write ``payload`` as JSON to ``path`` atomically (temp file + replace).

    A crash mid-write can never leave a truncated ``pins.json`` behind: the new
    content is fully written and fsync-ed to a sibling temp file, then
    ``os.replace``-d into place (CYTHON-PIN-001).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".pins-", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:  # ruff:ignore[suppressible-exception]
                tmp_path.unlink()
            except FileNotFoundError:
                pass


def _read_registry(root: Path) -> dict[str, str]:
    """Read and validate ``pins.json``, raising on corruption.

    Returns the alias→key mapping (invalid *individual* entries are filtered,
    preserving prior lenient behaviour for a well-formed dict).  A malformed
    JSON document or a non-object top level raises :class:`PinRegistryError`
    rather than silently returning an empty mapping — the latter would let a
    corrupt registry be overwritten (losing all pins) or let pinned entries be
    garbage-collected (CYTHON-PIN-001).
    """
    p = _pins_path(root)
    if not p.exists():
        return {}
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as e:
        raise PinRegistryError(f"cannot read pin registry {p}: {e}") from e
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise PinRegistryError(
            f"pin registry {p} is corrupt (invalid JSON); "
            f"back it up and remove it to reset pins"
        ) from e
    if not isinstance(data, dict):
        raise PinRegistryError(
            f"pin registry {p} is corrupt (expected a JSON object, "
            f"got {type(data).__name__})"
        )
    out: dict[str, str] = {}
    for k, v in data.items():
        if (
            isinstance(k, str)
            and isinstance(v, str)
            and _ALIAS_RE.fullmatch(k)
            and is_valid_key(v)
        ):
            out[k] = v
    return out


def _validate_alias(alias: str) -> None:
    if not isinstance(alias, str) or not alias or _ALIAS_RE.fullmatch(alias) is None:
        raise ValueError(
            "alias must be a non-empty identifier-like string: ^[A-Za-z_][A-Za-z0-9_]*$"
        )


def list_pins(cache_dir: str | Path | None = None) -> dict[str, str]:
    """
    List the current alias→key mappings.

    Parameters
    ----------
    cache_dir : str or pathlib.Path or None, default=None
        Cache root. If None, uses the default cache location.

    Returns
    -------
    dict[str, str]
        Mapping of alias to cache key. Returned mapping is a copy and can be mutated
        by the caller safely.
    """
    root = peek_cache_dir(cache_dir)
    if not root.exists():
        return {}

    # Raises PinRegistryError on a corrupt registry rather than silently
    # returning {} (CYTHON-PIN-001).
    return _read_registry(root)


def pin(
    key: str,
    *,
    alias: str,
    cache_dir: str | Path | None = None,
    overwrite: bool = False,
    lock_timeout_s: float = 60.0,
) -> str:
    """
    Pin a cache key under a human-friendly alias.

    Parameters
    ----------
    key : str
        Cache key (64 hex chars).
    alias : str
        Alias name (identifier-like).
    cache_dir : str or pathlib.Path or None, default=None
        Cache root. If None, uses the default cache location.
    overwrite : bool, default=False
        If False, collisions raise ValueError. If True, overwrite existing mapping.
    lock_timeout_s : float, default=60.0
        Max seconds to wait for the pin registry lock.

    Returns
    -------
    str
        The pinned key.

    Raises
    ------
    ValueError
        If alias/key are invalid or a collision occurs without overwrite.
    """
    _validate_alias(alias)
    if not is_valid_key(key):
        raise ValueError(f"Invalid cache key: {key!r}")

    root = resolve_cache_dir(cache_dir)
    lock_dir = _pins_lock_dir(root)
    lock_dir.parent.mkdir(parents=True, exist_ok=True)

    with build_lock(lock_dir, timeout_s=lock_timeout_s):
        current = _read_registry(root)
        if alias in current and current[alias] != key and not overwrite:
            raise ValueError(
                f"Alias collision: alias {alias!r} already points to a different key "
                f"({current[alias][:16]}...). Use overwrite=True to replace."
            )
        # strict one-to-one by default: prevent one key being pinned under multiple aliases
        if not overwrite:
            for a, k in current.items():
                if k == key and a != alias:
                    raise ValueError(
                        f"Key {key[:16]}... is already pinned as alias {a!r}. "
                        "Use overwrite=True to repin under a new alias."
                    )

        current[alias] = key
        _atomic_write_json(_pins_path(root), current)
    return key


def unpin(
    alias: str,
    *,
    cache_dir: str | Path | None = None,
    lock_timeout_s: float = 60.0,
) -> bool:
    """
    Remove an alias pin.

    Parameters
    ----------
    alias : str
        Alias to remove.
    cache_dir : str or pathlib.Path or None, default=None
        Cache root. If None, uses the default cache location.
    lock_timeout_s : float, default=60.0
        Max seconds to wait for the pin registry lock.

    Returns
    -------
    bool
        True if the alias existed and was removed, otherwise False.
    """
    _validate_alias(alias)
    root = peek_cache_dir(cache_dir)
    if not root.exists():
        return False

    lock_dir = _pins_lock_dir(root)
    with build_lock(lock_dir, timeout_s=lock_timeout_s):
        current = _read_registry(root)
        if alias not in current:
            return False
        del current[alias]
        p = _pins_path(root)
        if current:
            _atomic_write_json(p, current)
        else:
            # remove empty pins file
            try:  # noqa: SIM105
                p.unlink()
            except FileNotFoundError:
                pass
    return True


def resolve_pinned_key(alias: str, *, cache_dir: str | Path | None = None) -> str:
    """
    Resolve an alias to a cache key.

    Parameters
    ----------
    alias : str
        Alias name.
    cache_dir : str or pathlib.Path or None, default=None
        Cache root. If None, uses the default cache location.

    Returns
    -------
    str
        Cache key.

    Raises
    ------
    KeyError
        If alias is not pinned.
    ValueError
        If alias is invalid.
    """
    _validate_alias(alias)
    pins = list_pins(cache_dir)
    if alias not in pins:
        raise KeyError(f"Unknown pinned alias: {alias!r}")
    return pins[alias]
