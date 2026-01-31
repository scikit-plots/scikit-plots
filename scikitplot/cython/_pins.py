# scikitplot/cython/_pins.py
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
import re
from pathlib import Path

from ._cache import is_valid_key, peek_cache_dir, resolve_cache_dir
from ._lock import build_lock

_ALIAS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _pins_path(cache_root: Path) -> Path:
    return cache_root / "pins.json"


def _pins_lock_dir(cache_root: Path) -> Path:
    # A directory lock is portable and avoids partial write races.
    return cache_root / ".pins.lock"


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

    p = _pins_path(root)
    if not p.exists():
        return {}

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        # Strict: if pins.json is corrupted, return empty and let user repin.
        return {}

    out: dict[str, str] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if (
                isinstance(k, str)
                and isinstance(v, str)
                and _ALIAS_RE.fullmatch(k)
                and is_valid_key(v)
            ):
                out[k] = v
    return out


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
        current = list_pins(root)
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
        _pins_path(root).write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
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
        current = list_pins(root)
        if alias not in current:
            return False
        del current[alias]
        p = _pins_path(root)
        if current:
            p.write_text(
                json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
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
