"""
Canonical encoding: the one rule that turns a value into identity bytes.

Every durable identifier in this package — a plan fingerprint, an embedding
generation, a build generation, a document digest — is a hash of some value.
Those hashes were derived through ``repr``, which is a display format rather
than a serialisation, with two measured consequences: the same configuration
fingerprinted differently in a second interpreter, because a default ``repr``
embeds a memory address and a ``frozenset`` of strings reprs in hash order; and
equal mappings built in different orders fingerprinted differently within one
process.

This module is the single place that answers "what bytes represent this value",
so every identity in the package derives from one rule and they cannot drift
apart.

Design
------
The encoding is **type-tagged**: every value is written as its tag, then its
payload, so ``1``, ``True``, ``1.0`` and ``"1"`` cannot collide. It is
**length-prefixed**, so no concatenation of two values can be mistaken for a
third. It is **ordered**: mapping keys and set elements are sorted by their own
encoding, not by iteration order or by ``<``, which means a set of mixed types
still has one canonical order. And it is **closed**: a value the encoder cannot
represent deterministically is refused with the field that holds it, rather
than being encoded as something that happens to look stable.

Notes
-----
**User.** If :class:`CanonicalError` names a field in your configuration, that
value has no stable identity — typically a plain object, a lambda, or a
closure. Give the object a ``fingerprint`` attribute, make it a frozen
dataclass, or pass a name instead.

**Developer.** :data:`CANONICAL_VERSION` is written as the first bytes of every
encoding. Nothing reads an older version — this package keeps one format — but
the marker means a future change to this encoder produces visibly different
identities instead of silently colliding with the old ones. Changing the
encoding requires bumping it.

Only the standard library is imported, and the module lives inside the
submodule that needs it, so ``scikitplot.corpus`` derives its identities without
depending on any other submodule.

See Also
--------
scikitplot.corpus._validation : The row-index and count contract.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import decimal
import enum
import hashlib
import pathlib
import uuid
from typing import Any

__all__ = [
    "CANONICAL_VERSION",
    "CanonicalError",
    "canonical_bytes",
    "canonical_digest",
]

#: Version of this encoding, written as the first bytes of every result.
CANONICAL_VERSION = "c1"

_SEP = b"\x1f"


class CanonicalError(TypeError):
    """
    Raised when a value has no deterministic encoding.

    A subclass of :exc:`TypeError`, because an object that cannot be encoded is
    the wrong kind of value for an identity, not a bad quantity.
    """


def _tagged(tag: str, payload: bytes) -> bytes:
    """Return ``payload`` tagged with its type and its length."""
    raw = tag.encode("ascii")
    return raw + b":" + str(len(payload)).encode("ascii") + b":" + payload


def _encode(  # ruff: ignore[too-many-branches, too-many-return-statements]
    value: Any,
    path: str,
    seen: frozenset[int],
) -> bytes:
    """
    Encode one value, recursing with a cycle guard and a field path.

    Parameters
    ----------
    value : object
        Value to encode.
    path : str
        Dotted location within the document, used in refusal messages.
    seen : frozenset of int
        Identities of the containers currently being encoded. Passed by value
        so that a value repeated in two sibling branches is not mistaken for a
        cycle.

    Returns
    -------
    bytes
        The canonical encoding.

    Raises
    ------
    CanonicalError
        If the value has no deterministic encoding, naming ``path``.
    """
    where = path or "value"

    # An object that already names its own identity keeps it: this is the
    # existing package convention and the reason a plan can hold another plan.
    for attr in ("fingerprint", "plan_id"):
        named = getattr(value, attr, None)
        if isinstance(named, str):
            return _tagged("id", named.encode("utf-8"))

    if value is None:
        return _tagged("nil", b"")
    # Before the primitive branches: IntEnum subclasses int and StrEnum
    # subclasses str, so testing int first claimed them and collapsed
    # Colour.RED into 1. A member's identity is the member, not its value.
    if isinstance(value, enum.Enum):
        name = f"{type(value).__module__}.{type(value).__qualname__}.{value.name}"
        return _tagged("enum", name.encode("utf-8"))
    if isinstance(value, bool):
        return _tagged("bool", b"1" if value else b"0")
    if isinstance(value, int):
        return _tagged("int", str(value).encode("ascii"))
    if isinstance(value, float):
        # repr is round-trip exact for float and separates 0.0 from -0.0.
        return _tagged("float", repr(value).encode("ascii"))
    if isinstance(value, decimal.Decimal):
        return _tagged("dec", str(value).encode("ascii"))
    if isinstance(value, str):
        return _tagged("str", value.encode("utf-8"))
    if isinstance(value, (bytes, bytearray)):
        return _tagged("bytes", bytes(value))
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return _tagged("time", value.isoformat().encode("ascii"))
    if isinstance(value, uuid.UUID):
        return _tagged("uuid", str(value).encode("ascii"))
    if isinstance(value, pathlib.PurePath):
        return _tagged("path", value.as_posix().encode("utf-8"))
    if isinstance(value, type):
        name = f"{value.__module__}.{value.__qualname__}"
        return _tagged("type", name.encode("utf-8"))

    marker = id(value)
    if marker in seen:
        raise CanonicalError(
            f"{where} is part of a reference cycle, which has no canonical form."
        )
    nested = seen | {marker}

    if dataclasses.is_dataclass(value):
        fields = sorted(dataclasses.fields(value), key=lambda f: f.name)
        parts = [
            _tagged(
                "cls", f"{type(value).__module__}.{type(value).__qualname__}".encode()
            )
        ]
        for field in fields:
            parts.append(_tagged("key", field.name.encode("utf-8")))
            parts.append(
                _encode(
                    getattr(value, field.name),
                    f"{path}.{field.name}" if path else field.name,
                    nested,
                )
            )
        return _tagged("data", _SEP.join(parts))

    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            key_bytes = _encode(key, f"{where}<key>", nested)
            label = f"{path}.{key}" if path else str(key)
            items.append((key_bytes, _encode(item, label, nested)))
        items.sort(key=lambda pair: pair[0])
        return _tagged("map", _SEP.join(k + _SEP + v for k, v in items))

    if isinstance(value, (set, frozenset)):
        # Sorted by encoding, not by <: a set of mixed types still has one order.
        encoded = sorted(_encode(item, f"{where}<item>", nested) for item in value)
        tag = "set" if isinstance(value, set) else "fset"
        return _tagged(tag, _SEP.join(encoded))

    if isinstance(value, tuple):
        parts = [_encode(item, f"{where}[{i}]", nested) for i, item in enumerate(value)]
        return _tagged("tuple", _SEP.join(parts))

    if isinstance(value, list):
        parts = [_encode(item, f"{where}[{i}]", nested) for i, item in enumerate(value)]
        return _tagged("list", _SEP.join(parts))

    raise CanonicalError(
        f"{where} is a {type(value).__name__}, which has no canonical encoding; "
        "its identity would depend on its memory address and would differ "
        "between processes. Give it a `fingerprint` attribute, make it a frozen "
        "dataclass, or pass a name instead."
    )


def canonical_bytes(value: Any) -> bytes:
    """
    Return the canonical byte encoding of ``value``.

    Parameters
    ----------
    value : object
        Value to encode. Supported: ``None``, ``bool``, ``int``, ``float``,
        ``str``, ``bytes``, :class:`decimal.Decimal`, :class:`enum.Enum`, date
        and time objects, :class:`uuid.UUID`, :class:`pathlib.PurePath`,
        classes, dataclass instances, mappings, sets, tuples and lists, plus any
        object exposing a string ``fingerprint`` or ``plan_id``.

    Returns
    -------
    bytes
        An encoding prefixed with :data:`CANONICAL_VERSION`. Equal values encode
        identically in every process; distinct values encode differently.

    Raises
    ------
    CanonicalError
        If any part of the value has no deterministic encoding. The message
        names the dotted path of the offending field.

    Examples
    --------
    >>> canonical_bytes({"b": 2, "a": 1}) == canonical_bytes({"a": 1, "b": 2})
    True
    >>> canonical_bytes(1) == canonical_bytes(True)
    False
    """
    return CANONICAL_VERSION.encode("ascii") + _SEP + _encode(value, "", frozenset())


def canonical_digest(value: Any) -> str:
    """
    Return the full SHA-256 hex digest of ``value``'s canonical encoding.

    Parameters
    ----------
    value : object
        Value to identify.

    Returns
    -------
    str
        A 64-character lowercase hex digest. The full digest is what gets
        persisted; a prefix is for display only, because a truncated digest
        used as a durable identity narrows the collision bound for no benefit
        the storage layer needs.

    Raises
    ------
    CanonicalError
        Propagated from :func:`canonical_bytes`.

    Examples
    --------
    >>> len(canonical_digest({"a": 1}))
    64
    """
    return hashlib.sha256(canonical_bytes(value)).hexdigest()
