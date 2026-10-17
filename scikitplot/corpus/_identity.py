# scikitplot/corpus/_identity.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Document identity schemes: content-addressed SHA-256 or time-ordered UUIDv7.

Every durable identity in :mod:`scikitplot.corpus` is a full SHA-256 digest of
64 lowercase hex characters (:data:`IDENTITY_HEX_WIDTH`): ``doc_id``,
``content_hash``, the embedding-manifest fingerprint and the index generation.
This module adds the second, opt-in way to name a *document*: a UUID version 7
as defined by RFC 9562.

The two schemes answer different questions, so neither replaces the other.

==============  =================================  ===============================
scheme          ``doc_id`` is                      the same document built twice
==============  =================================  ===============================
``"sha256"``    a digest of source type, path,     gets the **same** id: rebuilds
(default)       chunk index and full content       upsert, qrels stay valid
``"uuid7"``     minted when the build assigns it:  gets a **new** id: rebuilds
                48-bit Unix time in ms, then a     append; deduplicate by
                monotonic counter and random bits  ``content_hash``, never by id
==============  =================================  ===============================

Notes
-----
**User.** Keep the default unless the store that receives the documents wants
time-ordered primary keys (index locality in a database, ids that sort by
creation, a millisecond timestamp for log correlation). With ``"uuid7"``:

- deduplicate on ``content_hash`` (still the content's SHA-256), never on
  ``doc_id``;
- evaluation judgments (qrels) and checkpoints keyed by ``doc_id`` hold only for
  the build that minted those ids;
- a ``doc_id`` carries the millisecond it was minted and so signals the order
  of creation (RFC 9562, section 8); do not use one where the creation time is
  confidential, and never as a secret or an access token ("UUIDs MUST NOT be
  used as security capabilities", section 8).

**Developer.** The generator follows RFC 9562 section 5.7 for the layout and
section 6.2, "Method 1", for monotonicity: the 12 bits of ``rand_a`` and the
first 30 bits of ``rand_b`` hold a 42-bit counter, seeded from the operating
system's CSPRNG (section 6.9) with its top bit clear at each new millisecond and
incremented within one; the last 32 bits are fresh random bits on every call.
The same layout is what CPython 3.14's :func:`uuid.uuid7` documents (a 48-bit
timestamp and a 42-bit counter). This module does not delegate to it, so that
Python 3.9 to 3.13 and 3.14+ produce identifiers by one implementation.

Ordering holds within a generator: every identifier is greater than the one
before, as an integer, as bytes and as canonical text. When the clock reads an
earlier millisecond than the last identifier (a clock step backwards), the last
timestamp is reused and the counter incremented (section 6.2, "Monotonic Error
Checking"). When the counter is exhausted within one millisecond, the timestamp
is advanced by one and the counter reseeded (section 6.2, "Counter Rollover
Handling"). Calls are serialised by a lock, and a forked child process starts a
fresh sequence, so parent and child never continue one counter.

Across processes and machines identifiers are unique by their random bits, not
ordered: RFC 9562 orders identifiers from one generator only.

References
----------
.. [1] RFC 9562, "Universally Unique IDentifiers (UUIDs)", sections 5.7, 6.2,
   6.9, 6.11 and 8. https://www.rfc-editor.org/rfc/rfc9562.html

Examples
--------
>>> from scikitplot.corpus import DocIdScheme, doc_id_scheme_of, uuid7
>>> u = uuid7()
>>> (u.version, doc_id_scheme_of(str(u)) is DocIdScheme.UUID7)
(7, True)
>>> doc_id_scheme_of("0" * 64) is DocIdScheme.SHA256
True
"""

from __future__ import annotations

import os
import re
import threading
import time
import uuid
from typing import Any, Callable, Iterable

from ._compat import StrEnum as _StrEnumBase

__all__: list[str] = [
    "IDENTITY_HEX_WIDTH",
    "DocIdScheme",
    "UUID7Generator",
    "assign_doc_ids",
    "doc_id_scheme_of",
    "uuid7",
    "uuid7_unix_ms",
]

#: Width, in lowercase hex characters, of every durable SHA-256 identity.
IDENTITY_HEX_WIDTH = 64

_HEX64 = re.compile(r"[0-9a-f]{64}")
# Canonical lowercase text form with version 7 and the RFC 9562 variant (0b10).
_UUID7_TEXT = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)

_TIMESTAMP_MAX = (1 << 48) - 1
_COUNTER_BITS = 42
_COUNTER_MAX = (1 << _COUNTER_BITS) - 1
# Seed: 41 random bits, top bit of the 42 clear, so that at least 2**41
# increments fit in one millisecond before a rollover (RFC 9562, 6.2).
_COUNTER_SEED_MASK = (1 << (_COUNTER_BITS - 1)) - 1
_TAIL_MASK = (1 << 32) - 1
_VERSION_7 = 0x7 << 76
_VARIANT_RFC = 0b10 << 62


class DocIdScheme(_StrEnumBase):
    """
    How a corpus build names its documents.

    Notes
    -----
    **SHA256** -- ``doc_id`` is the 64-hex SHA-256 of the document's source
    type, input path, chunk index and full content
    (:meth:`~scikitplot.corpus.CorpusDocument.make_doc_id`). Deterministic and
    idempotent: the default.

    **UUID7** -- ``doc_id`` is a freshly minted RFC 9562 UUID version 7 in its
    36-character canonical text form. Time-ordered within one build process;
    a rebuild mints new ids. See the module notes for what that changes.
    """

    SHA256 = "sha256"
    UUID7 = "uuid7"

    @classmethod
    def coerce(cls, value: DocIdScheme | str) -> DocIdScheme:
        """
        Return the member named by ``value``.

        Parameters
        ----------
        value : DocIdScheme or str
            A member, or its string value.

        Returns
        -------
        DocIdScheme
            The member.

        Raises
        ------
        ValueError
            If ``value`` names no member. The message lists the valid values.
        TypeError
            If ``value`` is neither a member nor a ``str``.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError(
                f"doc_id_scheme must be a DocIdScheme or str, got "
                f"{type(value).__name__}"
            )
        try:
            return cls(value)
        except ValueError:
            valid = ", ".join(repr(m.value) for m in cls)
            raise ValueError(
                f"unknown doc_id_scheme {value!r}; expected one of {valid}"
            ) from None


class UUID7Generator:
    """
    Monotonic generator of RFC 9562 UUID version 7 identifiers.

    Parameters
    ----------
    clock_ns : callable, optional
        Returns the current Unix time in nanoseconds. Default
        :func:`time.time_ns`. Injectable so that tests can step the clock.
    random_bytes : callable, optional
        ``random_bytes(n) -> bytes`` returning ``n`` unpredictable bytes.
        Default :func:`os.urandom`, the operating system's CSPRNG (RFC 9562,
        section 6.9).

    Notes
    -----
    **User.** Call the instance to get a :class:`uuid.UUID`. One process needs
    one generator: the module-level :func:`uuid7` is that instance.

    **Developer.** Thread-safe: the clock read, the counter step and the record
    of the result happen under one lock. Fork-safe for every instance, not only
    the module's: each call compares :func:`os.getpid` with the process that
    last used the instance and, in a forked child, calls :meth:`reset` before
    taking the lock. A lock held by another thread at the fork stays held in
    the child, and the child must not continue the parent's counter (RFC 9562,
    section 6.9: reseed on fork).

    Examples
    --------
    >>> ticks = iter([5_000_000, 5_000_000, 1_000_000])
    >>> make = UUID7Generator(clock_ns=lambda: next(ticks))
    >>> a, b, c = make(), make(), make()
    >>> a < b < c  # same millisecond, then a clock step backwards
    True
    >>> {uuid7_unix_ms(u) for u in (a, b, c)}
    {5}
    """

    def __init__(
        self,
        *,
        clock_ns: Callable[[], int] | None = None,
        random_bytes: Callable[[int], bytes] | None = None,
    ) -> None:
        self._clock_ns = time.time_ns if clock_ns is None else clock_ns
        self._random_bytes = os.urandom if random_bytes is None else random_bytes
        self._lock = threading.Lock()
        self._last_ms = -1
        self._last_counter = 0
        self._pid = os.getpid()

    def reset(self) -> None:
        """
        Forget the sequence, so the next identifier starts a new one.

        Notes
        -----
        Replaces the lock instead of acquiring it: called in a forked child,
        where another thread of the parent may have held it at the fork.
        """
        self._lock = threading.Lock()
        self._last_ms = -1
        self._last_counter = 0
        self._pid = os.getpid()

    def _random_int(self, n_bytes: int) -> int:
        """Return ``n_bytes`` random bytes as a non-negative integer."""
        raw = self._random_bytes(n_bytes)
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"random_bytes({n_bytes}) must return bytes, got {type(raw).__name__}"
            )
        data = bytes(raw)
        if len(data) != n_bytes:
            raise ValueError(
                f"random_bytes({n_bytes}) must return {n_bytes} bytes, got {len(data)}"
            )
        return int.from_bytes(data, "big")

    def _seed(self) -> tuple[int, int]:
        """Return a fresh ``(counter, tail)`` from 80 random bits."""
        rand = self._random_int(10)
        return (rand >> 32) & _COUNTER_SEED_MASK, rand & _TAIL_MASK

    def __call__(self) -> uuid.UUID:
        """
        Return the next identifier.

        Returns
        -------
        uuid.UUID
            Version 7, RFC 9562 variant, greater than every identifier this
            generator returned before.

        Raises
        ------
        OverflowError
            If the timestamp no longer fits 48 bits (the year 10889), or the
            clock reads before the Unix epoch.
        ValueError
            If ``random_bytes`` returns the wrong number of bytes.
        TypeError
            If ``random_bytes`` returns something other than bytes.
        """
        if os.getpid() != self._pid:  # a forked child: start a new sequence
            self.reset()
        with self._lock:
            now_ms = self._clock_ns() // 1_000_000
            if now_ms < 0:
                raise OverflowError(
                    f"UUIDv7 needs a time at or after the Unix epoch; the clock "
                    f"reads {now_ms} ms"
                )
            if now_ms > self._last_ms:
                timestamp = now_ms
                counter, tail = self._seed()
            else:
                # Same millisecond, or the clock stepped back: keep the last
                # timestamp and count on (RFC 9562, 6.2).
                timestamp = self._last_ms
                counter = self._last_counter + 1
                if counter > _COUNTER_MAX:
                    timestamp += 1
                    counter, tail = self._seed()
                else:
                    tail = self._random_int(4)
            if timestamp > _TIMESTAMP_MAX:
                raise OverflowError(
                    "UUIDv7 timestamp exceeds 48 bits (after the year 10889)"
                )
            value = (
                (timestamp << 80)
                | _VERSION_7
                | ((counter >> 30) << 64)
                | _VARIANT_RFC
                | ((counter & 0x3FFF_FFFF) << 32)
                | tail
            )
            # Record only after every step that can raise has succeeded.
            self._last_ms = timestamp
            self._last_counter = counter
        return uuid.UUID(int=value)


_DEFAULT_GENERATOR = UUID7Generator()


def uuid7() -> uuid.UUID:
    """
    Return a new RFC 9562 UUID version 7 from this process's generator.

    Returns
    -------
    uuid.UUID
        Time-ordered: greater than every identifier this function returned
        before in this process.

    Raises
    ------
    OverflowError
        If the system clock is outside the 48-bit millisecond range.

    See Also
    --------
    UUID7Generator : The generator, with an injectable clock and random source.
    uuid7_unix_ms : The millisecond timestamp an identifier carries.

    Examples
    --------
    >>> a, b = uuid7(), uuid7()
    >>> a < b and str(a) < str(b) and a.bytes < b.bytes
    True
    """
    return _DEFAULT_GENERATOR()


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    """Parse ``value`` into a :class:`uuid.UUID`, naming the value if it fails."""
    if isinstance(value, uuid.UUID):
        return value
    if not isinstance(value, str):
        raise TypeError(f"expected a UUID or str, got {type(value).__name__}")
    try:
        return uuid.UUID(value)
    except ValueError:
        raise ValueError(f"{value!r} is not a UUID") from None


def uuid7_unix_ms(value: uuid.UUID | str) -> int:
    """
    Return the Unix time in milliseconds that a UUID version 7 carries.

    Parameters
    ----------
    value : uuid.UUID or str
        A version-7 identifier, or its text form.

    Returns
    -------
    int
        The 48-bit ``unix_ts_ms`` field (RFC 9562, section 5.7).

    Raises
    ------
    ValueError
        If ``value`` is not a UUID, or not version 7 with the RFC 9562 variant.
    TypeError
        If ``value`` is neither a :class:`uuid.UUID` nor a ``str``.

    Examples
    --------
    The test vector of RFC 9562, appendix A.6 (2022-02-22 19:22:22 UTC):

    >>> uuid7_unix_ms("017f22e2-79b0-7cc3-98c4-dc0c0c07398f")
    1645557742000
    """
    parsed = _as_uuid(value)
    if parsed.variant != uuid.RFC_4122 or parsed.version != 7:  # noqa: PLR2004
        raise ValueError(
            f"{value!s} is not a UUID version 7 with the RFC 9562 variant "
            f"(version {parsed.version}, variant {parsed.variant!r})"
        )
    return parsed.int >> 80


def require_full_width(declared: object, owner: str) -> None:
    """
    Refuse a recorded SHA-256 identity that is not 64 hex characters long.

    Parameters
    ----------
    declared : object
        The identity a stored payload declares (a fingerprint).
    owner : str
        What declared it, for the message (``"EmbeddingManifest.from_dict"``).

    Raises
    ------
    ValueError
        If ``declared`` is not a ``str`` of exactly :data:`IDENTITY_HEX_WIDTH`
        lowercase hex characters. A 16-hex value gets a message saying it was
        written before identities were widened and must be rebuilt, so the
        caller is not told the payload was tampered with.

    Notes
    -----
    Internal to ``scikitplot.corpus``: every loader of a stored fingerprint
    calls it before comparing, so one width rule covers them all.
    """
    if isinstance(declared, str) and _HEX64.fullmatch(declared):
        return
    if isinstance(declared, str) and re.fullmatch(r"[0-9a-f]{1,63}", declared):
        raise ValueError(
            f"{owner}: the stored fingerprint {declared!r} has {len(declared)} "
            f"hex characters; this build records {IDENTITY_HEX_WIDTH}. It was "
            "written by a build before identities were widened to the full "
            "SHA-256 digest; rebuild the index (re-embed) with this build."
        )
    raise ValueError(
        f"{owner}: the stored fingerprint {declared!r} is not "
        f"{IDENTITY_HEX_WIDTH} lowercase hex characters"
    )


def doc_id_scheme_of(doc_id: object) -> DocIdScheme | None:
    """
    Return the scheme that produced ``doc_id``, judged by its exact form.

    Parameters
    ----------
    doc_id : object
        A document identifier.

    Returns
    -------
    DocIdScheme or None
        :attr:`DocIdScheme.SHA256` for exactly 64 lowercase hex characters,
        :attr:`DocIdScheme.UUID7` for the 36-character lowercase canonical form
        of a version-7 RFC 9562 UUID, ``None`` for anything else: a caller's
        own id, or a 16-hex id from a build before identities were widened.

    Notes
    -----
    The two forms cannot be confused: one has 64 characters and no hyphen, the
    other 36 with four. The test is purely syntactic; it cannot tell whether a
    64-hex string was produced by :meth:`CorpusDocument.make_doc_id`.

    Examples
    --------
    >>> doc_id_scheme_of("0" * 16) is None  # pre-widening width
    True
    """
    if not isinstance(doc_id, str):
        return None
    if _HEX64.fullmatch(doc_id):
        return DocIdScheme.SHA256
    if _UUID7_TEXT.fullmatch(doc_id):
        return DocIdScheme.UUID7
    return None


def assign_doc_ids(
    documents: Iterable[Any],
    scheme: DocIdScheme | str = DocIdScheme.SHA256,
    *,
    generator: Callable[[], uuid.UUID] | None = None,
) -> list[Any]:
    """
    Return ``documents`` named by ``scheme``, with parent links kept intact.

    Parameters
    ----------
    documents : iterable of CorpusDocument
        Documents in reading order. Each needs ``doc_id``, ``parent_doc_id``
        and ``replace(**changes)``.
    scheme : DocIdScheme or str, optional
        ``"sha256"`` (default) keeps the content-addressed ids the producers
        derived. ``"uuid7"`` mints one identifier per document, in order, so
        ``doc_id`` order is the order of ``documents``.
    generator : callable, optional
        Returns a :class:`uuid.UUID`; defaults to :func:`uuid7`. For a caller
        that keeps its own :class:`UUID7Generator`.

    Returns
    -------
    list of CorpusDocument
        New documents for ``"uuid7"``; the same objects for ``"sha256"``.

    Raises
    ------
    ValueError
        For an unknown ``scheme``. For ``"uuid7"``: if a ``parent_doc_id``
        names an id that two documents of the batch share, so its new id would
        be ambiguous; if ``generator`` returns something other than a version-7
        UUID, or an identifier not greater than the one before (a repeat or a
        step back would break uniqueness and order).

    Notes
    -----
    **User.** The builder and the pipeline call this once per source, right
    after it is read and before normalisation, enrichment and embedding, so no
    stage sees two names for one document.

    **Developer.** A parent link to a document of the batch follows it to its
    new id. A link to a document outside the batch is kept verbatim: it names
    a document stored elsewhere (a previous build, another source), which this
    call cannot rename, and :mod:`scikitplot.corpus._hierarchy` already treats
    such a parent as external. Nothing is minted until the whole batch has been
    read, so an iterable that fails part-way consumes no identifiers.

    Examples
    --------
    >>> from scikitplot.corpus import CorpusDocument
    >>> parent = CorpusDocument.create("a.txt", 0, "Part one.")
    >>> child = CorpusDocument.create("a.txt", 1, "Part one, section one.")
    >>> child = child.replace(parent_doc_id=parent.doc_id)
    >>> p, c = assign_doc_ids([parent, child], "uuid7")
    >>> (c.parent_doc_id == p.doc_id, p.doc_id < c.doc_id)
    (True, True)
    """
    resolved = DocIdScheme.coerce(scheme)
    rows = list(documents)
    if resolved is DocIdScheme.SHA256:
        return rows

    mint = uuid7 if generator is None else generator
    renamed: dict[str, str] = {}
    shared: set[str] = set()
    new_ids: list[str] = []
    previous: uuid.UUID | None = None
    for row in rows:
        minted = mint()
        if not isinstance(minted, uuid.UUID) or minted.version != 7:  # noqa: PLR2004
            raise ValueError(
                f"generator must return a version-7 uuid.UUID, got {minted!r}"
            )
        if previous is not None and minted <= previous:
            raise ValueError(
                f"generator returned {minted} after {previous}; identifiers "
                "must strictly increase, or ids repeat and lose their order"
            )
        previous = minted
        new_id = str(minted)
        old_id = row.doc_id
        if old_id in renamed:
            shared.add(old_id)
        renamed[old_id] = new_id
        new_ids.append(new_id)

    out: list[Any] = []
    for row, new_id in zip(rows, new_ids):
        parent = row.parent_doc_id
        changes: dict[str, Any] = {"doc_id": new_id}
        if parent in shared:
            raise ValueError(
                f"document {row.doc_id!r}: parent_doc_id {parent!r} is shared "
                "by more than one document of the batch, so its new id is "
                "ambiguous"
            )
        if parent and parent in renamed:
            changes["parent_doc_id"] = renamed[parent]
        out.append(row.replace(**changes))
    return out
