# scikitplot/corpus/_embeddings/_cache_policy.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Embedding cache policy (R3-5).

When a cache hit is provable, what a cached call did, and how big the cache
may grow.

A disk-cache hit must be equivalent to recomputing. The key already names
the texts, the model, the backend, the provider and the output settings, but
an *unpinned* model (``model_revision=""``) names whatever the hub serves
today: after an upstream change, the old vectors were served as if they were
the new model's. :class:`CachePolicy` says what to do about that, and bounds
the cache, which used to grow without limit.

Notes
-----
**User.** The default policy computes, and does not cache, the embeddings of
an unpinned model. Pin ``model_revision`` (a commit hash makes the cache
exact), or choose:

* ``CachePolicy(unpinned="resolve", revision_resolver=resolve_hub_revision)``
  resolves the revision once per engine, then loads and keys exactly that
  revision;
* ``CachePolicy(unpinned="allow")`` caches under the unpinned name, as before
  (you accept that an entry can outlive an upstream change).

``max_entries`` / ``max_bytes`` bound the cache; the least recently used
entries are evicted first.

**Developer.** There is no time-to-live on purpose: staleness is a question
of identity (which revision produced the vectors), which the key answers
exactly, not of age, which can only guess.
"""

from __future__ import annotations

import enum
import hashlib
import logging
import os
import pathlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

__all__ = [
    "CachePolicy",
    "CacheStatus",
    "CachedEmbeddings",
    "resolve_hub_revision",
]

logger = logging.getLogger(__name__)

#: Accepted values of :attr:`CachePolicy.unpinned`.
UNPINNED_MODES: tuple[str, ...] = ("bypass", "allow", "resolve")

#: A cache entry's file name: a 64-hex key and ``.npy``. Eviction touches
#: nothing else in the directory.
_ENTRY_RE = re.compile(r"[0-9a-f]{64}\.npy")

#: A Hub commit (git SHA-1, or SHA-256). A branch or tag can move, a commit
#: cannot, so only a commit makes a sentence-transformers cache hit exact.
_COMMIT_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


def is_commit(revision: str) -> bool:
    """Return whether ``revision`` is a full commit hash (40 or 64 hex).

    Parameters
    ----------
    revision : str
        A revision string.

    Returns
    -------
    bool
        ``True`` for a lowercase 40- or 64-hex string.
    """
    return isinstance(revision, str) and _COMMIT_RE.fullmatch(revision) is not None


class CacheStatus(str, enum.Enum):
    """What one cached embedding call did.

    Attributes
    ----------
    HIT : str
        Served from the disk cache.
    STORED : str
        Computed and stored.
    STORE_FAILED : str
        Computed; writing the entry failed (logged).
    TOO_LARGE : str
        Computed; the entry alone exceeds ``max_bytes``, so it is not kept
        (no other entry is evicted for it).
    NOT_STORABLE : str
        Computed; the result cannot be one entry (a document without an
        embedding, or embeddings of different shapes or dtypes).
    DISABLED : str
        Computed; the cache is disabled (``enable_cache=False``).
    BYPASS_UNIDENTIFIED : str
        Computed, not cached: a custom provider without an identity.
    BYPASS_UNPINNED : str
        Computed, not cached: the model revision is not pinned or resolved.
    """

    HIT = "hit"
    STORED = "stored"
    STORE_FAILED = "store_failed"
    TOO_LARGE = "too_large"
    NOT_STORABLE = "not_storable"
    DISABLED = "disabled"
    BYPASS_UNIDENTIFIED = "bypass_unidentified"
    BYPASS_UNPINNED = "bypass_unpinned"


class CachedEmbeddings(NamedTuple):
    """The result of :meth:`EmbeddingEngine.embed_cached`.

    Attributes
    ----------
    embeddings : numpy.ndarray
        ``(n_texts, dim)`` in the engine's dtype.
    status : CacheStatus
        What the call did.
    key : str or None
        The 64-hex cache key, or ``None`` when the cache was not consulted.
    revision : str or None
        The revision the key names (pinned, resolved, or the branch or tag
        given with ``unpinned="allow"``); ``None`` when empty or when the
        cache was not consulted.
    """

    embeddings: Any
    status: CacheStatus
    key: str | None
    revision: str | None


@dataclass(frozen=True)
class CachePolicy:
    """How :class:`EmbeddingEngine` uses its disk cache.

    Parameters
    ----------
    unpinned : {"bypass", "allow", "resolve"}, default="bypass"
        What to do for a model not pinned to a commit (no ``model_revision``,
        or a sentence-transformers branch or tag, which can move).
        ``"bypass"`` computes and does not cache. ``"allow"`` caches under
        the name given (an entry can outlive an upstream change of the
        model). ``"resolve"`` (sentence-transformers, empty
        ``model_revision``) calls ``revision_resolver`` once per engine, then
        loads and keys that commit; when it fails, for another backend, or
        for a branch or tag given in ``model_revision`` (which is not
        resolved), the call bypasses the cache.
    revision_resolver : callable or None, default=None
        ``(model_name) -> commit hash``; required by, and only valid with,
        ``unpinned="resolve"``. :func:`resolve_hub_revision` is one. An
        answer that is not a 40- or 64-hex commit counts as a failure.
    max_entries : int or None, default=None
        Most entries kept in the cache directory; ``None`` is unbounded.
    max_bytes : int or None, default=None
        Most bytes kept in the cache directory; ``None`` is unbounded.

    Raises
    ------
    ValueError
        On an unknown ``unpinned``, a resolver without ``"resolve"`` (or
        ``"resolve"`` without one), or a bound that is not a positive int.

    See Also
    --------
    EmbeddingEngine.embed_cached : Returns the :class:`CacheStatus`.

    Notes
    -----
    **User.** A custom provider's identity is its ``custom_fn_id``; a
    sentence-transformers model is identified by a commit in
    ``model_revision``; another backend by any ``model_revision`` you state.
    Each makes the cache usable whatever ``unpinned`` says.

    **Developer.** Eviction is least recently used: a hit refreshes the
    entry's modification time, and after a store the oldest entries (ties by
    name) are removed until both bounds hold. Only ``<64 hex>.npy`` files are
    counted or removed.

    Examples
    --------
    >>> CachePolicy(max_entries=1000).unpinned
    'bypass'
    >>> CachePolicy(unpinned="resolve")
    Traceback (most recent call last):
    ...
    ValueError: CachePolicy: unpinned='resolve' requires a revision_resolver (for example resolve_hub_revision).
    """

    unpinned: str = "bypass"
    revision_resolver: Callable[[str], str] | None = None
    max_entries: int | None = None
    max_bytes: int | None = None

    def __post_init__(self) -> None:
        """Validate the fields."""
        if self.unpinned not in UNPINNED_MODES:
            raise ValueError(
                f"CachePolicy: unpinned must be one of {UNPINNED_MODES};"
                f" got {self.unpinned!r}."
            )
        if self.unpinned == "resolve" and self.revision_resolver is None:
            raise ValueError(
                "CachePolicy: unpinned='resolve' requires a revision_resolver"
                " (for example resolve_hub_revision)."
            )
        if self.revision_resolver is not None:
            if self.unpinned != "resolve":
                raise ValueError(
                    "CachePolicy: revision_resolver is used only with"
                    f" unpinned='resolve'; got unpinned={self.unpinned!r}."
                )
            if not callable(self.revision_resolver):
                raise ValueError(
                    "CachePolicy: revision_resolver must be callable;"
                    f" got {self.revision_resolver!r}."
                )
        for name in ("max_entries", "max_bytes"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(
                    f"CachePolicy: {name} must be a positive int or None;"
                    f" got {value!r}."
                )

    @property
    def bounded(self) -> bool:
        """Whether a bound is set (eviction runs after each store)."""
        return self.max_entries is not None or self.max_bytes is not None


def resolve_hub_revision(model_name: str, *, timeout: float | None = None) -> str:
    """Return the Hugging Face Hub commit a model id currently points to.

    Parameters
    ----------
    model_name : str
        A full Hub repository id, ``"owner/name"``.
    timeout : float or None, default=None
        Request timeout in seconds, passed to the Hub client.

    Returns
    -------
    str
        The commit hash (``huggingface_hub.model_info(model_name).sha``).

    Raises
    ------
    ValueError
        If ``model_name`` is not ``"owner/name"`` (a short name such as
        ``"all-MiniLM-L6-v2"`` is completed by sentence-transformers' own
        rules, which this function does not guess), is an existing local
        directory, or the Hub reports no commit.
    ImportError
        If ``huggingface_hub`` is not installed.

    Notes
    -----
    **User.** The engine loads the revision this returns, so the cache key,
    the manifest and the model agree. It needs the network once per engine.
    """
    if (
        not isinstance(model_name, str)
        or model_name.count("/") != 1
        or not all(model_name.split("/"))
    ):
        raise ValueError(
            "resolve_hub_revision: model_name must be a full Hub id"
            f" 'owner/name'; got {model_name!r}."
        )
    if os.path.isdir(model_name):
        # sentence-transformers loads an existing directory, not the Hub id.
        raise ValueError(
            f"resolve_hub_revision: {model_name!r} is a local directory, which"
            " is loaded as it is on disk; it has no Hub revision."
        )
    from huggingface_hub import model_info  # noqa: PLC0415

    sha = model_info(model_name, timeout=timeout).sha
    if not isinstance(sha, str) or not sha:
        raise ValueError(
            f"resolve_hub_revision: the Hub reported no commit for {model_name!r}."
        )
    return sha


# ---------------------------------------------------------------------------
# Keys and the store, shared by both embedding engines
# ---------------------------------------------------------------------------


def hash_fields(fields: list[bytes]) -> str:
    """Return the SHA-256 hex digest of length-prefixed ``fields``.

    Parameters
    ----------
    fields : list of bytes
        Field values, in order.

    Returns
    -------
    str
        64-character lowercase hex digest. Field boundaries are unambiguous
        (``[b"ab", b"c"]`` and ``[b"a", b"bc"]`` differ).
    """
    h = hashlib.sha256()
    for value in fields:
        h.update(str(len(value)).encode("ascii"))
        h.update(b"\x1f")
        h.update(value)
        h.update(b"\x1e")
    return h.hexdigest()


def touch_entry(path: pathlib.Path) -> None:
    """Mark ``path`` as just used (the LRU order), best effort.

    Parameters
    ----------
    path : pathlib.Path
        A cache entry that was just read.

    Notes
    -----
    **Developer.** A read-only cache directory still serves hits; only the
    eviction order is then stale, so a failure is logged at debug level.
    """
    try:
        os.utime(path)
    except OSError as exc:
        logger.debug("embedding cache: could not refresh %s: %s", path.name, exc)


def _scan_entries(
    cache_dir: pathlib.Path, stored_name: str
) -> tuple[list[tuple[int, str, int]], int | None]:
    """List the cache entries other than ``stored_name``, and its size.

    Parameters
    ----------
    cache_dir : pathlib.Path
        The cache directory.
    stored_name : str
        The file name of the entry just stored.

    Returns
    -------
    entries : list of (int, str, int)
        ``(mtime_ns, name, size)`` of every other ``<64 hex>.npy`` regular
        file (symbolic links are not entries).
    stored_size : int or None
        The stored entry's size; ``None`` when the scan did not see it.

    Raises
    ------
    OSError
        If the directory cannot be listed.
    """
    entries: list[tuple[int, str, int]] = []
    stored_size: int | None = None
    with os.scandir(cache_dir) as it:
        for entry in it:
            if not _ENTRY_RE.fullmatch(entry.name):
                continue
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                st = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if entry.name == stored_name:
                stored_size = st.st_size
            else:
                entries.append((st.st_mtime_ns, entry.name, st.st_size))
    return entries, stored_size


def enforce_bounds(
    cache_dir: pathlib.Path,
    policy: CachePolicy,
    *,
    stored: pathlib.Path,
) -> bool:
    """Evict least recently used entries until ``policy``'s bounds hold.

    Parameters
    ----------
    cache_dir : pathlib.Path
        The cache directory.
    policy : CachePolicy
        The bounds.
    stored : pathlib.Path
        The entry just stored. It is never evicted to make room; it is
        removed only when it alone exceeds ``max_bytes``.

    Returns
    -------
    bool
        ``False`` when ``stored`` alone exceeds ``max_bytes`` and was removed
        (no other entry is touched then); ``True`` otherwise.

    Notes
    -----
    **Developer.** Order: modification time, then name, so the result is
    deterministic. A file another process already removed no longer counts;
    a file that cannot be removed (in use on Windows) is skipped and still
    counted, so the bounds are met at a later store. A directory that cannot
    be listed leaves the cache as it is (logged at debug level): the entry is
    stored, and bounding it is best effort, never a reason to fail the call.
    """
    if not policy.bounded:
        return True
    try:
        entries, stored_size = _scan_entries(cache_dir, stored.name)
    except OSError as exc:
        logger.debug("embedding cache: could not list %s: %s", cache_dir, exc)
        return True
    if (
        policy.max_bytes is not None
        and stored_size is not None
        and stored_size > policy.max_bytes
    ):
        # One result larger than the quota: drop it, keep every other entry.
        try:
            stored.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.debug("embedding cache: could not remove %s: %s", stored.name, exc)
        return False
    entries.sort()
    count = len(entries) + (stored_size is not None)
    total = (stored_size or 0) + sum(size for _, _, size in entries)
    for _, name, size in entries:
        if (policy.max_entries is None or count <= policy.max_entries) and (
            policy.max_bytes is None or total <= policy.max_bytes
        ):
            break
        try:
            (cache_dir / name).unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.debug("embedding cache: could not evict %s: %s", name, exc)
            continue
        count -= 1
        total -= size
    return True
