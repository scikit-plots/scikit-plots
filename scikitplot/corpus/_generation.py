# scikitplot/corpus/_generation.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Content-derived identity for a built index: :class:`IndexGeneration`.

Notes
-----
**User-focused.**  Every search result carries the generation of the index that
produced it.  Comparing generations answers *"was this result computed against
the index I am holding now?"* -- and, unlike a counter, it keeps answering that
across process restarts and persisted indexes.

**Developer-focused.**  Findings F-R01-06 and F-R04-01 measured the previous
behaviour::

    process A, after 2 builds  -> index_generation = 2
    fresh instance, 1 build    -> index_generation = 1
    persisted anywhere         -> False

``_generation`` was a plain per-instance counter starting at ``0``.  Within one
process it correctly flagged a result computed against a since-rebuilt index --
genuinely useful.  Across a process boundary it carried no information at all:
generation ``2`` in one process and generation ``2`` in the next were
indistinguishable, and there was no serialisation path by which they could even
be compared.

R04 classified ``build()`` as the **only** ``NON_IDEMPOTENT`` operation in the
package, purely because of that counter side effect.  A content-derived
identifier removes that: building the same documents with the same
configuration twice yields the *same* generation, which is the correct answer.
Rebuild-detection becomes "does this index match this content?", which is the
question a caller actually has.

The four components are exactly those specified by proposal P-I1-02:

``schema_version``
    Serialisation generation of the documents.
``embedding_manifest_id``
    Which model produced the vectors.  Per decision DEC-34 this *is* the
    :class:`~scikitplot.corpus.EmbeddingManifest` fingerprint -- deriving both
    from one value means they cannot disagree.
``document_digest``
    Order-independent digest of the ``doc_id`` set.
``backend``
    Which index implementation was built.

See Also
--------
scikitplot.corpus.EmbeddingManifest : supplies ``embedding_manifest_id``.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Iterable

__all__: list[str] = [
    "IndexGeneration",
    "derive_generation",
]

_FIELD_SEP = "\x1f"
_GENERATION_SCHEMA = "gen1"


def _document_digest(doc_ids: Iterable[str]) -> str:
    """Return an order-independent digest of a ``doc_id`` set.

    Notes
    -----
    **Developer.**  Sorted before hashing, so the *same documents supplied in a
    different order* produce the same digest.  Document order is not part of an
    index's identity -- two builds over the same corpus are the same index --
    whereas a different *set* of documents genuinely is a different index.
    """
    ordered = sorted(set(doc_ids))
    joined = _FIELD_SEP.join(f"{len(d)}:{d}" for d in ordered)
    body = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
    return f"{len(ordered)}-{body}"


@dataclasses.dataclass(frozen=True)
class IndexGeneration:
    """Identity of one built index.

    Parameters
    ----------
    schema_version : str
        Document serialisation generation.
    embedding_manifest_id : str or None
        Fingerprint of the embedding generation, or ``None`` for an index with
        no vectors.
    document_digest : str
        Order-independent digest of the indexed ``doc_id`` set.
    backend : str or None
        Name of the index backend, or ``None`` when no dense index was built.

    Examples
    --------
    >>> a = IndexGeneration("2.0", None, "2-abc", "bruteforce")
    >>> b = IndexGeneration("2.0", None, "2-abc", "bruteforce")
    >>> a == b and a.fingerprint == b.fingerprint
    True
    """

    schema_version: str
    embedding_manifest_id: str | None
    document_digest: str
    backend: str | None

    @property
    def fingerprint(self) -> str:
        """Stable content-derived identifier, 16 hex characters."""
        parts = [
            _GENERATION_SCHEMA,
            self.schema_version,
            self.embedding_manifest_id or "",
            self.document_digest,
            self.backend or "",
        ]
        raw = _FIELD_SEP.join(f"{len(p)}:{p}" for p in parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @property
    def document_count(self) -> int:
        """Number of documents indexed, read from the digest prefix."""
        return int(self.document_digest.split("-", 1)[0])

    def matches(self, other: IndexGeneration | None) -> bool:
        """Whether ``other`` describes the same index."""
        return other is not None and self.fingerprint == other.fingerprint

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping including the fingerprint."""
        data = dataclasses.asdict(self)
        data["fingerprint"] = self.fingerprint
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexGeneration:
        """Rebuild from :meth:`to_dict` output.

        Raises
        ------
        ValueError
            If a declared fingerprint disagrees with the fields supplied, which
            means the payload was altered.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        generation = cls(**{k: v for k, v in data.items() if k in known})
        declared = data.get("fingerprint")
        if declared is not None and declared != generation.fingerprint:
            raise ValueError(
                f"IndexGeneration.from_dict: declared fingerprint {declared!r} "
                f"does not match the fields supplied (computed "
                f"{generation.fingerprint!r}); the payload has been altered."
            )
        return generation

    def __str__(self) -> str:
        """Return the fingerprint, so formatting a generation is unambiguous."""
        return self.fingerprint


def derive_generation(
    documents: Iterable[Any],
    *,
    backend: str | None = None,
    schema_version: str | None = None,
) -> IndexGeneration:
    """Derive the generation identity for a set of documents.

    Parameters
    ----------
    documents : Iterable[Any]
        iterable of CorpusDocument
    backend : str or None, optional
        Name of the built index backend.
    schema_version : str or None, optional
        Overrides the package's current serialisation version, for tests and for
        loading a persisted index written by another build.

    Returns
    -------
    IndexGeneration

    Raises
    ------
    ValueError
        If the documents span more than one embedding manifest.  That condition
        is already refused at build time; deriving a single generation for
        incompatible vectors would silently paper over it.

    Notes
    -----
    **Developer.**  ``embedding_manifest_id`` is taken from the documents rather
    than from an engine, so a *persisted* index can have its generation recomputed
    from what it contains -- which is the whole point of moving off a counter.
    """
    from ._schema import _SCHEMA_VERSION  # noqa: PLC0415

    docs = list(documents)
    manifests = {
        getattr(doc, "embedding_manifest_id", None)
        for doc in docs
        if getattr(doc, "embedding", None) is not None
    }
    manifests.discard(None)
    if len(manifests) > 1:
        raise ValueError(
            f"cannot derive a single index generation for {len(manifests)} "
            f"embedding manifests {sorted(manifests)}; these vectors do not "
            "belong in one index."
        )

    return IndexGeneration(
        schema_version=schema_version or _SCHEMA_VERSION,
        embedding_manifest_id=next(iter(manifests), None),
        document_digest=_document_digest(getattr(doc, "doc_id", "") for doc in docs),
        backend=backend,
    )
