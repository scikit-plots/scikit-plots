# scikitplot/corpus/_embedding_manifest.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Provenance for a set of embeddings: :class:`EmbeddingManifest`.

An embedding vector on its own says nothing about what produced it.  This module
supplies the record that does, so vectors from different models cannot be mixed
without anyone noticing.

Notes
-----
**User-focused.**  A manifest describes one *embedding generation* -- one model,
one configuration, one set of vectors::

    manifest = EmbeddingManifest(
        provider="sentence_transformers",
        model="all-MiniLM-L6-v2",
        revision="c9745ed",
        dimension=384,
        dtype="float32",
        normalization="l2",
    )
    manifest.fingerprint  # stable, content-derived
    manifest.is_compatible(other)  # can these vectors share an index?

**Developer-focused.**  Review run R05 measured the defect this closes, and it
is the worst one the campaign found.  ``CorpusDocument`` had 54 fields of which
exactly one touched embeddings -- ``embedding: list[float] | None``, a bare
vector with no provenance.  Two consequences, both reproduced:

*Mixed dimensions* silently disabled dense search for the **entire corpus**::

    corpus = [384-dim doc, 768-dim doc, 384-dim doc]
    build() -> SUCCEEDED, has_embeddings=False, backend=None

*Same dimension, different models* was worse still::

    corpus = [2 docs in model-A space, 1 doc in model-B space]
    build()  -> SUCCEEDED, has_embeddings=True
    search() -> 3 ranked hits

Nothing detected it.  Not an exception, not a degradation, not a log line --
scored, ranked results computed across two incompatible vector spaces.  Every
other finding in the campaign concerned output that was *correct but
unreported*; this one produced output that was **incorrect and indistinguishable
from correct**.

The manifest **imports no embedding library**.  That is what makes it usable at
load time by code that no longer has -- or never had -- the model installed,
which is precisely the R05 exit gate: *"dense/vector indexing must be able to
validate embedding compatibility without importing the original embedding
model."*

See Also
--------
scikitplot.corpus._embeddings._embedding.EmbeddingEngine : produces manifests.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

__all__: list[str] = [
    "EmbeddingManifest",
    "IncompatibleEmbeddingsError",
]

#: Field separator for fingerprint derivation.  Length-prefixing every field
#: prevents ``("ab", "c")`` and ``("a", "bc")`` colliding -- the same discipline
#: the embedding cache key already uses (finding CORPUS-CACHE-001).
_FIELD_SEP = "\x1f"

#: Fingerprint scheme generation.  Bump when the preimage changes, so old and
#: new fingerprints cannot be mistaken for one another.
_FINGERPRINT_SCHEMA = "em1"


class IncompatibleEmbeddingsError(ValueError):
    """Raised when vectors from incompatible generations would be combined.

    Notes
    -----
    A subclass of :exc:`ValueError` so existing ``except ValueError`` handlers
    still catch it, while callers who want to react specifically to a
    provenance mismatch can.
    """


@dataclasses.dataclass(frozen=True)
class EmbeddingManifest:
    """Identity of one embedding generation.

    Parameters
    ----------
    provider : str
        Backend that produced the vectors, e.g. ``"sentence_transformers"``,
        ``"openai"``, ``"custom"``.
    model : str
        Model identifier as the provider names it.
    revision : str or None, optional
        Model revision/commit, when the provider exposes one.  ``None`` means
        *unpinned*, which is recorded rather than hidden: two unpinned
        generations of the same model are **not** assumed identical.
    dimension : int or None, optional
        Vector length.  ``None`` until the first vector is produced.
    dtype : str, optional
        Element type, e.g. ``"float32"``.
    normalization : str, optional
        ``"l2"``, ``"none"``, or a provider-specific scheme.
    preprocessing : str or None, optional
        Identifier for the text transformation applied before embedding.
    extra : dict, optional
        Additional JSON-compatible provenance.

    Notes
    -----
    **Developer.**  ``frozen=True`` because a manifest describes vectors that
    already exist.  A mutable manifest could drift away from the vectors it
    claims to describe, which is the failure it exists to prevent.

    Examples
    --------
    >>> a = EmbeddingManifest(provider="p", model="m", dimension=8)
    >>> b = EmbeddingManifest(provider="p", model="m", dimension=8)
    >>> a.fingerprint == b.fingerprint
    True
    >>> c = EmbeddingManifest(provider="p", model="other", dimension=8)
    >>> a.is_compatible(c)
    False
    """

    provider: str
    model: str
    revision: str | None = None
    dimension: int | None = None
    dtype: str = "float32"
    normalization: str = "none"
    preprocessing: str | None = None
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """Stable content-derived identifier for this generation.

        Returns
        -------
        str
            16 hex characters of a SHA-256 over every semantically relevant
            field, length-prefixed.

        Notes
        -----
        **Developer.**  This value is the ``embedding_model_id`` component of
        the content-derived *generation* identifier: a change here is exactly a
        change that should invalidate a built index.  Deriving both from one
        function means they cannot disagree.

        ``extra`` participates, sorted by key, so provider-specific provenance
        that changes the vectors also changes the fingerprint.
        """
        parts = [
            _FINGERPRINT_SCHEMA,
            self.provider,
            self.model,
            self.revision or "",
            "" if self.dimension is None else str(self.dimension),
            self.dtype,
            self.normalization,
            self.preprocessing or "",
        ]
        for key in sorted(self.extra):
            parts.extend((str(key), str(self.extra[key])))

        raw = _FIELD_SEP.join(f"{len(p)}:{p}" for p in parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def is_compatible(
        self,
        other: EmbeddingManifest,
    ) -> bool:
        """Whether vectors from ``self`` and ``other`` may share an index.

        Parameters
        ----------
        other : EmbeddingManifest
            EmbeddingManifest

        Returns
        -------
        bool

        Notes
        -----
        **Developer.**  Compatibility is fingerprint equality, deliberately.
        A looser rule -- "same dimension and model" -- would readmit the exact
        defect: two generations of one model with different normalization
        produce same-shaped vectors whose distances mean different things.
        Equality is the only rule that cannot be argued into unsoundness.
        """
        return self.fingerprint == other.fingerprint

    def require_compatible(self, other: EmbeddingManifest) -> None:
        """Raise unless ``other`` is compatible with ``self``.

        Raises
        ------
        IncompatibleEmbeddingsError
            Naming both generations and the fields that differ.
        """
        if self.is_compatible(other):
            return
        differing = sorted(
            field.name
            for field in dataclasses.fields(self)
            if getattr(self, field.name) != getattr(other, field.name)
        )
        raise IncompatibleEmbeddingsError(
            f"embeddings from generation {self.describe()} cannot be combined "
            f"with {other.describe()}; differing fields: {differing}. "
            "Vectors from different embedding generations occupy different "
            "spaces, so distances between them are meaningless."
        )

    def describe(self) -> str:
        """Return a compact human-readable identifier."""
        revision = f"@{self.revision}" if self.revision else ""
        dimension = f" dim={self.dimension}" if self.dimension is not None else ""
        return f"{self.provider}/{self.model}{revision}{dimension} [{self.fingerprint}]"

    def with_dimension(self, dimension: int) -> EmbeddingManifest:
        """Return a copy with ``dimension`` set.

        Notes
        -----
        **Developer.**  Dimension is often unknown until the first vector is
        produced.  Because it participates in the fingerprint, filling it in
        yields a *different* generation identity -- which is correct: a manifest
        that does not yet know its dimension has not described any vectors.
        """
        return dataclasses.replace(self, dimension=dimension)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping, including the fingerprint."""
        data = dataclasses.asdict(self)
        data["fingerprint"] = self.fingerprint
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingManifest:
        """Rebuild from :meth:`to_dict` output.

        Raises
        ------
        ValueError
            If the payload carries a fingerprint that does not match the fields
            it accompanies -- which means the payload was edited or corrupted.
        """
        known = {field.name for field in dataclasses.fields(cls)}
        manifest = cls(**{k: v for k, v in data.items() if k in known})

        declared = data.get("fingerprint")
        if declared is not None and declared != manifest.fingerprint:
            raise ValueError(
                f"EmbeddingManifest.from_dict: declared fingerprint {declared!r} "
                f"does not match the fields supplied (computed "
                f"{manifest.fingerprint!r}); the payload has been altered."
            )
        return manifest
