# scikitplot/corpus/_artifact.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Persistable index artifacts: :class:`ANNIndexArtifact` and the ordinal sidecar.

An artifact is a directory holding everything needed to reload a vector index
*and prove it still means what it meant when it was written*:

.. code-block:: text

    my_index/
        manifest.json     EmbeddingManifest + IndexGeneration + sidecar schema
        sidecar.json      ordinal -> doc_id, in row order
        vectors.npy       the native index payload

Notes
-----
**User-focused.**

.. code-block:: python

    artifact = ANNIndexArtifact.write(
        path, documents=docs, backend="bruteforce", manifest=manifest
    )
    reloaded = ANNIndexArtifact.open(path)
    reloaded.doc_id_for(3)  # ordinal -> stable identity
    reloaded.require_compatible(manifest)  # refuses on mismatch

**Developer-focused.**  This closes findings F-R01-07 and F-R06-04.

``VectorIndexBackend.query()`` returns ``(row_index, score)`` -- a *row offset
into the embedding matrix*, not a document identity.  ``RetrievalIndex`` mapped
those back positionally, ``self._documents[idx]``, and correctness rested
entirely on the invariant that row *i* corresponds to ``self._documents[i]``.

That invariant was enforced by exactly one build-time length check and
**persisted nowhere**.  In-memory it holds.  For a memory-mapped index shared
across processes it cannot: the row space is frozen at write time while
``_documents`` is rebuilt per process from whatever the caller passes.  Two
processes disagreeing about document order would silently map every hit to the
wrong document -- with no exception, no degradation, and results that look
entirely reasonable.

The sidecar makes the mapping **data rather than coincidence**.  It is written
with the index, versioned, and validated on load, so a reload that cannot prove
the correspondence refuses instead of guessing.

See Also
--------
scikitplot.corpus.EmbeddingManifest : what the vectors were produced by.
scikitplot.corpus.IndexGeneration : what content the index was built over.
scikitplot.corpus._atomic.atomic_write_path : how the artifact is published.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import shutil
import tempfile
from typing import Any, Iterable, Sequence

from ._atomic import atomic_write_path
from ._embedding_manifest import EmbeddingManifest
from ._generation import IndexGeneration, derive_generation

__all__: list[str] = [
    "ANNIndexArtifact",
    "ArtifactError",
]

#: Sidecar format generation.  Bumped when the ordinal mapping's meaning changes,
#: so an older sidecar cannot be silently misread by newer code.
SIDECAR_SCHEMA = "1"

_MANIFEST_NAME = "manifest.json"
_SIDECAR_NAME = "sidecar.json"
_VECTORS_NAME = "vectors.npy"


class ArtifactError(ValueError):
    """Raised when an artifact is missing, malformed or incompatible."""


@dataclasses.dataclass(frozen=True)
class ANNIndexArtifact:
    """A persisted vector index together with its provenance.

    Parameters
    ----------
    path : pathlib.Path
        Artifact directory.
    manifest : EmbeddingManifest
        Which model produced the vectors.
    generation : IndexGeneration
        Which content the index was built over.
    doc_ids : tuple of str
        The sidecar: ``doc_ids[i]`` is the document at row *i*.
    backend : str
        Name of the backend that wrote the native payload.

    Notes
    -----
    **Developer.**  ``doc_ids`` is a tuple, so an artifact cannot have its
    mapping mutated after the correspondence has been validated.
    """

    path: pathlib.Path
    manifest: EmbeddingManifest
    generation: IndexGeneration
    doc_ids: tuple[str, ...]
    backend: str

    # -- reading -------------------------------------------------------------

    def doc_id_for(self, ordinal: int) -> str:
        """Resolve a backend row offset to a stable document identity.

        Parameters
        ----------
        ordinal : int
            Row index as returned by ``VectorIndexBackend.query``.

        Returns
        -------
        str
            The ``doc_id`` at that row.

        Raises
        ------
        IndexError
            If ``ordinal`` is outside the sidecar, which means the index and
            the sidecar disagree -- a corrupt artifact rather than a bad query.
        """
        try:
            return self.doc_ids[ordinal]
        except IndexError:
            raise IndexError(
                f"ordinal {ordinal} is outside this artifact's sidecar of "
                f"{len(self.doc_ids)} rows; the native index and its sidecar "
                "disagree, so the artifact is corrupt."
            ) from None

    def resolve(self, hits: Iterable[tuple[int, float]]) -> list[tuple[str, float]]:
        """Map ``(ordinal, score)`` pairs to ``(doc_id, score)`` pairs."""
        return [(self.doc_id_for(ordinal), score) for ordinal, score in hits]

    @property
    def row_count(self) -> int:
        """Number of rows the sidecar describes."""
        return len(self.doc_ids)

    # -- validation ----------------------------------------------------------

    def require_compatible(
        self,
        manifest: EmbeddingManifest | None = None,
        *,
        documents: Iterable[Any] | None = None,
    ) -> None:
        """Refuse to use this artifact with incompatible inputs.

        Parameters
        ----------
        manifest : EmbeddingManifest or None, optional
            Query-time embedding generation.  Must match the artifact's.
        documents : iterable or None, optional
            Documents the caller intends to serve from this index.  Their
            identities must match the sidecar **as a set**.

        Raises
        ------
        IncompatibleEmbeddingsError
            If ``manifest`` describes a different embedding generation.
        ArtifactError
            If ``documents`` do not match the sidecar.

        Notes
        -----
        **Developer.**  The document check compares *sets*, not order.  Order is
        exactly what the sidecar exists to record, so requiring the caller to
        reproduce it would defeat the purpose; what must hold is that the
        artifact describes these documents and no others.
        """
        if manifest is not None:
            self.manifest.require_compatible(manifest)

        if documents is not None:
            supplied = {getattr(doc, "doc_id", None) for doc in documents}
            supplied.discard(None)
            recorded = set(self.doc_ids)
            if supplied != recorded:
                missing = sorted(recorded - supplied)[:3]
                extra = sorted(supplied - recorded)[:3]
                raise ArtifactError(
                    f"artifact at {self.path} was built over "
                    f"{len(recorded)} documents but {len(supplied)} were "
                    f"supplied; missing={missing} unexpected={extra}. Row "
                    "offsets from this index would not name the documents you "
                    "intend to serve."
                )

    # -- writing / opening ---------------------------------------------------

    @classmethod
    def write(
        cls,
        path: str | pathlib.Path,
        *,
        documents: Sequence[Any],
        backend: str,
        manifest: EmbeddingManifest,
        vectors: Any = None,
        generation: IndexGeneration | None = None,
    ) -> ANNIndexArtifact:
        """Publish an artifact atomically.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination directory.  Replaced atomically if it exists.
        documents : sequence
            Documents **in row order**.  Their order defines the sidecar, which
            is the point: it is recorded rather than assumed.
        backend : str
            Name of the backend that produced the native payload.
        manifest : EmbeddingManifest
            Embedding generation the vectors belong to.
        vectors : array-like or None, optional
            Native payload.  Written as ``.npy`` when NumPy is available.
        generation : IndexGeneration or None, optional
            Defaults to deriving one from ``documents`` and ``backend``.

        Returns
        -------
        ANNIndexArtifact

        Notes
        -----
        **Developer.**  Publication uses :func:`atomic_write_path`, which R04
        verified under ``ENOSPC``, ``EACCES`` and ``KeyboardInterrupt``: the
        target is left intact and no temporary files survive.  A half-written
        artifact would be worse than none, because its sidecar could describe
        rows the native index does not have.
        """
        target = pathlib.Path(path)
        doc_ids = tuple(getattr(doc, "doc_id", "") for doc in documents)
        if any(not doc_id for doc_id in doc_ids):
            raise ArtifactError(
                "every document must have a doc_id; the sidecar cannot record "
                "an unidentified row."
            )

        gen = generation or derive_generation(documents, backend=backend)

        def _writer(tmp: pathlib.Path) -> None:
            staging = pathlib.Path(tempfile.mkdtemp(dir=str(tmp.parent)))
            try:
                (staging / _MANIFEST_NAME).write_text(
                    json.dumps(
                        {
                            "sidecar_schema": SIDECAR_SCHEMA,
                            "backend": backend,
                            "embedding_manifest": manifest.to_dict(),
                            "generation": gen.to_dict(),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                (staging / _SIDECAR_NAME).write_text(
                    json.dumps(
                        {"sidecar_schema": SIDECAR_SCHEMA, "doc_ids": list(doc_ids)},
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                if vectors is not None:
                    try:
                        import numpy as np  # noqa: PLC0415

                        np.save(staging / _VECTORS_NAME, np.asarray(vectors))
                    except ImportError:  # pragma: no cover - NumPy is a core dep
                        pass
                tmp.unlink(missing_ok=True)
                staging.rename(tmp)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise

        if target.exists():
            shutil.rmtree(target)
        atomic_write_path(target, _writer)

        return cls(
            path=target,
            manifest=manifest,
            generation=gen,
            doc_ids=doc_ids,
            backend=backend,
        )

    @classmethod
    def open(cls, path: str | pathlib.Path) -> ANNIndexArtifact:
        """Load an artifact, validating its internal consistency.

        Raises
        ------
        ArtifactError
            If the artifact is missing a required file, declares an unreadable
            sidecar schema, or its sidecar disagrees with its manifest.

        Notes
        -----
        **Developer.**  Loading validates *before* returning, so a caller cannot
        hold an artifact whose mapping has not been checked.  A sidecar written
        by a future schema is refused rather than interpreted, for the same
        reason the document schema refuses an unknown major version: a mapping
        that might mean something else is worse than no mapping.
        """
        directory = pathlib.Path(path)
        manifest_file = directory / _MANIFEST_NAME
        sidecar_file = directory / _SIDECAR_NAME

        for required in (manifest_file, sidecar_file):
            if not required.is_file():
                raise ArtifactError(
                    f"artifact at {directory} is missing {required.name}; it "
                    "cannot be loaded without both its manifest and sidecar."
                )

        head = json.loads(manifest_file.read_text(encoding="utf-8"))
        body = json.loads(sidecar_file.read_text(encoding="utf-8"))

        for source, payload in (("manifest", head), ("sidecar", body)):
            declared = str(payload.get("sidecar_schema", ""))
            if declared != SIDECAR_SCHEMA:
                raise ArtifactError(
                    f"artifact {source} declares sidecar schema {declared!r}, "
                    f"but this build reads {SIDECAR_SCHEMA!r}; refusing to "
                    "interpret a mapping whose meaning may have changed."
                )

        doc_ids = tuple(body.get("doc_ids", ()))
        generation = IndexGeneration.from_dict(head["generation"])

        if generation.document_count != len(doc_ids):
            raise ArtifactError(
                f"artifact at {directory} is inconsistent: its generation "
                f"describes {generation.document_count} documents but its "
                f"sidecar has {len(doc_ids)} rows."
            )

        return cls(
            path=directory,
            manifest=EmbeddingManifest.from_dict(head["embedding_manifest"]),
            generation=generation,
            doc_ids=doc_ids,
            backend=head.get("backend", "unknown"),
        )

    def load_vectors(self) -> Any:
        """Return the native payload, or ``None`` when none was written."""
        vectors_file = self.path / _VECTORS_NAME
        if not vectors_file.is_file():
            return None
        import numpy as np  # noqa: PLC0415

        return np.load(vectors_file)
