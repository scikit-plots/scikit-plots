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
import logging
import os
import pathlib
import shutil
import tempfile
import uuid
from typing import Any, Iterable, Sequence

from ._atomic import atomic_write_bytes, atomic_write_path
from ._embedding_manifest import EmbeddingManifest
from ._generation import IndexGeneration, derive_generation
from ._validation import require_index

__all__: list[str] = [
    "ANNIndexArtifact",
    "ArtifactError",
]

#: Sidecar format generation.  Bumped when the ordinal mapping's meaning changes,
#: so an older sidecar cannot be silently misread by newer code.
SIDECAR_SCHEMA = "1"

_MANIFEST_NAME = "manifest.json"
_POINTER_NAME = "current.json"
_POINTER_SCHEMA = "pointer1"
#: Generations retained behind the pointer: the current one, and the one it
#: replaced. A reader that opened a generation keeps reading it until it is
#: pruned; keeping every build instead would grow without bound.
_RETAINED_GENERATIONS = 2
_SIDECAR_NAME = "sidecar.json"
_VECTORS_NAME = "vectors.npy"


logger = logging.getLogger(__name__)


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
            If ``ordinal`` is negative or outside the sidecar. Out of range
            means the index and the sidecar disagree -- a corrupt artifact
            rather than a bad query.
        TypeError
            If ``ordinal`` is not an integer. ``bool`` is rejected explicitly:
            it is an ``int`` subclass, so ``True`` would otherwise resolve to
            row 1.

        Notes
        -----
        **Developer.** The domain comes from
        :func:`scikitplot._utils._indexing.require_index`, so this boundary and
        the lexical index answer the same question the same way. Direct tuple
        indexing used to accept ``-1`` and resolve it to the last row, which
        turned a caller's off-by-one into a plausible wrong document.
        """
        try:
            position = require_index(ordinal, len(self.doc_ids), name="ordinal")
        except IndexError as exc:
            raise IndexError(
                f"{exc}; the native index and its sidecar disagree, so the "
                "artifact is corrupt."
            ) from None
        return self.doc_ids[position]

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
            # The strict default of EmbeddingManifest.require_compatible refuses
            # two unpinned manifests, because equal names are not evidence that
            # the same weights produced both sets of vectors. Refusing here would
            # make every artifact built without a pinned revision unopenable,
            # which is a usability break rather than the safety C05 asked for.
            # So this caller opts in explicitly and says so: the artifact is
            # usable, and the caller is told the match is unverified rather than
            # being left to assume it was checked.
            unverified = self.manifest.revision is None and manifest.revision is None
            self.manifest.require_compatible(manifest, assume_unpinned_match=unverified)
            if unverified:
                logger.warning(
                    "embedding compatibility for %s is unverified: neither the "
                    "artifact's manifest nor the supplied one has a resolved "
                    "revision, so nothing records that the same weights produced "
                    "both. Pin the revision to make this a real check.",
                    self.manifest.describe(),
                )

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
        seen: set[str] = set()
        repeated = sorted({d for d in doc_ids if d in seen or seen.add(d)})
        if repeated:
            raise ArtifactError(
                f"doc_id(s) {repeated} appear on more than one row of "
                f"{len(doc_ids)} supplied; a sidecar cannot map one identity to "
                "several rows, and the generation would record fewer documents "
                "than the sidecar holds. Deduplicate before publishing."
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

        # A generation is an immutable directory and publication updates a
        # small pointer beside it. Replacing in place overwrote the payload a
        # reader already had open: the handle kept an in-memory mapping whose
        # vectors on disk had become a different build's. Writing a new
        # directory and moving the pointer leaves the held generation readable.
        root = target
        root.mkdir(parents=True, exist_ok=True)
        generation_dir = root / f"generation-{gen.fingerprint}"
        if not generation_dir.is_dir():
            candidate = root / f".candidate-{uuid.uuid4().hex}"
            try:
                atomic_write_path(candidate, _writer)
                # Verification is what makes the pointer move safe: pointing at
                # an unread directory would publish an assumption.
                cls.open_generation(candidate)
                os.replace(candidate, generation_dir)
            except BaseException:
                shutil.rmtree(candidate, ignore_errors=True)
                raise

        previous = cls._current_generation_name(root)
        atomic_write_bytes(
            root / _POINTER_NAME,
            json.dumps(
                {"pointer_schema": _POINTER_SCHEMA, "generation": generation_dir.name},
                indent=2,
            ).encode("utf-8"),
        )
        cls._prune_generations(root, keep={generation_dir.name, previous})

        return cls(
            path=generation_dir,
            manifest=manifest,
            generation=gen,
            doc_ids=doc_ids,
            backend=backend,
        )

    @classmethod
    def _current_generation_name(cls, root: pathlib.Path) -> str | None:
        """Return the generation the pointer names, or ``None`` if unset."""
        pointer = root / _POINTER_NAME
        if not pointer.is_file():
            return None
        try:
            return str(json.loads(pointer.read_text(encoding="utf-8"))["generation"])
        except (ValueError, KeyError):
            return None

    @classmethod
    def _prune_generations(cls, root: pathlib.Path, keep: set) -> None:
        """Remove retained generations other than ``keep``."""
        wanted = {name for name in keep if name}
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith(  # ruff: ignore[collapsible-if]
                "generation-",
            ):
                if child.name not in wanted:
                    shutil.rmtree(child, ignore_errors=True)

    @classmethod
    def generations(cls, path: str | pathlib.Path) -> list[pathlib.Path]:
        """Return the generation directories retained under ``path``.

        Parameters
        ----------
        path : str or pathlib.Path
            Artifact root.

        Returns
        -------
        list of pathlib.Path
            Retained generations, sorted by name. A reader holding one of these
            keeps reading it until it is pruned.
        """
        root = pathlib.Path(path)
        if not root.is_dir():
            return []
        return sorted(
            child
            for child in root.iterdir()
            if child.is_dir() and child.name.startswith("generation-")
        )

    @classmethod
    def open(cls, path: str | pathlib.Path) -> ANNIndexArtifact:
        """Open the generation the artifact's pointer currently names.

        Parameters
        ----------
        path : str or pathlib.Path
            Artifact root, or a generation directory, which is opened directly.

        Returns
        -------
        ANNIndexArtifact

        Raises
        ------
        ArtifactError
            If the root has no readable pointer, or the generation it names is
            absent. A directory with no pointer is refused rather than guessed
            at: picking a generation by name would make the choice silently.
        """
        root = pathlib.Path(path)
        if (root / _MANIFEST_NAME).is_file() or (root / _SIDECAR_NAME).is_file():
            # A generation directory, opened directly. This is what a held
            # handle keeps working with after the pointer has moved on. A
            # directory holding one member but not the other is a damaged
            # generation, not a root, so it is routed here to be refused with a
            # message about the member it is missing rather than about a
            # pointer it was never supposed to have.
            return cls.open_generation(root)
        name = cls._current_generation_name(root)
        if name is None:
            raise ArtifactError(
                f"artifact at {root} has no readable {_POINTER_NAME}; without it "
                "there is no way to know which generation is current."
            )
        generation_dir = root / name
        if not generation_dir.is_dir():
            raise ArtifactError(
                f"artifact at {root} points at generation {name}, which is not "
                "present; it may have been pruned while this pointer was stale."
            )
        return cls.open_generation(generation_dir)

    @classmethod
    def open_generation(cls, path: str | pathlib.Path) -> ANNIndexArtifact:
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
