# scikitplot/corpus/tests/test__artifact.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`scikitplot.corpus._artifact` — the row-index-leakage gate.

The central test is :meth:`TestRowIndexLeakage.test_sidecar_survives_document_reordering`.
It reproduces the condition findings F-R01-07 and F-R06-04 describe: a persisted
index whose row space is frozen at write time, reloaded alongside documents in a
different order. Positional mapping returns the wrong document; the sidecar
returns the right one.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from .._artifact import SIDECAR_SCHEMA, ANNIndexArtifact, ArtifactError
from .._embedding_manifest import EmbeddingManifest, IncompatibleEmbeddingsError
from .._schema import CorpusDocument

__all__: "list[str]" = [
    "TestWriteAndOpen",
    "TestRowIndexLeakage",
    "TestValidation",
    "TestCorruption",
]


@pytest.fixture(name="manifest")
def _manifest() -> EmbeddingManifest:
    return EmbeddingManifest(provider="st", model="m", dimension=3)


@pytest.fixture(name="docs")
def _docs(manifest: EmbeddingManifest):
    return [
        CorpusDocument.create(
            input_path=f"f{i}.txt",
            chunk_index=i,
            text=f"text {i}",
            embedding=[float(i), 1.0, 0.0],
            embedding_manifest_id=manifest.fingerprint,
        )
        for i in range(3)
    ]


@pytest.fixture(name="artifact")
def _artifact(tmp_path: pathlib.Path, docs, manifest: EmbeddingManifest):
    return ANNIndexArtifact.write(
        tmp_path / "idx",
        documents=docs,
        backend="bruteforce",
        manifest=manifest,
        vectors=np.array([d.embedding for d in docs], dtype=np.float32),
    )


class TestWriteAndOpen:
    """Round-tripping an artifact."""

    def test_writes_manifest_sidecar_and_payload(self, artifact) -> None:
        names = {p.name for p in artifact.path.iterdir()}
        assert names == {"manifest.json", "sidecar.json", "vectors.npy"}

    def test_reopen_preserves_identity(self, artifact, manifest) -> None:
        reloaded = ANNIndexArtifact.open(artifact.path)
        assert reloaded.doc_ids == artifact.doc_ids
        assert reloaded.manifest == manifest
        assert reloaded.generation == artifact.generation
        assert reloaded.backend == "bruteforce"

    def test_vectors_round_trip(self, artifact, docs) -> None:
        loaded = ANNIndexArtifact.open(artifact.path).load_vectors()
        assert loaded.shape == (len(docs), 3)

    def test_write_replaces_an_existing_artifact(
        self, tmp_path: pathlib.Path, docs, manifest
    ) -> None:
        target = tmp_path / "idx"
        ANNIndexArtifact.write(
            target, documents=docs, backend="bruteforce", manifest=manifest
        )
        rewritten = ANNIndexArtifact.write(
            target, documents=docs[:2], backend="bruteforce", manifest=manifest
        )
        assert ANNIndexArtifact.open(target).row_count == 2 == rewritten.row_count


class TestRowIndexLeakage:
    """The finding this module exists for: F-R01-07 / F-R06-04."""

    def test_sidecar_survives_document_reordering(self, artifact, docs) -> None:
        """The exact condition a memory-mapped index creates.

        A persisted index's row space is frozen at write time, while the
        document list is rebuilt per process from whatever the caller passes.
        Positional mapping -- ``documents[ordinal]`` -- silently returns the
        wrong document. The sidecar records the correspondence instead of
        assuming it.
        """
        reloaded = ANNIndexArtifact.open(artifact.path)
        reordered = [docs[2], docs[0], docs[1]]

        # What positional mapping would have said, and what is actually true.
        assert reordered[1].doc_id != docs[1].doc_id
        assert reloaded.doc_id_for(1) == docs[1].doc_id

    def test_resolve_maps_a_whole_hit_list(self, artifact, docs) -> None:
        reloaded = ANNIndexArtifact.open(artifact.path)
        resolved = reloaded.resolve([(2, 0.9), (0, 0.5)])
        assert resolved == [(docs[2].doc_id, 0.9), (docs[0].doc_id, 0.5)]

    def test_out_of_range_ordinal_is_refused(self, artifact) -> None:
        """An ordinal past the sidecar means index and sidecar disagree."""
        with pytest.raises(IndexError, match="corrupt"):
            artifact.doc_id_for(99)

    def test_row_order_is_recorded_not_derived(self, artifact, docs) -> None:
        payload = json.loads((artifact.path / "sidecar.json").read_text())
        assert payload["doc_ids"] == [d.doc_id for d in docs]


class TestValidation:
    """Refusing to serve an artifact with inputs it does not describe."""

    def test_compatible_manifest_is_accepted(self, artifact, manifest) -> None:
        artifact.require_compatible(manifest)

    def test_different_embedding_generation_is_refused(self, artifact) -> None:
        other = EmbeddingManifest(provider="st", model="OTHER", dimension=3)
        with pytest.raises(IncompatibleEmbeddingsError):
            artifact.require_compatible(other)

    def test_document_set_is_checked_by_identity_not_order(
        self, artifact, docs
    ) -> None:
        """Order is what the sidecar records, so requiring it would be circular."""
        artifact.require_compatible(documents=[docs[2], docs[0], docs[1]])

    def test_different_document_set_is_refused(self, artifact, docs) -> None:
        with pytest.raises(ArtifactError, match="documents"):
            artifact.require_compatible(documents=docs[:2])

    def test_documents_without_ids_are_refused_at_write(
        self, tmp_path: pathlib.Path, manifest
    ) -> None:
        class Anonymous:
            doc_id = ""

        with pytest.raises(ArtifactError, match="doc_id"):
            ANNIndexArtifact.write(
                tmp_path / "idx",
                documents=[Anonymous()],
                backend="bruteforce",
                manifest=manifest,
            )


class TestCorruption:
    """A mapping that might mean something else is worse than no mapping."""

    def test_missing_sidecar_is_refused(self, artifact) -> None:
        (artifact.path / "sidecar.json").unlink()
        with pytest.raises(ArtifactError, match="sidecar.json"):
            ANNIndexArtifact.open(artifact.path)

    def test_missing_manifest_is_refused(self, artifact) -> None:
        (artifact.path / "manifest.json").unlink()
        with pytest.raises(ArtifactError, match="manifest.json"):
            ANNIndexArtifact.open(artifact.path)

    def test_unknown_sidecar_schema_is_refused(self, artifact) -> None:
        """Refused rather than interpreted, as with the document schema."""
        path = artifact.path / "sidecar.json"
        payload = json.loads(path.read_text())
        payload["sidecar_schema"] = "999"
        path.write_text(json.dumps(payload))
        with pytest.raises(ArtifactError, match="sidecar schema"):
            ANNIndexArtifact.open(artifact.path)

    def test_sidecar_disagreeing_with_generation_is_refused(self, artifact) -> None:
        """The generation records a count; the sidecar records the rows."""
        path = artifact.path / "sidecar.json"
        payload = json.loads(path.read_text())
        payload["doc_ids"] = payload["doc_ids"][:1]
        path.write_text(json.dumps(payload))
        with pytest.raises(ArtifactError, match="inconsistent"):
            ANNIndexArtifact.open(artifact.path)

    def test_current_schema_is_the_one_written(self, artifact) -> None:
        payload = json.loads((artifact.path / "manifest.json").read_text())
        assert payload["sidecar_schema"] == SIDECAR_SCHEMA
