# scikitplot/corpus/tests/test__embedding_manifest.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`scikitplot.corpus._embedding_manifest` and the F-R05-01 gate.

The gate reproduces review run R05's two measured cases exactly, and additionally
asserts the R05 exit gate itself: compatibility must be checkable *without
importing the embedding model*.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from .._embedding_manifest import EmbeddingManifest, IncompatibleEmbeddingsError
from .._schema import CorpusDocument
from .._similarity._similarity import RetrievalIndex

__all__: "list[str]" = [
    "TestFingerprint",
    "TestCompatibility",
    "TestSerialisation",
    "TestMixedGenerationGate",
]


def _manifest(model: str = "model-A", **kwargs) -> EmbeddingManifest:
    defaults = {"provider": "st", "model": model, "dimension": 3}
    defaults.update(kwargs)
    return EmbeddingManifest(**defaults)


def _doc(index: int, text: str, embedding, manifest) -> CorpusDocument:
    return CorpusDocument.create(
        input_path=f"f{index}.txt",
        chunk_index=index,
        text=text,
        embedding=embedding,
        embedding_manifest_id=manifest.fingerprint if manifest else None,
    )


class TestFingerprint:
    """Content-derived identity."""

    def test_is_deterministic(self) -> None:
        assert _manifest().fingerprint == _manifest().fingerprint

    @pytest.mark.parametrize(
        "field",
        ["provider", "model", "revision", "dtype", "normalization", "preprocessing"],
    )
    def test_every_semantic_field_changes_it(self, field: str) -> None:
        base = _manifest()
        changed = _manifest(**{field: "different"})
        assert base.fingerprint != changed.fingerprint

    def test_dimension_changes_it(self) -> None:
        assert _manifest().fingerprint != _manifest(dimension=8).fingerprint

    def test_extra_participates(self) -> None:
        assert _manifest().fingerprint != _manifest(extra={"pooling": "mean"}).fingerprint

    def test_fields_are_length_prefixed(self) -> None:
        """``('ab','c')`` and ``('a','bc')`` must not collide.

        The same discipline the embedding cache key already uses.
        """
        a = EmbeddingManifest(provider="ab", model="c")
        b = EmbeddingManifest(provider="a", model="bc")
        assert a.fingerprint != b.fingerprint

    def test_with_dimension_yields_a_new_identity(self) -> None:
        """A manifest that does not know its dimension has described nothing."""
        unsized = EmbeddingManifest(provider="p", model="m")
        sized = unsized.with_dimension(384)
        assert sized.dimension == 384
        assert sized.fingerprint != unsized.fingerprint


class TestCompatibility:
    """Compatibility is fingerprint equality, deliberately."""

    def test_identical_manifests_are_compatible(self) -> None:
        assert _manifest().is_compatible(_manifest())

    def test_different_models_are_not(self) -> None:
        assert not _manifest("A").is_compatible(_manifest("B"))

    def test_same_shape_different_normalization_is_not(self) -> None:
        """The case a looser rule would wrongly admit.

        Two generations of one model with different normalization produce
        same-shaped vectors whose distances mean different things, so a
        "same dimension and model" rule would be unsound.
        """
        a = _manifest(normalization="l2")
        b = _manifest(normalization="none")
        assert a.dimension == b.dimension and a.model == b.model
        assert not a.is_compatible(b)

    def test_unpinned_revisions_are_not_assumed_identical(self) -> None:
        assert not _manifest(revision=None).is_compatible(_manifest(revision="abc"))

    def test_require_compatible_names_the_differing_fields(self) -> None:
        with pytest.raises(IncompatibleEmbeddingsError, match="model"):
            _manifest("A").require_compatible(_manifest("B"))


class TestSerialisation:
    """Round-tripping, and tamper detection."""

    def test_round_trip(self) -> None:
        original = _manifest(revision="c9745ed", extra={"pooling": "mean"})
        restored = EmbeddingManifest.from_dict(json.loads(json.dumps(original.to_dict())))
        assert restored == original
        assert restored.fingerprint == original.fingerprint

    def test_altered_payload_is_rejected(self) -> None:
        """A fingerprint that disagrees with its fields means the payload was edited."""
        payload = _manifest().to_dict()
        payload["model"] = "swapped"
        with pytest.raises(ValueError, match="does not match"):
            EmbeddingManifest.from_dict(payload)

    def test_fingerprint_may_be_omitted(self) -> None:
        payload = _manifest().to_dict()
        del payload["fingerprint"]
        assert EmbeddingManifest.from_dict(payload) == _manifest()


class TestMixedGenerationGate:
    """Regression gate for F-R05-01 — the campaign's only failed exit gate."""

    def test_mixed_dimensions_are_refused(self) -> None:
        """R05 case G1.

        Previously ``build()`` SUCCEEDED, ``vstack`` failed, and dense search was
        silently disabled for the *entire* corpus -- one mis-dimensioned document
        cost semantic search over everything.
        """
        manifest = _manifest()
        docs = [
            _doc(0, "a", [0.1] * 384, manifest),
            _doc(1, "b", [0.2] * 768, manifest),
            _doc(2, "c", [0.3] * 384, manifest),
        ]
        with pytest.raises(IncompatibleEmbeddingsError, match="differing dimensions"):
            RetrievalIndex().build(docs)

    def test_same_dimension_different_models_are_refused(self) -> None:
        """R05 case G2 -- the worst defect the campaign found.

        Previously ``build()`` succeeded, ``has_embeddings`` was ``True``, and
        ``search()`` returned three ranked hits computed across two incompatible
        vector spaces. Not an exception, not a degradation, not a log line:
        output that was incorrect and indistinguishable from correct.
        """
        model_a, model_b = _manifest("model-A"), _manifest("model-B")
        docs = [
            _doc(0, "cat", [1.0, 0.0, 0.0], model_a),
            _doc(1, "dog", [0.0, 1.0, 0.0], model_a),
            _doc(2, "gato", [0.0, 0.0, 1.0], model_b),
        ]
        with pytest.raises(IncompatibleEmbeddingsError, match="different generations"):
            RetrievalIndex().build(docs)

    def test_single_generation_builds_normally(self) -> None:
        manifest = _manifest()
        docs = [
            _doc(0, "cat", [1.0, 0.0, 0.0], manifest),
            _doc(1, "dog", [0.0, 1.0, 0.0], manifest),
        ]
        index = RetrievalIndex()
        index.build(docs)
        assert index.has_embeddings

    def test_untagged_corpora_still_build(self) -> None:
        """Documents predating manifests must not be rejected wholesale."""
        docs = [
            _doc(0, "cat", [1.0, 0.0, 0.0], None),
            _doc(1, "dog", [0.0, 1.0, 0.0], None),
        ]
        index = RetrievalIndex()
        index.build(docs)
        assert index.has_embeddings

    def test_mixing_tagged_and_untagged_is_refused(self) -> None:
        """Untagged vectors cannot be *shown* compatible with tagged ones."""
        docs = [
            _doc(0, "cat", [1.0, 0.0, 0.0], _manifest()),
            _doc(1, "dog", [0.0, 1.0, 0.0], None),
        ]
        with pytest.raises(IncompatibleEmbeddingsError, match="no manifest"):
            RetrievalIndex().build(docs)

    def test_exit_gate_no_embedding_library_is_imported(self) -> None:
        """The R05 exit gate itself.

        *"Dense/vector indexing must be able to validate embedding compatibility
        without importing the original embedding model."*  Checked in a fresh
        interpreter, because an in-process check cannot tell who imported what.
        """
        source = (
            "import sys, json\n"
            "from scikitplot.corpus import EmbeddingManifest\n"
            "a = EmbeddingManifest(provider='p', model='m', dimension=8)\n"
            "b = EmbeddingManifest.from_dict(json.loads(json.dumps(a.to_dict())))\n"
            "assert a.is_compatible(b)\n"
            "watched = ('torch', 'sentence_transformers', 'transformers', 'tensorflow')\n"
            "print(','.join(m for m in watched if m in sys.modules))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", source], capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, proc.stderr
        loaded = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        assert loaded == "", f"manifest validation imported {loaded}"
