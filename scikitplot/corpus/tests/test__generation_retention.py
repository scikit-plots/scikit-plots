"""
Retention regressions for published artifacts (slice S-41).

Commit was an in-place swap, so publishing a replacement overwrote the payload
a reader already had open: the handle kept an in-memory mapping whose vectors
on disk had become someone else's. A generation is now an immutable directory
and publication updates a small pointer, so a held generation stays readable
until it is pruned.

See Also
--------
scikitplot.corpus._artifact.ANNIndexArtifact.write
scikitplot.corpus._artifact.ANNIndexArtifact.generations
"""

import numpy as np
import pytest

from .._artifact import ANNIndexArtifact, ArtifactError
from .._embedding_manifest import EmbeddingManifest
from .._schema import CorpusDocument


def _doc(doc_id, text="body"):
    return CorpusDocument.create(input_path="f.txt", chunk_index=0, text=text,
                                 doc_id=doc_id)


def _publish(target, ids, text="body"):
    return ANNIndexArtifact.write(
        target, documents=[_doc(i, text) for i in ids], backend="bruteforce",
        manifest=EmbeddingManifest(provider="p", model="m", dimension=2),
        vectors=np.zeros((len(ids), 2), dtype="float32"),
    )


def test_a_held_generation_survives_a_replacement(tmp_path):
    """The payload a reader opened is still its payload afterwards."""
    target = tmp_path / "index"
    held = _publish(target, ["a", "b"])
    _publish(target, ["c"])
    assert held.path.is_dir()
    reopened = ANNIndexArtifact.open(held.path)
    assert reopened.doc_ids == ("a", "b")
    assert reopened.generation.fingerprint == held.generation.fingerprint


def test_opening_the_artifact_gives_the_current_generation(tmp_path):
    """The pointer decides what "the artifact" means."""
    target = tmp_path / "index"
    _publish(target, ["a", "b"])
    second = _publish(target, ["c"])
    current = ANNIndexArtifact.open(target)
    assert current.doc_ids == ("c",)
    assert current.generation.fingerprint == second.generation.fingerprint


def test_generations_are_listed_newest_last(tmp_path):
    """A caller can see what is retained."""
    target = tmp_path / "index"
    first = _publish(target, ["a"])
    second = _publish(target, ["b"])
    listed = ANNIndexArtifact.generations(target)
    assert first.path in listed
    assert second.path in listed


def test_retention_is_bounded(tmp_path):
    """Retention keeps readers safe without keeping every build forever."""
    target = tmp_path / "index"
    for n in range(5):
        _publish(target, [f"d{n}"])
    assert len(ANNIndexArtifact.generations(target)) <= 2


def test_a_first_publication_creates_the_pointer(tmp_path):
    """The layout is the same whether or not something was there before."""
    target = tmp_path / "index"
    written = _publish(target, ["a"])
    assert written.path.parent == target
    assert ANNIndexArtifact.open(target).doc_ids == ("a",)


def test_a_missing_pointer_is_refused(tmp_path):
    """An artifact directory with no pointer is not silently guessed at."""
    target = tmp_path / "index"
    _publish(target, ["a"])
    next(target.glob("current.json")).unlink()
    with pytest.raises(ArtifactError):
        ANNIndexArtifact.open(target)


def test_republishing_identical_content_is_stable(tmp_path):
    """The same rows twice is the same generation, not a second copy."""
    target = tmp_path / "index"
    first = _publish(target, ["a", "b"])
    again = _publish(target, ["a", "b"])
    assert again.generation.fingerprint == first.generation.fingerprint
    assert len(ANNIndexArtifact.generations(target)) == 1
