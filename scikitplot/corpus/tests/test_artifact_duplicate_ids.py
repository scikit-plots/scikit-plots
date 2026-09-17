"""
Duplicate-identity regressions for artifact publication (slice S-2).

Two rows sharing one ``doc_id`` were previously accepted at write time and
refused at open time: the writer produced an artifact its own reader rejects,
and the recorded document count disagreed with the sidecar it was written
beside. These checks pin the refusal to the point where the mistake is still
cheap.

See Also
--------
scikitplot.corpus._artifact.ANNIndexArtifact.write
"""

import numpy as np
import pytest

from .._artifact import ANNIndexArtifact, ArtifactError
from .._embedding_manifest import EmbeddingManifest
from .._schema import CorpusDocument


def _document(doc_id, text="body"):
    """Return a document carrying an explicit, caller-supplied identity."""
    return CorpusDocument.create(
        input_path="f.txt", chunk_index=0, text=text, doc_id=doc_id
    )


def _manifest():
    """Return a minimal embedding manifest."""
    return EmbeddingManifest(provider="p", model="m", dimension=2)


def _vectors(rows):
    """Return a zero payload with ``rows`` rows."""
    return np.zeros((rows, 2), dtype="float32")


def test_duplicate_ids_are_refused_at_write(tmp_path):
    """Publication refuses the duplicate instead of producing an unreadable artifact."""
    target = tmp_path / "index"
    with pytest.raises(ArtifactError) as excinfo:
        ANNIndexArtifact.write(
            target,
            documents=[_document("same", "A"), _document("same", "B")],
            backend="bruteforce",
            manifest=_manifest(),
            vectors=_vectors(2),
        )
    assert "same" in str(excinfo.value)


def test_refusal_names_every_repeated_identity(tmp_path):
    """The message names the repeated identities, not just that one exists."""
    with pytest.raises(ArtifactError) as excinfo:
        ANNIndexArtifact.write(
            tmp_path / "index",
            documents=[_document("a"), _document("b"), _document("a"),
                       _document("b"), _document("c")],
            backend="bruteforce",
            manifest=_manifest(),
            vectors=_vectors(5),
        )
    message = str(excinfo.value)
    assert "'a'" in message and "'b'" in message
    assert "'c'" not in message


def test_nothing_is_published_when_the_write_is_refused(tmp_path):
    """A refused publication leaves no artifact and no staging directory behind."""
    target = tmp_path / "index"
    with pytest.raises(ArtifactError):
        ANNIndexArtifact.write(
            target,
            documents=[_document("same"), _document("same")],
            backend="bruteforce",
            manifest=_manifest(),
            vectors=_vectors(2),
        )
    assert not target.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_distinct_ids_still_publish_and_reopen(tmp_path):
    """The ordinary path is unchanged: distinct identities publish and reopen."""
    target = tmp_path / "index"
    written = ANNIndexArtifact.write(
        target,
        documents=[_document("a"), _document("b")],
        backend="bruteforce",
        manifest=_manifest(),
        vectors=_vectors(2),
    )
    reopened = ANNIndexArtifact.open(target)
    assert written.doc_ids == ("a", "b")
    assert reopened.doc_ids == ("a", "b")
    assert reopened.generation.document_count == 2


def test_writer_and_reader_agree(tmp_path):
    """Whatever the writer accepts, the reader accepts: no artifact refuses itself."""
    target = tmp_path / "index"
    ANNIndexArtifact.write(
        target,
        documents=[_document("a"), _document("b"), _document("c")],
        backend="bruteforce",
        manifest=_manifest(),
        vectors=_vectors(3),
    )
    artifact = ANNIndexArtifact.open(target)
    assert artifact.row_count == artifact.generation.document_count == 3
