"""
Publication regressions for artifact writes (slice S-4).

``write`` removed the destination before a replacement existed, so a failure at
any point afterwards left no artifact at all — the previous generation was gone
and the new one had never been written. Publication now builds a candidate,
verifies it by reopening it, and only then swaps, so a failure before the swap
leaves the previous generation exactly as it was.

See Also
--------
scikitplot.corpus._artifact.ANNIndexArtifact.write
"""

import errno
from unittest.mock import patch

import numpy as np
import pytest

from .. import _artifact as artifact_module
from .._artifact import ANNIndexArtifact, ArtifactError
from .._embedding_manifest import EmbeddingManifest
from .._schema import CorpusDocument


def _doc(doc_id, text="body"):
    return CorpusDocument.create(input_path="f.txt", chunk_index=0, text=text,
                                 doc_id=doc_id)


def _manifest():
    return EmbeddingManifest(provider="p", model="m", dimension=2)


def _publish(target, ids):
    return ANNIndexArtifact.write(
        target, documents=[_doc(i) for i in ids], backend="bruteforce",
        manifest=_manifest(), vectors=np.zeros((len(ids), 2), dtype="float32"),
    )


#: Points at which a publication can fail. ``commit_rename`` patches the rename
#: used both by the staging helper and by the swap, so it exercises the whole
#: commit path rather than one call.
INJECTION_POINTS = ["json_dumps", "vector_save", "commit_rename"]


def _patch_for(point):
    boom = OSError(errno.ENOSPC, "injected: no space left on device")
    if point == "json_dumps":
        return patch.object(artifact_module.json, "dumps", side_effect=boom)
    if point == "vector_save":
        return patch.object(artifact_module.np, "save", side_effect=boom)
    return patch.object(artifact_module.os, "replace", side_effect=boom)


@pytest.mark.parametrize("point", INJECTION_POINTS)
def test_a_failure_leaves_the_previous_generation_intact(tmp_path, point):
    """Whatever fails, what was published before is still published."""
    target = tmp_path / "index"
    before = _publish(target, ["a", "b"]).generation.fingerprint

    with pytest.raises(BaseException):  # noqa: B017 - any failure must preserve
        with _patch_for(point):
            _publish(target, ["c"])

    assert target.is_dir(), f"the previous generation was removed at {point}"
    reopened = ANNIndexArtifact.open(target)
    assert reopened.generation.fingerprint == before
    assert reopened.doc_ids == ("a", "b")


@pytest.mark.parametrize("point", INJECTION_POINTS)
def test_a_failure_leaves_no_candidate_behind(tmp_path, point):
    """A refused publication cleans up after itself."""
    target = tmp_path / "index"
    _publish(target, ["a", "b"])
    with pytest.raises(BaseException):  # noqa: B017
        with _patch_for(point):
            _publish(target, ["c"])
    strays = [p.name for p in tmp_path.iterdir() if p.name != "index"]
    assert strays == [], f"left behind at {point}: {strays}"


def test_a_first_publication_into_an_empty_directory_works(tmp_path):
    """Nothing to preserve is not a special case."""
    target = tmp_path / "index"
    written = _publish(target, ["a"])
    assert ANNIndexArtifact.open(target).generation.fingerprint == \
        written.generation.fingerprint


def test_a_failed_first_publication_leaves_nothing_usable(tmp_path):
    """
    A failure with no previous generation publishes nothing.

    Notes
    -----
    The artifact root is created before the candidate, so it may exist
    afterwards. What must not exist is a generation or a pointer: an empty
    directory is not a published artifact, and opening it is refused.
    """
    target = tmp_path / "index"
    with pytest.raises(BaseException):  # noqa: B017
        with _patch_for("json_dumps"):
            _publish(target, ["a"])
    assert ANNIndexArtifact.generations(target) == []
    assert not (target / "current.json").exists()
    with pytest.raises(ArtifactError):
        ANNIndexArtifact.open(target)


def test_a_successful_replacement_is_visible(tmp_path):
    """The ordinary path still replaces what was there."""
    target = tmp_path / "index"
    _publish(target, ["a", "b"])
    _publish(target, ["c"])
    reopened = ANNIndexArtifact.open(target)
    assert reopened.doc_ids == ("c",)
    assert [p.name for p in tmp_path.iterdir()] == ["index"]


def test_a_candidate_that_cannot_be_reopened_is_not_committed(tmp_path):
    """Verification is what makes the swap safe; without it the swap is a guess."""
    target = tmp_path / "index"
    before = _publish(target, ["a", "b"]).generation.fingerprint
    with patch.object(ANNIndexArtifact, "open_generation",
                      side_effect=ArtifactError("injected: candidate unreadable")):
        with pytest.raises(ArtifactError):
            _publish(target, ["c"])
    reopened = ANNIndexArtifact.open(target)
    assert reopened.generation.fingerprint == before
