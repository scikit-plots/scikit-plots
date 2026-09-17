"""
Shared row-index contract at the artifact boundary (slice S-14).

``doc_id_for`` indexed a Python tuple directly, so it inherited two behaviours a
row identifier must not have: ``-1`` resolved to the last row, and ``True``
resolved to row 1. Both turn a caller's off-by-one or type error into a
plausible wrong document rather than an error. The rule now comes from
:mod:`scikitplot._utils._indexing`, so it is the same rule the lexical index
applies.

See Also
--------
scikitplot.corpus._validation.require_index
"""

import numpy as np
import pytest

from .._validation import require_index
from .._artifact import ANNIndexArtifact
from .._embedding_manifest import EmbeddingManifest
from .._schema import CorpusDocument


@pytest.fixture
def artifact(tmp_path):
    """Return a published three-row artifact."""
    documents = [
        CorpusDocument.create(input_path="f.txt", chunk_index=i, text=f"t{i}",
                              doc_id=f"d{i}")
        for i in range(3)
    ]
    ANNIndexArtifact.write(
        tmp_path / "index", documents=documents, backend="bruteforce",
        manifest=EmbeddingManifest(provider="p", model="m", dimension=2),
        vectors=np.zeros((3, 2), dtype="float32"),
    )
    return ANNIndexArtifact.open(tmp_path / "index")


@pytest.mark.parametrize("ordinal", [0, 1, 2])
def test_valid_ordinals_resolve(artifact, ordinal):
    """The ordinary path is unchanged."""
    assert artifact.doc_id_for(ordinal) == f"d{ordinal}"


@pytest.mark.parametrize("ordinal", [-1, -3])
def test_negative_ordinals_are_refused(artifact, ordinal):
    """A negative ordinal is a caller mistake, not the last row."""
    with pytest.raises(IndexError):
        artifact.doc_id_for(ordinal)


@pytest.mark.parametrize("ordinal", [True, False])
def test_boolean_ordinals_are_refused(artifact, ordinal):
    """``True`` is not row 1."""
    with pytest.raises(TypeError):
        artifact.doc_id_for(ordinal)


@pytest.mark.parametrize("ordinal", [3, 99])
def test_out_of_range_ordinals_are_refused(artifact, ordinal):
    """Beyond the sidecar is still an IndexError, as it always was."""
    with pytest.raises(IndexError):
        artifact.doc_id_for(ordinal)


def test_numpy_integers_are_accepted(artifact):
    """Backends return NumPy integers; those are converted, then bounds-checked."""
    assert artifact.doc_id_for(np.int64(2)) == "d2"
    with pytest.raises(IndexError):
        artifact.doc_id_for(np.int64(3))


def test_float_ordinals_are_refused(artifact):
    """A float is not a row."""
    with pytest.raises(TypeError):
        artifact.doc_id_for(1.0)


def test_the_rule_is_the_shared_one(artifact):
    """The artifact answers exactly what the shared primitive answers."""
    for candidate in (-1, 0, 2, 3, True, 1.0):
        try:
            expected = require_index(candidate, artifact.row_count)
        except Exception as exc:  # noqa: BLE001
            with pytest.raises(type(exc)):
                artifact.doc_id_for(candidate)
        else:
            assert artifact.doc_id_for(candidate) == artifact.doc_ids[expected]
