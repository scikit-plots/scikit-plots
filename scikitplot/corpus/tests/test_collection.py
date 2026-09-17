"""
Collection regressions (slice S-10).

A collection is a *named, pinned* set of corpora. The design question this
settles is what it owns that a corpus does not: membership, and the generation
each member is pinned to. Binding a member to "whatever is current" would make
the same collection mean different things when read twice, which is the defect
the generation contract exists to prevent, one level up.

See Also
--------
scikitplot.corpus._collection.CorpusCollection
"""

import numpy as np
import pytest

from .._artifact import ANNIndexArtifact
from .._collection import CollectionError, CorpusCollection
from .._embedding_manifest import EmbeddingManifest
from .._schema import CorpusDocument


def _publish(target, ids):
    ANNIndexArtifact.write(
        target,
        documents=[CorpusDocument.create(input_path="f.txt", chunk_index=0,
                                         text=f"t{i}", doc_id=i) for i in ids],
        backend="bruteforce",
        manifest=EmbeddingManifest(provider="p", model="m", dimension=2),
        vectors=np.zeros((len(ids), 2), dtype="float32"),
    )
    return ANNIndexArtifact.open(target)


def test_a_member_is_pinned_to_the_generation_it_was_added_at(tmp_path):
    """A collection read twice means the same thing both times."""
    target = tmp_path / "docs"
    first = _publish(target, ["a", "b"])
    collection = CorpusCollection("kb").add("docs", target)
    _publish(target, ["c"])
    assert collection.member("docs").generation == first.generation.fingerprint


def test_a_pinned_member_still_opens_after_the_corpus_moves_on(tmp_path):
    """Retention makes the pin meaningful rather than merely recorded."""
    target = tmp_path / "docs"
    first = _publish(target, ["a", "b"])
    collection = CorpusCollection("kb").add("docs", target)
    _publish(target, ["c"])
    opened = collection.open("docs")
    assert opened.generation.fingerprint == first.generation.fingerprint
    assert opened.doc_ids == ("a", "b")


def test_duplicate_member_names_are_refused(tmp_path):
    """One name cannot mean two corpora."""
    a, b = tmp_path / "a", tmp_path / "b"
    _publish(a, ["x"])
    _publish(b, ["y"])
    collection = CorpusCollection("kb").add("docs", a)
    with pytest.raises(CollectionError):
        collection.add("docs", b)


def test_a_member_that_cannot_be_opened_is_refused_at_add_time(tmp_path):
    """The mistake is reported where it is made, not at first query."""
    with pytest.raises(CollectionError):
        CorpusCollection("kb").add("docs", tmp_path / "absent")


def test_identity_covers_membership_and_pins(tmp_path):
    """Two collections are the same collection only if they name the same builds."""
    a, b = tmp_path / "a", tmp_path / "b"
    _publish(a, ["x"])
    _publish(b, ["y"])
    one = CorpusCollection("kb").add("docs", a)
    same = CorpusCollection("kb").add("docs", a)
    other = CorpusCollection("kb").add("docs", b)
    renamed = CorpusCollection("kb").add("corpus", a)
    assert one.identity == same.identity
    assert one.identity != other.identity
    assert one.identity != renamed.identity


def test_identity_does_not_depend_on_insertion_order(tmp_path):
    """Membership is a set of pins, not the order someone added them."""
    a, b = tmp_path / "a", tmp_path / "b"
    _publish(a, ["x"])
    _publish(b, ["y"])
    forward = CorpusCollection("kb").add("one", a).add("two", b)
    reverse = CorpusCollection("kb").add("two", b).add("one", a)
    assert forward.identity == reverse.identity


def test_a_collection_is_immutable(tmp_path):
    """``add`` returns a new collection, so a held one cannot change underneath."""
    a = tmp_path / "a"
    _publish(a, ["x"])
    empty = CorpusCollection("kb")
    grown = empty.add("docs", a)
    assert empty.names == ()
    assert grown.names == ("docs",)
    assert empty.identity != grown.identity


def test_a_snapshot_round_trips(tmp_path):
    """A published collection reloads as the same pinned set."""
    a = tmp_path / "a"
    _publish(a, ["x"])
    collection = CorpusCollection("kb").add("docs", a)
    collection.save(tmp_path / "kb")
    reloaded = CorpusCollection.load(tmp_path / "kb")
    assert reloaded.identity == collection.identity
    assert reloaded.open("docs").doc_ids == ("x",)


def test_a_snapshot_is_a_generation_behind_a_pointer(tmp_path):
    """The same publication contract as every other artifact here."""
    a = tmp_path / "a"
    _publish(a, ["x"])
    CorpusCollection("kb").add("docs", a).save(tmp_path / "kb")
    assert (tmp_path / "kb" / "current.json").is_file()


def test_an_unknown_member_is_named(tmp_path):
    """A lookup failure says what was asked for and what exists."""
    with pytest.raises(CollectionError) as excinfo:
        CorpusCollection("kb").open("missing")
    assert "missing" in str(excinfo.value)
