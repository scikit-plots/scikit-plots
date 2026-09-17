"""
Generation-identity regressions (slice S-5).

``_document_digest`` hashed ``sorted(set(doc_ids))``, so a generation recorded
which documents were indexed and nothing else. Two consequences: changing a
document's text while keeping a caller-supplied ``doc_id`` left the identity
unmoved, and a duplicate identifier shrank the recorded count below the number
of rows actually written. A generation now binds the content and the row order
that were actually published.

See Also
--------
scikitplot.corpus._generation.derive_generation
"""

import pytest

from .._generation import derive_generation
from .._schema import CorpusDocument


def _doc(doc_id, text="body"):
    return CorpusDocument.create(input_path="f.txt", chunk_index=0, text=text,
                                 doc_id=doc_id)


def _fp(documents):
    return derive_generation(documents, backend="bruteforce").fingerprint


def test_changed_content_under_a_supplied_id_moves_the_identity():
    """The defect C06 recorded: same id, different text, same generation."""
    assert _fp([_doc("stable", "A")]) != _fp([_doc("stable", "B")])


def test_row_order_is_part_of_the_identity():
    """The sidecar maps positions to ids, so order is what was published."""
    assert _fp([_doc("a"), _doc("b")]) != _fp([_doc("b"), _doc("a")])


def test_the_same_rows_in_the_same_order_are_the_same_generation():
    """Identity is stable for an identical publication."""
    assert _fp([_doc("a"), _doc("b")]) == _fp([_doc("a"), _doc("b")])


def test_the_recorded_count_is_the_number_of_rows():
    """A duplicate id no longer shrinks the count below the rows written."""
    generation = derive_generation([_doc("same", "A"), _doc("same", "B")],
                                   backend="bruteforce")
    assert generation.document_count == 2


def test_a_different_document_set_is_a_different_generation():
    """The original property still holds."""
    assert _fp([_doc("a")]) != _fp([_doc("a"), _doc("b")])


def test_backend_remains_part_of_the_identity():
    """Two backends over the same rows are different builds."""
    documents = [_doc("a")]
    assert (derive_generation(documents, backend="bruteforce").fingerprint
            != derive_generation(documents, backend="annoy").fingerprint)


def test_an_empty_publication_has_an_identity():
    """Zero rows is a real generation, not an error."""
    generation = derive_generation([], backend="bruteforce")
    assert generation.document_count == 0
    assert generation.fingerprint


def test_the_digest_states_the_schema_it_was_built_under():
    """A later change to this derivation must be detectable, not silent."""
    generation = derive_generation([_doc("a")], backend="bruteforce")
    assert generation.schema_version
