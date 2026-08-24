# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the document hierarchy (F-R08-02) and schema versioning (F-R01-04)."""

from __future__ import annotations

import dataclasses

import pytest

from .._hierarchy import (
    DEFAULT_MAX_DEPTH,
    ancestors,
    children,
    descendants,
    validate_hierarchy,
)
from .._schema import _SCHEMA_VERSION, CorpusDocument
from .._storage._storage import InMemoryStorage, StorageQuery

__all__: "list[str]" = [
    "TestHierarchyValidation",
    "TestHierarchyTraversal",
    "TestHierarchyIsQueryable",
    "TestSchemaVersion",
]


def _doc(index: int, parent: "str | None" = None) -> CorpusDocument:
    doc = CorpusDocument.create(
        input_path=f"f{index}.txt", chunk_index=index, text=f"text {index}"
    )
    return dataclasses.replace(doc, parent_doc_id=parent)


@pytest.fixture(name="tree")
def _tree():
    """root -> (c1 -> g1, c2)."""
    root = _doc(0)
    c1 = _doc(1, root.doc_id)
    c2 = _doc(2, root.doc_id)
    g1 = _doc(3, c1.doc_id)
    return [root, c1, c2, g1]


class TestHierarchyValidation:
    """Referential integrity, cycles and depth (F-R08-02)."""

    def test_valid_tree_passes(self, tree) -> None:
        report = validate_hierarchy(tree)
        assert report.ok
        assert report.checked == 4
        assert report.max_depth_seen == 2

    def test_documents_without_parents_are_valid(self) -> None:
        """The common case today: R13/U-11 found no producer sets parent_doc_id."""
        report = validate_hierarchy([_doc(0), _doc(1)])
        assert report.ok
        assert report.max_depth_seen == 0

    def test_dangling_parent_is_reported(self) -> None:
        report = validate_hierarchy([_doc(0, "does-not-exist")])
        assert not report.ok
        assert [e.code for e in report.errors] == ["HIERARCHY_DANGLING_PARENT"]
        assert report.errors[0].details["parent_doc_id"] == "does-not-exist"

    def test_dangling_parent_tolerated_for_partial_pages(self) -> None:
        """A page of a larger corpus may legitimately reference an absent parent."""
        report = validate_hierarchy(
            [_doc(0, "elsewhere")], require_parent_present=False
        )
        assert report.ok

    def test_self_parent_is_reported_distinctly(self) -> None:
        """Reported separately from a general cycle: it is usually a typo."""
        doc = _doc(0)
        report = validate_hierarchy([dataclasses.replace(doc, parent_doc_id=doc.doc_id)])
        assert [e.code for e in report.errors] == ["HIERARCHY_SELF_PARENT"]

    def test_cycle_is_reported(self) -> None:
        a, b = _doc(0), _doc(1)
        a_cyclic = dataclasses.replace(a, parent_doc_id=b.doc_id)
        b_cyclic = dataclasses.replace(b, parent_doc_id=a.doc_id)
        report = validate_hierarchy([a_cyclic, b_cyclic])
        assert not report.ok
        assert all(e.code == "HIERARCHY_CYCLE" for e in report.errors)

    def test_depth_bound_is_enforced(self) -> None:
        docs = [_doc(0)]
        for i in range(1, 12):
            docs.append(_doc(i, docs[-1].doc_id))
        assert validate_hierarchy(docs, max_depth=5).errors
        assert validate_hierarchy(docs, max_depth=DEFAULT_MAX_DEPTH).ok

    def test_all_violations_reported_not_just_the_first(self) -> None:
        """Structured records, so a caller can fix everything in one pass."""
        report = validate_hierarchy([_doc(0, "missing-a"), _doc(1, "missing-b")])
        assert len(report.errors) == 2

    def test_raise_if_invalid(self, tree) -> None:
        validate_hierarchy(tree).raise_if_invalid()
        with pytest.raises(ValueError, match="invalid document hierarchy"):
            validate_hierarchy([_doc(0, "missing")]).raise_if_invalid()

    def test_report_is_serialisable(self) -> None:
        import json

        json.dumps(validate_hierarchy([_doc(0, "missing")]).to_dict())


class TestHierarchyTraversal:
    """Walking the hierarchy — impossible before F-R08-02 was fixed."""

    def test_children_are_direct_only_and_ordered(self, tree) -> None:
        root = tree[0]
        kids = children(tree, root.doc_id)
        assert [d.chunk_index for d in kids] == [1, 2]

    def test_descendants_are_transitive(self, tree) -> None:
        root = tree[0]
        assert {d.chunk_index for d in descendants(tree, root.doc_id)} == {1, 2, 3}

    def test_ancestors_walk_to_the_root(self, tree) -> None:
        grandchild = tree[3]
        assert [d.chunk_index for d in ancestors(tree, grandchild.doc_id)] == [1, 0]

    def test_traversal_terminates_on_a_cyclic_hierarchy(self) -> None:
        """
        Traversal must be safe on data validation would reject.

        Callers do not always validate first, so a cycle must terminate rather
        than loop forever.
        """
        a, b = _doc(0), _doc(1)
        a_cyclic = dataclasses.replace(a, parent_doc_id=b.doc_id)
        b_cyclic = dataclasses.replace(b, parent_doc_id=a.doc_id)
        docs = [a_cyclic, b_cyclic]
        assert len(list(descendants(docs, a_cyclic.doc_id))) <= 2
        assert len(ancestors(docs, a_cyclic.doc_id)) <= 2


class TestHierarchyIsQueryable:
    """The StorageQuery half of F-R08-02."""

    def test_parent_doc_id_filter(self, tree) -> None:
        store = InMemoryStorage()
        store.save_batch(tree)
        result = store.query(StorageQuery(parent_doc_id=tree[0].doc_id))
        assert {d.chunk_index for d in result.documents} == {1, 2}
        assert "parent_doc_id" in result.filter_support


class TestSchemaVersion:
    """Serialisation generation (F-R01-04)."""

    def test_to_dict_declares_a_version(self) -> None:
        payload = _doc(0).to_dict()
        assert payload["schema_version"] == _SCHEMA_VERSION

    def test_round_trip(self) -> None:
        doc = _doc(0)
        assert CorpusDocument.from_dict(doc.to_dict()).doc_id == doc.doc_id

    def test_versionless_payload_is_refused_not_guessed(self) -> None:
        """
        DEC-119: refusing is recoverable; mis-reading is not.

        Assuming a version-less payload matches the current format would fill
        absent fields with defaults, producing a document whose ``doc_id``
        disagrees with what ``make_doc_id`` computes for its own content --
        and nothing downstream could detect that.
        """
        payload = _doc(0).to_dict()
        del payload["schema_version"]
        with pytest.raises(ValueError, match="schema_version"):
            CorpusDocument.from_dict(payload)

    def test_incompatible_major_is_refused(self) -> None:
        payload = _doc(0).to_dict()
        payload["schema_version"] = "99.0"
        with pytest.raises(ValueError, match="incompatible"):
            CorpusDocument.from_dict(payload)

    def test_newer_minor_is_accepted(self) -> None:
        """MINOR is reserved for additive fields an older reader can ignore."""
        payload = _doc(0).to_dict()
        major = _SCHEMA_VERSION.split(".", 1)[0]
        payload["schema_version"] = f"{major}.999"
        assert CorpusDocument.from_dict(payload).doc_id == payload["doc_id"]

    def test_malformed_version_is_refused(self) -> None:
        payload = _doc(0).to_dict()
        payload["schema_version"] = "not-a-version"
        with pytest.raises(ValueError, match="malformed"):
            CorpusDocument.from_dict(payload)
