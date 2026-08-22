# scikitplot/corpus/_hierarchy.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Validation and traversal for the ``parent_doc_id`` document hierarchy.

:class:`~scikitplot.corpus.CorpusDocument` carries ``parent_doc_id``, a
self-referential reference to another document's ``doc_id``.  This module makes
that hierarchy *usable*: it can be validated, walked and queried.

Notes
-----
**User-focused.**  Given a collection of documents::

    report = validate_hierarchy(docs)
    if not report.ok:
        for record in report.errors:
            print(record)

    children_of_x = children(docs, "abc123")
    path_to_root = ancestors(docs, "def456")

**Developer-focused.**  Finding F-R08-02 recorded that ``parent_doc_id`` was
*recorded but inert*: a plain optional string with no referential integrity
(nothing checked the value named an existing document), no cycle prevention
(``A -> B -> A`` was representable), no depth bound, no traversal accessor, and
no ``StorageQuery`` field -- so it could be written and read back per document,
and could not be walked, validated or queried.

**A caveat worth stating plainly.**  Review run R13 answered unknown U-11: *no
reader, chunker or pipeline stage currently populates* ``parent_doc_id``.  Every
producer leaves it ``None``.  So on today's corpora these functions correctly
validate and traverse an *empty* relation.  They are the prerequisite for the
graph work (the ``parent_of``/``contains`` edges of the derived G0 view), not a
feature that lights up on existing data.  Producer-side population is tracked
separately.

See Also
--------
scikitplot.corpus._diagnostics.ErrorRecord : the diagnostic each violation carries.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Iterator

from ._diagnostics import ErrorCategory, ErrorRecord

__all__: list[str] = [
    "HierarchyReport",
    "ancestors",
    "children",
    "descendants",
    "validate_hierarchy",
]

#: Maximum ancestor chain length accepted by :func:`validate_hierarchy`.
#:
#: A bound is required because an unbounded chain makes every traversal's cost
#: unbounded too.  The value is generous for document nesting (a book with
#: part/chapter/section/paragraph nesting uses ~5) while still catching runaway
#: structures early.
DEFAULT_MAX_DEPTH = 64


@dataclasses.dataclass(frozen=True)
class HierarchyReport:
    """Outcome of validating a document hierarchy.

    Parameters
    ----------
    errors : list of ErrorRecord
        One record per violation found.
    checked : int
        Number of documents examined.
    max_depth_seen : int
        Deepest ancestor chain encountered.
    """

    errors: list[ErrorRecord] = dataclasses.field(default_factory=list)
    checked: int = 0
    max_depth_seen: int = 0

    @property
    def ok(self) -> bool:
        """Whether the hierarchy is free of violations."""
        return not self.errors

    def raise_if_invalid(self) -> None:
        """Raise if any violation was found.

        Raises
        ------
        ValueError
            Naming every violation.
        """
        if self.errors:
            joined = "; ".join(str(record) for record in self.errors)
            raise ValueError(f"invalid document hierarchy: {joined}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {
            "ok": self.ok,
            "checked": self.checked,
            "max_depth_seen": self.max_depth_seen,
            "errors": [record.to_dict() for record in self.errors],
        }


def _index(documents: Iterable[Any]) -> dict[str, Any]:
    """Return a ``doc_id -> document`` mapping."""
    return {doc.doc_id: doc for doc in documents}


def validate_hierarchy(
    documents: Iterable[Any],
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    require_parent_present: bool = True,
) -> HierarchyReport:
    """Check referential integrity, cycles and depth of the hierarchy.

    Parameters
    ----------
    documents : iterable of CorpusDocument
        The collection to validate.  Treated as a closed world: a
        ``parent_doc_id`` naming a document outside this collection counts as
        dangling unless ``require_parent_present`` is ``False``.
    max_depth : int, optional
        Maximum permitted ancestor chain length.
    require_parent_present : bool, optional
        When ``False``, a ``parent_doc_id`` absent from ``documents`` is
        tolerated.  Use for validating a partial page of a larger corpus, where
        the parent legitimately lives elsewhere.

    Returns
    -------
    HierarchyReport
        Violations found, as structured records rather than exceptions, so a
        caller can report all of them rather than only the first.

    Notes
    -----
    Three violation classes are detected, each with its own code:

    ``HIERARCHY_DANGLING_PARENT``
        ``parent_doc_id`` names a document not present.
    ``HIERARCHY_CYCLE``
        Following parents revisits a document.
    ``HIERARCHY_TOO_DEEP``
        The ancestor chain exceeds ``max_depth``.
    ``HIERARCHY_SELF_PARENT``
        A document is its own parent -- reported distinctly from a general
        cycle because it is almost always a copy-paste error rather than a
        structural one.

    Examples
    --------
    >>> from scikitplot.corpus import CorpusDocument
    >>> a = CorpusDocument.create(input_path="f.txt", chunk_index=0, text="a")
    >>> validate_hierarchy([a]).ok
    True
    """
    by_id = _index(documents)
    errors: list[ErrorRecord] = []
    deepest = 0

    for doc_id, doc in by_id.items():
        parent_id = getattr(doc, "parent_doc_id", None)
        if parent_id is None:
            continue

        if parent_id == doc_id:
            errors.append(
                ErrorRecord(
                    code="HIERARCHY_SELF_PARENT",
                    category=ErrorCategory.VALIDATION,
                    message="document is its own parent",
                    source_id=doc_id,
                )
            )
            continue

        if parent_id not in by_id:
            if require_parent_present:
                errors.append(
                    ErrorRecord(
                        code="HIERARCHY_DANGLING_PARENT",
                        category=ErrorCategory.VALIDATION,
                        message=f"parent_doc_id {parent_id!r} names no known document",
                        source_id=doc_id,
                        details={"parent_doc_id": parent_id},
                    )
                )
            continue

        # Walk to the root, detecting cycles and depth in one pass.
        seen = {doc_id}
        depth = 0
        cursor = parent_id
        while cursor is not None:
            depth += 1
            if cursor in seen:
                errors.append(
                    ErrorRecord(
                        code="HIERARCHY_CYCLE",
                        category=ErrorCategory.VALIDATION,
                        message=f"ancestor chain revisits {cursor!r}",
                        source_id=doc_id,
                        details={"cycle_at": cursor, "depth": depth},
                    )
                )
                break
            if depth > max_depth:
                errors.append(
                    ErrorRecord(
                        code="HIERARCHY_TOO_DEEP",
                        category=ErrorCategory.VALIDATION,
                        message=f"ancestor chain exceeds max_depth={max_depth}",
                        source_id=doc_id,
                        details={"max_depth": max_depth},
                    )
                )
                break
            seen.add(cursor)
            parent = by_id.get(cursor)
            if parent is None:
                break
            cursor = getattr(parent, "parent_doc_id", None)

        deepest = max(deepest, depth)

    return HierarchyReport(errors=errors, checked=len(by_id), max_depth_seen=deepest)


def children(
    documents: Iterable[Any],
    doc_id: str,
) -> list[Any]:
    """Return the direct children of ``doc_id``.

    Parameters
    ----------
    documents : Iterable[Any]
        iterable of CorpusDocument
    doc_id : str
        The parent's identifier.

    Returns
    -------
    list
        Direct children, ordered by ``chunk_index`` where available.
    """
    found = [doc for doc in documents if getattr(doc, "parent_doc_id", None) == doc_id]
    return sorted(found, key=lambda d: getattr(d, "chunk_index", 0) or 0)


def descendants(
    documents: Iterable[Any],
    doc_id: str,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> Iterator[Any]:
    """Yield all descendants of ``doc_id``, breadth-first.

    Parameters
    ----------
    documents : Iterable[Any]
        iterable of CorpusDocument
    doc_id : str
        The root's identifier.
    max_depth : int, optional
        Maximum levels to descend.

    Yields
    ------
    CorpusDocument

    Notes
    -----
    **Developer.**  Visited identifiers are tracked, so a cyclic hierarchy
    terminates instead of looping forever.  Traversal must stay safe on data
    that :func:`validate_hierarchy` would reject -- callers do not always
    validate first.
    """
    docs = list(documents)
    frontier = [doc_id]
    visited = {doc_id}
    depth = 0
    while frontier and depth < max_depth:
        depth += 1
        following: list[str] = []
        for parent_id in frontier:
            for child in children(docs, parent_id):
                if child.doc_id in visited:
                    continue
                visited.add(child.doc_id)
                following.append(child.doc_id)
                yield child
        frontier = following


def ancestors(
    documents: Iterable[Any],
    doc_id: str,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> list[Any]:
    """Return the chain from ``doc_id``'s parent up to the root.

    Parameters
    ----------
    documents : Iterable[Any]
        iterable of CorpusDocument
    doc_id : str
        The root's identifier.
    max_depth : int, optional
        Maximum levels to descend.

    Returns
    -------
    list
        Nearest ancestor first.  Stops safely on a cycle or a dangling parent.
    """
    by_id = _index(documents)
    chain: list[Any] = []
    seen = {doc_id}
    cursor = getattr(by_id.get(doc_id), "parent_doc_id", None)
    while cursor is not None and cursor not in seen and len(chain) < max_depth:
        parent = by_id.get(cursor)
        if parent is None:
            break
        chain.append(parent)
        seen.add(cursor)
        cursor = getattr(parent, "parent_doc_id", None)
    return chain
