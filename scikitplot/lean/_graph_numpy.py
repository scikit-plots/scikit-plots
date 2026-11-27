# scikitplot/lean/_graph_numpy.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The ``graph="numpy"`` backend of :class:`~scikitplot.lean.LeanIndex`.

Once every edge is resolved, the edges are packed into compressed rows
(CSR): ``indptr`` (one offset per node, ``int64``) and ``codes`` (one
``target * 4 + how`` per edge, ``int32`` while it fits). The whole-graph
steps then run vectorised:

- :func:`use_counts`: how many declarations, and files, name each node;
- :func:`breadth_first`: a breadth-first search forward (a dossier's
  closure) or along reversed edges (the declarations that reach a flagged
  one), level by level.

Every function gives exactly what the standard-library code gives: the same
counts, the same parent of every reached node and the same order. A FIFO
search discovers the nodes of one level in the order of their first
occurrence among the edges of the previous level, taken in that level's
order; :func:`breadth_first` keeps that first occurrence.

Notes
-----
**Developer.** NumPy is imported only by this module, which only the
``numpy`` backend imports. Name resolution stays in Python: it is string
and dictionary work, and on real projects it is most of the time.
"""

from __future__ import annotations

from array import array
from typing import Any, Sequence

import numpy as np

__all__: list[str] = []


def pack(edges_of: Any, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``(indptr, codes)`` for the edges of nodes ``0 .. n-1``.

    *edges_of(node)* is called once per node, in order, and returns the
    node's edges as ``target * 4 + how``, sorted. They are appended to one
    growing buffer of 4-byte (or 8-byte) codes, so no copy per node is kept.
    """
    wide = 4 * n >= 2**31
    buffer = array("q" if wide else "i")
    indptr = np.zeros(n + 1, dtype=np.int64)
    for node in range(n):
        buffer.extend(iter(edges_of(node)))  # any int sequence, any typecode
        indptr[node + 1] = len(buffer)
    codes = np.frombuffer(buffer, dtype=np.int64 if wide else np.int32).copy()
    return indptr, codes


def use_counts(
    indptr: np.ndarray, codes: np.ndarray, file_of: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return, per node, the declarations that name it and the files they are in.

    The same as counting along the edges in Python: a node named from two
    declarations of one file counts that file once.
    """
    targets = codes // 4
    in_degree = np.bincount(targets, minlength=n).astype(np.int64)
    if not len(targets):
        return in_degree, np.zeros(n, dtype=np.int64)
    sources = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    files = np.int64(file_of.max()) + 1
    pairs = np.unique(
        targets.astype(np.int64) * files + file_of[sources].astype(np.int64)
    )
    in_files = np.bincount(pairs // files, minlength=n).astype(np.int64)
    return in_degree, in_files


def reverse(
    indptr: np.ndarray, codes: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the reversed edges as CSR: per target, ``source * 4 + how``.

    Sources stay in ascending order for each target (a stable sort), as the
    Python code fills them.
    """
    targets = codes // 4
    sources = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    order = np.argsort(targets, kind="stable")
    rev_codes = (sources[order] * 4 + codes[order] % 4).astype(codes.dtype)
    rev_indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(targets, minlength=n), out=rev_indptr[1:])
    return rev_indptr, rev_codes


def breadth_first(
    indptr: np.ndarray,
    codes: np.ndarray,
    roots: Sequence[int],
    n: int,
    *,
    follow: np.ndarray | None = None,
) -> dict[int, tuple[int, int]]:
    """
    Return each reached node mapped to ``(parent, how)``; roots map to ``(-1, -1)``.

    Level by level from *roots*: the edges of the current level, in its
    order, are gathered; a node not yet reached takes the first edge that
    names it. With *follow* (a boolean mask), a reached node whose mask is
    false is kept but its edges are not followed. The dictionary's order is
    the order of discovery, as a FIFO search builds it.
    """
    reached = np.zeros(n, dtype=bool)
    parent = np.full(n, -1, dtype=np.int64)
    how = np.full(n, -1, dtype=np.int64)
    order: list[np.ndarray] = []
    frontier = np.asarray(list(dict.fromkeys(roots)), dtype=np.int64)
    reached[frontier] = True
    order.append(frontier)
    starts_all, stops_all = indptr[:-1], indptr[1:]
    while len(frontier):
        expanding = frontier if follow is None else frontier[follow[frontier]]
        if not len(expanding):
            break
        starts, stops = starts_all[expanding], stops_all[expanding]
        lengths = stops - starts
        total = int(lengths.sum())
        if not total:
            break
        # Positions of every edge of the level, in the level's order.
        owner = np.repeat(np.arange(len(expanding), dtype=np.int64), lengths)
        offsets = np.arange(total, dtype=np.int64) - np.repeat(
            np.cumsum(lengths) - lengths, lengths
        )
        level_codes = codes[np.repeat(starts, lengths) + offsets]
        targets = level_codes // 4
        fresh = ~reached[targets]
        if not fresh.any():
            break
        targets, level_codes, owner = targets[fresh], level_codes[fresh], owner[fresh]
        first = np.unique(targets, return_index=True)[1]
        first.sort()  # first occurrence, in the order they occur
        found = targets[first]
        reached[found] = True
        parent[found] = expanding[owner[first]]
        how[found] = level_codes[first] % 4
        order.append(found)
        frontier = found
    sequence = np.concatenate(order).tolist()
    parents, hows = parent[sequence].tolist(), how[sequence].tolist()
    return dict(zip(sequence, zip(parents, hows)))
