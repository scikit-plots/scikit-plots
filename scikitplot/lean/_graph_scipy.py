# scikitplot/lean/_graph_scipy.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The ``graph="scipy"`` backend of :class:`~scikitplot.lean.LeanIndex`.

The edges are packed as for ``graph="numpy"`` (:mod:`._graph_numpy`); the
breadth-first searches run in SciPy's compiled
:func:`scipy.sparse.csgraph.breadth_first_order`. Use counts and reversal
are the NumPy backend's.

Kept chiefly to benchmark and to cross-check the other backends: every
search gives exactly what the standard-library FIFO search gives (the same
nodes, order and parents), which the tests check on random graphs. SciPy
visits a row's neighbours in ascending order, which is the order the FIFO
search reads a packed row in (rows are sorted by target). Several roots
are joined to one extra source node; SciPy visits those ascending too, so
roots in another order take the NumPy search, which keeps their order
(LeanIndex passes them ascending). A node whose edges are not followed
has its row emptied; how each edge was found is looked up in the packed
edges.

Notes
-----
**User.** Every call builds a matrix of the whole project's edges, so a
call costs a few milliseconds whatever the closure's size (about 3.5 ms
on mathlib4); for ordinary dossiers the standard library or NumPy is
faster. Its value is an independent implementation to compare against.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

from . import _graph_numpy
from ._graph_numpy import reverse, use_counts

__all__: list[str] = []

#: SciPy takes 32-bit sparse indices below this size, 64-bit from it.
_INDEX_LIMIT = 2**31

# Re-exported so the backends answer to one interface.
_ = (reverse, use_counts)


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

    The contract of :func:`._graph_numpy.breadth_first`, computed by
    :func:`scipy.sparse.csgraph.breadth_first_order`.
    """
    roots = list(dict.fromkeys(int(r) for r in roots))
    if not roots:
        return {}
    if roots != sorted(roots):
        # SciPy visits a row's neighbours in ascending order, the extra
        # source's too: roots in another order would be visited differently
        # from a FIFO search. LeanIndex passes them ascending; any other
        # order takes the exact level-by-level search.
        return _graph_numpy.breadth_first(indptr, codes, roots, n, follow=follow)
    sources = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    targets = (codes // 4).astype(np.int64)
    keep_codes = codes
    if follow is not None:  # a node not followed keeps its place, loses its edges
        kept = follow[sources]
        sources, targets, keep_codes = sources[kept], targets[kept], codes[kept]
    # Rows of a graph of n + 1 nodes: the project's, then one extra source
    # whose neighbours are the roots (ascending, see above).
    counts = np.bincount(sources, minlength=n).astype(np.int64)
    counts = np.append(counts, len(roots))
    row_ptr = np.zeros(n + 2, dtype=np.int64)
    np.cumsum(counts, out=row_ptr[1:])
    columns = np.concatenate([targets, np.asarray(roots, dtype=np.int64)])
    index_type = np.int32 if max(n + 1, len(columns)) < _INDEX_LIMIT else np.int64
    graph = csr_matrix(
        (
            np.ones(len(columns), dtype=np.int8),
            columns.astype(index_type),
            row_ptr.astype(index_type),
        ),
        shape=(n + 1, n + 1),
    )
    order, predecessors = breadth_first_order(
        graph, n, directed=True, return_predecessors=True
    )
    order = order[1:]  # the extra source
    parents = predecessors[order].astype(np.int64)
    parents[parents == n] = -1  # roots
    hows = np.full(len(order), -1, dtype=np.int64)
    linked = parents >= 0
    if linked.any():
        # Edges are sorted by (source, target): find each tree edge's code.
        keys = sources * (n + 1) + targets
        wanted = parents[linked] * (n + 1) + order[linked]
        hows[linked] = keep_codes[np.searchsorted(keys, wanted)] % 4
    return dict(zip(order.tolist(), zip(parents.tolist(), hows.tolist())))
