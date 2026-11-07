# scikitplot/rank_bm25/_engine.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Precomputed sparse scoring: every observed term contribution, computed once.

Notes
-----
**User-focused.** An index scores a query by looking up contributions that
were computed when it was first queried (or when :meth:`BM25.prepare` was
called), instead of re-evaluating the formula for every document on every
query. Scores are **bit-identical** to the reference evaluation for every
built-in recipe; only the time changes. ``backend="numba"`` runs the same
loop compiled, and is used by ``backend="auto"`` when Numba is importable.

**Developer-focused.** Layout and the exactness argument.

*Layout.* Term-major compressed columns: for vocabulary id ``t`` its postings
are ``rows[indptr[t]:indptr[t + 1]]`` (ascending row order) and
``data[indptr[t]:indptr[t + 1]]``, where ``data`` holds the **full** recipe
contribution ``_term_contribution(idf[t], tf, doc_len[row])``, evaluated by the
recipe's own method on arrays.

*Exactness.* The reference (:meth:`BM25.get_scores`) adds one contribution
per query term, in query order, to every document: for document ``r`` it
computes ``((0.0 + c_1(r)) + c_2(r)) + ...``. The engine performs exactly that
sequence per element: for each query term, ``r`` receives the stored
contribution when it holds the term, else the term's *absent* contribution.
The common "add a baseline to everything, then add sparse deltas" form is
deliberately not used: it reassociates the sum and drifts in the last bits for
the recipes whose absent contribution is non-zero (BM25L, BM25+).

*Absent contribution.* For the built-in recipes it does not depend on document
length: a zero frequency makes the length-dependent part of every formula 0,
and the recipes define the empty-document case (``b == 1``, where the length
normaliser is 0) explicitly as that same value. One absent contribution per
query term therefore serves every row. The engine checks this when it is built
and refuses to build if any row differs, so a formula change that broke the
argument fails at once instead of scoring some rows differently from the
reference.

*Scope.* Only recipes whose ``_term_contribution`` is one of the audited
built-in implementations use the engine; a subclass that overrides the formula
falls back to the reference path, because the length-independence argument
above is a property of those formulas, not of every possible one.
"""

from __future__ import annotations

import os
from itertools import chain
from typing import Any

import numpy as np

__all__: list[str] = [
    "BACKENDS",
    "SparseScoreEngine",
    "accumulate",
    "numba_available",
    "resolve_backend",
    "top_positions",
]

#: Accepted ``backend`` values.
BACKENDS: tuple[str, ...] = ("auto", "numpy", "numba", "reference")

_NUMBA_KERNEL: Any = None
_NUMBA_IMPORT_ERROR: BaseException | None = None


def numba_available() -> bool:
    """
    Report whether the compiled backend can be used in this process.

    Returns
    -------
    bool
        ``True`` when Numba imports and the kernel compiles.

    Notes
    -----
    **Developer.** Importing Numba and compiling are both deferred to the
    first call, so importing :mod:`scikitplot.rank_bm25` never imports Numba.
    The environment variable ``SCIKITPLOT_RANK_BM25_DISABLE_NUMBA=1`` forces
    ``False``, which is how CI exercises the NumPy path on a machine that has
    Numba installed.
    """
    return _numba_kernel() is not None


def _numba_kernel():
    """Return the compiled accumulator, compiling it on first use, or ``None``."""
    global _NUMBA_KERNEL, _NUMBA_IMPORT_ERROR  # noqa: PLW0603 - one-time lazy init
    if os.environ.get("SCIKITPLOT_RANK_BM25_DISABLE_NUMBA") == "1":
        return None
    if _NUMBA_KERNEL is not None or _NUMBA_IMPORT_ERROR is not None:
        return _NUMBA_KERNEL
    try:
        import numba  # noqa: PLC0415 - optional dependency, deliberately lazy
    except ImportError as exc:
        _NUMBA_IMPORT_ERROR = exc
        return None

    # No fastmath: it permits reassociation, which is exactly what the
    # exactness argument forbids.
    #
    # Serial, not parallel=True: under Numba's "workqueue" threading layer
    # (the fallback where neither TBB nor OpenMP is available) two Python
    # threads calling a parallel kernel at once terminate the process
    # ("Concurrent access has been detected") -- reproduced with 8 threads.
    # A serial kernel is safe under every layer, and nogil=True lets
    # concurrent queries from different threads run truly in parallel, which
    # is the parallelism a service needs. parallel=True gave no measurable
    # single-query gain on 2 cores.
    #
    # cache=True keeps the compiled kernel on disk (next to the module, or in
    # Numba's per-user cache directory when the install is read-only).
    @numba.njit(fastmath=False, cache=True, nogil=True)
    def accumulate(  # ruff: ignore[too-many-positional-arguments]
        term_ids,
        absent,
        indptr,
        rows,
        data,
        n_rows,
    ):  # pragma: no cover - compiled
        n_terms = term_ids.shape[0]
        scores = np.zeros(n_rows)
        stamp = np.full(n_rows, -1, dtype=np.int64)
        value = np.zeros(n_rows)
        for i in range(n_terms):
            t = term_ids[i]
            if t >= 0:
                for j in range(indptr[t], indptr[t + 1]):
                    r = rows[j]
                    stamp[r] = i
                    value[r] = data[j]
            a = absent[i]
            for r in range(n_rows):
                if stamp[r] == i:
                    scores[r] += value[r]
                else:
                    scores[r] += a
        return scores

    _NUMBA_KERNEL = accumulate
    return _NUMBA_KERNEL


def _accumulate_numpy(  # ruff: ignore[too-many-positional-arguments]
    term_ids,
    absent,
    indptr,
    rows,
    data,
    n_rows,
):
    """
    Exact in-order accumulation with NumPy.

    Parameters
    ----------
    term_ids : numpy.ndarray of int64
        Vocabulary id per query occurrence, ``-1`` for an unknown term.
    absent : numpy.ndarray of float64
        Absent-term contribution per query occurrence.
    indptr, rows, data : numpy.ndarray
        The term-major postings.
    n_rows : int
        Number of documents.

    Returns
    -------
    numpy.ndarray of float64
        One score per document.
    """
    scores = np.zeros(n_rows)
    contribution = np.empty(n_rows)
    for t, a in zip(term_ids.tolist(), absent.tolist()):
        contribution.fill(a)
        if t >= 0:
            start, stop = indptr[t], indptr[t + 1]
            contribution[rows[start:stop]] = data[start:stop]
        scores += contribution
    return scores


def resolve_backend(requested: str) -> str:
    """
    Resolve ``"auto"``/``"numpy"``/``"numba"`` to the backend that will run.

    Parameters
    ----------
    requested : {"auto", "numpy", "numba"}
        The caller's request.

    Returns
    -------
    str
        ``"numba"`` or ``"numpy"``.

    Raises
    ------
    ValueError
        If ``requested`` is not one of the three values.
    ImportError
        If ``"numba"`` is requested and is not available.
    """
    if requested not in ("auto", "numpy", "numba"):
        raise ValueError(
            f"backend must be 'auto', 'numpy' or 'numba', got {requested!r}."
        )
    if requested == "numpy":
        return "numpy"
    if numba_available():
        return "numba"
    if requested == "numba":
        raise ImportError(
            "backend='numba' was requested but Numba is not importable (or "
            "SCIKITPLOT_RANK_BM25_DISABLE_NUMBA=1 is set)."
        )
    return "numpy"


def accumulate(  # ruff: ignore[too-many-positional-arguments]
    term_ids,
    absent,
    indptr,
    rows,
    data,
    n_rows,
    backend="numpy",
):
    """
    Sum per-term contributions for every row, in query order, exactly.

    Parameters
    ----------
    term_ids : numpy.ndarray of int64
        Column id per query occurrence; ``-1`` for a term with no column.
    absent : numpy.ndarray of float64
        Contribution added to rows that do not hold that occurrence's term.
    indptr, rows, data : numpy.ndarray
        Term-major postings: column ``t`` is ``rows[indptr[t]:indptr[t+1]]``
        with contributions ``data[indptr[t]:indptr[t+1]]``; rows unique
        within a column.
    n_rows : int
        Number of rows.
    backend : {"numpy", "numba"}, optional
        A resolved backend (see :func:`resolve_backend`).

    Returns
    -------
    numpy.ndarray of float64
        ``scores[r] = ((0.0 + c_1(r)) + c_2(r)) + ...``, the same sequence of
        IEEE additions on either backend.

    Notes
    -----
    **Developer.** Shared by :class:`SparseScoreEngine` and by the corpus
    keyword index, each of which supplies its own ``data``: the formula stays
    with its owner, and only the summation is shared.
    """
    if backend == "numba":
        return _numba_kernel()(term_ids, absent, indptr, rows, data, n_rows)
    return _accumulate_numpy(term_ids, absent, indptr, rows, data, n_rows)


class SparseScoreEngine:
    """
    Term-major precomputed contributions for one built index.

    Parameters
    ----------
    index : BM25
        A built index whose recipe is an audited built-in formula.

    Attributes
    ----------
    vocabulary : dict of str to int
        Term to column id.
    indptr : numpy.ndarray of int64
        Column offsets, length ``len(vocabulary) + 1``.
    rows : numpy.ndarray of int64
        Row of each stored contribution, ascending within a column.
    tfs : numpy.ndarray of int64
        Term frequency of each stored (row, term) pair; kept so the index can
        be persisted without its per-document dicts.
    data : numpy.ndarray of float64
        Full recipe contribution of each stored (row, term) pair.

    Raises
    ------
    RuntimeError
        If the recipe's absent-term contribution depends on document length,
        which the engine's exactness argument excludes.

    Notes
    -----
    **Developer.** Built from ``doc_freqs``/``doc_len``/``idf`` -- information
    the index already holds -- so it is a cache, never a second source of
    truth. It is discarded whenever those change; nothing here is persisted.
    """

    def __init__(self, index: Any) -> None:
        self._index = index
        doc_freqs = index.doc_freqs
        n_rows = len(doc_freqs)
        self.n_rows = n_rows
        terms = list(index.idf)
        self.vocabulary = {term: column for column, term in enumerate(terms)}
        vocabulary = self.vocabulary

        # One flat pass over the per-document dicts (row-major), then a stable
        # sort by column: rows are already ascending, and a stable sort keeps
        # them so within every column.
        # map/chain iterate in C; a generator expression here costs a Python
        # frame per stored pair and dominated the build.
        flat_columns = np.fromiter(
            map(vocabulary.__getitem__, chain.from_iterable(doc_freqs)),
            dtype=np.int64,
        )
        flat_tfs = np.fromiter(
            chain.from_iterable(map(dict.values, doc_freqs)),
            dtype=np.int64,
            count=flat_columns.size,
        )
        flat_rows = np.repeat(
            np.arange(n_rows, dtype=np.int64),
            np.fromiter((len(f) for f in doc_freqs), dtype=np.int64, count=n_rows),
        )
        order = np.argsort(flat_columns, kind="stable")
        rows = flat_rows[order]
        tfs = flat_tfs[order]
        indptr = np.zeros(len(terms) + 1, dtype=np.int64)
        np.cumsum(np.bincount(flat_columns, minlength=len(terms)), out=indptr[1:])

        idf = np.array([index.idf[term] or 0 for term in terms], dtype=np.float64)
        doc_len = np.asarray(index.doc_len, dtype=np.int64)
        columns = np.repeat(np.arange(len(terms), dtype=np.int64), np.diff(indptr))
        # The recipe's own vectorised method computes every stored value, so a
        # stored contribution is the very value the reference adds.
        data = np.asarray(
            index._term_contribution(idf[columns], tfs, doc_len[rows]),
            dtype=np.float64,
        )
        _require_length_independent_absent(index, doc_len)
        self.indptr = indptr
        self.rows = rows
        self.tfs = tfs
        self.data = data

    @classmethod
    def from_arrays(
        cls,
        index: Any,
        vocabulary,
        indptr,
        rows,
        tfs,
    ) -> SparseScoreEngine:
        """
        Rebuild an engine from persisted postings, without the per-document dicts.

        Notes
        -----
        **Developer.** The contributions are not persisted: they are the
        recipe's formula of the persisted counts, idf and document lengths, so
        they are recomputed here by the same vectorised call the builder
        makes, which gives the same values bit for bit (an integer count
        converts to float64 exactly whatever its stored width).
        """
        engine = cls.__new__(cls)
        engine._index = index
        engine.n_rows = len(index.doc_len)
        engine.vocabulary = {term: column for column, term in enumerate(vocabulary)}
        engine.indptr = np.asarray(indptr, dtype=np.int64)
        engine.rows = np.asarray(rows, dtype=np.int64)
        engine.tfs = np.asarray(tfs, dtype=np.int64)
        doc_len = np.asarray(index.doc_len, dtype=np.int64)
        idf = np.array([index.idf[term] or 0 for term in vocabulary], dtype=np.float64)
        columns = np.repeat(
            np.arange(len(vocabulary), dtype=np.int64),
            np.diff(engine.indptr),
        )
        engine.data = np.asarray(
            index._term_contribution(idf[columns], engine.tfs, doc_len[engine.rows]),
            dtype=np.float64,
        )
        _require_length_independent_absent(index, doc_len)
        return engine

    def doc_freqs(self) -> list[dict[str, int]]:
        """Rebuild the per-document term counts (row order) from the arrays."""
        terms = list(self.vocabulary)
        out: list[dict[str, int]] = [{} for _ in range(self.n_rows)]
        indptr, rows, tfs = self.indptr, self.rows, self.tfs
        for column, term in enumerate(terms):
            for j in range(int(indptr[column]), int(indptr[column + 1])):
                out[int(rows[j])][term] = int(tfs[j])
        return out

    def nbytes(self) -> int:
        """Return the bytes held by the engine's arrays."""
        return int(
            self.indptr.nbytes + self.rows.nbytes + self.tfs.nbytes + self.data.nbytes
        )

    def _absent(self, idf: float) -> float:
        """Absent-term contribution of a term, by the recipe's formula."""
        return float(self._index._term_contribution(idf, 0, self._index.avgdl))

    def scores(self, query: list[str], backend: str) -> np.ndarray:
        """
        Score every document, bit-identical to the reference evaluation.

        Parameters
        ----------
        query : list of str
            Validated query tokens; duplicates count once per occurrence.
        backend : {"numpy", "numba"}
            Resolved backend.

        Returns
        -------
        numpy.ndarray of float64
            One score per document, in row order.
        """
        index = self._index
        idfs = [index.idf.get(term) or 0 for term in query]
        term_ids = np.array(
            [self.vocabulary.get(term, -1) for term in query], dtype=np.int64
        )
        absent = np.array([self._absent(idf) for idf in idfs], dtype=np.float64)
        return accumulate(
            term_ids, absent, self.indptr, self.rows, self.data, self.n_rows, backend
        )

    def candidate_mask(self, query: list[str]) -> np.ndarray:
        """Return a boolean mask of documents holding at least one query term."""
        mask = np.zeros(self.n_rows, dtype=bool)
        for term in set(query):
            column = self.vocabulary.get(term)
            if column is not None:
                mask[self.rows[self.indptr[column] : self.indptr[column + 1]]] = True
        return mask


def _require_length_independent_absent(index: Any, doc_len: np.ndarray) -> None:
    """
    Refuse a recipe whose absent-term contribution varies with document length.

    Parameters
    ----------
    index : BM25
        The index the engine is built for.
    doc_len : numpy.ndarray of int64
        Every document's length, in row order.

    Raises
    ------
    RuntimeError
        If any row's zero-frequency contribution is non-finite or differs from
        the one :meth:`SparseScoreEngine._absent` uses for every row.

    Notes
    -----
    **Developer.** The engine adds one absent contribution per query term to
    every row that lacks the term; that is exact only if the value is the same
    for every length. Probing with ``idf = 1.0`` is sufficient: each built-in
    formula is ``idf`` times a length term, so a length dependence shows at
    any non-zero ``idf``. The check is one vectorised evaluation per build.
    """
    probe = np.asarray(
        index._term_contribution(1.0, np.zeros(doc_len.size, dtype=np.int64), doc_len),
        dtype=np.float64,
    )
    generic = float(index._term_contribution(1.0, 0, index.avgdl))
    irregular = np.flatnonzero(~(np.isfinite(probe) & (probe == generic)))
    if irregular.size:
        raise RuntimeError(
            f"recipe {index.RECIPE_ID!r}: the absent-term contribution of "
            f"{irregular.size} row(s) (first: row {int(irregular[0])}) differs "
            f"from {generic!r}; the precomputed engine requires it to be the "
            "same for every document length."
        )


def top_positions(
    scores: np.ndarray,
    n: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Return up to ``n`` row positions, highest score first, ties by lower row.

    Parameters
    ----------
    scores : numpy.ndarray of float64
        One score per document.
    n : int
        Maximum number of positions.
    mask : numpy.ndarray of bool or None, optional
        Rows eligible for selection; all rows when ``None``.

    Returns
    -------
    numpy.ndarray of int64
        Selected rows, in rank order.

    Notes
    -----
    **Developer.** Equivalent to ``argsort(-scores, kind="stable")[:n]`` over
    the eligible rows -- the reference order -- without sorting everything:
    the ``n``-th best value is found by partition, then *every* row tied with
    it is kept before the stable sort, so a tie straddling the cut-off is
    resolved by row order exactly as the full sort resolves it. Scores holding
    ``NaN`` take the full stable sort, which defines their placement.
    """
    candidates = np.arange(scores.size, dtype=np.int64)
    if mask is not None:
        candidates = candidates[mask]
    if n <= 0 or candidates.size == 0:
        return candidates[:0]
    keys = -scores[candidates]
    if n < candidates.size and not np.isnan(keys).any():
        cutoff = np.partition(keys, n - 1)[n - 1]
        keep = keys <= cutoff
        candidates = candidates[keep]
        keys = keys[keep]
    order = np.argsort(keys, kind="stable")
    return candidates[order[:n]]
