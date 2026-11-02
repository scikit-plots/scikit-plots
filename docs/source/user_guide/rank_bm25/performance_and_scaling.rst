.. _rank-bm25-performance-and-scaling:

Performance and scaling
=======================

Every query term's contribution to every document that holds it is computed
once per index -- on the first query, or when you call ``prepare()`` -- and
stored term by term. A query then adds those stored values, in query order,
for each document; documents without a term receive that term's absent-term
value. The results are bit-identical to evaluating the formula directly
(``backend="reference"``), so switching backend never changes a score, a tie
order or ``identity``.

.. code-block:: python

   from scikitplot.rank_bm25 import BM25Okapi

   index = BM25Okapi(tokenized_corpus, doc_ids=ids).prepare()  # pay costs at start-up
   index.backend          # "numba" when Numba is installed, else "numpy"
   index.get_top_ids(query_tokens, n=10)

Backends
--------

``backend="auto"`` (default)
    Numba when it can be imported, NumPy otherwise.
``backend="numpy"``
    Vectorised NumPy; no compilation.
``backend="numba"``
    The same loop compiled; raises ``ImportError`` at construction when Numba
    is not installed. Install it with the ``rank-bm25`` extra.
``backend="reference"``
    The direct formula evaluation every other backend is tested against.

An index can be queried from several threads at once on every backend. The
compiled kernel is single-threaded and releases the GIL, so concurrent queries
from different threads run in parallel; it starts no worker threads of its
own, under any Numba threading layer.

Numba is imported only when the compiled backend is first used. Compiling the
kernel takes under a second in a new process and is cached on disk after
that; ``SCIKITPLOT_RANK_BM25_DISABLE_NUMBA=1`` forces NumPy. A subclass that
overrides the scoring formula always uses the reference evaluation.

Costs
-----

Measured on a synthetic 200 000-document corpus (about 20 million stored
document--term pairs, two CPU cores): ``get_top_ids`` took about 600 ms per
four-term query with the reference evaluation and about 2.5 ms with either
NumPy or Numba. Building the stored contributions took about 5 s, roughly the
cost of building the index itself, once per index. On two cores Numba was no
faster than NumPy; measure on your own hardware before relying on a
difference.

Memory: the stored postings and contributions take about 24 bytes per
document--term pair. An index built in the process also keeps its
per-document term counts; an index loaded from a ``bm25-index/2`` generation
does not, until a path that needs them runs (see
:ref:`rank-bm25-persistence-and-reproducibility`).

``get_batch_scores`` still evaluates the formula for the requested rows only,
which is the cheaper path when you score a handful of documents.

Tokenization can use several processes (``workers=N``); query scoring runs in
the calling process.

Move to a larger search system when you need distributed indexing, concurrent
mutation, service-level persistence, very large corpora, advanced analyzers, or
hybrid/vector retrieval at a scale that exceeds an in-process index. BM25
remains useful as a lexical leg in a hybrid retriever.
