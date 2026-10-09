.. currentmodule:: scikitplot.rank_bm25

.. _rank-bm25-index:

======================================================================
Rank-BM25 user guide
======================================================================

``scikitplot.rank_bm25`` provides lexical document ranking with three BM25
recipes: ``BM25Okapi``, ``BM25L``, and ``BM25Plus``.  Use it when relevance
should come from query terms and their collection statistics rather than from a
vector embedding or a learned model.

The implementation is intentionally explicit about two boundaries that matter
in production retrieval:

* **text analysis is part of the index** -- documents and queries must be
  tokenized consistently; and
* **a hit should name a document** -- supply stable ``doc_ids`` when building an
  index and prefer ``get_top_ids`` over rebinding row positions later.

Installation
============

Rank-BM25 is included in the full ``scikit-plots`` distribution.  It is also
available as the project-maintained partial distribution
``scikit-plots-rank-bm25``::

   python -m pip install scikit-plots-rank-bm25

The partial distribution ships ``scikitplot.rank_bm25`` and depends on NumPy
and the dependency-light Scikit-Plots core.  Do not install it alongside the
full ``scikit-plots`` distribution because the full distribution already owns
these files.  See :doc:`../../affiliated/index` for the partial-distribution
model and compatibility guidance.

Quick start: bind results to document identities
================================================

Tokenize the corpus, assign one stable identity per row, then query with tokens
produced by the same analysis policy::

   from scikitplot.rank_bm25 import BM25Okapi

   documents = [
       "Hello there good man!",
       "It is quite windy in London",
       "How is the weather today?",
   ]
   doc_ids = ["greeting", "london-weather", "weather-question"]
   corpus = [text.lower().split() for text in documents]

   index = BM25Okapi(corpus, doc_ids=doc_ids)
   query = "windy london".split()

   hits = index.get_top_ids(query, n=2)
   # [('london-weather', <score>)]

``get_top_ids`` returns ``(doc_id, score)`` pairs in descending score order.  It
scores only documents containing at least one query term, so an out-of-vocabulary
query returns an empty list instead of arbitrary zero-score documents.  Equal
scores are ordered deterministically by the original corpus row.

The example deliberately does not assert an exact floating-point score.  Ranking
is the user-facing result; exact values depend on the selected recipe, corpus
statistics, and parameters.

Choose a BM25 recipe
====================

The public module exports one base class and three concrete scorers:

.. list-table::
   :header-rows: 1
   :widths: 18 25 25 32

   * - Class
     - Recipe identifier
     - Defaults
     - Distinguishing behavior
   * - ``BM25Okapi``
     - ``okapi-epsilon-floor/1``
     - ``k1=1.5``, ``b=0.75``, ``epsilon=0.25``
     - Floors negative IDF values at ``epsilon * average_idf``.
   * - ``BM25L``
     - ``bm25l-delta/1``
     - ``k1=1.5``, ``b=0.75``, ``delta=0.5``
     - Adds ``delta`` to normalized term frequency.
   * - ``BM25Plus``
     - ``bm25plus-delta/1``
     - ``k1=1.5``, ``b=0.75``, ``delta=1``
     - Adds ``delta`` to the term contribution.

``BM25`` is the shared base class.  Its scoring hooks are intentionally not
implemented, so normal users should instantiate one of the three concrete
classes above.

The ``recipe`` property exposes the named, versioned formula attached to an
index.  This is more precise than assuming that a class name implies every
implementation detail::

   index = BM25Okapi([['alpha'], ['beta']])
   print(index.recipe.identifier)
   # okapi-epsilon-floor/1

Parameter domains are validated when the index is built:

* ``k1`` must be finite and greater than zero;
* ``b`` must be finite and between zero and one, inclusive;
* ``epsilon`` and ``delta`` must be finite and non-negative; and
* booleans are rejected even though Python treats ``bool`` as a subclass of
  ``int``.

Tokenization is part of correctness
===================================

Without ``tokenizer=`` the corpus must already be an iterable of token
sequences.  A raw string document is rejected rather than being indexed one
character at a time::

   from scikitplot.rank_bm25 import BM25Okapi

   corpus = [
       ['alpha', 'beta'],
       ['beta', 'gamma'],
   ]
   index = BM25Okapi(corpus)

Every token must be a string.  An empty corpus is invalid, and a corpus in which
all documents are empty is invalid; individual empty documents are allowed when
at least one document contains tokens.

You can instead pass raw documents plus a tokenizer::

   index = BM25Okapi(
       ['alpha beta', 'beta gamma'],
       tokenizer=str.split,
   )

Tokenization runs in the current process by default.  ``workers=N`` opts into a
multiprocessing pool.  With multiple workers the tokenizer must therefore be
picklable; lambdas and closures are not reliable choices for that mode.

Queries are token sequences too.  Apply the same case handling, stopword policy,
stemming policy, and other normalization used for the indexed corpus.  Do not
rely on the scorer to infer or repair a different query analysis policy.

Retrieval, scoring, and diagnostics
===================================

Use the method that matches the question you are asking.

``get_top_ids(query, n=5)``
   Preferred retrieval interface when the index was built with ``doc_ids``.
   Returns stable identities and scores and uses the sparse postings path.

``candidate_count(query)``
   Counts rows containing at least one query term.  This is useful for
   understanding how many documents the sparse retrieval path will score.

``get_scores(query)``
   Returns one NumPy score per corpus row, including zeros for non-matches.
   This whole-corpus vector is useful for diagnostics, evaluation, or custom
   downstream ranking, but it is not the sparse retrieval path.

``get_batch_scores(query, doc_ids)``
   Scores selected **row positions**.  The argument name ``doc_ids`` here is
   historical: these values are integer positions, not the stable string
   identities supplied to the constructor.  Negative, boolean, and
   out-of-range positions are rejected.

``get_top_n(query, documents, n=5)``
   Maps ranked row positions into a sequence supplied at query time.  The
   sequence length is checked, but its ordering cannot be verified.  Use it
   only while ``documents`` is in exactly the same row order as the corpus used
   to build the index.  Prefer ``doc_ids`` plus ``get_top_ids`` when results can
   cross process, storage, or application boundaries.

Stable identities and result binding
====================================

``doc_ids`` must contain exactly one identity per corpus row and identities must
be unique.  Values are converted to strings when the index is built.  The
resulting tuple is exposed as ``index.doc_ids`` in row order.

This binding prevents a common retrieval error: a row offset from one index
being accidentally applied to a reordered document sequence.  It also makes
saved indexes useful without requiring the original document objects to be
loaded merely to identify a hit.

Counts such as ``n`` are non-negative integers.  ``n=0`` is valid and returns an
empty result; negative values and booleans are rejected.  Asking
``get_top_ids`` for identities on an index built without ``doc_ids`` raises an
error instead of guessing.

Persistence and reproducibility
===============================

``save(directory)`` publishes index state as a generation behind
``current.json``.  ``load(directory)`` accepts either the publication root or a
specific generation directory.  Publication writes a candidate first, reloads
it, then moves the pointer, so a failed save leaves the previously published
generation selected.

A saved index stores derived lexical state rather than the original document
objects, but that does **not** make the artifact non-sensitive.  The JSON state
contains term strings, per-document term frequencies, IDF values, document
lengths, scoring parameters, and ``doc_ids`` when provided.  Protect a saved
index according to the sensitivity of that vocabulary and metadata.

Pre-tokenized indexes can be saved directly::

   index = BM25Okapi(
       [['alpha', 'beta'], ['beta', 'gamma']],
       doc_ids=['a', 'b'],
   )
   generation = index.save('bm25-index')
   restored = BM25Okapi.load('bm25-index')

An index built with a bare callable ``tokenizer=`` cannot be saved because the
callable has no durable analyzer identity.  The implementation has an analyzer
declaration mechanism for reproducible tokenization, but that declaration is
currently not exported from the public ``scikitplot.rank_bm25`` namespace.
Treat that analyzer API as internal until the public contract is resolved; do
not build application code on its private import path.

Build identity
--------------

``index.identity`` is a SHA-256 digest of the scoring recipe, declared analyzer,
and scoring parameters.  It answers **how** an index was built.  It deliberately
does not identify the corpus contents, so equal ``identity`` values do not mean
two indexes contain the same documents.

Performance choices
===================

The implementation exposes both dense and sparse paths rather than hiding the
trade-off:

* ``get_top_ids`` uses an inverted postings view and only accumulates scores for
  rows containing query terms;
* ``candidate_count`` lets you inspect that candidate set size;
* ``get_scores`` allocates a value for every corpus row; and
* ``workers`` affects corpus tokenization during construction, not BM25 query
  scoring.

The postings map is derived lazily from the index's document-frequency records
and cached after first use.  Choose a larger-scale search backend when your
operational requirements exceed an in-process Python/NumPy lexical index; this
module does not claim distributed indexing, concurrent mutation, or a storage
service.

Failure modes worth handling
============================

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Situation
     - Behavior / fix
   * - Raw string passed as a corpus row without ``tokenizer=``
     - Refused.  Pre-tokenize documents or supply a tokenizer.
   * - Bare string used as a query
     - The public contract is token sequences.  Pass ``query.split()`` or your
       matching analysis pipeline.
   * - Empty corpus or corpus with no tokens anywhere
     - Refused because collection statistics cannot be derived.
   * - Duplicate or wrong-length ``doc_ids``
     - Refused at construction time.
   * - Negative/bool/out-of-range row position
     - Refused instead of using Python negative indexing or NumPy boolean-mask
       semantics.
   * - ``get_top_ids`` without constructor ``doc_ids``
     - Refused; use stable IDs or the positional ``get_top_n`` path explicitly.
   * - Query has no indexed terms
     - Sparse identity retrieval returns no hits; dense ``get_scores`` still
       returns a zero vector with one entry per corpus row.
   * - Multiple tokenization workers with an unpicklable tokenizer
     - The multiprocessing error is re-raised with guidance to remove
       ``workers=`` or use a picklable tokenizer.
   * - Saving an index built with a bare callable tokenizer
     - Refused because the tokenizer cannot be identified durably.

Public surface and provenance
=============================

The stable import path is::

   from scikitplot.rank_bm25 import BM25, BM25Okapi, BM25L, BM25Plus

``BM25Okapi``, ``BM25L``, and ``BM25Plus`` are the implemented concrete
recipes.  BM25-Adpt and BM25T appear only as unimplemented source placeholders
and are not public exports.

The Rank-BM25 implementation is adapted from the ``rank_bm25`` project and the
adapted source files retain their Apache-2.0 notice.  The Scikit-Plots partial
distribution records the combined project licensing in its package metadata.
For package ownership and installation boundaries, see
:doc:`../../affiliated/index`.

See also
========

* :ref:`rank_bm25_examples` -- executable, offline gallery examples.
* :doc:`../mcp/index` -- retrieval adapters and hybrid lexical/vector workflows.
* :doc:`../annoy/index` -- approximate nearest-neighbor vector retrieval.
* :doc:`../corpus/index` -- corpus ingestion and retrieval workflows.
* :doc:`../../affiliated/index` -- project-maintained partial distributions.
