.. _rank_bm25_examples:

Rank-BM25
=========

.. currentmodule:: scikitplot.rank_bm25

Examples for :mod:`~scikitplot.rank_bm25` are arranged as a learning path for
lexical retrieval rather than as isolated API demonstrations.  They use only
small in-memory corpora, make no network requests, and keep result identities
bound to the index whenever a result could leave the process.

The examples intentionally avoid inventing a plotting layer for a text-ranking
API.  Sphinx-Gallery still uses ``plot_*.py`` filenames, but these examples
render their useful output as code and text.

Install
-------

The full Scikit-Plots distribution already includes Rank-BM25::

    python -m pip install scikit-plots

For a smaller installation that owns only the Rank-BM25 package and the
Scikit-Plots core dependencies::

    python -m pip install scikit-plots-rank-bm25

Do not install the full and partial distributions into the same environment;
they own overlapping ``scikitplot`` package files.  See
:ref:`affiliated-packages-index` for the partial-distribution
model.

Start here
----------

1. **Quick start and stable identities** — build a :class:`BM25Okapi` index,
   retrieve by document ID, see deterministic tie ordering, and verify that an
   out-of-vocabulary query returns no sparse hits.
2. **Choose a recipe** — run the same corpus through :class:`BM25Okapi`,
   :class:`BM25L`, and :class:`BM25Plus`; inspect the versioned recipe identity
   and compare rankings without treating one score scale as interchangeable
   with another.
3. **Sparse retrieval versus dense diagnostics** — use
   :meth:`BM25.candidate_count`, :meth:`BM25.get_top_ids`, and
   :meth:`BM25.get_scores` for the jobs they actually perform.
4. **Save and reload an index** — publish a pre-tokenized index to a temporary
   directory, reload it, and verify that identities and retrieval survive the
   round trip.

Which example should I use?
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Goal
     - Example
   * - Add lexical search to an application
     - :ref:`sphx_glr_auto_examples_rank_bm25_plot_rank_bm25_quickstart_script.py`
   * - Decide among Okapi, BM25L, and BM25+
     - :ref:`sphx_glr_auto_examples_rank_bm25_plot_rank_bm25_recipes_script.py`
   * - Understand candidate filtering and whole-corpus scores
     - :ref:`sphx_glr_auto_examples_rank_bm25_plot_rank_bm25_sparse_dense_script.py`
   * - Persist an index across process starts
     - :ref:`sphx_glr_auto_examples_rank_bm25_plot_rank_bm25_persistence_script.py`

What the gallery treats as the contract
----------------------------------------

The examples follow the current public behavior rather than assumptions from
other BM25 libraries:

* corpus rows are token sequences unless ``tokenizer=`` is supplied;
* stable ``doc_ids`` are preferred over caller-owned row offsets;
* :meth:`BM25.get_top_ids` uses sparse candidate retrieval and an unknown term
  can therefore return an empty result;
* :meth:`BM25.get_scores` is a dense diagnostic vector with one value per
  corpus row;
* equal retrieval scores are ordered by original corpus position;
* exact floating-point values are not used as cross-recipe compatibility
  assertions; and
* persistence examples use pre-tokenized data because the public namespace does
  not currently expose the durable analyzer declaration used by the internal
  tokenizer persistence path.

The corpus is deliberately tiny.  These are correctness and workflow examples,
not performance benchmarks or evidence that an in-process NumPy index is the
right backend for every corpus size.

.. seealso::

   * :ref:`rank-bm25-index` — concepts,
     contracts, failure modes, persistence boundaries, and performance choices.
   * :ref:`corpus-index` — corpus ingestion and
     retrieval workflows.
   * :ref:`annoy-index` — approximate
     nearest-neighbor vector retrieval.
