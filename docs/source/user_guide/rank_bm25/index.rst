.. currentmodule:: scikitplot.rank_bm25

.. _rank-bm25-index:

BM25 lexical ranking
====================

``scikitplot.rank_bm25`` provides in-process lexical ranking with Okapi BM25, BM25L and BM25Plus. It makes tokenization policy, stable document identity, sparse retrieval, persistence and recipe identity explicit rather than hiding them behind a generic search interface.

.. important::

   Text analysis is part of index correctness. Tokenize documents and queries with the same policy, bind rows to stable ``doc_ids`` when results leave the index process, and protect saved lexical state according to the sensitivity of the indexed vocabulary.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   recipes_and_tokenization
   retrieval_and_identity
   persistence_and_reproducibility
   performance_and_scaling
   python_api
   troubleshooting
   usage_scenarios

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Getting started
      :link: rank-bm25-getting-started
      :link-type: ref

      Build an index with stable doc_ids and retrieve lexical matches.

   .. grid-item-card:: Recipes and tokenization
      :link: rank-bm25-recipes-and-tokenization
      :link-type: ref

      Choose Okapi/BM25L/BM25Plus and keep document/query analysis consistent.

   .. grid-item-card:: Retrieval and identity
      :link: rank-bm25-retrieval-and-identity
      :link-type: ref

      Use sparse top-id retrieval, dense scores, stable ties, and candidate diagnostics correctly.

   .. grid-item-card:: Persistence and reproducibility
      :link: rank-bm25-persistence-and-reproducibility
      :link-type: ref

      Save generation-based state, understand build identity, and protect derived lexical artifacts.

   .. grid-item-card:: Performance and scaling
      :link: rank-bm25-performance-and-scaling
      :link-type: ref

      Sparse postings, dense vectors, tokenizer workers, and when to move to a larger search system.

   .. grid-item-card:: Python API
      :link: rank-bm25-python-api
      :link-type: ref

      BM25 classes, recipe/identity properties, validation behavior, and internal analyzer boundary.

   .. grid-item-card:: Troubleshooting
      :link: rank-bm25-troubleshooting
      :link-type: ref

      Tokenization mismatch, OOV queries, doc-id binding, persistence, and parameter validation.

.. grid:: 1 1 1 1
   :gutter: 2

   .. grid-item-card:: Usage Scenarios
      :link: rank_bm25-usage-scenarios-index
      :link-type: ref

Which page should I use?
------------------------

Start with **Getting started**. Then choose the page that owns the decision
you are making: architecture and lifecycle, configuration, persistence,
performance, public API, or limits. The guide intentionally separates those
concerns so examples do not accidentally become policy.

Examples
--------

See :ref:`rank_bm25_examples` for executable examples that accompany this guide.

See also
--------

* :ref:`mcp-index`
* :ref:`corpus-index`
