.. _rank-bm25-performance-and-scaling:

Performance and scaling
=======================

Sparse retrieval uses an inverted postings view and accumulates scores only for
rows containing query terms. Dense ``get_scores`` must touch/allocate one score
per corpus row.

The postings map is derived lazily and cached. Corpus tokenization can use
multiple workers, but query scoring is not turned into a distributed service by
that setting.

Move to a larger search system when you need distributed indexing, concurrent
mutation, service-level persistence, very large corpora, advanced analyzers, or
hybrid/vector retrieval at a scale that exceeds an in-process NumPy/Python
index. BM25 remains useful as a lexical leg in a hybrid retriever.
