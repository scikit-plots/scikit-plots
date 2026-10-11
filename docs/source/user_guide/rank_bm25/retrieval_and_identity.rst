.. _rank-bm25-retrieval-and-identity:

Retrieval and identity
======================

``get_top_ids(query, n)``
   Sparse retrieval over rows containing at least one query term. Requires
   constructor ``doc_ids`` and returns ``(doc_id, score)`` pairs.

``candidate_count(query)``
   Reports how many rows the sparse query can score.

``get_scores(query)``
   Allocates a dense score vector for every corpus row; useful for diagnostics
   or evaluation rather than routine sparse retrieval.

``get_batch_scores(query, doc_ids)``
   Historical parameter name: values are integer row positions, not stable
   string identities.

``get_top_n(query, documents, n)``
   Maps row positions into a caller-provided sequence. Its length is validated
   but its ordering cannot be proven; prefer ``doc_ids`` for durable binding.

Ties are deterministic by original corpus row.
