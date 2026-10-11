.. _rank-bm25-troubleshooting:

Troubleshooting
===============

No results
   Inspect tokenization and ``candidate_count``. If no query token exists in the
   vocabulary, sparse retrieval correctly returns an empty result.

Ranking looks wrong
   Confirm query and corpus use the same normalization/tokenization policy and
   that the intended BM25 recipe/parameters are in use.

Wrong document attached to a hit
   Prefer constructor ``doc_ids`` plus ``get_top_ids``. ``get_top_n`` cannot
   verify that a later document sequence has the original index row order.

Save rejected
   A bare callable tokenizer has no durable analyzer declaration. Use a
   persistable analysis configuration or keep the index ephemeral.

Parameter rejected
   ``k1`` must be finite and positive; ``b`` is in ``[0, 1]``; delta/epsilon are
   finite and non-negative; booleans are rejected as numeric parameters/counts.
