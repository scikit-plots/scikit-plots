.. _mcp-retrieval-and-fusion:

Retrieval and fusion
====================

``DocsRetriever`` is the retrieval protocol. Concrete implementations include a
small in-memory BM25 demo retriever, Corpus/vector adapters, and hybrid
composition.

:func:`reciprocal_rank_fusion` combines ranked lists without pretending their
raw scores are directly comparable. :class:`HybridRetriever` records per-leg
status and can operate in fail-soft or strict modes.

:class:`RetrievalOutcome` distinguishes states such as success, empty,
degraded, failed and cancelled. Preserve that distinction in application code:
an empty result is not the same as a failed retrieval backend.

Result limits and chunk-size limits are enforced at the MCP boundary so a
retriever cannot accidentally turn one query into unbounded output.

Reading retrieval results as evidence
-------------------------------------

A retrieval rank means “this passage matched the query under this retriever.” It
does **not** mean “this claim is true.” Keep these signals separate:

* rank/score -- relevance evidence, not truth probability;
* ``source_uri``/citation -- provenance, not authority;
* agreement across sources -- corroboration worth checking, not guaranteed
  independence; and
* an empty result -- no supporting passage found in the searched corpus, not
  proof that a claim is false.

A robust workflow retrieves candidates, inspects source/version/date metadata,
compares independent evidence, keeps conflicts visible, then states conclusions
in proportion to the evidence.
