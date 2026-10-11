.. _mcp-corpus-and-annoy:

Corpus and Annoy
================

Use Corpus when evidence originates from real files, URLs, archives or media.
MCP should receive retrieval-ready evidence rather than become another ingestion
framework.

``CorpusIndexRetriever`` composes a Corpus-style index with optional embeddings
and supports lexical/vector/hybrid matching. ``CorpusAnnoyRetriever`` adapts an
embedder plus vector index and document lookup function.

Annoy results are approximate similarity candidates. Preserve source metadata
and citations through the adapter so a client can trace a returned chunk back
to the evidence that produced it.
