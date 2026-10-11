.. _mcp-python-api:

Python API
==========

Dependency-light core
---------------------

.. autosummary::
   :nosignatures:

   DocsRetriever
   RetrievedChunk
   SearchCoordinator
   RetrievalOutcome
   LegRecord
   HybridRetriever
   Bm25Retriever
   CorpusIndexRetriever
   CorpusAnnoyRetriever
   InMemoryBm25Retriever
   reciprocal_rank_fusion
   server_capabilities
   effective_server_capabilities
   server_runtime_status

Lazy server tier
----------------

``SearchService``, ``SearchDocsOutput``, ``CitationOutput`` and ``create_server``
are public but loaded lazily so a base installation can use retrieval without
pydantic/MCP SDK imports.
