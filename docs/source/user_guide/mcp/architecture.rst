.. _mcp-architecture:

Architecture
============

.. code-block:: text

   sources -> Corpus/index/retriever -> RetrievedChunk / RetrievalOutcome
                                      |
                                      v
                              scikitplot.mcp adapter
                                      |
                                      v
                         official MCP SDK / transport

The package intentionally has no in-package JSON-RPC implementation branded as
MCP. Protocol behavior belongs to the supported official SDK.

The dependency-light core contains retrieval contracts, hybrid fusion, outcomes
and demo retrieval. Pydantic-backed output models and server construction are
resolved lazily on first access.
