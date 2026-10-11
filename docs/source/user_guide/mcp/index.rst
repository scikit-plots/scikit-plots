.. currentmodule:: scikitplot.mcp

.. _mcp-index:

Model Context Protocol Retrieval
================================

``scikitplot.mcp`` is a documentation-retrieval and MCP adapter layer. The base import remains independent of the MCP SDK and pydantic; retrieval can be built and tested first, then exposed through the official MCP server layer when that optional tier is installed.

.. important::

   MCP owns wire/adapter semantics, not evidence truth. Corpus owns ingestion and retrieval data semantics, Annoy owns vector mechanics, and the official MCP SDK owns the protocol implementation. Keep those boundaries explicit.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   architecture
   retrieval_and_fusion
   corpus_and_annoy
   capabilities_and_runtime
   server_cli_security
   deployment_and_operations
   python_api
   troubleshooting
   usage_scenarios

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Getting started
      :link: mcp-getting-started
      :link-type: ref

      Use the dependency-light retrieval core before adding an MCP server.

   .. grid-item-card:: Architecture
      :link: mcp-architecture
      :link-type: ref

      Separate retrieval semantics, MCP wire/server semantics, Corpus, and vector index ownership.

   .. grid-item-card:: Retrieval and fusion
      :link: mcp-retrieval-and-fusion
      :link-type: ref

      BM25/demo retrieval, hybrid retrieval, reciprocal-rank fusion, statuses, and deterministic truncation.

   .. grid-item-card:: Corpus and Annoy
      :link: mcp-corpus-and-annoy
      :link-type: ref

      Compose Corpus evidence and vector search without moving ownership into MCP.

   .. grid-item-card:: Capabilities and runtime
      :link: mcp-capabilities-and-runtime
      :link-type: ref

      Inspect advertised/effective server capabilities and runtime status.

   .. grid-item-card:: Server, CLI, and security
      :link: mcp-server-cli-security
      :link-type: ref

      Lazy SDK/pydantic server layer, stdio/network transport, validation, limits, and remote-bind grants.

   .. grid-item-card:: Deployment and operations
      :link: mcp-deployment-and-operations
      :link-type: ref

      Local stdio/HTTP, Docker, remote gateways, health checks, and runtime controls.

   .. grid-item-card:: Python API
      :link: mcp-python-api
      :link-type: ref

      Core contracts, retrievers, outcomes, server exports, and validation helpers.

   .. grid-item-card:: Troubleshooting
      :link: mcp-troubleshooting
      :link-type: ref

      Optional SDKs, capability mismatches, degraded retrieval legs, and server startup.

.. grid:: 1 1 1 1
   :gutter: 2

   .. grid-item-card:: Usage Scenarios
      :link: mcp-usage-scenarios-index
      :link-type: ref

      Use the dependency-light retrieval core before adding an MCP server.
Which page should I use?
------------------------

Start with **Getting started**. Then choose the page that owns the decision
you are making: architecture and lifecycle, configuration, persistence,
performance, public API, or limits. The guide intentionally separates those
concerns so examples do not accidentally become policy.

Examples
--------

See :ref:`mcp_examples` for executable examples that accompany this guide.

See also
--------

* :ref:`corpus-index`
* :ref:`annoy-index`
* :ref:`rank-bm25-index`

Further reading
---------------

.. seealso::
  * https://github.com/modelcontextprotocol/python-sdk
  * https://github.com/semantica-agi/semantica
