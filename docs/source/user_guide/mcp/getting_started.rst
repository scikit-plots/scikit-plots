.. _mcp-getting-started:

Getting started
===============

Start with retrieval without a server:

.. code-block:: python

   from scikitplot.mcp import builtin_demo_retriever

   retriever = builtin_demo_retriever()
   hits = retriever.search("local documentation search", k=3)

This path does not require the MCP SDK. It lets you validate chunk identity,
ranking, citations and truncation before protocol/network concerns enter the
system.

Only create a server after the retrieval result contract is satisfactory. The
server exports are lazy and may require pydantic and the ``[mcp]`` optional
dependency tier.
