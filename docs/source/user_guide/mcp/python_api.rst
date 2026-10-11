.. Do not use here directive ``autosummary``

.. _mcp-python-api:

Python API
==========

Lazy server tier
----------------

``SearchService``, ``SearchDocsOutput``, ``CitationOutput`` and ``create_server``
are public but loaded lazily so a base installation can use retrieval without
pydantic/MCP SDK imports.
