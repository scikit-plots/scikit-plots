.. _mcp-server-cli-security:

Server, CLI, and security
=========================

The server layer is optional. Accessing ``SearchService``, ``SearchDocsOutput``,
``CitationOutput`` or ``create_server`` loads the pydantic-backed server module;
creating a real MCP server additionally needs the supported MCP SDK.

Prefer ``stdio`` for local tool integration when a network listener is not
required. Remote/network binding is a separate security decision and must obey
the server's explicit grant/validation policy.

The server validates counts, indexes, document identifiers, query/resource
sizes and structured truncation. Those limits protect the adapter boundary; they
do not sanitize arbitrary source content or make retrieved instructions safe to
execute.
