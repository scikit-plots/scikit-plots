.. _mcp-troubleshooting:

Troubleshooting
===============

Import succeeds, server symbol fails
   The base retrieval tier is intentionally lighter than the server tier. Check
   that the optional pydantic/MCP dependencies match the supported Python
   version and package range.

Hybrid result is degraded
   Inspect the per-leg records in :class:`RetrievalOutcome`. Do not collapse a
   degraded multi-retriever result into ordinary success if an application needs
   complete retrieval coverage.

No results
   Verify source/index population, query tokenization/embedding, requested
   ``k``, and document-id binding before changing fusion weights.

Capability mismatch
   Compare advertised and effective capabilities rather than inferring runtime
   readiness from whether a module imports.
