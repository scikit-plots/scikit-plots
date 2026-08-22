# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Optional, read-only adapters that expose scikit-plots documentation retrieval
to agent frameworks.

Design rules (see ``_maintenance/RULESET.md``):

* **Read-only.** Adapters expose only ``search_docs``; no writes, no graph logic.
* **Optional.** A framework is imported lazily; its absence raises an actionable
  error, and the framework-neutral core still works.
* **No inverted dependency.** Importing this package must not require the MCP SDK
  or any agent framework. The in-process toolkit is backed by the Tier-L
  coordinator (``SearchCoordinator``), which imports neither the MCP SDK nor
  ``pydantic``, so it genuinely runs on the Legacy Retrieval tier.
"""  # ruff: ignore[missing-blank-line-after-summary]

__all__ = []
