"""
Stable UI ownership contract shared by collection-style Sphinx adapters.

This module is deliberately dependency-free so every layer can assert the same
contract without importing Sphinx or docutils.  The live result status belongs
to the document collection root as a sibling immediately after the controls
shell; it is never a child of ``.sk-collection-controls``.
"""

from __future__ import annotations

COLLECTION_UI_CONTRACT = "controls-status-results-v4"
CONTRACT_CLASS = "sk-collection-controls-status-results-v4"
STATUS_CLASS = "sk-collection-status"
STATUS_SOURCE_ATTRIBUTE = "data-sk-collection-status-source"
STATUS_SOURCE_DOCUMENT = "document"
STATUS_PLACEMENT_ATTRIBUTE = "data-sk-collection-status-placement"
STATUS_PLACEMENT_SIBLING = "sibling"

__all__ = [
    "COLLECTION_UI_CONTRACT",
    "CONTRACT_CLASS",
    "STATUS_CLASS",
    "STATUS_PLACEMENT_ATTRIBUTE",
    "STATUS_PLACEMENT_SIBLING",
    "STATUS_SOURCE_ATTRIBUTE",
    "STATUS_SOURCE_DOCUMENT",
]
