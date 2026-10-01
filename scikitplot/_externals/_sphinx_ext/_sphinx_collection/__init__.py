# scikitplot/_externals/_sphinx_ext/_sphinx_collection/__init__.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Domain-agnostic collection infrastructure, lazily exported.

Filtering, sorting, grouping, browser metadata/assets, and section rendering are
shared by ``gallery-grid`` and typed adapters such as ``youtube-gallery``.
The package initializer intentionally imports nothing eagerly: data-only helpers
(``_yaml`` and the YouTube catalog model/sync path) must remain usable without
Sphinx/docutils installed.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "COLLECTION_UI_CONTRACT": (".contract", "COLLECTION_UI_CONTRACT"),
    "CONTRACT_CLASS": (".contract", "CONTRACT_CLASS"),
    "STATUS_CLASS": (".contract", "STATUS_CLASS"),
    "CONTAINER_CLASS": (".assets", "CONTAINER_CLASS"),
    "SEARCHABLE_CLASS": (".assets", "SEARCHABLE_CLASS"),
    "SEARCH_VARIANTS": (".assets", "SEARCH_VARIANTS"),
    "collection_asset_revision": (".setup", "collection_asset_revision"),
    "collection_assets_outdated": (".setup", "collection_assets_outdated"),
    "ensure_assets": (".setup", "ensure_assets"),
    "register_collection_asset_revision": (
        ".setup",
        "register_collection_asset_revision",
    ),
    "remember_collection_asset_revision": (
        ".setup",
        "remember_collection_asset_revision",
    ),
    "verify_collection_assets": (".setup", "verify_collection_assets"),
    "SECTION_STYLES": (".sections", "SECTION_STYLES"),
    "render_sections": (".sections", "render_sections"),
    "sections_allowed": (".sections", "sections_allowed"),
    "FilterError": (".select", "FilterError"),
    "Selection": (".select", "Selection"),
    "apply_selection": (".select", "apply_selection"),
    "group_records": (".select", "group_records"),
    "has_field": (".select", "has_field"),
    "parse_filter": (".select", "parse_filter"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Resolve one public symbol on demand and cache it in this module."""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
