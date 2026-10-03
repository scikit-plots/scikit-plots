"""
Shared search-control presentation contract for Sphinx collection UIs.

The module is intentionally dependency-light. Directive packages reuse the
collection browser's canonical presentation names plus one conflict-resolution
policy, without importing one another or duplicating option parsing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._sphinx_collection.assets import SEARCH_VARIANTS

SEARCH_VARIANT_KEYS = ("search-variant", "search_variant")


def search_variant_option(argument: str | None) -> str | None:
    """
    Validate an optional search variant directive option.

    ``None`` is preserved so the same converter can back both a traditional
    valueless activation flag (``:interactive:``) and the shorthand
    ``:interactive: classic``.  Sphinx/Docutils option validators receive
    ``None`` when an option has no argument.
    """

    if argument is None:
        return None
    value = str(argument).strip().lower()
    if not value:
        return None
    if value not in SEARCH_VARIANTS:
        raise ValueError("search variant must be 'pill-overflow' or 'classic'")
    return value


def resolve_search_variant(
    options: Mapping[str, Any],
    default: str,
    *,
    activation_keys: Sequence[str] = ("interactive", "searchable"),
) -> str:
    """
    Resolve one unambiguous per-directive search presentation.

    Dedicated ``search-variant`` / ``search_variant`` options and optional
    values carried by activation options all participate.  Supplying the same
    value more than once is harmless; conflicting values fail closed instead
    of depending on incidental option order.
    """

    candidates: list[tuple[str, str]] = []
    for key in (*SEARCH_VARIANT_KEYS, *activation_keys):
        if key not in options:
            continue
        value = options.get(key)
        if value in (None, ""):
            continue
        normalized = search_variant_option(str(value))
        if normalized:
            candidates.append((key, normalized))

    distinct = tuple(dict.fromkeys(value for _key, value in candidates))
    if len(distinct) > 1:
        detail = ", ".join(f"{key}={value}" for key, value in candidates)
        raise ValueError(
            "conflicting search variants were supplied ("
            + detail
            + "); choose one presentation"
        )

    resolved = distinct[0] if distinct else str(default).strip().lower()
    if resolved not in SEARCH_VARIANTS:
        raise ValueError("search variant must be 'pill-overflow' or 'classic'")
    return resolved
