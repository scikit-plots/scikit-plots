"""
Create bounded, explicit metadata for local controls on static galleries.

Only fields requested by filter, sort, or search options cross into HTML. Values
are reduced to JSON-safe scalars/lists, serialized with strict JSON, and escaped
before entering a raw HTML carrier. The module performs no network access and
contains no client-side behavior; ``assets.py`` owns progressive enhancement.
"""

from __future__ import annotations

import datetime
import html
import json
import math
import re

from docutils import nodes

from .select import get_field


def collection_id(argument):
    """Validate a stable per-page identity for optional browser persistence."""
    value = argument.strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", value):
        raise ValueError(
            "collection-id must start with a letter and contain at most 64 letters, digits, underscores or hyphens"
        )
    return value


def field_names(argument):
    """Parse a nonempty, ordered set of safe field paths from an option."""
    names = tuple(
        dict.fromkeys(part.strip() for part in argument.split(",") if part.strip())
    )
    if not names or any(
        not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*", name)
        for name in names
    ):
        raise ValueError("expected comma-separated field names, e.g. category,tags")
    return names


def _value(value):
    """Convert supported metadata to finite, JSON-safe browser values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None]
    return str(value)


def _field_tuple(value):
    """Normalize parsed field-name options to one ordered tuple."""
    if not value:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(value)


def record_for_browser(item, options):
    """
    Return searchable metadata while excluding unrelated record fields.

    Generic gallery cards keep their link alternative in the default search
    corpus because authors often use it as descriptive content. Typed adapters
    may provide ``_sk_collection_search_base`` to keep accessibility-only link
    prose (for example ``Open … on YouTube``) out of search without weakening
    the accessible link itself.
    """
    facets = _field_tuple(options.get("filter-fields", ()))
    sorts = _field_tuple(options.get("sort-fields", ("title",)))
    extra = _field_tuple(options.get("search-fields", ()))
    fields = {
        key: _value(get_field(item, key)) for key in dict.fromkeys((*facets, *sorts))
    }
    base = item.get("_sk_collection_search_base")
    if base is None:
        search_values = [item.get("title", ""), item.get("link-alt", "")]
    elif isinstance(base, (list, tuple)):
        search_values = list(base)
    else:
        search_values = [base]
    search_values += [get_field(item, key) for key in dict.fromkeys((*facets, *extra))]
    return {
        "title": str(item.get("title", "")),
        "fields": fields,
        "search": " ".join(str(v) for v in search_values if v is not None),
    }


def metadata_node(records, options):
    """Return an escaped, versioned HTML metadata carrier for one gallery."""
    payload = {
        "version": 1,
        "interactive": "interactive" in options,
        "facets": list(_field_tuple(options.get("filter-fields", ()))),
        "sorts": list(_field_tuple(options.get("sort-fields", ("title",)))),
        "records": records,
        "collectionId": options.get("collection-id", ""),
    }
    encoded = html.escape(
        json.dumps(payload, ensure_ascii=False, allow_nan=False), quote=True
    )
    return nodes.raw(
        "",
        '<span hidden class="sk-collection-data">' + encoded + "</span>",
        format="html",
    )
