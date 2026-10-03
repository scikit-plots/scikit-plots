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

from .contract import (
    STATUS_CLASS,
    STATUS_SOURCE_ATTRIBUTE,
    STATUS_SOURCE_DOCUMENT,
)
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


_STATUS_NODE_SOURCE_KEY = "sk_collection_status_source"


def status_node(count):
    """
    Return the build-time live-result placeholder for one enhanced gallery.

    The status belongs to the collection root, never the controls shell.  It is
    emitted by the document builder so the controls -> status -> results
    relationship is structural before JavaScript runs, matching AI Learn's
    explorer contract.  JavaScript only updates/unhides this existing node.
    ``hidden`` keeps the static/no-JavaScript gallery free of enhancement-only
    result text.

    ``nodes.raw`` stores its first argument as ``rawsource`` and its second as
    rendered text.  Keep both populated and attach a Docutils-side sentinel so
    sibling directives can verify ownership structurally instead of scraping
    serialized HTML.
    """
    count = max(0, int(count))
    markup = (
        f'<p hidden class="{STATUS_CLASS}" role="status" '
        'aria-live="polite" aria-atomic="true" '
        f'{STATUS_SOURCE_ATTRIBUTE}="{STATUS_SOURCE_DOCUMENT}">'
        f"{count} of {count} cards</p>"
    )
    node = nodes.raw(markup, markup, format="html")
    node[_STATUS_NODE_SOURCE_KEY] = STATUS_SOURCE_DOCUMENT
    return node


def is_document_status_node(node):
    """
    Return whether *node* is the shared document-owned status placeholder.

    The structured node sentinel is authoritative.  The HTML-marker fallback
    accepts a serialized/raw node from an older compatible producer, but never
    depends on ``rawsource`` being populated.
    """
    if not isinstance(node, nodes.raw):
        return False
    if node.get(_STATUS_NODE_SOURCE_KEY) == STATUS_SOURCE_DOCUMENT:
        return True
    markup = node.astext()
    return (
        f'class="{STATUS_CLASS}"' in markup
        and f'{STATUS_SOURCE_ATTRIBUTE}="{STATUS_SOURCE_DOCUMENT}"' in markup
    )


def metadata_node(records, options):
    """Return an escaped, versioned HTML metadata carrier for one gallery."""
    payload = {
        "version": 1,
        "interactive": "interactive" in options,
        "searchVariant": options.get("search-variant", "pill-overflow"),
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
