"""
Tests for ``_sphinx_collection._browser``: option validators, the bounded
browser record, the document-owned status node and the escaped metadata
carrier.
"""

from __future__ import annotations

import datetime as dt
import html
import importlib
import json
import re
import sys
from pathlib import Path

import pytest
from docutils import nodes

ROOT = Path(__file__).resolve().parents[2]
HOST_ROOT = ROOT.parents[2]


def _module(name: str):
    externals = str(HOST_ROOT / "scikitplot" / "_externals")
    if externals not in sys.path:
        sys.path.insert(0, externals)
    return importlib.import_module("_sphinx_ext._sphinx_collection." + name)


@pytest.fixture(scope="module")
def browser():
    return _module("_browser")


@pytest.fixture(scope="module")
def contract():
    return _module("contract")


def _payload(node):
    """Decode the JSON carried by a metadata node the way a browser would."""
    markup = node.astext()
    match = re.fullmatch(
        r'<span hidden class="sk-collection-data">([^<>]*)</span>', markup
    )
    assert match, markup
    return json.loads(html.unescape(match.group(1)))


# -- collection_id ------------------------------------------------------------


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        pytest.param("a", "a", id="one-letter"),
        pytest.param("Videos", "Videos", id="word"),
        pytest.param("my-gallery_2", "my-gallery_2", id="hyphen-underscore-digit"),
        pytest.param("  padded  ", "padded", id="trimmed"),
        pytest.param("a" * 64, "a" * 64, id="max-length-64"),
    ],
)
def test_collection_id_accepts(browser, argument, expected):
    assert browser.collection_id(argument) == expected


@pytest.mark.parametrize(
    "argument",
    [
        pytest.param("", id="empty"),
        pytest.param("   ", id="blank"),
        pytest.param("a" * 65, id="too-long-65"),
        pytest.param("a" * 10_000, id="very-long"),
        pytest.param("1abc", id="leading-digit"),
        pytest.param("-abc", id="leading-hyphen"),
        pytest.param("_abc", id="leading-underscore"),
        pytest.param("a b", id="inner-space"),
        pytest.param("a.b", id="dot"),
        pytest.param("a/b", id="slash"),
        pytest.param("../etc", id="path-traversal"),
        pytest.param("a\nb", id="inner-newline"),
        pytest.param("été", id="non-ascii-letter"),
        pytest.param('a"onload="x', id="attribute-breakout"),
        pytest.param("<script>", id="markup"),
        pytest.param("a\x00", id="nul"),
    ],
)
def test_collection_id_rejects(browser, argument):
    with pytest.raises(ValueError, match="collection-id must start with a letter"):
        browser.collection_id(argument)


# -- field_names --------------------------------------------------------------


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        pytest.param("category", ("category",), id="single"),
        pytest.param("category,tags", ("category", "tags"), id="pair"),
        pytest.param(" b , a ", ("b", "a"), id="order-kept-and-trimmed"),
        pytest.param("a,b,a,b", ("a", "b"), id="duplicates-collapsed"),
        pytest.param("a,,b,", ("a", "b"), id="empty-parts-skipped"),
        pytest.param("author.name", ("author.name",), id="dotted"),
        pytest.param("link-alt,_x,a1", ("link-alt", "_x", "a1"), id="allowed-chars"),
    ],
)
def test_field_names_accepts(browser, argument, expected):
    result = browser.field_names(argument)
    assert result == expected
    assert isinstance(result, tuple)


@pytest.mark.parametrize(
    "argument",
    [
        pytest.param("", id="empty"),
        pytest.param(" , ,", id="only-separators"),
        pytest.param("a..b", id="empty-segment"),
        pytest.param(".a", id="leading-dot"),
        pytest.param("a.", id="trailing-dot"),
        pytest.param("1a", id="leading-digit"),
        pytest.param("a b", id="space-inside"),
        pytest.param("ok,<script>", id="markup-among-valid"),
        pytest.param("a;b", id="semicolon"),
        pytest.param("été", id="non-ascii"),
        pytest.param("a\nb", id="inner-newline"),
        pytest.param("__proto__.x y", id="prototype-junk"),
    ],
)
def test_field_names_rejects(browser, argument):
    with pytest.raises(ValueError, match="comma-separated field names"):
        browser.field_names(argument)


@pytest.mark.parametrize(
    "validator",
    [
        pytest.param("collection_id", id="collection-id"),
        pytest.param("field_names", id="field-names"),
    ],
)
def test_option_validators_reject_a_missing_value_with_value_error(browser, validator):
    # docutils calls an option converter with ``None`` for ``:option:`` written
    # without a value and only turns ValueError/TypeError into a directive error.
    with pytest.raises((ValueError, TypeError)):
        getattr(browser, validator)(None)


# -- record_for_browser -------------------------------------------------------

ITEM = {
    "title": "Intro to Plots",
    "link-alt": "Open the intro",
    "category": "tutorial",
    "tags": ["python", None, "charts"],
    "stars": 12,
    "published": dt.date(2024, 5, 1),
    "author": {"name": "Ada", "email": "ada@example.invalid"},
    "secret": "do-not-leak",
}


def test_record_defaults_expose_only_title(browser):
    record = browser.record_for_browser(ITEM, {})
    assert record == {
        "title": "Intro to Plots",
        "fields": {"title": "Intro to Plots"},
        "search": "Intro to Plots Open the intro",
    }


def test_record_contains_only_requested_fields(browser):
    options = {
        "filter-fields": ("category", "tags"),
        "sort-fields": ("stars", "published"),
        "search-fields": ("author.name",),
    }
    record = browser.record_for_browser(ITEM, options)
    assert sorted(record) == ["fields", "search", "title"]
    assert record["fields"] == {
        "category": "tutorial",
        "tags": ["python", "charts"],
        "stars": 12,
        "published": "2024-05-01",
    }
    assert list(record["fields"]) == ["category", "tags", "stars", "published"]
    assert "Ada" in record["search"]
    assert "tutorial" in record["search"]
    serialized = json.dumps(record)
    assert "do-not-leak" not in serialized
    assert "ada@example.invalid" not in serialized


def test_search_fields_are_searchable_but_not_exposed_as_fields(browser):
    record = browser.record_for_browser(ITEM, {"search-fields": ("secret",)})
    assert "do-not-leak" in record["search"]
    assert "secret" not in record["fields"]


def test_sort_only_fields_are_not_added_to_the_search_corpus(browser):
    record = browser.record_for_browser(ITEM, {"sort-fields": ("stars", "category")})
    assert record["fields"] == {"stars": 12, "category": "tutorial"}
    assert record["search"] == "Intro to Plots Open the intro"


@pytest.mark.parametrize(
    "option",
    [
        pytest.param("category, tags", id="comma-string"),
        pytest.param(("category", "tags"), id="tuple"),
        pytest.param(["category", "tags"], id="list"),
    ],
)
def test_field_options_accept_strings_and_sequences(browser, option):
    record = browser.record_for_browser(ITEM, {"filter-fields": option})
    assert list(record["fields"]) == ["category", "tags", "title"]


@pytest.mark.parametrize(
    "empty",
    [
        pytest.param("", id="empty-string"),
        pytest.param((), id="empty-tuple"),
        pytest.param(None, id="none"),
    ],
)
def test_empty_field_options_expose_nothing(browser, empty):
    options = {"filter-fields": empty, "sort-fields": empty, "search-fields": empty}
    record = browser.record_for_browser(ITEM, options)
    assert record["fields"] == {}
    assert record["search"] == "Intro to Plots Open the intro"


def test_field_named_by_both_filter_and_sort_appears_once(browser):
    options = {"filter-fields": ("category",), "sort-fields": ("category", "title")}
    record = browser.record_for_browser(ITEM, options)
    assert list(record["fields"]) == ["category", "title"]


def test_missing_fields_become_null_not_errors(browser):
    record = browser.record_for_browser(
        {"title": "T"}, {"filter-fields": ("nope", "a.b"), "sort-fields": ()}
    )
    assert record == {"title": "T", "fields": {"nope": None, "a.b": None}, "search": "T "}


def test_record_for_item_without_title(browser):
    record = browser.record_for_browser({}, {})
    assert record["title"] == ""
    assert record["fields"] == {"title": None}
    assert record["search"].strip() == ""


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        pytest.param("Only this", "Only this", id="string-base"),
        pytest.param(["Title", "Channel"], "Title Channel", id="list-base"),
        pytest.param(("Title", None, "X"), "Title X", id="tuple-base-drops-null"),
        pytest.param([], "", id="empty-base"),
    ],
)
def test_search_base_replaces_title_and_link_alt(browser, base, expected):
    item = dict(ITEM, _sk_collection_search_base=base)
    record = browser.record_for_browser(item, {})
    assert record["search"] == expected
    assert "Open the intro" not in record["search"]
    assert record["title"] == "Intro to Plots"


def test_search_base_still_appends_facets_and_extra_fields(browser):
    item = dict(ITEM, _sk_collection_search_base="Base")
    record = browser.record_for_browser(
        item, {"filter-fields": ("category",), "search-fields": ("author.name",)}
    )
    assert record["search"] == "Base tutorial Ada"


def test_record_does_not_mutate_item_or_options(browser):
    item = {"title": "T", "tags": ["a", None], "_sk_collection_search_base": ["x"]}
    options = {"filter-fields": ("tags",)}
    snapshot = (json.dumps(item), dict(options))
    browser.record_for_browser(item, options)
    assert (json.dumps(item), options) == snapshot


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("text", "text", id="string"),
        pytest.param(True, True, id="bool"),
        pytest.param(7, 7, id="int"),
        pytest.param(1.5, 1.5, id="finite-float"),
        pytest.param(float("nan"), None, id="nan"),
        pytest.param(float("inf"), None, id="positive-infinity"),
        pytest.param(float("-inf"), None, id="negative-infinity"),
        pytest.param(dt.date(2024, 1, 2), "2024-01-02", id="date"),
        pytest.param(
            dt.datetime(2024, 1, 2, 3, 4, 5), "2024-01-02T03:04:05", id="datetime"
        ),
        pytest.param(["a", None, 1], ["a", "1"], id="list-stringified-nulls-dropped"),
        pytest.param(("a", "b"), ["a", "b"], id="tuple-becomes-list"),
        pytest.param([], [], id="empty-list"),
        pytest.param({"k": 1}, "{'k': 1}", id="mapping-stringified"),
    ],
)
def test_browser_values_are_json_safe(browser, value, expected):
    record = browser.record_for_browser({"title": "T", "v": value}, {"sort-fields": ("v",)})
    result = record["fields"]["v"]
    assert result == expected
    assert type(result) is type(expected)
    json.dumps(record, allow_nan=False)


def test_unicode_and_long_values_survive(browser):
    title = "Ünïcødé 日本語 🎉 " + "x" * 100_000
    record = browser.record_for_browser({"title": title}, {})
    assert record["title"] == title
    assert record["fields"]["title"] == title


# -- status_node --------------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "shown"),
    [
        pytest.param(0, 0, id="zero"),
        pytest.param(1, 1, id="one"),
        pytest.param(1043, 1043, id="many"),
        pytest.param(-5, 0, id="negative-clamped"),
        pytest.param("7", 7, id="numeric-string"),
        pytest.param(True, 1, id="bool"),
    ],
)
def test_status_node_markup(browser, contract, count, shown):
    node = browser.status_node(count)
    assert isinstance(node, nodes.raw)
    assert node["format"] == "html"
    assert node.astext() == (
        '<p hidden class="sk-collection-status" role="status" '
        'aria-live="polite" aria-atomic="true" '
        'data-sk-collection-status-source="document">'
        f"{shown} of {shown} cards</p>"
    )
    # Both the raw source and the rendered payload are populated.
    assert node.rawsource == node.astext()
    assert contract.STATUS_CLASS in node.astext()


@pytest.mark.parametrize(
    "count",
    [
        pytest.param("<script>", id="markup"),
        pytest.param("many", id="word"),
        pytest.param("", id="empty"),
        pytest.param(None, id="none"),
        pytest.param([3], id="list"),
    ],
)
def test_status_node_refuses_non_numeric_counts(browser, count):
    with pytest.raises((ValueError, TypeError)):
        browser.status_node(count)


def test_status_nodes_are_independent(browser):
    first, second = browser.status_node(1), browser.status_node(2)
    assert first is not second
    assert "1 of 1" in first.astext()
    assert "2 of 2" in second.astext()


# -- is_document_status_node --------------------------------------------------


def test_shared_status_node_is_recognised(browser):
    assert browser.is_document_status_node(browser.status_node(3)) is True


def test_status_node_recognised_after_copy(browser):
    node = browser.status_node(3)
    assert browser.is_document_status_node(node.deepcopy()) is True


def test_sentinel_alone_is_authoritative(browser, contract):
    node = nodes.raw("", "<p>unrelated</p>", format="html")
    node["sk_collection_status_source"] = contract.STATUS_SOURCE_DOCUMENT
    assert browser.is_document_status_node(node) is True


def test_legacy_markup_without_rawsource_is_recognised(browser):
    markup = (
        '<p hidden class="sk-collection-status" '
        'data-sk-collection-status-source="document">1 of 1 cards</p>'
    )
    node = nodes.raw("", markup, format="html")
    assert node.rawsource == ""
    assert browser.is_document_status_node(node) is True


@pytest.mark.parametrize(
    "node",
    [
        pytest.param(nodes.paragraph(text="x"), id="paragraph"),
        pytest.param(nodes.Text("sk-collection-status"), id="text"),
        pytest.param(None, id="none"),
        pytest.param("<p class=\"sk-collection-status\">", id="string"),
        pytest.param(nodes.raw("", "", format="html"), id="empty-raw"),
        pytest.param(nodes.raw("", "<p>other</p>", format="html"), id="unrelated-raw"),
        pytest.param(
            nodes.raw("", '<p class="sk-collection-status">1</p>', format="html"),
            id="class-without-source-marker",
        ),
        pytest.param(
            nodes.raw(
                "",
                '<p data-sk-collection-status-source="document">1</p>',
                format="html",
            ),
            id="source-marker-without-class",
        ),
        pytest.param(
            nodes.raw(
                "",
                '<p class="sk-collection-status" '
                'data-sk-collection-status-source="runtime">1</p>',
                format="html",
            ),
            id="runtime-created-status",
        ),
    ],
)
def test_other_nodes_are_not_status_nodes(browser, node):
    assert browser.is_document_status_node(node) is False


def test_wrong_sentinel_value_falls_back_to_markup(browser):
    node = nodes.raw("", "<p>other</p>", format="html")
    node["sk_collection_status_source"] = "runtime"
    assert browser.is_document_status_node(node) is False


def test_paragraph_wrapping_status_markup_is_not_a_status_node(browser):
    text = browser.status_node(1).astext()
    assert browser.is_document_status_node(nodes.paragraph(text=text)) is False


# -- metadata_node ------------------------------------------------------------


def test_metadata_defaults(browser):
    node = browser.metadata_node([], {})
    assert isinstance(node, nodes.raw)
    assert node["format"] == "html"
    assert _payload(node) == {
        "version": 1,
        "interactive": False,
        "searchVariant": "pill-overflow",
        "facets": [],
        "sorts": ["title"],
        "records": [],
        "collectionId": "",
    }


def test_metadata_reflects_options(browser):
    options = {
        "interactive": None,  # a docutils flag: present with value None
        "search-variant": "classic",
        "filter-fields": ("category", "tags"),
        "sort-fields": "title, stars",
        "collection-id": "videos",
    }
    records = [{"title": "A", "fields": {"category": "x"}, "search": "A x"}]
    payload = _payload(browser.metadata_node(records, options))
    assert payload == {
        "version": 1,
        "interactive": True,
        "searchVariant": "classic",
        "facets": ["category", "tags"],
        "sorts": ["title", "stars"],
        "records": records,
        "collectionId": "videos",
    }


def test_metadata_key_order_is_stable(browser):
    first = browser.metadata_node([{"title": "A"}], {"filter-fields": ("b", "a")})
    second = browser.metadata_node([{"title": "A"}], {"filter-fields": ("b", "a")})
    assert first.astext() == second.astext()
    assert list(_payload(first)) == [
        "version",
        "interactive",
        "searchVariant",
        "facets",
        "sorts",
        "records",
        "collectionId",
    ]


@pytest.mark.parametrize(
    "hostile",
    [
        pytest.param("</span><script>alert(1)</script>", id="closing-tag-and-script"),
        pytest.param('" onmouseover="alert(1)', id="double-quote-breakout"),
        pytest.param("' onfocus='alert(1)", id="single-quote-breakout"),
        pytest.param("<!-- --><img src=x onerror=alert(1)>", id="comment-and-img"),
        pytest.param("&lt;script&gt; &amp; &#x27;", id="pre-escaped-entities"),
        pytest.param("</SPAN >", id="uppercase-closing-tag"),
        pytest.param("   line separators", id="js-line-separators"),
        pytest.param("back\\slash \"quoted\" \n newline \t tab", id="json-escapes"),
        pytest.param("Ünïcødé 日本語 🎉", id="unicode"),
    ],
)
def test_metadata_escapes_hostile_text_and_round_trips(browser, hostile):
    records = [{"title": hostile, "fields": {hostile: [hostile]}, "search": hostile}]
    node = browser.metadata_node(records, {"collection-id": hostile})
    markup = node.astext()
    prefix, suffix = '<span hidden class="sk-collection-data">', "</span>"
    assert markup.startswith(prefix)
    assert markup.endswith(suffix)
    body = markup[len(prefix) : -len(suffix)]
    # Nothing inside the carrier can open/close an element or leave an attribute.
    assert not set("<>\"'") & set(body)
    payload = json.loads(html.unescape(body))
    assert payload["records"] == records
    assert payload["collectionId"] == hostile


def test_metadata_keeps_non_ascii_readable(browser):
    node = browser.metadata_node([{"title": "日本語"}], {})
    assert "日本語" in node.astext()


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
    ],
)
def test_metadata_uses_strict_json(browser, value):
    with pytest.raises(ValueError, match="not JSON compliant"):
        browser.metadata_node([{"title": "A", "fields": {"v": value}}], {})


def test_metadata_refuses_unserializable_records(browser):
    with pytest.raises(TypeError):
        browser.metadata_node([{"title": object()}], {})


def test_records_built_by_record_for_browser_always_serialize(browser):
    item = {
        "title": "T",
        "v": float("nan"),
        "d": dt.datetime(2024, 1, 1, 12),
        "tags": ["a", None],
        "obj": object,
    }
    options = {"filter-fields": ("v", "d", "tags", "obj", "missing")}
    record = browser.record_for_browser(item, options)
    payload = _payload(browser.metadata_node([record], options))
    assert payload["records"][0]["fields"]["v"] is None
    assert payload["records"][0]["fields"]["d"] == "2024-01-01T12:00:00"
    assert payload["facets"] == ["v", "d", "tags", "obj", "missing"]


def test_large_record_set_round_trips(browser):
    records = [
        {"title": f"Item {n}", "fields": {"n": n}, "search": f"Item {n}"}
        for n in range(2_000)
    ]
    assert _payload(browser.metadata_node(records, {}))["records"] == records
