"""
Tests for ``_sphinx_collection.select``: the closed filter grammar, the
total/stable sort, pagination and grouping shared by collection directives.

Expectations come from the module's docstrings, not from its implementation.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import importlib
import itertools
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOST_ROOT = ROOT.parents[2]


def _select_module():
    externals = str(HOST_ROOT / "scikitplot" / "_externals")
    if externals not in sys.path:
        sys.path.insert(0, externals)
    return importlib.import_module("_sphinx_ext._sphinx_collection.select")


@pytest.fixture(scope="module")
def select():
    return _select_module()


def _titles(records):
    return [record["title"] for record in records]


def _matches(select, text, record):
    terms = select.parse_filter(text)
    return all(term.matches(record) for term in terms)


# -- field access -------------------------------------------------------------


@pytest.mark.parametrize(
    ("record", "path", "expected"),
    [
        pytest.param({"a": 1}, "a", 1, id="top-level"),
        pytest.param({"a": {"b": {"c": 3}}}, "a.b.c", 3, id="nested"),
        pytest.param({"a": 1}, "missing", None, id="missing"),
        pytest.param({"a": {"b": 1}}, "a.x", None, id="missing-leaf"),
        pytest.param({"a": 1}, "a.b", None, id="descend-into-scalar"),
        pytest.param({"a": [{"b": 1}]}, "a.b", None, id="descend-into-list"),
        pytest.param({"a": None}, "a", None, id="present-null"),
        pytest.param({}, "a", None, id="empty-record"),
        pytest.param({"a": 1}, "", None, id="empty-path"),
        pytest.param({"a": 0}, "a", 0, id="falsy-zero-kept"),
        pytest.param({"a": False}, "a", False, id="falsy-false-kept"),
        pytest.param({"naïve": "é"}, "naïve", "é", id="unicode-key"),
    ],
)
def test_get_field(select, record, path, expected):
    result = select.get_field(record, path)
    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize(
    ("record", "path", "expected"),
    [
        pytest.param({"a": 1}, "a", True, id="present"),
        pytest.param({"a": None}, "a", True, id="present-null"),
        pytest.param({"a": {"b": None}}, "a.b", True, id="nested-null"),
        pytest.param({"a": 1}, "b", False, id="missing"),
        pytest.param({"a": 1}, "a.b", False, id="through-scalar"),
        pytest.param({"a": None}, "a.b", False, id="through-null"),
        pytest.param({}, "a", False, id="empty-record"),
    ],
)
def test_has_field_distinguishes_null_from_missing(select, record, path, expected):
    assert select.has_field(record, path) is expected


def test_field_access_does_not_mutate_record(select):
    record = {"a": {"b": [1, 2]}}
    before = copy.deepcopy(record)
    select.get_field(record, "a.b")
    select.has_field(record, "a.x.y")
    assert record == before


# -- parse_filter -------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("a=b", ("a", "=", "b"), id="equal"),
        pytest.param("a!=b", ("a", "!=", "b"), id="not-equal"),
        pytest.param("a~b", ("a", "~", "b"), id="contains"),
        pytest.param("a!~b", ("a", "!~", "b"), id="not-contains"),
        pytest.param("a^b", ("a", "^", "b"), id="prefix"),
        pytest.param("a$b", ("a", "$", "b"), id="suffix"),
        pytest.param("a:b|c", ("a", ":", "b|c"), id="any-of"),
        pytest.param("a>1", ("a", ">", "1"), id="greater"),
        pytest.param("a>=1", ("a", ">=", "1"), id="greater-equal-not-split"),
        pytest.param("a<1", ("a", "<", "1"), id="less"),
        pytest.param("a<=1", ("a", "<=", "1"), id="less-equal-not-split"),
        pytest.param("  a  =  b  ", ("a", "=", "b"), id="whitespace-trimmed"),
        pytest.param("author.name=x", ("author.name", "=", "x"), id="dotted"),
        pytest.param("link-alt~x", ("link-alt", "~", "x"), id="hyphen-field"),
        pytest.param("_p=1", ("_p", "=", "1"), id="underscore-field"),
        pytest.param('a=""', ("a", "=", ""), id="quoted-empty"),
        pytest.param("a='>5'", ("a", "=", ">5"), id="quoted-operator-chars"),
        pytest.param('t~"a, b"', ("t", "~", "a, b"), id="quoted-comma"),
        pytest.param("a=b=c", ("a", "=", "b=c"), id="operator-inside-value"),
        pytest.param("a=x y z", ("a", "=", "x y z"), id="spaces-in-value"),
        pytest.param("a=日本語", ("a", "=", "日本語"), id="unicode-value"),
        pytest.param("d>2024-01-01", ("d", ">", "2024-01-01"), id="date-value"),
    ],
)
def test_parse_filter_single_term(select, text, expected):
    (term,) = select.parse_filter(text)
    assert (term.field, term.operator, term.operand) == expected
    assert term.negated is False


def test_parse_filter_docstring_examples(select):
    parsed = select.parse_filter("category=tutorial, stars>100")
    assert [(t.field, t.operator, t.operand) for t in parsed] == [
        ("category", "=", "tutorial"),
        ("stars", ">", "100"),
    ]
    term = select.parse_filter("!deprecated")[0]
    assert (term.field, term.operator, term.operand, term.negated) == (
        "deprecated",
        "",
        "",
        True,
    )


@pytest.mark.parametrize(
    ("text", "field", "negated"),
    [
        pytest.param("featured", "featured", False, id="bare"),
        pytest.param("!deprecated", "deprecated", True, id="negated"),
        pytest.param("! deprecated", "deprecated", True, id="negated-spaced"),
        pytest.param("a.b", "a.b", False, id="dotted"),
    ],
)
def test_parse_filter_presence_terms(select, text, field, negated):
    (term,) = select.parse_filter(text)
    assert (term.field, term.operator, term.operand) == (field, "", "")
    assert term.negated is negated


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param("   ", id="blank"),
        pytest.param(None, id="none"),
        pytest.param(",", id="comma-only"),
        pytest.param(" , ,, ", id="commas-and-blanks"),
    ],
)
def test_parse_filter_empty_expression_selects_everything(select, text):
    assert select.parse_filter(text) == []


def test_parse_filter_preserves_term_order_and_skips_empty_terms(select):
    parsed = select.parse_filter("b=1,, a=2 ,!c,")
    assert [(t.field, t.negated) for t in parsed] == [
        ("b", False),
        ("a", False),
        ("c", True),
    ]


def test_parse_filter_keeps_duplicate_terms(select):
    parsed = select.parse_filter("a=1, a=1")
    assert len(parsed) == 2
    assert parsed[0] == parsed[1]


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("category ??? tutorial", id="typo-operator"),
        pytest.param("a=", id="missing-value"),
        pytest.param("a=   ", id="blank-value"),
        pytest.param("a==b", id="double-equals"),
        pytest.param("a=>5", id="reversed-operator"),
        pytest.param("a=<5", id="value-starts-with-operator"),
        pytest.param("=b", id="missing-field"),
        pytest.param("!", id="bare-bang"),
        pytest.param("!a=b", id="negated-comparison"),
        pytest.param("a!b", id="lone-bang-operator"),
        pytest.param("1a=b", id="field-starts-with-digit"),
        pytest.param("a..b=c", id="empty-path-segment"),
        pytest.param("a.=c", id="trailing-dot"),
        pytest.param("a b", id="space-in-bare-field"),
        pytest.param("a=b\nc", id="newline-in-term"),
        pytest.param('a="x', id="unbalanced-double-quote"),
        pytest.param("a='x", id="unbalanced-single-quote"),
        pytest.param("a=1, b='x, c=2", id="unbalanced-in-later-term"),
        pytest.param("été=1", id="non-ascii-field"),
        pytest.param("__import__('os').system('id')", id="python-call"),
        pytest.param("<script>alert(1)</script>", id="html-as-term"),
        pytest.param("a\x00=1", id="nul-in-field"),
    ],
)
def test_parse_filter_rejects_malformed_terms(select, text):
    with pytest.raises(select.FilterError):
        select.parse_filter(text)


def test_filter_error_is_value_error_and_names_the_term(select):
    assert issubclass(select.FilterError, ValueError)
    with pytest.raises(select.FilterError, match=r"category \?\?\? tutorial"):
        select.parse_filter("ok=1, category ??? tutorial")


def test_hostile_operand_is_an_inert_literal(select, monkeypatch):
    import builtins

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("filter text reached an evaluator")

    monkeypatch.setattr(builtins, "eval", explode)
    monkeypatch.setattr(builtins, "exec", explode)
    payload = "__import__('os').system('id')"
    (term,) = select.parse_filter('title="' + payload + '"')
    assert term.operand == payload
    assert term.matches({"title": payload}) is True
    assert term.matches({"title": "harmless"}) is False


def test_operator_table_is_closed_and_longest_first(select):
    tokens = [token for token, _ in select.OPERATORS]
    assert sorted(tokens) == sorted(
        ["=", "!=", "~", "!~", "^", "$", ":", ">", ">=", "<", "<="]
    )
    assert len(set(tokens)) == len(tokens)
    for index, token in enumerate(tokens):
        for earlier in tokens[:index]:
            assert not (token.startswith(earlier) and token != earlier), (
                earlier,
                token,
            )


# -- FilterTerm.matches -------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "record", "expected"),
    [
        pytest.param("k=tutorial", {"k": "Tutorial"}, True, id="eq-casefold"),
        pytest.param("k=tutor", {"k": "tutorial"}, False, id="eq-not-substring"),
        pytest.param("k=STRASSE", {"k": "straße"}, True, id="eq-unicode-casefold"),
        pytest.param("tags=python", {"tags": ["Python", "r"]}, True, id="eq-list-any"),
        pytest.param("tags=pyth", {"tags": ["Python"]}, False, id="eq-list-whole-atom"),
        pytest.param("flag=true", {"flag": True}, True, id="eq-bool-as-text"),
        pytest.param("n=5", {"n": 5}, True, id="eq-int"),
        pytest.param("k!=b", {"k": "a"}, True, id="ne-different"),
        pytest.param("k!=A", {"k": "a"}, False, id="ne-same-casefold"),
        pytest.param("k!=a", {"k": ["a", "b"]}, False, id="ne-list-has-atom"),
        pytest.param("t~lo wo", {"t": "Hello World"}, True, id="contains"),
        pytest.param("t~xyz", {"t": "Hello World"}, False, id="contains-miss"),
        pytest.param("tags~yth", {"tags": ["python", "r"]}, True, id="contains-list"),
        pytest.param("t!~xyz", {"t": "Hello"}, True, id="not-contains"),
        pytest.param("t!~ELL", {"t": "Hello"}, False, id="not-contains-hit"),
        pytest.param("t^hell", {"t": "Hello World"}, True, id="prefix"),
        pytest.param("t^world", {"t": "Hello World"}, False, id="prefix-miss"),
        pytest.param("tags^c", {"tags": ["ab", "cd"]}, True, id="prefix-list-any"),
        pytest.param("t$WORLD", {"t": "Hello World"}, True, id="suffix"),
        pytest.param("t$hello", {"t": "Hello World"}, False, id="suffix-miss"),
        pytest.param("k:python | julia", {"k": "Julia"}, True, id="any-of-hit"),
        pytest.param("k:python|r", {"k": "julia"}, False, id="any-of-miss"),
        pytest.param("tags:python|r", {"tags": ["x", "R"]}, True, id="any-of-list"),
        pytest.param("stars>100", {"stars": 200}, True, id="gt-number"),
        pytest.param("stars>100", {"stars": 100}, False, id="gt-equal-is-false"),
        pytest.param("stars>10", {"stars": "9"}, False, id="gt-numeric-string"),
        pytest.param("stars>9", {"stars": "10"}, True, id="gt-not-lexical"),
        pytest.param("stars>=200", {"stars": 200}, True, id="ge-equal"),
        pytest.param("stars<200", {"stars": 200}, False, id="lt-equal-is-false"),
        pytest.param("stars<=200", {"stars": 200}, True, id="le-equal"),
        pytest.param("v<2", {"v": 1.5}, True, id="lt-float"),
        pytest.param("v>-1", {"v": 0}, True, id="gt-negative-operand"),
        pytest.param("d>2024-01-01", {"d": dt.date(2024, 5, 1)}, True, id="gt-date"),
        pytest.param("d<2024-01-01", {"d": dt.date(2024, 5, 1)}, False, id="lt-date"),
        pytest.param(
            "d>=2024-05-01", {"d": dt.datetime(2024, 5, 1)}, True, id="ge-datetime"
        ),
        pytest.param(
            "d<=2024-05-01T12:00:00+02:00",
            {"d": "2024-05-01T10:00:00Z"},
            True,
            id="le-same-instant-across-zones",
        ),
        pytest.param(
            "d<2024-05-01T12:00:00+02:00",
            {"d": "2024-05-01T10:00:00Z"},
            False,
            id="lt-same-instant-across-zones",
        ),
        pytest.param("d>2024-01-09", {"d": "2024-01-10"}, True, id="gt-iso-strings"),
        pytest.param("t>apple", {"t": "Banana"}, True, id="gt-text-casefold"),
        pytest.param("t<apple", {"t": "Banana"}, False, id="lt-text-casefold"),
    ],
)
def test_operator_semantics(select, text, record, expected):
    assert _matches(select, text, record) is expected


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("k=x", id="equal"),
        pytest.param("k!=x", id="not-equal"),
        pytest.param("k~x", id="contains"),
        pytest.param("k!~x", id="not-contains"),
        pytest.param("k^x", id="prefix"),
        pytest.param("k$x", id="suffix"),
        pytest.param("k:x|y", id="any-of"),
        pytest.param("k>1", id="greater"),
        pytest.param("k>=1", id="greater-equal"),
        pytest.param("k<1", id="less"),
        pytest.param("k<=1", id="less-equal"),
        pytest.param("k", id="presence"),
    ],
)
@pytest.mark.parametrize(
    "record",
    [pytest.param({}, id="missing"), pytest.param({"k": None}, id="null")],
)
def test_missing_field_satisfies_no_comparison(select, text, record):
    # "A missing field satisfies only a negated presence test."
    assert _matches(select, text, record) is False


@pytest.mark.parametrize(
    ("value", "present"),
    [
        pytest.param("x", True, id="text"),
        pytest.param(["a"], True, id="non-empty-list"),
        pytest.param({"a": 1}, True, id="non-empty-mapping"),
        pytest.param(True, True, id="true"),
        pytest.param(7, True, id="positive-number"),
        pytest.param(None, False, id="null"),
        pytest.param("", False, id="empty-text"),
        pytest.param([], False, id="empty-list"),
        pytest.param({}, False, id="empty-mapping"),
        pytest.param(False, False, id="false"),
    ],
)
def test_presence_and_negated_presence_are_complementary(select, value, present):
    record = {"k": value}
    assert select.FilterTerm("k").matches(record) is present
    assert select.FilterTerm("k", negated=True).matches(record) is (not present)


def test_missing_field_satisfies_negated_presence(select):
    assert select.FilterTerm("gone", negated=True).matches({"k": 1}) is True
    assert select.FilterTerm("gone").matches({"k": 1}) is False


def test_nested_field_filter(select):
    record = {"author": {"name": "Ada"}}
    assert _matches(select, "author.name=ada", record) is True
    assert _matches(select, "author.email=ada", record) is False


def test_unknown_operator_is_an_error_not_a_fallback(select):
    term = select.FilterTerm("k", "??", "x")
    with pytest.raises(select.FilterError, match="unknown operator"):
        term.matches({"k": "x"})


def test_filter_term_is_frozen_and_hashable(select):
    term = select.FilterTerm("k", "=", "x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        term.field = "other"
    assert term == select.FilterTerm("k", "=", "x")
    assert len({term, select.FilterTerm("k", "=", "x")}) == 1


def test_very_long_operand_and_value(select):
    long_text = "x" * 200_000
    (term,) = select.parse_filter("k~" + long_text)
    assert term.operand == long_text
    assert term.matches({"k": "a" + long_text + "b"}) is True
    assert term.matches({"k": long_text[:-1]}) is False


@pytest.mark.parametrize(
    ("text", "record"),
    [
        pytest.param("stars>100", {"stars": "unknown"}, id="word-vs-number"),
        pytest.param("stars>100", {"stars": [1, 2]}, id="list-vs-number"),
    ],
)
def test_ordered_comparison_of_non_comparable_values_is_false(select, text, record):
    # _compare: "None when the two are not comparable (which makes any ordered
    # comparison against them false ..., so one odd record cannot fail a build)".
    assert _matches(select, text, record) is False


# -- parse_sort ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("", [], id="empty"),
        pytest.param(None, [], id="none"),
        pytest.param(" , ,", [], id="blank-keys-skipped"),
        pytest.param("title", [("title", False)], id="ascending"),
        pytest.param("-title", [("title", True)], id="descending"),
        pytest.param("- title", [("title", True)], id="descending-spaced"),
        pytest.param(
            "category,-published",
            [("category", False), ("published", True)],
            id="docstring-example",
        ),
        pytest.param(" a , -b.c ", [("a", False), ("b.c", True)], id="trimmed-dotted"),
        pytest.param("link-alt", [("link-alt", False)], id="hyphen-inside-name"),
        pytest.param("a,a", [("a", False), ("a", False)], id="duplicates-kept"),
    ],
)
def test_parse_sort(select, text, expected):
    assert select.parse_sort(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("-", id="bare-minus"),
        pytest.param("--a", id="double-minus"),
        pytest.param("a b", id="space-in-name"),
        pytest.param("a.", id="trailing-dot"),
        pytest.param(".a", id="leading-dot"),
        pytest.param("1a", id="leading-digit"),
        pytest.param("a,+b", id="plus-prefix"),
        pytest.param("a;drop", id="punctuation"),
        pytest.param("a\nb", id="embedded-newline"),
        pytest.param("été", id="non-ascii"),
    ],
)
def test_parse_sort_rejects_invalid_keys(select, text):
    with pytest.raises(select.FilterError, match="sort key"):
        select.parse_sort(text)


# -- Selection ----------------------------------------------------------------


def test_empty_selection_defaults(select):
    selection = select.Selection()
    assert tuple(selection.terms) == ()
    assert tuple(selection.sort_keys) == ()
    assert selection.group_by == ""
    assert selection.limit is None
    assert selection.offset == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"limit": -1}, id="negative-limit"),
        pytest.param({"offset": -1}, id="negative-offset"),
        pytest.param({"limit": -5, "offset": -5}, id="both-negative"),
    ],
)
def test_selection_rejects_negative_bounds(select, kwargs):
    with pytest.raises(select.FilterError, match="must not be negative"):
        select.Selection(**kwargs)
    with pytest.raises(select.FilterError, match="must not be negative"):
        select.Selection.from_text(**kwargs)


def test_selection_accepts_zero_bounds(select):
    selection = select.Selection(limit=0, offset=0)
    assert (selection.limit, selection.offset) == (0, 0)


def test_selection_is_frozen(select):
    selection = select.Selection()
    with pytest.raises(dataclasses.FrozenInstanceError):
        selection.limit = 3


def test_from_text_parses_every_part(select):
    selection = select.Selection.from_text(
        filter_text="kind=demo, !old",
        sort_text="kind,-stars",
        group_by="  kind  ",
        limit=5,
        offset=2,
    )
    assert [(t.field, t.operator, t.operand, t.negated) for t in selection.terms] == [
        ("kind", "=", "demo", False),
        ("old", "", "", True),
    ]
    assert list(selection.sort_keys) == [("kind", False), ("stars", True)]
    assert selection.group_by == "kind"
    assert (selection.limit, selection.offset) == (5, 2)


def test_from_text_none_group_by_means_ungrouped(select):
    assert select.Selection.from_text(group_by=None).group_by == ""


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"filter_text": "a ??? b"}, id="bad-filter"),
        pytest.param({"sort_text": "a b"}, id="bad-sort"),
    ],
)
def test_from_text_propagates_parse_errors(select, kwargs):
    with pytest.raises(select.FilterError):
        select.Selection.from_text(**kwargs)


# -- apply_selection ----------------------------------------------------------

DATA = [
    {"title": "Beta", "stars": 30, "kind": "demo"},
    {"title": "alpha", "stars": 200, "kind": "tutorial"},
    {"title": "Gamma", "kind": "tutorial"},
]


def test_apply_selection_docstring_examples(select):
    chosen, total = select.apply_selection(
        DATA, select.Selection.from_text(sort_text="title")
    )
    assert (_titles(chosen), total) == (["alpha", "Beta", "Gamma"], 3)
    chosen, total = select.apply_selection(
        DATA, select.Selection.from_text(filter_text="kind=tutorial, stars>100")
    )
    assert (_titles(chosen), total) == (["alpha"], 1)
    chosen, _ = select.apply_selection(
        DATA, select.Selection.from_text(sort_text="-stars")
    )
    assert _titles(chosen) == ["alpha", "Beta", "Gamma"]


def test_empty_selection_returns_everything_in_source_order(select):
    chosen, total = select.apply_selection(DATA, select.Selection())
    assert chosen == DATA
    assert chosen is not DATA
    assert total == 3


def test_apply_selection_on_empty_collection(select):
    selection = select.Selection.from_text(
        filter_text="a=1", sort_text="a", limit=3, offset=2
    )
    assert select.apply_selection([], selection) == ([], 0)


def test_apply_selection_accepts_a_one_shot_iterable(select):
    chosen, total = select.apply_selection(
        (record for record in DATA), select.Selection.from_text(sort_text="title")
    )
    assert (_titles(chosen), total) == (["alpha", "Beta", "Gamma"], 3)


def test_apply_selection_does_not_mutate_input(select):
    data = copy.deepcopy(DATA)
    select.apply_selection(
        data, select.Selection.from_text(sort_text="-title", limit=1, offset=1)
    )
    assert data == DATA


def test_selected_records_are_the_original_objects(select):
    chosen, _ = select.apply_selection(
        DATA, select.Selection.from_text(sort_text="title")
    )
    assert chosen[0] is DATA[1]


def test_terms_are_combined_with_and(select):
    chosen, total = select.apply_selection(
        DATA, select.Selection.from_text(filter_text="kind=tutorial, !stars")
    )
    assert (_titles(chosen), total) == (["Gamma"], 1)


@pytest.mark.parametrize(
    ("limit", "offset", "expected"),
    [
        pytest.param(None, 0, ["a", "b", "c", "d", "e"], id="no-window"),
        pytest.param(2, 0, ["a", "b"], id="limit"),
        pytest.param(None, 3, ["d", "e"], id="offset"),
        pytest.param(2, 1, ["b", "c"], id="offset-then-limit"),
        pytest.param(0, 0, [], id="limit-zero"),
        pytest.param(99, 0, ["a", "b", "c", "d", "e"], id="limit-beyond-end"),
        pytest.param(None, 5, [], id="offset-at-end"),
        pytest.param(3, 99, [], id="offset-beyond-end"),
        pytest.param(10**12, 10**12, [], id="huge-bounds"),
    ],
)
def test_pagination_applies_after_sort_and_reports_full_total(
    select, limit, offset, expected
):
    data = [{"title": letter} for letter in "edcba"]
    chosen, total = select.apply_selection(
        data,
        select.Selection.from_text(sort_text="title", limit=limit, offset=offset),
    )
    assert _titles(chosen) == expected
    assert total == 5


def test_total_counts_matches_before_pagination(select):
    data = [{"title": str(n), "even": n % 2 == 0} for n in range(10)]
    chosen, total = select.apply_selection(
        data, select.Selection.from_text(filter_text="even", limit=2)
    )
    assert (len(chosen), total) == (2, 5)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param([10, 9, 100], [9, 10, 100], id="numbers"),
        pytest.param(["10", "9", "100"], ["9", "10", "100"], id="numeric-strings"),
        pytest.param(["10", 9, 2.5], [2.5, 9, "10"], id="mixed-numeric-forms"),
        pytest.param(["10", "9", "x"], ["10", "9", "x"], id="mixed-column-is-text"),
        pytest.param(["b", "A", "c"], ["A", "b", "c"], id="text-case-insensitive"),
        pytest.param(
            ["2024-03-01", "2023-12-31", "2024-01-15"],
            ["2023-12-31", "2024-01-15", "2024-03-01"],
            id="iso-dates",
        ),
        pytest.param(
            [dt.date(2024, 3, 1), dt.datetime(2023, 12, 31, 8), "2024-01-15"],
            [dt.datetime(2023, 12, 31, 8), "2024-01-15", dt.date(2024, 3, 1)],
            id="date-objects-and-strings",
        ),
        pytest.param(
            ["2024-01-01T12:00:00+05:00", "2024-01-01T08:00:00Z"],
            ["2024-01-01T12:00:00+05:00", "2024-01-01T08:00:00Z"],
            id="instants-compared-in-utc",
        ),
        pytest.param(
            ["2024-13-45", "2024-02-01", "2024-01-01 junk"],
            ["2024-01-01 junk", "2024-02-01", "2024-13-45"],
            id="invalid-dates-make-a-text-column",
        ),
        pytest.param([True, False, True], [False, True, True], id="booleans-as-text"),
        pytest.param(["é", "z", "a"], ["a", "z", "é"], id="unicode-text"),
    ],
)
def test_sort_is_type_aware_per_column(select, values, expected):
    data = [{"v": value} for value in values]
    chosen, _ = select.apply_selection(data, select.Selection.from_text(sort_text="v"))
    assert [record["v"] for record in chosen] == expected
    chosen, _ = select.apply_selection(
        data, select.Selection.from_text(sort_text="-v")
    )
    assert [record["v"] for record in chosen] == expected[::-1]


@pytest.mark.parametrize(
    "sort_text",
    [pytest.param("stars", id="ascending"), pytest.param("-stars", id="descending")],
)
def test_records_missing_the_sort_key_sort_last_both_directions(select, sort_text):
    data = [
        {"title": "none-1"},
        {"title": "low", "stars": 1},
        {"title": "null", "stars": None},
        {"title": "high", "stars": 9},
        {"title": "none-2"},
    ]
    chosen, total = select.apply_selection(
        data, select.Selection.from_text(sort_text=sort_text)
    )
    assert total == 5
    head = ["low", "high"] if sort_text == "stars" else ["high", "low"]
    # Never dropped, always last, and in source order among themselves.
    assert _titles(chosen) == [*head, "none-1", "null", "none-2"]


@pytest.mark.parametrize(
    "sort_text",
    [pytest.param("rank", id="ascending"), pytest.param("-rank", id="descending")],
)
def test_ties_fall_back_to_source_order(select, sort_text):
    data = [{"title": f"t{index}", "rank": index % 2} for index in range(8)]
    chosen, _ = select.apply_selection(
        data, select.Selection.from_text(sort_text=sort_text)
    )
    evens = ["t0", "t2", "t4", "t6"]
    odds = ["t1", "t3", "t5", "t7"]
    expected = evens + odds if sort_text == "rank" else odds + evens
    assert _titles(chosen) == expected


def test_multi_key_sort_uses_priority_order(select):
    data = [
        {"title": "a", "kind": "x", "n": 1},
        {"title": "b", "kind": "y", "n": 3},
        {"title": "c", "kind": "x", "n": 2},
        {"title": "d", "kind": "y", "n": 1},
        {"title": "e", "kind": "x"},
    ]
    chosen, _ = select.apply_selection(
        data, select.Selection.from_text(sort_text="kind,-n")
    )
    assert _titles(chosen) == ["c", "a", "e", "b", "d"]


def test_sort_by_nested_field(select):
    data = [
        {"title": "b", "author": {"name": "Zed"}},
        {"title": "a", "author": {"name": "amy"}},
        {"title": "c", "author": {}},
    ]
    chosen, _ = select.apply_selection(
        data, select.Selection.from_text(sort_text="author.name")
    )
    assert _titles(chosen) == ["a", "b", "c"]


def test_sort_result_is_independent_of_input_order_for_distinct_keys(select):
    base = [{"title": f"t{n}", "n": value} for n, value in enumerate([5, "3", 9.5, 1])]
    selection = select.Selection.from_text(sort_text="n")
    results = {
        tuple(_titles(select.apply_selection(list(order), selection)[0]))
        for order in itertools.permutations(base)
    }
    assert results == {("t3", "t1", "t0", "t2")}


def test_repeated_application_is_identical(select):
    data = [
        {"title": f"t{n % 7}", "n": n % 3, "tags": ["a", "b"][n % 2]}
        for n in range(50)
    ]
    selection = select.Selection.from_text(
        filter_text="tags:a|b", sort_text="-n,title", limit=20, offset=3
    )
    first = select.apply_selection(data, selection)
    assert all(select.apply_selection(data, selection) == first for _ in range(3))


def test_sorting_very_long_text_values(select):
    data = [{"title": "b" * 100_000}, {"title": "a" * 100_000}]
    chosen, _ = select.apply_selection(
        data, select.Selection.from_text(sort_text="title")
    )
    assert [record["title"][0] for record in chosen] == ["a", "b"]


def test_sort_stays_total_when_a_column_contains_nan(select):
    nan = float("nan")
    data = [
        {"title": "three", "v": 3},
        {"title": "nan", "v": nan},
        {"title": "one", "v": 1},
    ]
    chosen, _ = select.apply_selection(data, select.Selection.from_text(sort_text="v"))
    finite = [title for title in _titles(chosen) if title != "nan"]
    assert finite == ["one", "three"]


# -- group_records ------------------------------------------------------------


def _groups(select, data, group_by):
    return [
        (label, [record["n"] for record in records])
        for label, records in select.group_records(
            data, select.Selection.from_text(group_by=group_by)
        )
    ]


def test_group_records_docstring_examples(select):
    data = [{"n": 1, "kind": "demo"}, {"n": 2}, {"n": 3, "kind": "demo"}]
    assert _groups(select, data, "kind") == [("demo", [1, 3]), ("Ungrouped", [2])]
    tagged = [{"n": 1, "tags": ["a", "b"]}, {"n": 2, "tags": ["b"]}]
    assert _groups(select, tagged, "tags") == [("a", [1]), ("b", [1, 2])]


def test_no_group_by_returns_one_unlabelled_section(select):
    data = [{"n": 1}, {"n": 2}]
    result = select.group_records(data, select.Selection())
    assert result == [("", data)]
    assert result[0][1] is not data


def test_group_records_on_empty_input(select):
    assert select.group_records([], select.Selection()) == [("", [])]
    assert select.group_records([], select.Selection(group_by="kind")) == []


def test_section_order_follows_first_appearance(select):
    data = [{"n": 1, "k": "z"}, {"n": 2, "k": "a"}, {"n": 3, "k": "z"}, {"n": 4, "k": "m"}]
    assert _groups(select, data, "k") == [("z", [1, 3]), ("a", [2]), ("m", [4])]


def test_ungrouped_section_is_always_last(select):
    data = [{"n": 1}, {"n": 2, "k": "a"}, {"n": 3, "k": None}, {"n": 4, "k": "b"}]
    assert _groups(select, data, "k") == [("a", [2]), ("b", [4]), ("Ungrouped", [1, 3])]
    assert select.UNGROUPED_LABEL == "Ungrouped"


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="null"),
        pytest.param("", id="empty-text"),
        pytest.param("   ", id="blank-text"),
        pytest.param([], id="empty-list"),
        pytest.param([None], id="list-of-null"),
        pytest.param(["", " "], id="list-of-blanks"),
    ],
)
def test_empty_grouping_values_are_kept_as_ungrouped(select, value):
    data = [{"n": 1, "k": value}]
    assert _groups(select, data, "k") == [("Ungrouped", [1])]


@pytest.mark.parametrize(
    ("value", "label"),
    [
        pytest.param(0, "0", id="zero"),
        pytest.param(False, "False", id="false"),
        pytest.param(2024, "2024", id="integer"),
        pytest.param("  padded  ", "padded", id="label-trimmed"),
        pytest.param("日本語", "日本語", id="unicode"),
        pytest.param("<script>x</script>", "<script>x</script>", id="markup-verbatim"),
    ],
)
def test_scalar_grouping_values_become_labels(select, value, label):
    assert _groups(select, [{"n": 1, "k": value}], "k") == [(label, [1])]


def test_list_valued_record_appears_once_per_distinct_element(select):
    data = [{"n": 1, "tags": ["a", "b", "a", " a "]}, {"n": 2, "tags": ("b", "c")}]
    assert _groups(select, data, "tags") == [("a", [1]), ("b", [1, 2]), ("c", [2])]


def test_no_record_is_lost_by_grouping(select):
    data = [
        {"n": 1, "k": "a"},
        {"n": 2},
        {"n": 3, "k": ["a", None]},
        {"n": 4, "k": {"nested": 1}},
        {"n": 5, "k": 0},
    ]
    sections = select.group_records(data, select.Selection(group_by="k"))
    seen = {record["n"] for _, records in sections for record in records}
    assert seen == {1, 2, 3, 4, 5}
    labels = [label for label, _ in sections]
    assert len(labels) == len(set(labels))
    assert all(label for label in labels)


def test_group_by_nested_field(select):
    data = [{"n": 1, "a": {"b": "x"}}, {"n": 2, "a": {}}, {"n": 3, "a": "scalar"}]
    assert _groups(select, data, "a.b") == [("x", [1]), ("Ungrouped", [2, 3])]


def test_group_records_does_not_mutate_input(select):
    data = [{"n": 1, "tags": ["a", "a"]}, {"n": 2}]
    before = copy.deepcopy(data)
    select.group_records(data, select.Selection(group_by="tags"))
    assert data == before


def test_public_names_are_exported(select):
    assert sorted(select.__all__) == sorted(
        [
            "OPERATORS",
            "FilterError",
            "FilterTerm",
            "Selection",
            "apply_selection",
            "get_field",
            "group_records",
            "has_field",
            "parse_filter",
            "parse_sort",
        ]
    )
    assert all(hasattr(select, name) for name in select.__all__)
