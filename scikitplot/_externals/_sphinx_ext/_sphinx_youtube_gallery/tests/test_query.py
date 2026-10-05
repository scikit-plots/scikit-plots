"""
Tests for :mod:`.query`: the YouTube-shaped face over the shared selection engine.

Covers eager :class:`Query` validation, the bounded ``match-regex`` subset,
translation into the shared ``Selection`` type, the YouTube-specific filter
terms (identity, text, regex, publication interval), and the public
``apply_query`` / ``group_records`` entry points over video and channel
catalogs.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import warnings

import pytest

from .. import query as query_module
from ..model import (
    CatalogError,
    derive_channel_records,
    normalize_catalog,
    normalize_gallery_catalog,
    normalize_record,
)
from ..query import GROUP_KEYS, SORT_KEYS, Query, apply_query, as_record, group_records

UTC = dt.timezone.utc
UC_A = "UC" + "a" * 22
UC_B = "UC" + "b" * 22
PL_A = "PL" + "a" * 16
PL_B = "PL" + "b" * 16
A, B, C, D, E_ = (letter * 11 for letter in "abcde")


def day(year, month=1, dom=1):
    """Return an aware UTC midnight."""
    return dt.datetime(year, month, dom, tzinfo=UTC)


@pytest.fixture
def catalog():
    """Five videos exercising every queryable field, in a fixed source order."""
    return normalize_catalog(
        [
            {
                "id": A,
                "title": "Beta regression",
                "description": "All about PCA",
                "channel": "StatQuest",
                "channel_id": UC_A,
                "handle": "statquest",
                "playlist": "Intro",
                "playlist_id": PL_A,
                "position": 1,
                "published": "2022-05-01",
                "duration": 600,
                "tags": ["stats", "PCA"],
                "fields": {"category": "ml", "audience": {"level": "beginner"}},
            },
            {
                "id": B,
                "title": "alpha blending",
                "description": "",
                "channel": "Other",
                "channel_id": UC_B,
                "playlist": "Advanced",
                "playlist_id": PL_B,
                "position": 0,
                "published": "2024-01-01",
                "duration": 30,
                "tags": ["graphics"],
                "fields": {"category": "gfx"},
            },
            {
                "id": C,
                "title": "Gamma correction",
                "description": "colour and pca tricks",
                "channel": "StatQuest",
                "channel_id": UC_A,
                "playlist": "Intro",
                "playlist_id": PL_A,
                "position": 0,
                "published": "2023-01-01T00:00:00Z",
                "tags": ["stats", "graphics", "pca"],
            },
            {"id": D, "title": "delta (undated)"},
            {
                "id": E_,
                "title": "Epsilon",
                "published": "2023-12-31T23:59:59Z",
                "duration": 30,
                "tags": ["stats"],
            },
        ]
    )


def titles(records):
    """Return the first word of each record title, for compact assertions."""
    return [record.title.split()[0] for record in records]


def select(catalog, **keys):
    """Apply a query and return the selected first-words."""
    selected, _ = apply_query(catalog, Query(**keys))
    return titles(selected)


# -- Query validation ---------------------------------------------------------


class TestQueryValidation:
    def test_the_default_query_is_valid_and_selects_everything(self, catalog):
        selected, total = apply_query(catalog, Query())
        assert selected == catalog
        assert total == len(catalog)

    @pytest.mark.parametrize("key", SORT_KEYS)
    @pytest.mark.parametrize("prefix", ["", "-"], ids=["asc", "desc"])
    def test_every_builtin_sort_key_is_accepted(self, key, prefix):
        assert Query(sort_by=prefix + key).sort_by == prefix + key

    @pytest.mark.parametrize("key", GROUP_KEYS)
    def test_every_builtin_group_key_is_accepted(self, key):
        assert Query(group_by=key).group_by == key

    @pytest.mark.parametrize(
        "path", ["category", "audience.level", "_private", "a-b", "a1.b2.c3"]
    )
    def test_a_safe_custom_field_path_is_accepted(self, path):
        assert Query(sort_by=path, group_by=path).group_by == path

    @pytest.mark.parametrize(
        "key",
        ["bogus key", "a..b", ".a", "a.", "1a", "a/b", "a b", "title;x", "<b>",
         "a\nb", "café"],
        ids=["space", "double-dot", "leading-dot", "trailing-dot", "leading-digit",
             "slash", "inner-space", "semicolon", "html", "newline", "non-ascii"],
    )
    def test_an_unsafe_key_is_rejected_for_sort_and_group(self, key):
        with pytest.raises(CatalogError, match="invalid sort key"):
            Query(sort_by=key)
        with pytest.raises(CatalogError, match="invalid group key"):
            Query(group_by=key)

    def test_the_sort_error_lists_the_alternatives(self):
        with pytest.raises(CatalogError) as caught:
            Query(sort_by="no way")
        message = str(caught.value)
        assert "'no way'" in message
        assert all(repr(key) in message for key in SORT_KEYS)
        assert "'-'" in message

    def test_the_group_error_lists_the_alternatives(self):
        with pytest.raises(CatalogError) as caught:
            Query(group_by="no way")
        message = str(caught.value)
        assert all(repr(key) in message for key in GROUP_KEYS)

    def test_an_empty_group_key_is_rejected(self):
        with pytest.raises(CatalogError, match="invalid group key ''"):
            Query(group_by="")

    def test_match_and_match_regex_are_mutually_exclusive(self):
        with pytest.raises(CatalogError, match="mutually exclusive"):
            Query(match="a", match_regex="b")

    def test_an_invalid_match_regex_is_rejected_at_construction(self):
        with pytest.raises(CatalogError, match="not supported"):
            Query(match_regex="(a|b)+")

    @pytest.mark.parametrize("limit", [-1, -100])
    def test_a_negative_limit_is_rejected(self, limit):
        with pytest.raises(CatalogError, match=f"limit must not be negative, got {limit}"):
            Query(limit=limit)

    def test_a_negative_offset_is_rejected(self):
        with pytest.raises(CatalogError, match="offset must not be negative, got -1"):
            Query(offset=-1)

    @pytest.mark.parametrize("limit", [None, 0, 1, 10**6])
    def test_a_non_negative_limit_is_accepted(self, limit):
        assert Query(limit=limit).limit == limit

    @pytest.mark.parametrize(
        "since, until",
        [(day(2024), day(2024)), (day(2024, 6), day(2024))],
        ids=["equal", "reversed"],
    )
    def test_an_empty_interval_is_rejected(self, since, until):
        with pytest.raises(CatalogError) as caught:
            Query(since=since, until=until)
        message = str(caught.value)
        assert str(since.date()) in message and str(until.date()) in message

    def test_an_open_ended_interval_is_accepted(self):
        assert Query(since=day(2024)).until is None
        assert Query(until=day(2024)).since is None

    def test_a_query_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            Query().limit = 3


# -- the bounded regular-expression subset ------------------------------------


class TestSafeRegex:
    @pytest.mark.parametrize(
        "pattern",
        ["abc", "^a.c$", r"\d{2}", "[abc]", "[a-z]{1,3}", "colou?r", "a{0,100}",
         "a{100}", "[(*+|]", r"a\.b", r"[a\]b]", "]", "", r"\(literal\)"],
        ids=["literal", "anchors-dot", "escape-repeat", "class", "class-repeat",
             "optional", "max-range", "max-exact", "metachars-in-class",
             "escaped-dot", "escaped-bracket-in-class", "stray-bracket", "empty",
             "escaped-parens"],
    )
    def test_the_supported_subset_compiles_case_insensitively(self, pattern):
        compiled = query_module._compile_safe_regex(pattern)
        assert compiled.pattern == pattern
        assert compiled.flags & query_module.re.IGNORECASE

    @pytest.mark.parametrize(
        "pattern, construct",
        [("(a)", "("), ("a)", ")"), ("a*", "*"), ("a+", "+"), ("a|b", "|"),
         ("(?i)a", "("), ("(a+)+$", "("), ("(?=a)", "(")],
        ids=["group", "close-paren", "star", "plus", "alternation", "inline-flag",
             "nested-quantifier", "lookahead"],
    )
    def test_unbounded_and_grouping_constructs_are_rejected(self, pattern, construct):
        with pytest.raises(CatalogError) as caught:
            query_module._compile_safe_regex(pattern)
        assert f"construct {construct!r} is not supported" in str(caught.value)

    @pytest.mark.parametrize(
        "pattern", [r"\1", r"a\9", r"\g<1>", r"\k<n>"], ids=["one", "nine", "g", "k"]
    )
    def test_backreferences_are_rejected(self, pattern):
        with pytest.raises(CatalogError, match="backreferences"):
            query_module._compile_safe_regex(pattern)

    @pytest.mark.parametrize(
        "pattern", ["a??", "a{2}?", "a?{2}", "a{1}{2}"],
        ids=["lazy-optional", "lazy-repeat", "repeat-optional", "double-repeat"],
    )
    def test_adjacent_quantifiers_are_rejected(self, pattern):
        with pytest.raises(CatalogError, match="adjacent"):
            query_module._compile_safe_regex(pattern)

    @pytest.mark.parametrize(
        "pattern, fragment",
        [
            ("a\\", "incomplete escape"),
            ("a{2", "unterminated bounded repeat"),
            ("a{x}", "invalid bounded repeat {x}"),
            ("a{,3}", "invalid bounded repeat {,3}"),
            ("a{}", "invalid bounded repeat {}"),
            ("a{-1}", "invalid bounded repeat {-1}"),
            ("a{3,2}", "0 <= m <= n <= 100"),
            ("a{0,101}", "0 <= m <= n <= 100"),
            ("a{101}", "0 <= m <= n <= 100"),
            ("a{2,}", "0 <= m <= n <= 100"),
            ("[abc", "unterminated character class"),
            ("{2}", "invalid match-regex"),
            ("[]", "invalid match-regex"),
            ("[z-a]", "invalid match-regex"),
        ],
        ids=["dangling-escape", "open-brace", "non-numeric", "no-lower", "empty-brace",
             "negative", "reversed", "upper-over", "exact-over", "unbounded-upper",
             "open-class", "nothing-to-repeat", "empty-class", "bad-range"],
    )
    def test_malformed_patterns_are_rejected_with_a_reason(self, pattern, fragment):
        with pytest.raises(CatalogError) as caught:
            query_module._compile_safe_regex(pattern)
        assert fragment in str(caught.value)

    def test_the_length_limit_is_inclusive(self):
        limit = query_module.MAX_REGEX_LENGTH
        assert query_module._compile_safe_regex("a" * limit).pattern == "a" * limit
        with pytest.raises(CatalogError, match=f"is {limit + 1} characters"):
            query_module._compile_safe_regex("a" * (limit + 1))

    def test_every_rejection_is_a_catalog_error_never_a_warning(self):
        # The project runs with warnings as errors; a pattern the subset
        # accepts must compile silently.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            query_module._compile_safe_regex(r"[a-z]{2,5}\d?$")


# -- to_selection -------------------------------------------------------------


class TestToSelection:
    def test_an_empty_query_is_an_empty_selection(self):
        selection = Query().to_selection()
        assert list(selection.terms) == []
        assert list(selection.sort_keys) == []
        assert selection.group_by == ""
        assert (selection.limit, selection.offset) == (None, 0)

    @pytest.mark.parametrize(
        "sort_by, expected",
        [("title", [("title", False)]), ("-title", [("title", True)]),
         ("none", []), ("-none", []), ("", []), ("-", []),
         ("audience.level", [("audience.level", False)])],
        ids=["asc", "desc", "none", "desc-none", "empty", "bare-dash", "path"],
    )
    def test_sort_translation(self, sort_by, expected):
        assert list(Query(sort_by=sort_by).to_selection().sort_keys) == expected

    @pytest.mark.parametrize(
        "group_by, expected", [("none", ""), ("playlist", "playlist"), ("a.b", "a.b")]
    )
    def test_group_translation(self, group_by, expected):
        assert Query(group_by=group_by).to_selection().group_by == expected

    def test_pagination_is_carried_over(self):
        selection = Query(limit=3, offset=2).to_selection()
        assert (selection.limit, selection.offset) == (3, 2)

    def test_one_term_per_predicate_in_a_fixed_order(self):
        selection = Query(
            channel="c", playlist="p", tags=("x", "y"), match="m", since=day(2024)
        ).to_selection()
        assert [type(term).__name__ for term in selection.terms] == [
            "_AnyOfTerm",
            "_AnyOfTerm",
            "FilterTerm",
            "FilterTerm",
            "_TextTerm",
            "_IntervalTerm",
        ]

    def test_since_and_until_share_one_interval_term(self):
        selection = Query(since=day(2023), until=day(2024)).to_selection()
        assert len(selection.terms) == 1

    def test_an_any_of_term_needs_at_least_one_field(self):
        with pytest.raises(ValueError, match="must not be empty"):
            query_module._AnyOfTerm((), "x")


# -- as_record ----------------------------------------------------------------


class TestAsRecord:
    def test_a_video_is_projected_with_year_and_custom_fields(self, catalog):
        record = as_record(catalog[0])
        assert record["id"] == A
        assert record["title"] == "Beta regression"
        assert record["year"] == "2022"
        assert record["published"] == day(2022, 5)
        assert record["tags"] == ["stats", "PCA"]
        assert record["handle"] == "statquest"
        assert record["video_count"] is None
        assert record["category"] == "ml"
        assert record["audience"] == {"level": "beginner"}
        assert record["_video"] is catalog[0]

    def test_an_undated_video_has_an_unknown_year(self):
        assert as_record(normalize_record(A))["year"] == "unknown"

    def test_a_channel_is_projected_with_its_video_count(self):
        (channel,) = derive_channel_records(
            [normalize_record({"id": A, "handle": "Foo"})]
        )
        record = as_record(channel)
        assert record["video_count"] == 1
        assert record["handle"] == "Foo"
        assert record["url"] == "https://www.youtube.com/@Foo"
        assert record["_video"] is channel

    def test_custom_fields_cannot_shadow_core_keys(self):
        # Normalisation reserves every core name, so flattening is safe.
        with pytest.raises(CatalogError, match="reserved"):
            normalize_record({"id": A, "fields": {"year": "1900"}})


# -- filters ------------------------------------------------------------------


class TestChannelAndPlaylistFilters:
    @pytest.mark.parametrize(
        "wanted",
        ["StatQuest", "statquest", "  STATQUEST ", UC_A, "@statquest", UC_A.upper()],
        ids=["name", "lower", "padded-upper", "channel-id", "at-handle", "id-upper"],
    )
    def test_channel_matches_name_id_or_handle_case_insensitively(self, catalog, wanted):
        # Both StatQuest videos share the name and id; the handle is recorded
        # on one of them only, and any one equivalent field is enough.
        assert select(catalog, channel=wanted) == ["Beta", "Gamma"]

    def test_a_handle_recorded_on_one_video_matches_only_that_video(self):
        catalog = normalize_catalog(
            [{"id": A, "handle": "Foo"}, {"id": B, "handle": "Bar"}, {"id": C}]
        )
        selected, total = apply_query(catalog, Query(channel="@foo"))
        assert [record.id for record in selected] == [A]
        assert total == 1

    def test_channel_is_an_exact_identity_not_a_substring(self, catalog):
        assert select(catalog, channel="Stat") == []
        assert select(catalog, channel="Quest") == []

    @pytest.mark.parametrize("wanted", ["Intro", "intro", PL_A], ids=["name", "lower", "id"])
    def test_playlist_matches_name_or_id(self, catalog, wanted):
        assert select(catalog, playlist=wanted) == ["Beta", "Gamma"]

    def test_channel_and_playlist_combine_with_and(self, catalog):
        assert select(catalog, channel="StatQuest", playlist="Advanced") == []
        assert select(catalog, channel="Other", playlist="Advanced") == ["alpha"]

    def test_an_unknown_identity_matches_nothing(self, catalog):
        selected, total = apply_query(catalog, Query(channel="nobody"))
        assert (selected, total) == ([], 0)

    def test_a_record_without_a_channel_is_not_matched_by_a_blank_like_value(self):
        catalog = normalize_catalog([{"id": A}, {"id": B, "channel": "X"}])
        assert [r.id for r in apply_query(catalog, Query(channel="x"))[0]] == [B]


class TestTagFilter:
    def test_one_tag(self, catalog):
        assert select(catalog, tags=("stats",)) == ["Beta", "Gamma", "Epsilon"]

    def test_all_tags_are_required(self, catalog):
        assert select(catalog, tags=("stats", "graphics")) == ["Gamma"]

    def test_tags_match_case_insensitively(self, catalog):
        assert select(catalog, tags=("pca",)) == ["Beta", "Gamma"]
        assert select(catalog, tags=("PCA",)) == ["Beta", "Gamma"]

    def test_a_tag_is_matched_whole(self, catalog):
        assert select(catalog, tags=("stat",)) == []

    def test_an_unknown_tag_matches_nothing(self, catalog):
        assert select(catalog, tags=("nope",)) == []


class TestTextFilters:
    def test_match_searches_title_and_description(self, catalog):
        assert select(catalog, match="pca") == ["Beta", "Gamma"]

    def test_match_is_case_insensitive(self, catalog):
        assert select(catalog, match="ALPHA") == ["alpha"]

    def test_match_does_not_search_other_fields(self, catalog):
        assert select(catalog, match="StatQuest") == []
        assert select(catalog, match="stats") == []

    def test_match_does_not_span_the_title_description_boundary(self):
        catalog = normalize_catalog([{"id": A, "title": "foo", "description": "bar"}])
        assert apply_query(catalog, Query(match="foobar"))[1] == 0
        assert apply_query(catalog, Query(match="foo bar"))[1] == 0

    def test_match_treats_regex_metacharacters_literally(self, catalog):
        assert select(catalog, match="(undated)") == ["delta"]
        assert select(catalog, match=".*") == []

    def test_match_folds_unicode_case(self):
        catalog = normalize_catalog([{"id": A, "title": "STRASSE Çay"}])
        assert apply_query(catalog, Query(match="straße"))[1] == 1
        assert apply_query(catalog, Query(match="çAY"))[1] == 1

    def test_match_regex_searches_title_and_description(self, catalog):
        assert select(catalog, match_regex="^gamma") == ["Gamma"]
        assert select(catalog, match_regex="pc[a-z]") == ["Beta", "Gamma"]
        assert select(catalog, match_regex="tricks$") == ["Gamma"]

    def test_match_regex_dot_does_not_cross_the_field_boundary(self):
        catalog = normalize_catalog([{"id": A, "title": "foo", "description": "bar"}])
        assert apply_query(catalog, Query(match_regex="foo.bar"))[1] == 0
        assert apply_query(catalog, Query(match_regex="foo.{0,3}bar"))[1] == 0

    def test_anchors_work_on_a_title_without_a_description(self):
        catalog = normalize_catalog([{"id": A, "title": "Lecture 12"}])
        assert apply_query(catalog, Query(match_regex=r"^lecture \d{1,2}$"))[1] == 1
        assert apply_query(catalog, Query(match_regex=r"^ecture"))[1] == 0

    @pytest.mark.xfail(
        strict=True,
        reason="title and description are joined; '^'/'$' anchor the joined text only",
    )
    @pytest.mark.parametrize(
        "pattern",
        [r"^lecture \d{1,2}$", r"\d$", "^about"],
        ids=["whole-title", "title-end", "description-start"],
    )
    def test_anchors_apply_to_each_field(self, pattern):
        # Docstring: "True if the pattern matches either field", and anchors
        # are part of the documented subset.
        catalog = normalize_catalog(
            [{"id": A, "title": "Lecture 12", "description": "about PCA"}]
        )
        assert apply_query(catalog, Query(match_regex=pattern))[1] == 1

    def test_match_regex_is_case_insensitive(self, catalog):
        assert select(catalog, match_regex="EPSILON") == ["Epsilon"]

    def test_match_regex_only_inspects_a_bounded_prefix(self):
        limit = query_module.MAX_REGEX_TEXT
        catalog = normalize_catalog(
            [{"id": A, "title": "t", "description": "x" * limit + "needle"}]
        )
        assert apply_query(catalog, Query(match_regex="needle"))[1] == 0
        assert apply_query(catalog, Query(match_regex="x{3}"))[1] == 1


class TestIntervalFilter:
    def test_since_is_inclusive(self, catalog):
        assert select(catalog, since=day(2023)) == ["alpha", "Gamma", "Epsilon"]

    def test_until_is_exclusive(self, catalog):
        assert select(catalog, until=day(2023)) == ["Beta"]
        assert select(catalog, until=day(2024)) == ["Beta", "Gamma", "Epsilon"]

    def test_the_interval_is_half_open(self, catalog):
        assert select(catalog, since=day(2023), until=day(2024)) == ["Gamma", "Epsilon"]

    def test_an_undated_video_is_excluded_from_any_interval(self, catalog):
        assert "delta" not in select(catalog, since=day(1900))
        assert "delta" not in select(catalog, until=day(2999))

    def test_an_undated_video_is_kept_without_an_interval(self, catalog):
        assert "delta" in select(catalog)


# -- sorting, pagination, totals ----------------------------------------------


class TestSortAndPaginate:
    def test_no_sort_keeps_catalog_order(self, catalog):
        assert select(catalog) == ["Beta", "alpha", "Gamma", "delta", "Epsilon"]

    def test_title_sort_ignores_case(self, catalog):
        assert select(catalog, sort_by="title") == [
            "alpha", "Beta", "delta", "Epsilon", "Gamma",
        ]

    def test_descending_title_sort(self, catalog):
        assert select(catalog, sort_by="-title") == [
            "Gamma", "Epsilon", "delta", "Beta", "alpha",
        ]

    def test_published_sort_puts_undated_last_in_both_directions(self, catalog):
        assert select(catalog, sort_by="published") == [
            "Beta", "Gamma", "Epsilon", "alpha", "delta",
        ]
        assert select(catalog, sort_by="-published") == [
            "alpha", "Epsilon", "Gamma", "Beta", "delta",
        ]

    def test_numeric_sort_is_numeric_and_stable_for_ties(self, catalog):
        # alpha and Epsilon tie on 30 seconds: catalog order is preserved.
        assert select(catalog, sort_by="duration") == [
            "alpha", "Epsilon", "Beta", "Gamma", "delta",
        ]

    def test_position_sort(self, catalog):
        assert select(catalog, sort_by="position")[:3] == ["alpha", "Gamma", "Beta"]

    def test_sort_by_a_custom_field(self, catalog):
        assert select(catalog, sort_by="category")[:2] == ["alpha", "Beta"]

    def test_sort_by_a_nested_custom_field(self, catalog):
        assert select(catalog, sort_by="audience.level")[0] == "Beta"

    def test_sorting_is_deterministic_and_does_not_mutate_the_catalog(self, catalog):
        before = list(catalog)
        first = select(catalog, sort_by="-title")
        assert select(catalog, sort_by="-title") == first
        assert catalog == before

    def test_limit_applies_after_sorting_and_total_counts_matches(self, catalog):
        selected, total = apply_query(catalog, Query(sort_by="title", limit=2))
        assert titles(selected) == ["alpha", "Beta"]
        assert total == 5

    def test_offset_then_limit(self, catalog):
        selected, total = apply_query(
            catalog, Query(sort_by="title", offset=1, limit=2)
        )
        assert titles(selected) == ["Beta", "delta"]
        assert total == 5

    def test_total_counts_filter_matches_not_the_catalog(self, catalog):
        selected, total = apply_query(catalog, Query(tags=("stats",), limit=1))
        assert titles(selected) == ["Beta"]
        assert total == 3

    def test_limit_zero_selects_nothing_but_reports_the_total(self, catalog):
        assert apply_query(catalog, Query(limit=0)) == ([], 5)

    def test_an_offset_past_the_end_selects_nothing(self, catalog):
        assert apply_query(catalog, Query(offset=99)) == ([], 5)

    def test_a_limit_beyond_the_end_is_harmless(self, catalog):
        assert len(apply_query(catalog, Query(limit=99))[0]) == 5

    def test_any_iterable_of_records_is_accepted(self, catalog):
        selected, total = apply_query(iter(catalog), Query(sort_by="title", limit=1))
        assert titles(selected) == ["alpha"]
        assert total == 5

    def test_the_original_record_objects_are_returned(self, catalog):
        selected, _ = apply_query(catalog, Query(sort_by="title"))
        assert all(any(record is original for original in catalog) for record in selected)


class TestFieldPresenceValidation:
    @pytest.mark.parametrize("key", ["categories", "audience.nope", "nope.level"])
    def test_a_sort_field_absent_everywhere_is_an_error(self, catalog, key):
        with pytest.raises(CatalogError) as caught:
            apply_query(catalog, Query(sort_by=key))
        message = str(caught.value)
        assert message.startswith(f"option ':sort:' field {key!r} is not present")
        assert "fields:" in message

    def test_the_descending_prefix_is_not_part_of_the_reported_field(self, catalog):
        with pytest.raises(CatalogError, match=r"':sort:' field 'nope'"):
            apply_query(catalog, Query(sort_by="-nope"))

    def test_a_group_field_absent_everywhere_is_an_error(self, catalog):
        with pytest.raises(CatalogError, match=r"':group-by:' field 'categories'"):
            apply_query(catalog, Query(group_by="categories"))
        with pytest.raises(CatalogError, match=r"':group-by:' field 'categories'"):
            group_records(catalog, Query(group_by="categories"))

    def test_a_field_present_on_only_some_records_is_fine(self, catalog):
        assert len(select(catalog, sort_by="category")) == 5

    def test_a_field_whose_only_value_is_none_still_exists(self):
        catalog = normalize_catalog([{"id": A, "fields": {"category": None}}])
        assert apply_query(catalog, Query(sort_by="category"))[1] == 1

    def test_an_empty_catalog_cannot_contradict_a_field_name(self):
        assert apply_query([], Query(sort_by="anything", group_by="whatever")) == ([], 0)
        assert group_records([], Query(group_by="whatever")) == []

    def test_presence_is_checked_before_filtering(self, catalog):
        # A typo must not be masked by a filter that happens to match nothing.
        with pytest.raises(CatalogError, match="not present"):
            apply_query(catalog, Query(channel="nobody", sort_by="categories"))


# -- grouping -----------------------------------------------------------------


class TestGroupRecords:
    def test_no_grouping_is_one_unlabelled_section(self, catalog):
        assert group_records(catalog, Query()) == [("", catalog)]

    def test_no_grouping_of_nothing_is_one_empty_section(self):
        assert group_records([], Query()) == [("", [])]

    def test_playlist_groups_follow_first_appearance_with_ungrouped_last(self, catalog):
        sections = group_records(catalog, Query(group_by="playlist"))
        assert [(label, titles(group)) for label, group in sections] == [
            ("Intro", ["Beta", "Gamma"]),
            ("Advanced", ["alpha"]),
            ("Ungrouped", ["delta", "Epsilon"]),
        ]

    def test_ungrouped_is_last_even_when_it_appears_first(self, catalog):
        reordered = [catalog[3], *catalog[:3]]
        labels = [label for label, _ in group_records(reordered, Query(group_by="channel"))]
        assert labels == ["StatQuest", "Other", "Ungrouped"]

    def test_year_groups_use_unknown_for_undated_videos(self, catalog):
        sections = group_records(catalog, Query(group_by="year"))
        assert [(label, titles(group)) for label, group in sections] == [
            ("2022", ["Beta"]),
            ("2024", ["alpha"]),
            ("2023", ["Gamma", "Epsilon"]),
            ("unknown", ["delta"]),
        ]

    def test_grouping_by_a_custom_field(self, catalog):
        sections = group_records(catalog, Query(group_by="category"))
        assert [(label, len(group)) for label, group in sections] == [
            ("ml", 1), ("gfx", 1), ("Ungrouped", 3),
        ]

    def test_grouping_by_a_nested_custom_field(self, catalog):
        sections = group_records(catalog, Query(group_by="audience.level"))
        assert [(label, len(group)) for label, group in sections] == [
            ("beginner", 1), ("Ungrouped", 4),
        ]

    def test_every_record_lands_in_exactly_one_scalar_group(self, catalog):
        sections = group_records(catalog, Query(group_by="playlist"))
        grouped = [record for _, group in sections for record in group]
        assert sorted(record.id for record in grouped) == sorted(
            record.id for record in catalog
        )

    def test_groups_hold_the_original_record_objects(self, catalog):
        sections = group_records(catalog, Query(group_by="playlist"))
        assert sections[0][1][0] is catalog[0]

    def test_hostile_group_labels_are_returned_as_plain_text(self):
        catalog = normalize_catalog([{"id": A, "playlist": "<script>x</script>"}])
        assert group_records(catalog, Query(group_by="playlist"))[0][0] == (
            "<script>x</script>"
        )


# -- channel catalogs ---------------------------------------------------------


class TestChannelCatalogs:
    @pytest.fixture
    def channels(self):
        _, records = normalize_gallery_catalog(
            {
                "channels": [
                    {"id": "@Zed", "tags": ["x"]},
                    {"handle": "Alpha", "channel_id": UC_A, "title": "alpha one"},
                    {"id": UC_B, "title": "Mid", "description": "about zed"},
                ]
            }
        )
        return records

    def test_channels_sort_by_title(self, channels):
        selected, total = apply_query(channels, Query(sort_by="title"))
        assert [record.title for record in selected] == ["@Zed", "alpha one", "Mid"]
        assert total == 3

    @pytest.mark.parametrize(
        "wanted, expected",
        [("@zed", ["@Zed"]), ("zed", ["@Zed"]), (UC_A, ["alpha one"]),
         ("alpha", ["alpha one"]), ("Mid", ["Mid"])],
        ids=["at-handle", "bare-handle", "channel-id", "handle-of-id", "title"],
    )
    def test_channel_filter_over_channel_cards(self, channels, wanted, expected):
        selected, _ = apply_query(channels, Query(channel=wanted))
        assert [record.title for record in selected] == expected

    def test_match_searches_channel_descriptions(self, channels):
        selected, _ = apply_query(channels, Query(match="zed"))
        assert [record.title for record in selected] == ["@Zed", "Mid"]

    def test_an_interval_excludes_authored_channels_which_carry_no_date(self, channels):
        assert apply_query(channels, Query(since=day(1900))) == ([], 0)


class TestModuleSurface:
    def test_public_names_are_exported(self):
        for name in query_module.__all__:
            assert hasattr(query_module, name), name

    def test_position_is_a_sort_key_and_year_a_group_key(self):
        assert "position" in SORT_KEYS and "none" in SORT_KEYS
        assert "year" in GROUP_KEYS and "none" in GROUP_KEYS
