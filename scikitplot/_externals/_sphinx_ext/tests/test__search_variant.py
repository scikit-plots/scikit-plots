"""Tests for :mod:`_search_variant`: one search presentation per directive."""

from __future__ import annotations

import pytest

from .._search_variant import (
    SEARCH_VARIANT_KEYS,
    resolve_search_variant,
    search_variant_option,
)
from .._sphinx_collection.assets import SEARCH_VARIANTS


class TestOption:
    @pytest.mark.parametrize("value", sorted(SEARCH_VARIANTS))
    def test_a_known_variant_is_returned(self, value):
        assert search_variant_option(value) == value

    @pytest.mark.parametrize("value", sorted(SEARCH_VARIANTS))
    def test_case_and_surrounding_space_are_normalised(self, value):
        assert search_variant_option("  " + value.upper() + " ") == value

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_no_value_stays_none(self, value):
        assert search_variant_option(value) is None

    @pytest.mark.parametrize("value", ["modern", "pill", "classic,pill-overflow", "0"])
    def test_an_unknown_variant_is_refused(self, value):
        with pytest.raises(ValueError, match="search variant must be"):
            search_variant_option(value)


class TestResolve:
    def test_the_default_applies_when_nothing_is_given(self):
        assert resolve_search_variant({}, "classic") == "classic"

    def test_the_default_is_normalised(self):
        assert resolve_search_variant({}, " Classic ") == "classic"

    def test_an_invalid_default_is_refused(self):
        with pytest.raises(ValueError, match="search variant must be"):
            resolve_search_variant({}, "modern")

    @pytest.mark.parametrize("key", SEARCH_VARIANT_KEYS)
    def test_a_dedicated_option_wins_over_the_default(self, key):
        assert resolve_search_variant({key: "classic"}, "pill-overflow") == "classic"

    @pytest.mark.parametrize("key", ["interactive", "searchable"])
    def test_an_activation_option_may_carry_the_variant(self, key):
        assert resolve_search_variant({key: "classic"}, "pill-overflow") == "classic"

    @pytest.mark.parametrize("flag", [None, ""])
    def test_a_valueless_activation_flag_does_not_choose(self, flag):
        assert resolve_search_variant({"interactive": flag}, "pill-overflow") == "pill-overflow"

    def test_the_same_value_twice_is_not_a_conflict(self):
        options = {"search-variant": "classic", "interactive": "CLASSIC"}
        assert resolve_search_variant(options, "pill-overflow") == "classic"

    def test_conflicting_values_are_refused_and_named(self):
        options = {"search-variant": "classic", "interactive": "pill-overflow"}
        with pytest.raises(ValueError) as caught:
            resolve_search_variant(options, "classic")
        message = str(caught.value)
        assert "conflicting search variants" in message
        assert "search-variant=classic" in message and "interactive=pill-overflow" in message

    def test_the_result_does_not_depend_on_option_order(self):
        first = {"interactive": "classic", "search_variant": "classic"}
        second = {"search_variant": "classic", "interactive": "classic"}
        assert resolve_search_variant(first, "pill-overflow") == resolve_search_variant(second, "pill-overflow")

    def test_custom_activation_keys_replace_the_defaults(self):
        options = {"interactive": "classic", "live": "pill-overflow"}
        assert resolve_search_variant(options, "classic", activation_keys=("live",)) == "pill-overflow"

    def test_an_invalid_option_value_is_refused(self):
        with pytest.raises(ValueError, match="search variant must be"):
            resolve_search_variant({"search-variant": "modern"}, "classic")
