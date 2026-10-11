"""
Tests for :mod:`scikitplot.cleanprompt._canonical`.

Notes
-----
**Developer notes.** The length property is what lets a match in canonical
text be a span in the original, so it is tested over random text drawn from
the whole Basic Multilingual Plane and the mathematical letters, not only
over the characters the table was written for.
"""

from __future__ import annotations

import random

import pytest

from .._canonical import canonical, value_pattern
from .._surrogates import surrogate_for


class TestCanonical:
    @pytest.mark.parametrize("seed", range(10))
    def test_length_is_preserved_for_any_text(self, seed):
        rng = random.Random(seed)
        points = [rng.randrange(0, 0x10000) for _ in range(2000)]
        points += [rng.randrange(0x1D400, 0x1D800) for _ in range(200)]
        text = "".join(chr(p) for p in points if not 0xD800 <= p <= 0xDFFF)
        assert len(canonical(text)) == len(text)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("a b\tc\nd e", "a b c d e"),
            ("O’Brien ʼx `y", "O'Brien 'x 'y"),
            ("a–b—c−d", "a-b-c-d"),
            ("Ｍａｒ", "Mar"),
            ("\U0001d40c", "M"),
            ("ﬁ", "ﬁ"),  # expands under NFKC, so it is left alone
        ],
    )
    def test_forms(self, raw, expected):
        assert canonical(raw) == expected


class TestValuePattern:
    PATTERN = value_pattern(["Marion Holt", "O'Brien", "+1 555 010 4477"])

    @pytest.mark.parametrize(
        "text",
        [
            "Marion Holt",
            "MARION HOLT",
            "marion holt",
            "Marion\nHolt",
            "Marion  \tHolt",
            "Ｍarion Holt",
            "O’Brien",
            "o'brien",
            "call +1 555 010 4477.",
            "Marion Holt's",
        ],
    )
    def test_equivalent_writings_match(self, text):
        assert self.PATTERN.search(canonical(text))

    @pytest.mark.parametrize(
        "text",
        [
            "Holt, Marion",
            "M. Holt",
            "Holt",
            "Marionette Holtz",
            "xMarion Holt",
            "OBrien",
        ],
    )
    def test_rewordings_and_parts_do_not(self, text):
        assert not self.PATTERN.search(canonical(text))

    def test_no_values_matches_nothing(self):
        assert not value_pattern([]).search("anything")
        assert not value_pattern(["", "   "]).search("   ")


class TestSurrogatesAvoidHeldValues:
    """CP-071: a stand-in never contains a value the conversation holds."""

    def test_forbidden_candidates_are_skipped(self):
        held = value_pattern(["Marion"])
        chosen = surrogate_for(
            "EMAIL", 1, forbidden=lambda one: bool(held.search(canonical(one)))
        )
        assert chosen is not None and "marion" not in chosen.lower()

    def test_end_to_end_the_model_never_sees_a_held_name(self):
        from .. import FluentCleanPrompt

        guard = FluentCleanPrompt().style("surrogate").guard()
        guard.outgoing(
            "name,email\nMarion,bob@example.org\nDevin,x@example.org\n", "csv"
        )
        safe = guard.outgoing("mail ann@example.com and cc@example.net")
        assert "marion" not in safe.lower() and "devin" not in safe.lower()
        assert guard.incoming(safe) == "mail ann@example.com and cc@example.net"


class TestDetectionView:
    """
    ``CP-098``: the view and the way back from it.

    Notes
    -----
    **Developer notes.** The property every caller depends on is that a span
    in the view maps to the slice of the original that *is* the value as
    written. It is checked over random text salted with format characters,
    not only over the examples below.
    """

    def test_ascii_has_no_view(self):
        from .._canonical import detection_view

        assert detection_view("plain text, 192.0.2.10 and a@b.co\n\t") is None

    def test_text_the_view_would_not_change_has_no_view(self):
        from .._canonical import detection_view

        assert detection_view("café, naïve, 東京, Ελλάδα") is None

    def test_format_characters_are_removed(self):
        from .._canonical import detection_view

        for char in ("\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u00ad",
                     "\u202e", "\u2066", "\U000e0041"):
            view = detection_view(f"ab{char}cd")
            assert view.text == "abcd", repr(char)
            assert view.source_span(0, 4) == (0, 5)

    def test_compatibility_forms_are_folded_in_place(self):
        from .._canonical import detection_view

        view = detection_view("\uff41\uff44\uff41\uff20\uff45\uff58\uff41\uff4d\uff50\uff4c\uff45\uff0e\uff43\uff4f\uff4d")
        assert view.text == "ada@example.com"
        assert view.shifts == ()

    def test_ascii_control_whitespace_is_kept(self):
        """A line break stays a line break: patterns must not learn to span lines."""
        from .._canonical import detection_view

        view = detection_view("a\u00a0b\nc\td")
        assert view.text == "a b\nc\td"

    def test_unicode_spaces_and_dashes_fold(self):
        from .._canonical import detection_view

        assert detection_view("+1\u2011555\u00a00100").text == "+1-555 0100"

    def test_edges_are_not_widened(self):
        """Invisible characters just outside a value stay outside it."""
        from .._canonical import detection_view

        text = "\u200bab\u200bc\u200b"
        view = detection_view(text)
        assert view.text == "abc"
        start, end = view.source_span(0, 3)
        assert text[start:end] == "ab\u200bc"

    @pytest.mark.parametrize("seed", range(20))
    def test_every_view_span_maps_onto_the_same_characters(self, seed):
        from .._canonical import detection_view

        rng = random.Random(seed)
        salt = ("\u200b", "\u200d", "\u2060", "\u00ad", "\u202e", "\ufeff")
        word = "".join(rng.choice("abcdefghij") for _ in range(rng.randrange(5, 60)))
        pieces = []
        for char in word:
            pieces.append(char)
            while rng.random() < 0.35:
                pieces.append(rng.choice(salt))
        text = "".join(pieces)
        view = detection_view(text)
        assert view.text == word
        for _ in range(25):
            start = rng.randrange(0, len(word))
            end = rng.randrange(start + 1, len(word) + 1)
            source_start, source_end = view.source_span(start, end)
            kept = "".join(c for c in text[source_start:source_end] if c not in salt)
            assert kept == word[start:end]
            assert text[source_start] not in salt and text[source_end - 1] not in salt

    def test_a_salted_text_costs_no_more_per_lookup(self):
        """Lookups are logarithmic: a zero-width character after every letter."""
        import time

        from .._canonical import detection_view

        text = "\u200b".join("a" * 200_000)
        view = detection_view(text)
        started = time.perf_counter()
        for index in range(0, len(view.text), 97):
            view.source_index(index)
        assert time.perf_counter() - started < 2.0
