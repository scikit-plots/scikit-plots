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
