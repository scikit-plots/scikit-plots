"""
Tests for :mod:`scikitplot.cleanprompt._hooks`.

Notes
-----
**Developer notes.** Each validator is checked against a published canonical
positive and a single-digit mutation of it, so a validator that accepted
everything, or nothing, fails here rather than in a pack's examples.
"""

from __future__ import annotations

import re

import pytest

from .._hooks import VALIDATORS, get_validator, validator_names


def _accepts(name, value):
    return get_validator(name)(re.fullmatch(r".+", value))


class TestRegistry:
    def test_names_are_sorted_and_match_the_registry(self):
        assert validator_names() == tuple(sorted(VALIDATORS))

    def test_none_means_no_validator(self):
        assert get_validator(None) is None

    def test_unknown_name_raises_with_the_known_ones(self):
        with pytest.raises(KeyError) as caught:
            get_validator("eval")
        assert "luhn" in str(caught.value)


@pytest.mark.parametrize(
    ("name", "good", "bad"),
    [
        ("luhn", "4242 4242 4242 4242", "4242 4242 4242 4241"),
        ("iban_mod97", "GB82 WEST 1234 5698 7654 32", "GB82 WEST 1234 5698 7654 33"),
        ("tckn", "10000000146", "10000000147"),
        ("nhs_mod11", "943 476 5919", "943 476 5918"),
        ("npi", "1234567893", "1234567894"),
        ("aba_routing", "011000015", "011000016"),
    ],
)
def test_checksum_accepts_the_canonical_value_and_rejects_a_mutation(name, good, bad):
    assert _accepts(name, good) is True
    assert _accepts(name, bad) is False


@pytest.mark.parametrize("name", sorted(VALIDATORS))
@pytest.mark.parametrize("junk", ["", "abc", "0" * 40, "12-", "٣٤"])
def test_every_validator_is_total(name, junk):
    """Pure and total: junk returns a bool and never raises."""
    match = re.fullmatch(r"(?s).*", junk)
    assert get_validator(name)(match) in (True, False)


def test_repeated_digits_are_refused():
    assert _accepts("not_repeated_digit", "1111111111") is False
    assert _accepts("not_repeated_digit", "1234567890") is True
