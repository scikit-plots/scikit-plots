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


class TestVinCheck:
    """CP-117: the VIN check digit of 49 CFR 565.15."""

    _PROSE = re.compile(r"(?i)\b[A-HJ-NPR-Z0-9]{17}\b")
    _LABELLED = re.compile(
        r"(?i)(?:\b(?P<label>VIN)\s*:?\s*)?\b(?P<value>[A-HJ-NPR-Z0-9]{17})\b"
    )

    @pytest.mark.parametrize(
        "vin",
        [
            "1M8GDM9AXKP042788",
            "1HGCM82633A004352",
            "JH4KA4650JC000403",
            "1m8gdm9axkp042788",
        ],
    )
    def test_valid_check_digits(self, vin):
        assert VALIDATORS["vin_check"](self._PROSE.fullmatch(vin))

    @pytest.mark.parametrize(
        "vin",
        [
            "1M8GDM9A1KP042788",  # wrong check digit
            "11111111111111111",  # passes the arithmetic, a placeholder
            "WVWZZZ3BZWE689725",  # no check digit, as written outside North America
            "1M8GDM9AXKP04278A",  # the last four characters must be numeric
        ],
    )
    def test_refused_without_a_label(self, vin):
        assert not VALIDATORS["vin_check"](self._PROSE.fullmatch(vin))

    def test_a_label_accepts_any_well_formed_vin(self):
        validate = VALIDATORS["vin_check"]
        assert validate(self._LABELLED.fullmatch("VIN: WVWZZZ3BZWE689725"))
        assert not validate(self._LABELLED.fullmatch("WVWZZZ3BZWE689725"))
        assert not validate(self._LABELLED.fullmatch("VIN: 11111111111111111"))

    def test_the_remainder_ten_is_x(self):
        # 1M8GDM9AXKP042788: the weighted sum leaves 10, written X in position 9.
        assert VALIDATORS["vin_check"](self._PROSE.fullmatch("1M8GDM9AXKP042788"))
        assert not VALIDATORS["vin_check"](self._PROSE.fullmatch("1M8GDM9A0KP042788"))


class TestDeTaxId:
    """CP-119: the German tax identification number's published checks."""

    _RE = re.compile(r"\b[1-9]\d(?:\d{9}|(?: \d{3}){3})\b")

    @pytest.mark.parametrize(
        "number",
        [
            "86504957138",
            "86 504 957 138",
            "12345678911",  # 1 twice
            "11213456783",  # 1 three times, never three in a row
            "86504957314",
        ],
    )
    def test_valid(self, number):
        assert VALIDATORS["de_tax_id"](self._RE.fullmatch(number))

    @pytest.mark.parametrize(
        "number",
        [
            "86504957139",  # wrong check digit
            "47198320652",  # every digit once: none repeated
            "11123456786",  # three equal digits in a row
            "11223456789",  # two digits repeated
        ],
    )
    def test_refused(self, number):
        assert not VALIDATORS["de_tax_id"](self._RE.fullmatch(number))

    def test_the_published_test_number_has_its_check_digit_but_is_not_personal(self):
        # 02476291358, from the published specification: its check digit is 8,
        # and a leading 0 marks a test number, which is refused.
        loose = re.fullmatch(r"\d{11}", "02476291358")
        assert not VALIDATORS["de_tax_id"](loose)

    def test_a_value_group_is_read_without_its_label(self):
        labelled = re.fullmatch(
            r"(?P<label>IdNr) (?P<value>\d{11})", "IdNr 86504957138"
        )
        assert VALIDATORS["de_tax_id"](labelled)
