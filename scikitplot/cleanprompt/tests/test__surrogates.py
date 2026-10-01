"""
Tests for :mod:`scikitplot.cleanprompt._surrogates`.

Notes
-----
**Developer notes.** A surrogate is an ordinary phrase, so it does not have the
one property a bracket label gets for free: being obviously not part of the
text. Everything that protects it has to be asserted rather than assumed.

Three properties carry the design, and each has a failure that is worse than a
crash because it is silent.

*Uniqueness.* Two values sharing a stand-in cannot be told apart on the way
back, so one of them restores to the other's value — ``CP-001`` arriving from
the restoration side.

*Non-collision with the document.* If the text already contains the phrase a
surrogate would use, restoration cannot tell the user's own words from the
substitution and rewrites both.

*The credential exclusion.* A plausible-looking card number or access key can
be mistaken for real by a person or acted on by a system. Those kinds must keep
their placeholders, and the test names them rather than trusting the table.
"""

from __future__ import annotations

import pytest

from .._surrogates import (
    DEFAULT_STYLE,
    STYLES,
    SURROGATE_KINDS,
    surrogate_for,
)


class TestWhichKindsAreSurrogated:
    """The split between an invented value and a placeholder."""

    def test_linguistic_kinds_get_a_stand_in(self):
        for kind in ("PERSON", "ORG", "GPE", "LOC", "FAC"):
            assert surrogate_for(kind, 1) is not None

    @pytest.mark.parametrize(
        "kind",
        [
            "CREDIT_CARD",
            "IBAN",
            "SSN_US",
            "AWS_ACCESS_KEY",
            "JWT",
            "PRIVATE_KEY",
            "MAC",
            "IPV4",
            "IPV6",
        ],
    )
    def test_credentials_keep_their_placeholder(self, kind):
        """A plausible-looking credential is a hazard, not a convenience."""
        assert surrogate_for(kind, 1) is None

    def test_norp_keeps_its_placeholder(self):
        """It is adjectival; an invented demonym reads as nonsense."""
        assert surrogate_for("NORP", 1) is None

    def test_an_unknown_kind_keeps_its_placeholder(self):
        assert surrogate_for("SOMETHING_NEW", 1) is None

    def test_the_declared_set_is_what_is_implemented(self):
        for kind in SURROGATE_KINDS:
            assert surrogate_for(kind, 1) is not None


class TestUniqueness:
    """Two values must never share a stand-in."""

    def test_consecutive_people_are_distinct(self):
        names = [surrogate_for("PERSON", i) for i in range(1, 30)]
        assert len(set(names)) == len(names)

    def test_consecutive_people_do_not_share_a_surname(self):
        """Sharing one reads as a family and implies a relationship."""
        surnames = [surrogate_for("PERSON", i).split()[-1] for i in range(1, 6)]
        assert len(set(surnames)) == len(surnames)

    def test_the_bank_is_exhausted_before_it_repeats(self):
        names = {surrogate_for("PERSON", i) for i in range(1, 577)}
        assert len(names) == 576

    def test_an_already_issued_name_is_skipped(self):
        first = surrogate_for("PERSON", 1)
        second = surrogate_for("PERSON", 1, avoid={first})
        assert second != first

    def test_many_values_stay_unique_under_avoidance(self):
        issued = set()
        for ordinal in range(1, 200):
            name = surrogate_for("PERSON", ordinal, avoid=issued)
            assert name not in issued
            issued.add(name)
        assert len(issued) == 199


class TestCollisionWithTheDocument:
    """A stand-in must not be a phrase the text already uses."""

    def test_a_name_present_in_the_source_is_skipped(self):
        taken = surrogate_for("PERSON", 1)
        other = surrogate_for("PERSON", 1, source="a note about {0}".format(taken))
        assert other != taken

    def test_it_gives_up_rather_than_looping(self):
        """A pathological source falls back to a placeholder, not a hang."""
        every = " ".join(surrogate_for("PERSON", i) for i in range(1, 300))
        assert surrogate_for("PERSON", 1, source=every) is None

    def test_giving_up_is_reported_as_none(self):
        every = " ".join(surrogate_for("GPE", i) for i in range(1, 60))
        assert surrogate_for("GPE", 1, source=every) is None


class TestReservedForms:
    """Generated contact details must not be able to reach anybody."""

    def test_an_address_uses_the_reserved_domain(self):
        """RFC 2606 reserves .invalid permanently, so it cannot resolve."""
        assert surrogate_for("EMAIL", 1).endswith("@example.invalid")

    def test_a_url_uses_the_reserved_domain(self):
        assert surrogate_for("URL", 1).startswith("https://example.invalid/")

    def test_a_telephone_number_is_in_the_fiction_range(self):
        """+1 555 0100-0199 is reserved for fiction in the NANP."""
        for ordinal in (1, 2, 50, 99):
            number = surrogate_for("PHONE", ordinal)
            assert number.startswith("+1 555 0")
            assert 100 <= int(number.rsplit(" ", 1)[1]) <= 199

    def test_addresses_are_distinct_per_value(self):
        addresses = {surrogate_for("EMAIL", i) for i in range(1, 40)}
        assert len(addresses) == 39


class TestDeterminism:
    """The same value must get the same stand-in every run."""

    def test_the_same_ordinal_gives_the_same_name(self):
        assert surrogate_for("PERSON", 7) == surrogate_for("PERSON", 7)

    def test_different_kinds_do_not_share_a_stand_in(self):
        person = surrogate_for("PERSON", 1)
        org = surrogate_for("ORG", 1)
        assert person != org

    def test_it_does_not_depend_on_call_order(self):
        first = [surrogate_for("GPE", i) for i in (3, 1, 2)]
        second = [surrogate_for("GPE", i) for i in (1, 2, 3)]
        assert first == [second[2], second[0], second[1]]


class TestRejections:
    """Bad input, refused rather than absorbed."""

    @pytest.mark.parametrize("ordinal", [0, -1, -100])
    def test_an_ordinal_below_one_is_refused(self, ordinal):
        with pytest.raises(ValueError) as caught:
            surrogate_for("PERSON", ordinal)
        assert "at least 1" in str(caught.value)

    def test_the_default_style_is_the_unmistakable_one(self):
        """A bracket label cannot be mistaken for a real value; a name can."""
        assert DEFAULT_STYLE == "placeholder"
        assert set(STYLES) == {"placeholder", "surrogate"}


class TestEndToEnd:
    """The property that matters: the values come back."""

    @staticmethod
    def _policy():
        from .. import DEFAULT_POLICY, TagStyle

        return DEFAULT_POLICY.evolve(tag_style=TagStyle(style="surrogate"))

    def test_a_round_trip_is_exact(self):
        from .. import Redactor, restore

        policy = self._policy()
        text = "Mail ada@example.com or call +1 555 010 4477 from 192.168.1.10."
        result = Redactor(policy=policy).redact(text)
        assert restore(result.text, result.vault, policy=policy).text == text

    def test_no_value_survives_in_the_redacted_text(self):
        from .. import Redactor

        result = Redactor(policy=self._policy()).redact("Mail ada@example.com")
        assert "ada@example.com" not in result.text

    def test_the_redacted_text_reads_as_prose(self):
        """The point of the style: no bracket tokens where a name belongs."""
        from .. import Redactor

        result = Redactor(policy=self._policy()).redact("Mail ada@example.com now")
        assert "[EMAIL" not in result.text
        assert "@example.invalid" in result.text

    def test_a_credential_still_gets_a_placeholder(self):
        from .. import Redactor

        result = Redactor(policy=self._policy()).redact("card 4242 4242 4242 4242")
        assert "[CREDIT_CARD-1]" in result.text

    def test_a_model_reply_restores(self):
        """A stand-in appearing anywhere in the reply, not only where it was."""
        from .. import Redactor, restore

        policy = self._policy()
        result = Redactor(policy=policy).redact("Mail ada@example.com")
        stand_in = result.entries[0].label
        reply = "I have written to {0} twice already.".format(stand_in)
        assert "ada@example.com" in restore(reply, result.vault, policy=policy).text

    def test_the_two_styles_have_different_grammar_fingerprints(self):
        """A vault written in one style must not be read as the other."""
        from .. import DEFAULT_POLICY

        assert (
            self._policy().tag_style.fingerprint
            != DEFAULT_POLICY.tag_style.fingerprint
        )

    def test_reading_a_surrogate_vault_as_placeholders_is_refused(self):
        from .. import DEFAULT_POLICY, PolicyError, Redactor, restore

        policy = self._policy()
        result = Redactor(policy=policy).redact("Mail ada@example.com")
        with pytest.raises(PolicyError):
            restore(result.text, result.vault, policy=DEFAULT_POLICY)

    def test_placeholders_remain_the_default(self):
        from .. import Redactor

        result = Redactor().redact("Mail ada@example.com")
        assert result.text == "Mail [EMAIL-1]"

    def test_a_stand_in_does_not_claim_part_of_a_longer_one(self):
        """CP-001 on the restoration side: longest key wins."""
        from .. import Redactor, restore

        policy = self._policy()
        result = Redactor(policy=policy).redact(
            "Mail ada@example.com and bob@example.com"
        )
        reply = " and ".join(entry.label for entry in result.entries)
        restored = restore(reply, result.vault, policy=policy).text
        assert "ada@example.com" in restored and "bob@example.com" in restored

    def test_repeated_values_share_one_stand_in(self):
        from .. import Redactor

        result = Redactor(policy=self._policy()).redact(
            "Mail ada@example.com, then ada@example.com again"
        )
        assert len(result.entries) == 1
        assert result.text.count(result.entries[0].label) == 2


class TestDoctests:
    """Every documented example runs."""

    def test_doctests_pass(self):
        import doctest

        from .. import _surrogates

        assert doctest.testmod(_surrogates, verbose=False).failed == 0
