"""
Tests for :mod:`scikitplot.cleanprompt._patterns`.

Notes
-----
**Developer notes.** :class:`TestDeclaredExamples` executes the ``examples_yes``
and ``examples_no`` carried by every :class:`PatternSpec`. That makes the
examples a contract rather than documentation: a pattern change that breaks a
stated example fails the suite, and a new pattern that states no negative
example is rejected by :meth:`TestLibraryHygiene.test_every_pattern_states_a_negative`.
"""

from __future__ import annotations

import re
import time

import pytest

from .. import PATTERNS, PatternError, PatternSpec, default_patterns, get_pattern
from .._patterns import luhn_ok
from ._isolated import in_subprocess


def _accepted(spec, text):
    """Return the matches of ``spec`` that survive its validator."""
    return [
        match.group()
        for match in spec.compiled().finditer(text)
        if spec.validate is None or spec.validate(match)
    ]


class TestDeclaredExamples:
    """Every stated example is executed."""

    @pytest.mark.parametrize("kind", sorted(PATTERNS))
    def test_positive_examples_match(self, kind):
        spec = PATTERNS[kind]
        for example in spec.examples_yes:
            assert _accepted(spec, example), "{0} should match {1!r}".format(
                kind, example
            )

    @pytest.mark.parametrize("kind", sorted(PATTERNS))
    def test_negative_examples_do_not_match(self, kind):
        spec = PATTERNS[kind]
        for example in spec.examples_no:
            assert not _accepted(spec, example), "{0} should not match {1!r}".format(
                kind, example
            )


class TestLibraryHygiene:
    """Rules every entry in the library must satisfy."""

    @pytest.mark.parametrize("kind", sorted(PATTERNS))
    def test_kind_matches_its_key(self, kind):
        assert PATTERNS[kind].kind == kind

    @pytest.mark.parametrize("kind", sorted(PATTERNS))
    def test_every_pattern_states_an_intent(self, kind):
        assert len(PATTERNS[kind].intent) > 20

    @pytest.mark.parametrize("kind", sorted(PATTERNS))
    def test_every_pattern_states_a_positive(self, kind):
        assert PATTERNS[kind].examples_yes

    @pytest.mark.parametrize("kind", sorted(PATTERNS))
    def test_every_pattern_states_a_negative(self, kind):
        assert PATTERNS[kind].examples_no

    @pytest.mark.parametrize("kind", sorted(PATTERNS))
    def test_kind_is_safe_for_the_placeholder_grammar(self, kind):
        from .. import DEFAULT_TAG_STYLE

        assert DEFAULT_TAG_STYLE.render(kind, 1).endswith("-1]")

    @pytest.mark.parametrize("kind", sorted(PATTERNS))
    def test_pattern_compiles(self, kind):
        assert PATTERNS[kind].compiled() is not None

    @pytest.mark.parametrize("kind", sorted(PATTERNS))
    def test_pattern_cannot_match_empty(self, kind):
        """A zero-width match would yield no span and hide a broken pattern."""
        assert PATTERNS[kind].compiled().fullmatch("") is None


#: Inputs built to make a careless expression backtrack, by a short name.
#:
#: The name is the test id. The payloads are four thousand characters long,
#: and as ids they put seventy-eight four-kilobyte lines into every verbose
#: run (``CP-091``).
ADVERSARIAL = {
    "letters": "a" * 4000,
    "digits": "1" * 4000,
    "at-runs": ("a" * 200 + "@") * 20,
    "dash-runs": ("-" * 100 + "1") * 40,
    "dotted-url": "http://" + "a." * 900,
    "colons": ":" * 4000,
}

#: Seconds one pattern may spend on one payload.
MATCH_BUDGET = 2.0

#: Seconds before the child is killed: the budget, plus starting an interpreter
#: and importing the package on a slow machine.
KILL_AFTER = 30.0

_TIME_ONE_MATCH = (
    "import time\n"
    "from scikitplot.cleanprompt._patterns import PATTERNS\n"
    "compiled = PATTERNS[{0!r}].compiled()\n"
    "payload = {1!r}\n"
    "started = time.monotonic()\n"
    "compiled.findall(payload)\n"
    "print(time.monotonic() - started)\n"
)


class TestNoCatastrophicBacktracking:
    """
    Adversarial inputs must not blow up.

    Notes
    -----
    **Developer notes.** Each match runs in a child interpreter with a
    deadline. Timed in the pytest process, a pattern that backtracks without
    bound would never return, the assertion after it would never run, and the
    whole run would stop on a test that reports nothing. Here that pattern
    fails within :data:`KILL_AFTER` seconds, by name.
    """

    @pytest.mark.parametrize("kind", sorted(PATTERNS))
    @pytest.mark.parametrize("payload", sorted(ADVERSARIAL))
    def test_bounded_time(self, kind, payload):
        output = in_subprocess(
            _TIME_ONE_MATCH.format(kind, ADVERSARIAL[payload]), timeout=KILL_AFTER
        )
        assert float(output) < MATCH_BUDGET

    def test_a_pattern_that_never_returns_fails_instead_of_hanging(self):
        """The deadline itself: a child that does not finish is killed."""
        started = time.monotonic()
        with pytest.raises(AssertionError, match="still running after"):
            in_subprocess("import time\ntime.sleep(60)\n", timeout=1.0)
        assert time.monotonic() - started < 20

    def test_no_test_id_carries_a_payload(self):
        assert max(len(name) for name in ADVERSARIAL) < 20
        assert len(ADVERSARIAL) == 6


class TestLuhn:
    """The payment-card checksum."""

    @pytest.mark.parametrize(
        "digits", ["4242424242424242", "4111111111111111", "5555555555554444"]
    )
    def test_valid(self, digits):
        assert luhn_ok(digits)

    @pytest.mark.parametrize(
        "digits",
        ["4242424242424241", "1234567890123456", "", "abcd", "12345678901"],
    )
    def test_invalid(self, digits):
        assert not luhn_ok(digits)

    def test_length_bounds(self):
        assert not luhn_ok("0" * 11)
        assert not luhn_ok("0" * 20)


class TestEmailDomainCharacters:
    """``CP-005`` stated precisely: the domain admits no pipe, ever."""

    def test_no_match_spans_a_pipe(self):
        spec = get_pattern("EMAIL")
        for text in ("a@b.a|b", "x@y.z|w.com", "user@host.com|evil"):
            for match in _accepted(spec, text):
                assert "|" not in match

    def test_the_upstream_expression_did_span_it(self):
        """Documents what changed, so the fix cannot be quietly reverted."""
        upstream = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        assert upstream.search("a@b.a|b").group() == "a@b.a|b"

    def test_the_valid_prefix_is_still_found(self):
        """Stopping at the pipe must not mean detecting nothing at all."""
        assert _accepted(get_pattern("EMAIL"), "a@b.co|x") == ["a@b.co"]

    def test_a_pipe_in_the_local_part_is_permitted_by_rfc_5322(self):
        """``|`` is an atext character; the domain is where it is excluded."""
        assert _accepted(get_pattern("EMAIL"), "we|rd@example.com") == [
            "we|rd@example.com"
        ]


class TestIban:
    """The ISO 13616 mod-97 check."""

    @pytest.mark.parametrize(
        "value",
        ["GB82 WEST 1234 5698 7654 32", "DE89370400440532013000", "FR1420041010050500013M02606"],
    )
    def test_valid(self, value):
        assert _accepted(get_pattern("IBAN"), value) != []

    @pytest.mark.parametrize("value", ["GB00 WEST 1234 5698 7654 32", "DE89370400440532013001"])
    def test_invalid_checksum(self, value):
        assert _accepted(get_pattern("IBAN"), value) == []


class TestSsn:
    """Structural issuance rules for a US Social Security number."""

    def test_valid(self):
        assert _accepted(get_pattern("SSN_US"), "123-45-6789") == ["123-45-6789"]

    @pytest.mark.parametrize(
        "value", ["000-45-6789", "666-45-6789", "900-45-6789", "123-00-6789", "123-45-0000"]
    )
    def test_never_issued(self, value):
        assert _accepted(get_pattern("SSN_US"), value) == []

    def test_unhyphenated_is_not_matched(self):
        assert _accepted(get_pattern("SSN_US"), "123456789") == []


class TestIpv4:
    """Octet range validation."""

    @pytest.mark.parametrize("value", ["0.0.0.0", "8.8.8.8", "255.255.255.255"])
    def test_valid(self, value):
        assert _accepted(get_pattern("IPV4"), value) == [value]

    @pytest.mark.parametrize("value", ["256.1.1.1", "1.2.3", "1.2.3.4.5", "999.999.999.999"])
    def test_invalid(self, value):
        assert _accepted(get_pattern("IPV4"), value) == []


class TestPatternSpecValidation:
    """Construction-time checks."""

    def test_uncompilable_pattern_is_refused(self):
        with pytest.raises(PatternError, match="does not compile"):
            PatternSpec(kind="BAD", pattern="(unclosed", intent="x" * 30)

    def test_missing_intent_is_refused(self):
        with pytest.raises(PatternError, match="no stated intent"):
            PatternSpec(kind="BAD", pattern="a", intent="")

    def test_spec_is_hashable_and_frozen(self):
        spec = get_pattern("EMAIL")
        assert hash(spec) is not None
        with pytest.raises(Exception):
            spec.kind = "OTHER"


class TestSelection:
    """Library lookup and the default set."""

    def test_get_pattern(self):
        assert get_pattern("EMAIL").kind == "EMAIL"

    def test_unknown_kind_names_the_alternatives(self):
        with pytest.raises(PatternError) as caught:
            get_pattern("NOPE")
        assert "EMAIL" in str(caught.value)
        assert caught.value.name == "NOPE"

    def test_default_set_is_ordered_by_priority_then_name(self):
        specs = default_patterns()
        keys = [(-spec.priority, spec.kind) for spec in specs]
        assert keys == sorted(keys)

    def test_default_set_is_stable(self):
        assert default_patterns() == default_patterns()

    def test_private_key_outranks_everything(self):
        assert get_pattern("PRIVATE_KEY").priority == max(
            spec.priority for spec in PATTERNS.values()
        )


class TestRealisticDocuments:
    """Whole-document behaviour of the pattern set."""

    def test_private_key_block_is_taken_whole(self, redactor):
        _key = "KEY"  # {_key}
        text = (
            f"key follows\n-----BEGIN RSA PRIVATE {_key}-----\n"
            "MIIBOgIBAAJBAK+1234/abc=\nAAAAB3Nza\n"
            f"-----END RSA PRIVATE {_key}-----\ndone"
        )
        result = redactor.redact(text)
        assert result.text == "key follows\n[PRIVATE_KEY-1]\ndone"

    def test_url_absorbs_an_embedded_address(self, redactor):
        result = redactor.redact("see https://ex.com/u?m=a@b.co now")
        assert result.stats.entries == 1
        assert result.entries[0].kind == "URL"

    def test_jwt_is_detected(self, redactor):
        token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.dBjftJeZ4CVPmB92K27uhbUJU1p1r"
        assert "[JWT-1]" in redactor.redact("bearer " + token).text

    def test_prose_is_left_alone(self, redactor):
        text = (
            "We shipped version 1.26.4 on 2024-01-15 after 12345678 downloads, "
            "up 45 percent over 3 months."
        )
        assert redactor.redact(text).text == text
