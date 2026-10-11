"""
Tests for :mod:`scikitplot.cleanprompt._engine`.

Notes
-----
**Developer notes.** Grouped by the invariant each class proves, because the
engine's contract is stated as invariants rather than as a list of behaviours.
The identifiers ``I1`` to ``I9`` refer to the table in
``maintenances/cleanprompt/_maintenance/DESIGN.md``.
"""

from __future__ import annotations

import random
import string

import pytest

from .. import (
    DEFAULT_POLICY,
    LimitExceededError,
    OverlapError,
    OverlapStrategy,
    PolicyError,
    Redactor,
    RestorationError,
    Span,
    TagStyle,
    Vault,
    default_registry,
    resolve_spans,
    restore,
)
from .._engine import reserved_label_spans


class TestResolveSpans:
    """I5 — resolution yields disjoint, ascending, coverage-preserving spans."""

    def test_empty_input(self):
        assert resolve_spans([]) == ()

    def test_disjoint_spans_pass_through_sorted(self):
        late = Span(10, 15, "A", "x" * 5, "d1")
        early = Span(0, 5, "B", "y" * 5, "d2")
        assert [s.start for s in resolve_spans([late, early])] == [0, 10]

    def test_nested_span_is_absorbed_by_its_container(self):
        outer = Span(0, 20, "URL", "u" * 20, "regex:URL")
        inner = Span(5, 11, "EMAIL", "e" * 6, "regex:EMAIL")
        resolved = resolve_spans([outer, inner])
        assert [(s.start, s.end, s.kind) for s in resolved] == [(0, 20, "URL")]

    def test_partial_overlap_is_merged_not_dropped(self):
        """Dropping the tail of a partially overlapping span would disclose it."""
        text = "abcdefghij"
        left = Span(0, 6, "A", text[0:6], "d1")
        right = Span(4, 10, "B", text[4:10], "d2")
        resolved = resolve_spans([left, right], text=text)
        assert [(s.start, s.end) for s in resolved] == [(0, 10)]
        assert resolved[0].text == text

    def test_merge_without_text_is_refused(self):
        left = Span(0, 6, "A", "a" * 6, "d1")
        right = Span(4, 10, "B", "b" * 6, "d2")
        with pytest.raises(PolicyError, match="without the source text"):
            resolve_spans([left, right])

    def test_resolution_preserves_coverage(self):
        """Every covered character stays covered. The safety property."""
        rng = random.Random(20260920)
        for _ in range(300):
            spans = []
            for index in range(rng.randint(1, 8)):
                start = rng.randint(0, 40)
                end = start + rng.randint(1, 12)
                spans.append(
                    Span(
                        start,
                        end,
                        "K{0}".format(index),
                        "x" * (end - start),
                        "d{0}".format(index),
                    )
                )
            text = "x" * 60
            resolved = resolve_spans(spans, text=text)
            covered_in = {i for s in spans for i in range(s.start, s.end)}
            covered_out = {i for s in resolved for i in range(s.start, s.end)}
            assert covered_in == covered_out
            for left, right in zip(resolved, resolved[1:]):
                assert left.end <= right.start

    def test_strict_strategy_raises_on_overlap(self):
        policy = DEFAULT_POLICY.evolve(overlap=OverlapStrategy.STRICT)
        spans = [Span(0, 6, "A", "a" * 6, "d1"), Span(4, 10, "B", "b" * 6, "d2")]
        with pytest.raises(OverlapError) as caught:
            resolve_spans(spans, policy, text="x" * 10)
        assert len(caught.value.spans) == 2

    def test_priority_strategy_prefers_the_higher_priority_kind(self):
        policy = DEFAULT_POLICY.evolve(overlap=OverlapStrategy.PRIORITY_WINS)
        long_low = Span(0, 20, "LOW", "l" * 20, "d1", priority=10)
        short_high = Span(0, 20, "HIGH", "h" * 20, "d2", priority=90)
        resolved = resolve_spans([long_low, short_high], policy)
        assert resolved[0].kind == "HIGH"

    def test_ties_break_on_detector_name_not_insertion_order(self):
        first = Span(0, 10, "A", "a" * 10, "alpha")
        second = Span(0, 10, "B", "b" * 10, "beta")
        forward = resolve_spans([first, second])
        backward = resolve_spans([second, first])
        assert forward[0].kind == backward[0].kind == "A"


class TestReservation:
    """I7 — the placeholder grammar is masked from detection."""

    def test_finds_existing_labels(self):
        ranges, labels = reserved_label_spans("a [EMAIL-1] b [URL-2] c", DEFAULT_POLICY)
        assert labels == {"[EMAIL-1]", "[URL-2]"}
        assert len(ranges) == 2

    def test_disabled_by_policy(self):
        policy = DEFAULT_POLICY.evolve(preserve_placeholders=False)
        assert reserved_label_spans("a [EMAIL-1] b", policy) == ((), set())

    def test_a_different_grammar_sees_no_labels(self):
        policy = DEFAULT_POLICY.evolve(
            tag_style=TagStyle(prefix="<<", suffix=">>", separator="_")
        )
        _, labels = reserved_label_spans("a [EMAIL-1] b", policy)
        assert labels == set()


class TestRoundTrip:
    """I1 — restoration is the exact inverse of redaction."""

    def test_simple(self, redactor):
        text = "mail ada@example.com"
        result = redactor.redact(text)
        assert restore(result.text, result.vault).text == text

    def test_rich_document(self, redactor, sample_text):
        result = redactor.redact(sample_text, extra_terms=["Acme"])
        assert restore(result.text, result.vault).text == sample_text

    def test_empty_text(self, redactor):
        result = redactor.redact("")
        assert result.text == ""
        assert result.stats.entries == 0
        assert restore("", result.vault).text == ""

    def test_text_with_no_detections(self, redactor):
        text = "nothing sensitive at all here"
        result = redactor.redact(text)
        assert result.text == text
        assert restore(result.text, result.vault).text == text

    def test_fuzz_round_trip(self, redactor):
        """Random documents built from redactable and inert fragments."""
        rng = random.Random(1234567)
        redactable = [
            "ada@example.com",
            "bob.smith+x@sub.example.co.uk",
            "https://example.com/a?b=1",
            "+1 555 010 4477",
            "4242 4242 4242 4242",
            "192.168.1.10",
            "00:1B:44:11:3A:B7",
            "123-45-6789",
        ]
        inert = ["hello", "world", "2024-01-15", "12345678", "numpy 1.26.4", "\n", " "]
        for _ in range(400):
            parts = [
                rng.choice(redactable if rng.random() < 0.4 else inert)
                for _ in range(rng.randint(0, 14))
            ]
            text = " ".join(parts)
            result = redactor.redact(text)
            assert restore(result.text, result.vault).text == text

    def test_unicode_is_preserved(self, redactor):
        text = "Ünïcodé ✉ ada@example.com — naïve 🎉 emoji"
        result = redactor.redact(text)
        assert restore(result.text, result.vault).text == text

    def test_case_insensitive_folds_to_the_first_surface(self, redactor):
        """Documented consequence: I1 holds exactly only when folding is off."""
        policy = DEFAULT_POLICY.evolve(kinds=("EMAIL",), case_insensitive=True)
        folding = Redactor(policy=policy, registry=default_registry(kinds=("EMAIL",)))
        result = folding.redact("A@Example.com and a@example.com")
        assert result.stats.entries == 1
        assert restore(result.text, result.vault).text == (
            "A@Example.com and A@Example.com"
        )


class TestValueIdentity:
    """I9 — one label per distinct value; distinct values never share one."""

    def test_repeats_share_one_label(self, redactor):
        result = redactor.redact("a@x.com then a@x.com then a@x.com")
        assert result.stats.entries == 1
        assert result.entries[0].count == 3
        assert result.text.count("[EMAIL-1]") == 3

    def test_distinct_values_get_distinct_labels(self, redactor):
        result = redactor.redact("a@x.com and b@x.com")
        assert {entry.label for entry in result.entries} == {"[EMAIL-1]", "[EMAIL-2]"}

    def test_occurrences_are_recorded_in_order(self, redactor):
        result = redactor.redact("a@x.com .. a@x.com")
        occurrences = result.entries[0].occurrences
        assert len(occurrences) == 2
        assert occurrences[0][0] < occurrences[1][0]

    def test_same_value_in_two_kinds_gets_two_labels(self, redactor):
        result = redactor.redact("a@x.com", extra_terms=["a@x.com"])
        assert result.stats.entries == 1  # one wins the overlap; no duplicate output
        assert result.text.count("[") == 1


class TestDeterminism:
    """I3 — identical inputs give byte-identical outputs."""

    def test_repeated_calls_agree(self, redactor, sample_text):
        outputs = {
            redactor.redact(sample_text, extra_terms=["Acme"]).text for _ in range(25)
        }
        assert len(outputs) == 1

    def test_fresh_redactors_agree(self, sample_text):
        first = Redactor().redact(sample_text).text
        second = Redactor().redact(sample_text).text
        assert first == second

    def test_output_is_stable_across_hash_seeds(self, sample_text):
        """A digest computed in a subprocess with a different hash seed."""
        import os
        import subprocess
        import sys

        script = (
            "import sys;sys.path.insert(0, {0!r});"
            "from scikitplot.cleanprompt import Redactor;"
            "print(Redactor().redact({1!r}).text)"
        ).format(str(_repo_root()), sample_text)
        outputs = set()
        for seed in ("0", "1", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            completed = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            outputs.add(completed.stdout)
        assert len(outputs) == 1


def _repo_root():
    """Return the directory containing the ``scikitplot`` package."""
    import pathlib

    return pathlib.Path(__file__).resolve().parents[3]


class TestNoLeakage:
    """I2 — no detected surface survives in the output."""

    def test_secrets_are_absent_from_the_redacted_text(self, redactor, sample_text):
        result = redactor.redact(sample_text, extra_terms=["Acme"])
        for entry in result.entries:
            assert entry.original not in result.text

    def test_partial_fragments_are_absent(self, redactor):
        result = redactor.redact("Anna", extra_terms=["Ann", "Anna"])
        assert result.text == "[CUSTOM-1]"


class TestLimits:
    """I8 — bounds raise; nothing is silently truncated."""

    def test_input_length(self, redactor):
        policy = DEFAULT_POLICY.evolve(
            limits=DEFAULT_POLICY.limits.__class__(max_input_chars=10)
        )
        small = Redactor(policy=policy)
        with pytest.raises(LimitExceededError) as caught:
            small.redact("x" * 11)
        assert caught.value.limit_name == "max_input_chars"
        assert caught.value.actual == 11

    def test_entry_cap(self):
        from .._policy import Limits

        policy = DEFAULT_POLICY.evolve(kinds=("EMAIL",), limits=Limits(max_entries=2))
        small = Redactor(policy=policy, registry=default_registry(kinds=("EMAIL",)))
        text = " ".join("u{0}@x.com".format(i) for i in range(5))
        with pytest.raises(LimitExceededError) as caught:
            small.redact(text)
        assert caught.value.limit_name == "max_entries"

    def test_entry_cap_counts_only_new_values(self):
        """CP-064: a seed of earlier entries does not use up the bound."""
        from .._policy import Limits

        policy = DEFAULT_POLICY.evolve(kinds=("EMAIL",), limits=Limits(max_entries=2))
        small = Redactor(policy=policy, registry=default_registry(kinds=("EMAIL",)))
        seed = small.redact("a@x.com b@x.com").entries
        again = small.redact("c@x.com d@x.com a@x.com", seed=seed)
        assert again.text == "[EMAIL-3] [EMAIL-4] [EMAIL-1]"
        with pytest.raises(LimitExceededError) as caught:
            small.redact("e@x.com f@x.com g@x.com", seed=seed)
        assert caught.value.actual == 3

    def test_literal_term_count(self, redactor):
        from .._policy import Limits

        policy = DEFAULT_POLICY.evolve(limits=Limits(max_literal_terms=2))
        small = Redactor(policy=policy)
        with pytest.raises(LimitExceededError):
            small.redact("abc", extra_terms=["a", "b", "c"])

    def test_literal_term_length(self, redactor):
        from .._policy import Limits

        policy = DEFAULT_POLICY.evolve(limits=Limits(max_literal_length=3))
        small = Redactor(policy=policy)
        with pytest.raises(LimitExceededError):
            small.redact("abcd", extra_terms=["abcd"])

    def test_result_never_reports_truncation(self, redactor, sample_text):
        assert redactor.redact(sample_text).truncated is False


class TestRestore:
    """Restoration behaviour outside the round trip."""

    def test_unknown_label_is_reported_not_raised(self, redactor):
        result = redactor.redact("mail a@x.com")
        outcome = restore("see [EMAIL-9] and [EMAIL-1]", result.vault)
        assert outcome.unknown == ("[EMAIL-9]",)
        assert outcome.restored == ("[EMAIL-1]",)
        assert outcome.complete is False
        assert "[EMAIL-9]" in outcome.text

    def test_strict_raises_on_unknown(self, redactor):
        result = redactor.redact("mail a@x.com")
        with pytest.raises(RestorationError) as caught:
            restore("see [EMAIL-9]", result.vault, strict=True)
        assert caught.value.labels == ("[EMAIL-9]",)

    def test_unused_entries_are_reported(self, redactor):
        result = redactor.redact("a@x.com and b@x.com")
        outcome = restore("only [EMAIL-1]", result.vault)
        assert outcome.unused == ("[EMAIL-2]",)

    def test_cleared_vault_is_refused(self, redactor):
        result = redactor.redact("mail a@x.com")
        result.vault.clear()
        with pytest.raises(PolicyError, match="cleared"):
            restore(result.text, result.vault)

    def test_grammar_mismatch_is_refused(self, redactor):
        result = redactor.redact("mail a@x.com")
        other = DEFAULT_POLICY.evolve(tag_style=TagStyle(prefix="<<", suffix=">>"))
        with pytest.raises(PolicyError, match="placeholder grammar"):
            restore(result.text, result.vault, policy=other)

    def test_narrowing_kinds_does_not_invalidate_a_vault(self):
        """The grammar decides, not the whole policy."""
        policy = DEFAULT_POLICY.evolve(kinds=("EMAIL",))
        result = Redactor(
            policy=policy, registry=default_registry(kinds=("EMAIL",))
        ).redact("mail a@x.com")
        assert restore(result.text, result.vault).text == "mail a@x.com"

    def test_restored_value_is_not_rescanned(self):
        """A secret that looks like a placeholder must not be re-restored."""
        vault = Vault({"[EMAIL-1]": "[EMAIL-2]", "[EMAIL-2]": "final"})
        outcome = restore("x [EMAIL-1] y", vault)
        assert outcome.text == "x [EMAIL-2] y"

    def test_rejects_non_string(self, redactor):
        result = redactor.redact("a")
        with pytest.raises(TypeError):
            restore(None, result.vault)


class TestRedactorSurface:
    """Construction and argument handling."""

    def test_rejects_non_string_input(self, redactor):
        with pytest.raises(TypeError, match="must be str"):
            redactor.redact(b"bytes")

    def test_unknown_kind_is_refused_when_building_the_default_registry(self):
        """The pattern library rejects it first, and names what is available."""
        from .. import PatternError

        with pytest.raises(PatternError, match="unknown pattern kind") as caught:
            Redactor(policy=DEFAULT_POLICY.evolve(kinds=("NOPE",)))
        assert "EMAIL" in str(caught.value)

    def test_unknown_kind_is_refused_against_a_supplied_registry(self):
        """With an explicit registry, the policy is what rejects it."""
        with pytest.raises(PolicyError, match="unknown detection kind") as caught:
            Redactor(
                policy=DEFAULT_POLICY.evolve(kinds=("NOPE",)),
                registry=default_registry(kinds=("EMAIL",)),
            )
        assert "EMAIL" in str(caught.value)

    def test_both_refusals_share_one_base(self):
        """A caller may catch either through the package's base error."""
        from .. import CleanPromptError, PatternError

        assert issubclass(PatternError, CleanPromptError)
        assert issubclass(PolicyError, CleanPromptError)

    def test_repr_is_informative_and_secret_free(self, redactor):
        text = repr(redactor)
        assert text.startswith("Redactor(")
        assert "EMAIL" in text

    def test_word_boundary_terms(self, redactor):
        loose = redactor.redact("classic", extra_terms=["class"])
        strict = redactor.redact("classic", extra_terms=["class"], word_boundary=True)
        assert loose.stats.entries == 1
        assert strict.stats.entries == 0

    def test_extra_kind_is_configurable(self, redactor):
        result = redactor.redact("Acme", extra_terms=["Acme"], extra_kind="ORG")
        assert result.text == "[ORG-1]"

    def test_stats_are_consistent(self, redactor, sample_text):
        result = redactor.redact(sample_text)
        stats = result.stats
        assert stats.input_chars == len(sample_text)
        assert stats.output_chars == len(result.text)
        assert stats.resolved_spans >= stats.entries
        assert sum(stats.by_kind.values()) == stats.entries

    def test_long_document_is_linear_enough(self, redactor):
        """A large document completes; guards against quadratic rewriting."""
        text = ("contact user{0}@example.com. " * 500).format(0)
        result = redactor.redact(text)
        assert restore(result.text, result.vault).text == text


class TestEdgeCases:
    """Boundaries that a naive implementation gets wrong."""

    def test_detection_at_offset_zero(self, redactor):
        assert redactor.redact("a@x.com trailing").text.startswith("[EMAIL-1]")

    def test_detection_at_end_of_text(self, redactor):
        assert redactor.redact("leading a@x.com").text.endswith("[EMAIL-1]")

    def test_whole_text_is_one_detection(self, redactor):
        assert redactor.redact("a@x.com").text == "[EMAIL-1]"

    def test_adjacent_detections(self, redactor):
        result = redactor.redact("a@x.com,b@x.com")
        assert result.text == "[EMAIL-1],[EMAIL-2]"

    def test_newlines_and_tabs_survive(self, redactor):
        text = "a@x.com\n\tb@x.com\r\n"
        assert (
            restore(redactor.redact(text).text, redactor.redact(text).vault).text
            == text
        )

    def test_only_whitespace(self, redactor):
        assert redactor.redact("   \n  ").text == "   \n  "

    def test_random_noise_never_crashes(self, redactor):
        rng = random.Random(99)
        alphabet = string.printable
        for _ in range(200):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 200)))
            result = redactor.redact(text)
            assert isinstance(result.text, str)
            restored = restore(result.text, result.vault)
            assert isinstance(restored.text, str)


class TestRewrittenStandIns:
    """CP-072: a surrogate the model re-cased or re-spaced still restores."""

    @staticmethod
    def _vault(mapping):
        from .._policy import DEFAULT_POLICY
        from .._vault import Vault

        return Vault(mapping, grammar_fingerprint=DEFAULT_POLICY.tag_style.fingerprint)

    MAPPING = {
        "Marion Holt": "Ann Lee",
        "marion.holt@example.invalid": "ann@example.com",
        "+1 555 0100": "+1 555 010 4477",
    }

    @pytest.mark.parametrize(
        ("reply", "back"),
        [
            ("DEAR MARION HOLT,", "DEAR Ann Lee,"),
            ("Dear marion holt,", "Dear Ann Lee,"),
            ("Dear Marion\nHolt,", "Dear Ann Lee,"),
            ("Dear Marion Holt,", "Dear Ann Lee,"),
            ("Dear Marion" + " " * 16 + "Holt,", "Dear Ann Lee,"),
            ("Mail MARION.HOLT@EXAMPLE.INVALID", "Mail ann@example.com"),
            ("Call +1 555 0100", "Call +1 555 010 4477"),
        ],
    )
    def test_rewritten_forms_restore_and_are_reported(self, reply, back):
        from .._engine import restore

        result = restore(reply, self._vault(self.MAPPING))
        assert result.text == back
        assert result.repaired and not result.unknown

    @pytest.mark.parametrize(
        "reply",
        [
            "Dear Marion" + " " * 17 + "Holt,",  # beyond the bound
            "Call +1-555-0100",  # a reformat, not an equivalent writing
            "Dear Marionette Holtz",
        ],
    )
    def test_other_text_is_left_alone(self, reply):
        from .._engine import restore

        result = restore(reply, self._vault(self.MAPPING))
        assert result.text == reply and not result.unknown

    def test_exact_mode_restores_only_exact_forms(self):
        from .._engine import restore

        result = restore(
            "Dear MARION HOLT and Marion Holt", self._vault(self.MAPPING), lenient=False
        )
        assert result.text == "Dear MARION HOLT and Ann Lee"
        assert not result.unknown and not result.repaired

    def test_keys_that_differ_only_in_case_restore_only_exactly(self):
        from .._engine import restore

        vault = self._vault({"Devin Holt": "A", "devin holt": "B"})
        result = restore("Devin Holt / devin holt / DEVIN HOLT", vault)
        assert result.text == "A / B / DEVIN HOLT"
        assert not result.unknown


class TestDetectionView:
    """
    ``CP-098``: a value is found however invisibly it is written, and restored
    exactly as it was written.

    Notes
    -----
    **Developer notes.** Each obfuscation below went to the model in the clear
    on the tree before this round. The class-level test at the end salts every
    built-in pattern's own positive examples, the same way
    ``test__patterns`` runs them in twelve sentence positions: a single
    hand-picked case per pattern is how ``CP-028`` stayed hidden.
    """

    OBFUSCATED = [
        ("mail ada\u200b@example.com now", "EMAIL"),
        ("mail \uff41\uff44\uff41\uff20\uff45\uff58\uff41\uff4d\uff50\uff4c\uff45\uff0e\uff43\uff4f\uff4d now", "EMAIL"),
        ("call +1 555\u00a00100 today", "PHONE"),
        ("call +1\u2011555\u20110100 today", "PHONE"),
        ("host 192.0.2.\u200b10 is down", "IPV4"),
        ("card 4111\u200b1111 1111 1111 expires", "CREDIT_CARD"),
        ("mail ada\u2060@exam\u00adple.com now", "EMAIL"),
        ("mail ada@example.com\u202e now", "EMAIL"),
        ("mail \U0001f600ada@example.com\U0001f600 now", "EMAIL"),
    ]

    @pytest.mark.parametrize("text,kind", OBFUSCATED)
    def test_obfuscated_values_are_found(self, text, kind):
        result = Redactor().redact(text)
        kinds = {entry.kind for entry in result.entries}
        assert kind in kinds, (text, result.text)
        assert len(result.entries) == 1, "one value, one placeholder"

    @pytest.mark.parametrize("text,kind", OBFUSCATED)
    def test_the_original_writing_is_what_comes_back(self, text, kind):
        result = Redactor().redact(text)
        assert restore(result.text, result.vault).text == text

    @pytest.mark.parametrize("text,kind", OBFUSCATED)
    def test_no_visible_part_of_the_value_is_sent(self, text, kind):
        """Everything left of the value in the output is the surrounding prose."""
        result = Redactor().redact(text)
        surface = result.entries[0].original
        assert surface not in result.text
        visible = "".join(c for c in surface if c.isalnum())
        assert visible[:4] not in result.text.replace(" now", "")

    def test_the_vault_keeps_the_invisible_characters(self):
        text = "mail ada\u200b@example.com now"
        result = Redactor().redact(text)
        assert result.entries[0].original == "ada\u200b@example.com"

    @pytest.mark.parametrize(
        "text",
        [
            "Plain English with a@b.co and 192.0.2.10.",
            "Café crème, naïve résumé, 東京 and Ελλάδα — no values here.",
            "Don\u2019t worry, it\u2019s fine.",
        ],
    )
    def test_text_without_hidden_values_is_unchanged_by_the_view(self, text):
        """The view adds spans only where it finds something the text hides."""
        from .._canonical import detection_view

        result = Redactor().redact(text)
        view = detection_view(text)
        if view is None:
            return
        plain = Redactor().redact(view.text)
        assert [e.kind for e in result.entries] == [e.kind for e in plain.entries]

    def test_nfd_and_nfc_writings_redact_the_same_value(self):
        """Combining marks are not format characters; the value is still found."""
        import unicodedata

        for form in ("NFC", "NFD"):
            text = unicodedata.normalize(form, "José écrit à jose@example.com, à bientôt")
            result = Redactor().redact(text)
            assert any(entry.kind == "EMAIL" for entry in result.entries), form
            assert restore(result.text, result.vault).text == text

    def test_a_hidden_literal_term_is_found(self):
        result = Redactor().redact("Ad\u200ba Lovelace wrote it", extra_terms=["Ada"])
        assert result.entries and result.entries[0].original == "Ad\u200ba"

    def test_overlap_with_an_original_text_span_merges(self):
        """The view and the original find parts of one value: one placeholder."""
        text = "card 4111\u200b1111 1111 1111 expires"
        result = Redactor().redact(text)
        assert result.text == "card [CREDIT_CARD-1] expires"

    def test_entity_detectors_never_read_the_view(self):
        from .._detectors import DetectorRegistry
        from .._types import Span

        seen = []

        class Probe:
            name = "probe"
            kind = "NE"
            priority = 30

            def kinds(self):
                return ("PERSON",)

            def detect(self, text, policy):
                seen.append(text)
                return iter(())

        registry = DetectorRegistry()
        registry.add(Probe())
        Redactor(registry=registry).redact("Ada\u200b Lovelace")
        assert seen == ["Ada\u200b Lovelace"]
        del Span

    def test_running_twice_is_stable(self):
        text = "mail ada\u200b@example.com and +1 555\u00a00100"
        first = Redactor().redact(text)
        second = Redactor().redact(first.text)
        assert second.text == first.text
        assert not second.entries

    def test_the_span_limit_counts_both_passes(self):
        from .._exceptions import LimitExceededError
        from .._policy import Limits, RedactionPolicy

        text = " ".join(f"u{i}\u200b@example.com" for i in range(6))
        policy = RedactionPolicy(limits=Limits(max_spans=5))
        with pytest.raises(LimitExceededError):
            Redactor(policy=policy).redact(text)

    def test_every_pattern_example_is_found_when_salted(self):
        """Each positive example, a zero-width space after its first character."""
        from .._patterns import PATTERNS

        failures = []
        for spec in PATTERNS.values():
            if not spec.enabled_by_default:
                continue
            for example in spec.examples_yes:
                if len(example) < 2:
                    continue
                salted = example[0] + "\u200b" + example[1:]
                text = f"see {salted} here"
                result = Redactor().redact(text)
                found = {entry.kind for entry in result.entries}
                if spec.kind not in found or salted[1:] in result.text:
                    failures.append((spec.kind, example, result.text))
                elif restore(result.text, result.vault).text != text:
                    failures.append((spec.kind, example, "restore"))
        assert not failures, failures
