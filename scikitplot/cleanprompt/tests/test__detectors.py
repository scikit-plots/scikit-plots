"""Tests for :mod:`scikitplot.cleanprompt._detectors`."""

from __future__ import annotations

import pytest

from .. import (
    DEFAULT_POLICY,
    Detector,
    DetectorError,
    DetectorRegistry,
    LimitExceededError,
    LiteralDetector,
    PatternError,
    PolicyError,
    RegexDetector,
    default_registry,
    get_pattern,
)
from .._policy import Limits


class TestDetectorBase:
    """The protocol every detector satisfies."""

    def test_detect_is_abstract(self):
        with pytest.raises(NotImplementedError, match="must implement detect"):
            list(Detector("d", "K").detect("x", DEFAULT_POLICY))

    def test_empty_name_is_refused(self):
        with pytest.raises(PolicyError, match="name"):
            Detector("", "K")

    def test_empty_kind_is_refused(self):
        with pytest.raises(PolicyError, match="kind"):
            Detector("d", "")

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_confidence_range(self, bad):
        with pytest.raises(PolicyError, match=r"\[0.0, 1.0\]"):
            Detector("d", "K", confidence=bad)

    def test_repr(self):
        assert "name='d'" in repr(Detector("d", "K"))


class TestRegexDetector:
    """The pattern-library adapter."""

    def test_finds_matches(self):
        detector = RegexDetector(get_pattern("EMAIL"))
        spans = list(detector.detect("write to a@b.co please", DEFAULT_POLICY))
        assert [(s.start, s.text) for s in spans] == [(9, "a@b.co")]

    def test_inherits_the_spec_metadata(self):
        detector = RegexDetector(get_pattern("PHONE"))
        assert detector.kind == "PHONE"
        assert detector.priority == get_pattern("PHONE").priority
        assert detector.confidence == get_pattern("PHONE").confidence

    def test_name_is_derived_and_stable(self):
        assert RegexDetector(get_pattern("URL")).name == "regex:URL"

    def test_validator_rejections_are_not_yielded(self):
        detector = RegexDetector(get_pattern("CREDIT_CARD"))
        assert list(detector.detect("1234 5678 9012 3456", DEFAULT_POLICY)) == []

    def test_spans_are_ascending(self):
        detector = RegexDetector(get_pattern("EMAIL"))
        spans = list(detector.detect("a@x.co b@x.co c@x.co", DEFAULT_POLICY))
        assert [s.start for s in spans] == sorted(s.start for s in spans)

    def test_span_text_matches_the_source(self):
        text = "mail a@b.co now"
        for span in RegexDetector(get_pattern("EMAIL")).detect(text, DEFAULT_POLICY):
            assert span.text == text[span.start : span.end]


class TestLiteralDetector:
    """Exact-string detection."""

    def test_longest_alternative_wins_at_a_position(self):
        spans = list(LiteralDetector(["Ann", "Anna"]).detect("Anna", DEFAULT_POLICY))
        assert [s.text for s in spans] == ["Anna"]

    def test_order_given_does_not_matter(self):
        for terms in (["Ann", "Anna"], ["Anna", "Ann"]):
            spans = list(LiteralDetector(terms).detect("Anna", DEFAULT_POLICY))
            assert [s.text for s in spans] == ["Anna"]

    def test_terms_are_escaped_not_interpreted(self):
        """``a.b`` is the literal three characters, not "a, any char, b"."""
        text = "axb and a.b"
        spans = list(LiteralDetector(["a.b"]).detect(text, DEFAULT_POLICY))
        assert [s.start for s in spans] == [text.index("a.b")]
        assert [s.text for s in spans] == ["a.b"]

    def test_regex_metacharacters_are_literal(self):
        for term in ["a+b", "(x)", "[y]", "c*d", "^z$", "a|b", "\\n"]:
            spans = list(
                LiteralDetector([term]).detect("prefix " + term, DEFAULT_POLICY)
            )
            assert [s.text for s in spans] == [term]

    def test_terms_are_stripped_and_deduplicated(self):
        detector = LiteralDetector(["  Acme ", "Acme", "Beta"])
        assert set(detector.terms) == {"Acme", "Beta"}

    def test_empty_term_is_refused(self):
        with pytest.raises(PolicyError, match="empty or whitespace-only"):
            LiteralDetector(["ok", "   "])

    def test_non_string_term_is_refused(self):
        with pytest.raises(PolicyError, match="must be strings"):
            LiteralDetector([1])

    def test_no_terms_yields_nothing(self):
        assert list(LiteralDetector([]).detect("anything", DEFAULT_POLICY)) == []

    def test_word_boundary_mode(self):
        loose = list(LiteralDetector(["class"]).detect("classic", DEFAULT_POLICY))
        strict = list(
            LiteralDetector(["class"], word_boundary=True).detect(
                "classic", DEFAULT_POLICY
            )
        )
        assert len(loose) == 1
        assert strict == []

    def test_word_boundary_still_matches_a_whole_word(self):
        spans = list(
            LiteralDetector(["class"], word_boundary=True).detect(
                "a class here", DEFAULT_POLICY
            )
        )
        assert [s.text for s in spans] == ["class"]

    def test_custom_kind_and_name(self):
        detector = LiteralDetector(["x"], kind="ORG", name="orgs")
        assert detector.kind == "ORG"
        assert detector.name == "orgs"

    def test_default_priority_is_above_the_patterns(self):
        """An explicitly named term beats a generic pattern."""
        assert LiteralDetector(["x"]).priority > get_pattern("PHONE").priority

    def test_single_pass_over_many_terms(self):
        """Cost is linear in the text, not in text times terms."""
        terms = ["term{0}".format(i) for i in range(500)]
        detector = LiteralDetector(terms)
        spans = list(detector.detect("term499 and term0", DEFAULT_POLICY))
        assert {s.text for s in spans} == {"term499", "term0"}


class TestDetectorRegistry:
    """Composition and error attribution."""

    def test_add_returns_self_for_chaining(self):
        registry = DetectorRegistry()
        assert registry.add(LiteralDetector(["a"])) is registry

    def test_duplicate_name_is_refused(self):
        registry = DetectorRegistry([LiteralDetector(["a"], name="dup")])
        with pytest.raises(PolicyError, match="already registered"):
            registry.add(LiteralDetector(["b"], name="dup"))

    def test_kinds_are_sorted_and_unique(self):
        registry = default_registry(kinds=["URL", "EMAIL"])
        assert registry.kinds() == ("EMAIL", "URL")

    def test_select_filters_by_kind(self):
        registry = default_registry(kinds=["URL", "EMAIL"])
        assert [d.name for d in registry.select(["EMAIL"])] == ["regex:EMAIL"]

    def test_len_and_iter(self):
        registry = default_registry(kinds=["EMAIL"])
        assert len(registry) == 1
        assert [d.kind for d in registry] == ["EMAIL"]

    def test_repr_names_its_detectors(self):
        assert "regex:EMAIL" in repr(default_registry(kinds=["EMAIL"]))

    def test_detect_all_collects_every_detector(self):
        registry = default_registry(kinds=["EMAIL", "URL"])
        spans = registry.detect_all("a@b.co https://x.com", DEFAULT_POLICY)
        assert {s.kind for s in spans} == {"EMAIL", "URL"}

    def test_a_failing_detector_is_named(self):
        class Boom(Detector):
            def detect(self, text, policy):
                raise RuntimeError("inner failure")
                yield  # pragma: no cover

        registry = DetectorRegistry([Boom("boom", "K")])
        with pytest.raises(DetectorError) as caught:
            registry.detect_all("x", DEFAULT_POLICY)
        assert caught.value.detector == "boom"
        assert "inner failure" in str(caught.value)

    def test_span_cap_is_enforced_during_collection(self):
        policy = DEFAULT_POLICY.evolve(limits=Limits(max_spans=5))
        registry = default_registry(kinds=["EMAIL"])
        text = " ".join("u{0}@x.co".format(i) for i in range(50))
        with pytest.raises(LimitExceededError) as caught:
            registry.detect_all(text, policy)
        assert caught.value.limit_name == "max_spans"

    def test_min_confidence_filters(self):
        policy = DEFAULT_POLICY.evolve(min_confidence=0.9)
        registry = default_registry(kinds=["PHONE"])
        spans = registry.detect_all("+1 555 010 4477", policy)
        assert spans == []

    def test_min_confidence_keeps_certain_detections(self):
        policy = DEFAULT_POLICY.evolve(min_confidence=0.9)
        registry = default_registry(kinds=["EMAIL"])
        assert registry.detect_all("a@b.co", policy) != []

    def test_explicit_detector_list_overrides_the_registry(self):
        registry = default_registry(kinds=["EMAIL"])
        spans = registry.detect_all(
            "Acme", DEFAULT_POLICY, detectors=[LiteralDetector(["Acme"])]
        )
        assert [s.kind for s in spans] == ["CUSTOM"]


class TestDefaultRegistry:
    """The convenience builder."""

    def test_defaults_to_the_whole_library(self):
        from .. import default_patterns

        assert len(default_registry()) == len(default_patterns())

    def test_selected_kinds(self):
        assert default_registry(kinds=["EMAIL"]).kinds() == ("EMAIL",)

    def test_unknown_kind_is_refused(self):
        with pytest.raises(PatternError, match="unknown pattern kind"):
            default_registry(kinds=["NOPE"])

    def test_literal_terms_add_one_detector(self):
        registry = default_registry(kinds=["EMAIL"], literal_terms=["Acme"])
        assert len(registry) == 2
        assert "CUSTOM" in registry.kinds()

    def test_no_literal_detector_when_no_terms(self):
        assert len(default_registry(kinds=["EMAIL"], literal_terms=[])) == 1

    def test_literal_kind_is_configurable(self):
        registry = default_registry(
            kinds=["EMAIL"], literal_terms=["Acme"], literal_kind="ORG"
        )
        assert "ORG" in registry.kinds()


class TestSpanPurity:
    """I6 — detectors never observe rewritten text."""

    def test_every_detector_receives_the_same_text(self):
        seen = []

        class Recorder(Detector):
            def detect(self, text, policy):
                seen.append(text)
                return iter(())

        registry = DetectorRegistry(
            [Recorder("r1", "A"), Recorder("r2", "B"), Recorder("r3", "C")]
        )
        registry.detect_all("original text", DEFAULT_POLICY)
        assert seen == ["original text"] * 3

    def test_a_detector_cannot_mutate_the_shared_input(self):
        """Strings are immutable; this documents the guarantee explicitly."""
        text = "a@b.co"
        registry = default_registry(kinds=["EMAIL"])
        registry.detect_all(text, DEFAULT_POLICY)
        assert text == "a@b.co"
