"""
Tests for :mod:`scikitplot.cleanprompt._ner`.

Notes
-----
**Developer notes.** spaCy and its models are large and are usually absent from
a test environment, so the assertions here split into three groups.

1. Everything that holds with no spaCy at all: construction, naming, the label
   policy, and the shape of the failure. These always run.
2. Behaviour against a stub pipeline injected into the module's cache. This
   proves the span translation and the composition with structural detectors
   without needing a model, and it is where the ``CP-006`` guarantee is tested
   under a detector that behaves like a real recogniser.
3. Live model tests, skipped when the tier is unavailable.

A stub is legitimate here because the contract under test is *this module's*
translation of entity offsets into spans, not spaCy's recognition quality. The
lane that would prove the latter is marked ``UNAVAILABLE`` in the evidence
record rather than being claimed by a stub.
"""

from __future__ import annotations

import pytest

from .. import DEFAULT_POLICY, CapabilityError, Redactor, default_registry, restore
from .. import _capabilities as caps
from .._ner import DEFAULT_ENTITY_LABELS, DEFAULT_MODEL, NerDetector, spacy_detector
from ._tiers import skip_reason, engine_ready, engine_skip_reason

NER_AVAILABLE = caps.probe("ner").available


class _Entity:
    """Minimal stand-in for a spaCy ``Span``."""

    def __init__(self, text, start_char, end_char, label):
        self.text = text
        self.start_char = start_char
        self.end_char = end_char
        self.label_ = label


class _Doc:
    def __init__(self, ents):
        self.ents = ents


class _StubPipeline:
    """A pipeline that tags every occurrence of a fixed set of words."""

    max_length = 1000

    def __init__(self, words):
        self.words = words
        self.calls = []

    def __call__(self, text):
        import re

        ents = []
        for word, label in self.words.items():
            for match in re.finditer(re.escape(word), text):
                ents.append(_Entity(match.group(), match.start(), match.end(), label))
        ents.sort(key=lambda entity: entity.start_char)
        self.calls.append(text)
        return _Doc(ents)


@pytest.fixture()
def stub_detector(monkeypatch):
    """Return a detector wired to a stub pipeline, with the cache isolated."""
    from .. import _ner

    monkeypatch.setattr(_ner, "_PIPELINES", {})
    detector = NerDetector(model="stub")
    pipeline = _StubPipeline({"Ada Lovelace": "PERSON", "Acme": "ORG", "EMAIL": "ORG"})
    _ner._PIPELINES[(detector.model, detector._disable)] = pipeline
    return detector, pipeline


class TestConstruction:
    """Building a detector costs nothing."""

    def test_no_import_at_construction(self):
        import sys

        before = "spacy" in sys.modules
        spacy_detector()
        assert ("spacy" in sys.modules) == before

    def test_default_model(self):
        assert spacy_detector().model == DEFAULT_MODEL

    def test_name_includes_the_model(self):
        assert spacy_detector(model="xx_test").name == "ner:xx_test"

    def test_registry_kind(self):
        assert spacy_detector().kind == "NE"

    def test_default_labels_exclude_ordinary_prose(self):
        """Redacting dates and cardinals destroys meaning and hides nothing."""
        for label in ("CARDINAL", "ORDINAL", "DATE", "TIME", "MONEY", "PERCENT", "QUANTITY"):
            assert label not in DEFAULT_ENTITY_LABELS

    def test_default_labels_include_the_personal_ones(self):
        for label in ("PERSON", "ORG", "GPE", "LOC"):
            assert label in DEFAULT_ENTITY_LABELS

    def test_all_labels_mode(self):
        assert spacy_detector(labels=None).labels is None
        assert spacy_detector(labels=None).kinds() == ()

    def test_kinds_reports_the_categories(self):
        detector = spacy_detector(labels=["PERSON", "ORG"])
        assert detector.kinds() == ("ORG", "PERSON")

    def test_kind_prefix(self):
        detector = spacy_detector(labels=["PERSON"], kind_prefix="NE_")
        assert detector.kinds() == ("NE_PERSON",)

    def test_priority_is_below_the_structural_patterns(self):
        """An address recognised by both should be labelled EMAIL, not ORG."""
        from .. import get_pattern

        assert spacy_detector().priority < get_pattern("EMAIL").priority

    def test_disabled_components_are_normalised(self):
        assert spacy_detector(disable=["b", "a"])._disable == ("a", "b")


class TestUnavailableTier:
    """The failure is actionable and arrives before the heavy import."""

    def test_detect_raises_capability_error(self, monkeypatch):
        from .. import _ner

        monkeypatch.setattr(_ner, "_PIPELINES", {})
        monkeypatch.setattr(caps, "_installed_version", lambda _n: None)
        with pytest.raises(CapabilityError) as caught:
            list(spacy_detector().detect("text", DEFAULT_POLICY))
        assert caught.value.tier == "ner"
        assert "pip install" in str(caught.value)

    def test_missing_model_is_misconfigured_not_absent(self, monkeypatch):
        """Installed spaCy plus a missing model is a different problem."""
        import sys
        import types

        from .. import _ner

        monkeypatch.setattr(_ner, "_PIPELINES", {})
        monkeypatch.setattr(caps, "_installed_version", lambda _n: "3.7.2")
        fake = types.ModuleType("spacy")

        def load(name, disable=None):
            raise OSError("[E050] Can't find model {0!r}".format(name))

        fake.load = load
        monkeypatch.setitem(sys.modules, "spacy", fake)

        with pytest.raises(CapabilityError) as caught:
            list(spacy_detector(model="xx_missing").detect("t", DEFAULT_POLICY))
        assert caught.value.status == "MISCONFIGURED"
        assert "python -m spacy download xx_missing" in str(caught.value)

    def test_capability_error_survives_the_registry(self, monkeypatch):
        """It must not be buried inside a DetectorError."""
        from .. import _ner

        monkeypatch.setattr(_ner, "_PIPELINES", {})
        monkeypatch.setattr(caps, "_installed_version", lambda _n: None)
        registry = default_registry(kinds=["EMAIL"])
        registry.add(spacy_detector())
        with pytest.raises(CapabilityError) as caught:
            Redactor(registry=registry).redact("text")
        assert "pip install" in str(caught.value)


class TestStubbedDetection:
    """Span translation, without needing a model."""

    def test_entities_become_spans(self, stub_detector):
        detector, _ = stub_detector
        text = "Ada Lovelace joined Acme"
        spans = list(detector.detect(text, DEFAULT_POLICY))
        assert [(s.start, s.kind, s.text) for s in spans] == [
            (0, "PERSON", "Ada Lovelace"),
            (20, "ORG", "Acme"),
        ]

    def test_span_text_matches_the_source(self, stub_detector):
        detector, _ = stub_detector
        text = "Ada Lovelace joined Acme"
        for span in detector.detect(text, DEFAULT_POLICY):
            assert span.text == text[span.start : span.end]

    def test_unselected_labels_are_dropped(self, monkeypatch, stub_detector):
        from .. import _ner

        detector = NerDetector(model="stub", labels=["PERSON"])
        _ner._PIPELINES[(detector.model, detector._disable)] = _StubPipeline(
            {"Ada Lovelace": "PERSON", "Acme": "ORG"}
        )
        spans = list(detector.detect("Ada Lovelace joined Acme", DEFAULT_POLICY))
        assert [s.kind for s in spans] == ["PERSON"]

    def test_pipeline_is_cached(self, stub_detector):
        detector, pipeline = stub_detector
        list(detector.detect("Acme", DEFAULT_POLICY))
        list(detector.detect("Acme", DEFAULT_POLICY))
        assert len(pipeline.calls) == 2  # two documents, one load

    def test_document_ceiling_is_raised_to_the_policy_limit(self, stub_detector):
        detector, pipeline = stub_detector
        list(detector.detect("Acme", DEFAULT_POLICY))
        assert pipeline.max_length >= DEFAULT_POLICY.limits.max_input_chars

    def test_pipeline_failure_is_attributed(self, monkeypatch):
        from .. import _ner
        from .. import DetectorError

        class Boom:
            max_length = 10**9

            def __call__(self, text):
                raise RuntimeError("model exploded")

        detector = NerDetector(model="boom")
        monkeypatch.setattr(_ner, "_PIPELINES", {(detector.model, detector._disable): Boom()})
        with pytest.raises(DetectorError) as caught:
            list(detector.detect("text", DEFAULT_POLICY))
        assert caught.value.detector == "ner:boom"
        assert "model exploded" in str(caught.value)


class TestCompositionWithStructuralDetectors:
    """``CP-006`` under a recogniser that would tag inside a placeholder."""

    def test_ner_never_sees_an_inserted_placeholder(self, stub_detector):
        """The stub tags the literal word ``EMAIL``; it must not fire on a tag."""
        detector, pipeline = stub_detector
        registry = default_registry(kinds=["EMAIL"])
        registry.add(detector)
        result = Redactor(registry=registry).redact("mail ada@example.com at Acme")
        assert pipeline.calls == ["mail ada@example.com at Acme"]
        assert "[EMAIL-1]" in result.text
        assert "[[" not in result.text

    def test_structural_detection_wins_an_overlap(self, monkeypatch):
        """An address matched by both is labelled EMAIL, not ORG."""
        from .. import _ner

        detector = NerDetector(model="stub", labels=["ORG"])
        _ner._PIPELINES[(detector.model, detector._disable)] = _StubPipeline(
            {"ada@example.com": "ORG"}
        )
        registry = default_registry(kinds=["EMAIL"])
        registry.add(detector)
        result = Redactor(registry=registry).redact("mail ada@example.com")
        assert result.text == "mail [EMAIL-1]"

    def test_round_trip_with_entities(self, stub_detector):
        detector, _ = stub_detector
        registry = default_registry(kinds=["EMAIL"])
        registry.add(detector)
        text = "Ada Lovelace at Acme, mail ada@example.com"
        result = Redactor(registry=registry).redact(text)
        assert restore(result.text, result.vault).text == text


@pytest.mark.skipif(not engine_ready("spacy"), reason=engine_skip_reason("spacy"))
class TestLiveModel:
    """Exercised only where spaCy and a model are installed."""

    def test_detects_a_person(self):
        registry = default_registry(kinds=["EMAIL"])
        registry.add(spacy_detector())
        result = Redactor(registry=registry).redact("Ada Lovelace wrote the notes.")
        assert result.stats.entries >= 1

    def test_round_trip(self):
        registry = default_registry()
        registry.add(spacy_detector())
        text = "Ada Lovelace wrote to ada@example.com from London."
        result = Redactor(registry=registry).redact(text)
        assert restore(result.text, result.vault).text == text
