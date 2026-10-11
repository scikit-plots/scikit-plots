"""
Tests for :mod:`scikitplot.cleanprompt._nltk`.

Notes
-----
**Developer notes — what is actually being tested here.**

NLTK's chunker returns a tree of *tokens*, not character offsets, while this
pipeline is built entirely on character spans over the original string. The
whole module exists to carry offsets through tokenization rather than recover
them afterwards, so that is the contract these tests defend:

    ``span.text == text[span.start:span.end]`` — always, for every span.

:class:`TestOffsetAlignment` asserts it over text containing tabs, newlines,
directional quotation marks, accented letters and astral-plane emoji, which is
exactly the class of input where a "find the entity again with ``str.find``"
implementation silently drifts. A drift of one character is not a cosmetic
defect in a redaction tool: it leaves a fragment of the value in the clear and
corrupts restoration.

The assertions split into two groups.

1. Everything that holds with no NLTK at all: construction, the label policy,
   the shape of each distinct failure, and the corpus accounting. These always
   run, and they never import NLTK.
2. Live behaviour, skipped when the tier or its corpora are unavailable.

There is deliberately **no stub chunker**. Stubbing NLTK here would test that a
fake returns what it was told to return; the contract under test is the
translation of real tokenizer output into spans, and only the real tokenizer
produces the boundary cases that matter. Where a stub *is* legitimate — proving
error attribution, where the exception's origin is the point — it is a
one-method fake and is named as such.
"""

from __future__ import annotations

import sys

import pytest

from .. import DEFAULT_POLICY, CapabilityError, Redactor, default_registry, restore
from .. import _capabilities as caps
from .._engines import CANONICAL_LABELS
from .._exceptions import DetectorError
from .._nltk import (
    REQUIRED_CORPORA,
    NltkDetector,
    _missing_corpora,
    corpora_status,
    nltk_detector,
)

NLTK_READY = caps.probe("nltk").available and corpora_status()["available"]

requires_nltk = pytest.mark.skipif(
    not NLTK_READY, reason="the 'nltk' tier or its corpora are unavailable"
)


def _spans(detector: NltkDetector, text: str) -> list:
    """Return the detector's spans for ``text``, eagerly."""
    return list(detector.detect(text, DEFAULT_POLICY))


class TestConstruction:
    """Building a detector costs nothing and imports nothing."""

    def test_no_import_at_construction(self):
        before = "nltk" in sys.modules
        nltk_detector()
        assert ("nltk" in sys.modules) == before

    def test_registry_name_and_kind(self):
        detector = nltk_detector()
        assert detector.name == "nltk"
        assert detector.kind == "NE"

    def test_default_language_is_english(self):
        assert nltk_detector().language == "en"

    def test_kinds_reports_the_categories(self):
        assert nltk_detector(labels=["PERSON", "ORG"]).kinds() == ("ORG", "PERSON")

    def test_kind_prefix_applies_to_kinds(self):
        detector = nltk_detector(labels=["PERSON"], kind_prefix="NE_")
        assert detector.kinds() == ("NE_PERSON",)

    def test_all_labels_mode(self):
        detector = nltk_detector(labels=None)
        assert detector.labels is None
        assert detector.kinds() == ()

    def test_priority_is_below_the_structural_patterns(self):
        """An address matched by both must be labelled EMAIL, not ORG."""
        from .. import get_pattern

        assert nltk_detector().priority < get_pattern("EMAIL").priority

    def test_confidence_is_below_spacy(self):
        """A documented judgement about relative quality, asserted so it stays."""
        from .._engines import ENGINES

        assert ENGINES["nltk"].confidence < ENGINES["spacy"].confidence

    def test_explicit_priority_and_confidence_override_the_engine(self):
        detector = nltk_detector(priority=5, confidence=0.25)
        assert (detector.priority, detector.confidence) == (5, 0.25)


class TestUnsupportedLanguage:
    """English-only, and the failure says so before anything is imported."""

    def test_non_english_raises_before_importing_nltk(self, monkeypatch):
        """The language check precedes ``require``, so it needs no tier."""
        monkeypatch.setattr(caps, "_installed_version", lambda _n: None)
        with pytest.raises(CapabilityError) as caught:
            _spans(nltk_detector(language="tr"), "Mustafa Kemal")
        assert caught.value.status == "MISCONFIGURED"
        assert caught.value.tier == "nltk"

    def test_the_remedy_names_the_other_engine(self):
        with pytest.raises(CapabilityError) as caught:
            _spans(nltk_detector(language="de"), "Angela")
        message = str(caught.value)
        assert "--ner-engine spacy" in message
        assert "--lang de" in message


class TestUnavailableTier:
    """Each distinct cause produces the message that matches it."""

    def test_absent_tier_raises_capability_error(self, monkeypatch):
        monkeypatch.setattr(caps, "_installed_version", lambda _n: None)
        with pytest.raises(CapabilityError) as caught:
            _spans(nltk_detector(), "Ada Lovelace")
        assert caught.value.tier == "nltk"
        assert "pip install" in str(caught.value)

    def test_missing_corpora_is_misconfigured_not_absent(self, monkeypatch):
        """Installed NLTK plus missing data is a different problem."""
        pytest.importorskip("nltk")  # MISCONFIGURED means installed; absent is ABSENT
        from .. import _nltk

        monkeypatch.setattr(caps, "_installed_version", lambda _n: "3.9")
        monkeypatch.setattr(_nltk, "missing_data", lambda _m: ["words"])
        with pytest.raises(CapabilityError) as caught:
            _spans(nltk_detector(), "Ada Lovelace")
        assert caught.value.status == "MISCONFIGURED"
        assert "nltk.download('words')" in str(caught.value)

    def test_the_remedy_is_a_runnable_command(self, monkeypatch):
        """A user must be able to paste the hint, not decode it."""
        pytest.importorskip("nltk")  # MISCONFIGURED means installed; absent is ABSENT
        from .. import _nltk

        monkeypatch.setattr(caps, "_installed_version", lambda _n: "3.9")
        monkeypatch.setattr(_nltk, "missing_data", lambda _m: ["punkt", "words"])
        with pytest.raises(CapabilityError) as caught:
            _spans(nltk_detector(), "Ada Lovelace")
        hint = caught.value.install_hint
        assert hint.startswith('python -c "import nltk;')
        assert hint.endswith('"')
        assert "nltk.download('punkt')" in hint

    def test_capability_error_survives_the_registry(self, monkeypatch):
        """It must not be buried inside a DetectorError."""
        monkeypatch.setattr(caps, "_installed_version", lambda _n: None)
        registry = default_registry(kinds=["EMAIL"])
        registry.add(nltk_detector())
        with pytest.raises(CapabilityError) as caught:
            Redactor(registry=registry).redact("text")
        assert "pip install" in str(caught.value)


class TestCorpusAccounting:
    """A package is present when *any* of its candidate paths resolves."""

    class _Data:
        """Stand-in for ``nltk.data``; resolves only the listed paths."""

        def __init__(self, resolvable):
            self.resolvable = set(resolvable)
            self.asked = []

        def find(self, path):
            self.asked.append(path)
            if path in self.resolvable:
                return path
            raise LookupError(path)

    class _Nltk:
        def __init__(self, data):
            self.data = data

    def _module(self, resolvable):
        return self._Nltk(self._Data(resolvable))

    def test_nothing_missing_when_every_package_resolves(self):
        every = [paths[0] for _package, paths in REQUIRED_CORPORA]
        assert _missing_corpora(self._module(every)) == []

    def test_the_legacy_name_satisfies_the_lookup(self):
        """NLTK 3.8.2 split packages; either name is enough."""
        legacy = [paths[-1] for _package, paths in REQUIRED_CORPORA]
        assert _missing_corpora(self._module(legacy)) == []

    def test_a_package_with_no_resolvable_path_is_reported(self):
        every = [paths[0] for _package, paths in REQUIRED_CORPORA]
        assert _missing_corpora(self._module(every[1:])) == [REQUIRED_CORPORA[0][0]]

    def test_missing_packages_are_reported_in_declaration_order(self):
        declared = [package for package, _paths in REQUIRED_CORPORA]
        assert _missing_corpora(self._module([])) == declared

    def test_an_unexpected_error_counts_as_absent(self):
        """A broken data directory must not crash a diagnostic."""

        class Exploding:
            data = type("D", (), {"find": staticmethod(lambda _p: 1 / 0)})()

        assert _missing_corpora(Exploding()) == [p for p, _ in REQUIRED_CORPORA]

    def test_status_reports_absent_tier_without_raising(self, monkeypatch):
        monkeypatch.setattr(caps, "_installed_version", lambda _n: None)
        status = corpora_status()
        assert status["available"] is False
        assert status["missing"] == [p for p, _ in REQUIRED_CORPORA]
        assert "pip install" in status["install_hint"]

    def test_status_has_a_stable_shape(self):
        status = corpora_status()
        assert set(status) == {"available", "missing", "install_hint"}
        assert isinstance(status["available"], bool)
        assert isinstance(status["missing"], list)

    def test_status_hint_is_empty_when_nothing_is_missing(self):
        status = corpora_status()
        if status["available"]:
            assert status["install_hint"] == ""


class TestLoadability:
    """
    ``CP-100``: present on disk is not the same as loadable by this NLTK.

    Notes
    -----
    **Developer notes.** NLTK 3.9 moved the tagger to
    ``averaged_perceptron_tagger_eng`` and the chunker to
    ``maxent_ne_chunker_tab``. :func:`_missing_corpora` accepts either name of
    a group, so with only the older packages NLTK 3.10.3 was reported ready
    and the first sentence failed in :func:`nltk.pos_tag`. Measured before the
    fix: ``doctor`` ready, ``inspect`` exit 1 with ``LookupError``. After:
    not ready, exit 69, the remedy naming the current package.

    The stand-ins here are one-method fakes in the sense the module notes
    allow: the point is *which* step raised ``LookupError``, which is the
    exception's origin, not recognition quality.
    """

    class _Nltk:
        """Every path resolves; ``pos_tag`` and ``ne_chunk`` raise as told."""

        def __init__(self, tagger_ok=True, chunker_ok=True):
            self.tagger_ok, self.chunker_ok = tagger_ok, chunker_ok
            self.data = type("D", (), {"find": staticmethod(lambda path: path)})()

        def pos_tag(self, tokens):
            if not self.tagger_ok:
                raise LookupError("averaged_perceptron_tagger_eng")
            return [(token, "NNP") for token in tokens]

        def ne_chunk(self, tagged):
            if not self.chunker_ok:
                raise LookupError("maxent_ne_chunker_tab")
            return tagged

    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch):
        pytest.importorskip("nltk")  # the tokenizers are NLTK's own classes
        from .. import _nltk

        monkeypatch.setattr(_nltk, "_RESOURCES", {})
        monkeypatch.setattr(_nltk, "_build_chunker", lambda: None)
        return _nltk

    def test_everything_loadable_is_ready_and_cached(self, _fresh):
        assert _fresh.missing_data(self._Nltk()) == []
        assert {"sentence", "word", "chunker"} <= set(_fresh._RESOURCES)

    def test_a_tagger_this_nltk_cannot_load_is_reported(self, _fresh):
        assert _fresh.missing_data(self._Nltk(tagger_ok=False)) == ["averaged_perceptron_tagger"]

    def test_a_chunker_this_nltk_cannot_load_is_reported(self, _fresh):
        assert _fresh.missing_data(self._Nltk(chunker_ok=False)) == ["maxent_ne_chunker"]

    def test_both_are_reported_in_one_answer(self, _fresh):
        """A user fixing one should not discover the other afterwards."""
        missing = _fresh.missing_data(self._Nltk(tagger_ok=False, chunker_ok=False))
        assert missing == ["averaged_perceptron_tagger", "maxent_ne_chunker"]

    def test_a_failure_is_not_cached(self, _fresh):
        """Data downloaded mid-process is believed on the next call."""
        assert _fresh.missing_data(self._Nltk(tagger_ok=False))
        assert "chunker" not in _fresh._RESOURCES
        assert _fresh.missing_data(self._Nltk()) == []

    def test_the_path_check_still_comes_first(self, _fresh):
        """Absent packages are all named at once, without running anything."""
        ran = []

        def absent(path):
            raise LookupError(path)

        class Absent(self._Nltk):
            def __init__(self):
                super().__init__()
                self.data = type("D", (), {"find": staticmethod(absent)})()

            def pos_tag(self, tokens):
                ran.append("tagger")
                return super().pos_tag(tokens)

        assert _fresh.missing_data(Absent()) == [p for p, _ in REQUIRED_CORPORA]
        assert ran == []

    def test_the_remedy_names_every_package_that_satisfies_a_group(self, _fresh):
        command = _fresh.download_command(["averaged_perceptron_tagger", "maxent_ne_chunker"])
        for name in (
            "averaged_perceptron_tagger_eng",
            "averaged_perceptron_tagger",
            "maxent_ne_chunker_tab",
            "maxent_ne_chunker",
        ):
            assert f"nltk.download('{name}')" in command
        assert "punkt" not in command, "only the groups asked for"

    def test_the_detector_refuses_with_the_remedy(self, _fresh, monkeypatch):
        monkeypatch.setattr(caps, "_installed_version", lambda _n: "3.10.3")
        monkeypatch.setattr(_fresh, "missing_data", lambda _m: ["averaged_perceptron_tagger"])
        with pytest.raises(CapabilityError) as caught:
            _spans(nltk_detector(), "Ada Lovelace")
        assert caught.value.status == "MISCONFIGURED"
        assert "averaged_perceptron_tagger_eng" in caught.value.install_hint


@requires_nltk
class TestOffsetAlignment:
    """
    The guarantee the module exists for: spans index the *original* text.

    Notes
    -----
    **Developer notes.** Each case is a character class that breaks offset
    recovery by search. Tabs and newlines because a sentence tokenizer
    normalises them; directional quotes because Treebank rewrites ``"`` into
    `````` and ``''`` in its non-span mode; accented letters because
    they are multi-byte but single-character and an implementation that counted
    bytes anywhere would drift; emoji because they are astral-plane and would
    drift on any UTF-16 assumption.
    """

    CASES = {
        "plain": "Ada Lovelace worked in London with Charles Babbage.",
        "tabs": "Ada Lovelace\tworked in\tLondon today.",
        "newlines": "Ada Lovelace\nworked in London.\n\nCharles Babbage agreed.",
        "curly_quotes": "“Ada Lovelace” worked in ‘London’ then.",
        "straight_quotes": '"Ada Lovelace" worked in "London" that year.',
        "accents": "Zoë François met Renée in Montréal.",
        "emoji": "\U0001f680 Ada Lovelace \U0001f680 worked in London \U0001f389 today.",
        "mixed": (
            "\U0001f4cc “Zoë François”\n\tmet Ada Lovelace "
            "in Montréal \U0001f389\nthen left for London."
        ),
        "brackets": "Ada Lovelace [1] worked in London (England) then.",
        "repeated": "London is not London, but London is London.",
    }

    @pytest.mark.parametrize("name", sorted(CASES))
    def test_span_text_equals_the_source_slice(self, name):
        text = self.CASES[name]
        for span in _spans(nltk_detector(), text):
            assert span.text == text[span.start : span.end], (
                "span {0}..{1} of case {2!r} does not index its own text".format(
                    span.start, span.end, name
                )
            )

    @pytest.mark.parametrize("name", sorted(CASES))
    def test_spans_are_well_formed_and_in_bounds(self, name):
        text = self.CASES[name]
        for span in _spans(nltk_detector(), text):
            assert 0 <= span.start < span.end <= len(text)

    @pytest.mark.parametrize("name", sorted(CASES))
    def test_spans_are_in_ascending_order(self, name):
        starts = [span.start for span in _spans(nltk_detector(), self.CASES[name])]
        assert starts == sorted(starts)

    @pytest.mark.parametrize("name", sorted(CASES))
    def test_every_case_round_trips_exactly(self, name):
        text = self.CASES[name]
        registry = default_registry()
        registry.add(nltk_detector())
        result = Redactor(registry=registry).redact(text)
        assert restore(result.text, result.vault).text == text

    def test_offsets_survive_a_leading_run_of_whitespace(self):
        """A sentence tokenizer that trimmed silently would shift every span."""
        text = "\n\n\t  Ada Lovelace worked in London."
        for span in _spans(nltk_detector(), text):
            assert span.text == text[span.start : span.end]

    def test_a_repeated_entity_is_not_collapsed_onto_the_first_occurrence(self):
        """The defect that offset-by-search implementations always have."""
        text = "London is not London, but London is London."
        starts = [s.start for s in _spans(nltk_detector(), text) if s.text == "London"]
        assert len(set(starts)) == len(starts)
        assert len(starts) >= 2


@requires_nltk
class TestLiveDetection:
    """Behaviour against the real chunker."""

    def test_detects_a_person(self):
        spans = _spans(nltk_detector(), "Ada Lovelace wrote the notes.")
        assert any(span.kind == "PERSON" for span in spans)

    def test_every_kind_is_canonical(self):
        """NLTK's ORGANIZATION/LOCATION/GSP must never reach a vault."""
        text = (
            "Ada Lovelace joined the Analytical Society of London and "
            "travelled to France with Charles Babbage."
        )
        for span in _spans(nltk_detector(labels=None), text):
            assert span.kind in CANONICAL_LABELS

    def test_the_uncanonical_names_never_appear(self):
        text = "Ada Lovelace joined the Analytical Society of London."
        kinds = {span.kind for span in _spans(nltk_detector(labels=None), text)}
        assert not kinds & {"ORGANIZATION", "LOCATION", "FACILITY", "GSP", "PER"}

    def test_unselected_labels_are_dropped(self):
        text = "Ada Lovelace joined the Analytical Society of London."
        kinds = {span.kind for span in _spans(nltk_detector(labels=["PERSON"]), text)}
        assert kinds <= {"PERSON"}

    def test_kind_prefix_reaches_the_spans(self):
        detector = nltk_detector(labels=["PERSON"], kind_prefix="NE_")
        kinds = {s.kind for s in _spans(detector, "Ada Lovelace wrote the notes.")}
        assert kinds <= {"NE_PERSON"}

    def test_empty_text_yields_nothing(self):
        assert _spans(nltk_detector(), "") == []

    def test_whitespace_only_text_yields_nothing(self):
        assert _spans(nltk_detector(), "  \n\t  ") == []

    def test_text_with_no_entities_yields_nothing(self):
        assert _spans(nltk_detector(), "the cat sat on the mat.") == []

    def test_detection_is_deterministic(self):
        text = "Ada Lovelace worked in London with Charles Babbage."
        first = [(s.start, s.end, s.kind) for s in _spans(nltk_detector(), text)]
        second = [(s.start, s.end, s.kind) for s in _spans(nltk_detector(), text)]
        assert first == second

    def test_the_detector_holds_no_state_between_documents(self):
        detector = nltk_detector()
        text = "Ada Lovelace worked in London."
        _spans(detector, "Charles Babbage went to Paris.")
        assert [(s.start, s.kind) for s in _spans(detector, text)] == [
            (s.start, s.kind) for s in _spans(nltk_detector(), text)
        ]


@requires_nltk
class TestCompositionWithStructuralDetectors:
    """``CP-006``: the chunker never sees an inserted placeholder."""

    def test_structural_detection_wins_an_overlap(self):
        registry = default_registry(kinds=["EMAIL"])
        registry.add(nltk_detector())
        result = Redactor(registry=registry).redact("Mail ada@example.com today.")
        assert "[EMAIL-1]" in result.text
        assert "ada@example.com" not in result.text

    def test_no_placeholder_is_redacted_twice(self):
        registry = default_registry()
        registry.add(nltk_detector())
        result = Redactor(registry=registry).redact(
            "Ada Lovelace mailed ada@example.com from London on 2024-01-15."
        )
        assert "[[" not in result.text
        assert "]]" not in result.text

    def test_round_trip_with_structural_and_entity_detectors(self):
        registry = default_registry()
        registry.add(nltk_detector())
        text = (
            "Ada Lovelace wrote to ada@example.com from London, "
            "card 4242 4242 4242 4242, host 192.168.1.10."
        )
        result = Redactor(registry=registry).redact(text)
        assert restore(result.text, result.vault).text == text

    def test_a_vault_written_under_nltk_restores_under_spacy_kinds(self):
        """Canonical labels are what makes the engines interchangeable."""
        registry = default_registry(kinds=["EMAIL"])
        registry.add(nltk_detector())
        result = Redactor(registry=registry).redact("Ada Lovelace, London.")
        for entry in result.entries:
            assert entry.kind in CANONICAL_LABELS or entry.kind == "EMAIL"


class TestErrorAttribution:
    """A failure inside NLTK is named, never swallowed or re-branded."""

    def _ready(self, monkeypatch, nltk, sentence, word, chunker=None):
        """Bypass the readiness checks with a fixed set of NLTK objects."""
        from .. import _nltk

        monkeypatch.setattr(
            NltkDetector,
            "_ensure_ready",
            lambda _self: (nltk, sentence, word, chunker),
        )
        return _nltk

    def test_sentence_tokenizer_failure_is_attributed(self, monkeypatch):
        class Boom:
            @staticmethod
            def span_tokenize(_text):
                raise RuntimeError("punkt exploded")

        self._ready(monkeypatch, None, Boom(), None)
        with pytest.raises(DetectorError) as caught:
            _spans(nltk_detector(), "Ada Lovelace")
        assert caught.value.detector == "nltk"
        assert "punkt exploded" in str(caught.value)

    def test_chunker_failure_is_attributed(self, monkeypatch):
        class Sentences:
            @staticmethod
            def span_tokenize(text):
                return [(0, len(text))]

        class Words:
            @staticmethod
            def span_tokenize(text):
                return [(0, len(text))]

        class Nltk:
            """Both names must exist: ``nltk.ne_chunk`` is looked up first."""

            @staticmethod
            def pos_tag(_tokens):
                raise RuntimeError("tagger exploded")

            @staticmethod
            def ne_chunk(_tagged):  # pragma: no cover - pos_tag raises first
                raise AssertionError("unreachable")

        self._ready(monkeypatch, Nltk(), Sentences(), Words())
        with pytest.raises(DetectorError) as caught:
            _spans(nltk_detector(), "Ada Lovelace")
        assert caught.value.detector == "nltk"
        assert "tagger exploded" in str(caught.value)

    def test_the_original_exception_is_chained(self, monkeypatch):
        class Boom:
            @staticmethod
            def span_tokenize(_text):
                raise ValueError("root cause")

        self._ready(monkeypatch, None, Boom(), None)
        with pytest.raises(DetectorError) as caught:
            _spans(nltk_detector(), "Ada Lovelace")
        assert isinstance(caught.value.__cause__, ValueError)
