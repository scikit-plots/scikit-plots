"""
Tests for :mod:`scikitplot.cleanprompt._engines`.

Notes
-----
**Developer notes.** The class that matters is :class:`TestCanonicalVocabulary`.
Without it a vault written under NLTK would not restore under spaCy, because
the placeholder would carry a different label for the same category.
"""

from __future__ import annotations

import pytest

from .. import _capabilities as caps
from .._engines import (
    CANONICAL_LABELS,
    DEFAULT_ENGINE,
    DEFAULT_ENTITY_LABELS,
    ENGINE_MODES,
    ENGINES,
    build_detectors,
    canonical_label,
    describe_engines,
    resolve_engine,
)
from .._exceptions import PolicyError


class TestCanonicalVocabulary:
    """One vocabulary, so engines are interchangeable."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("ORGANIZATION", "ORG"),   # NLTK
            ("ORG", "ORG"),            # spaCy
            ("LOCATION", "LOC"),       # NLTK
            ("LOC", "LOC"),            # spaCy
            ("FACILITY", "FAC"),       # NLTK
            ("FAC", "FAC"),            # spaCy
            ("GSP", "GPE"),            # NLTK geo-social-political
            ("GPE", "GPE"),
            ("PERSON", "PERSON"),
            ("PER", "PERSON"),         # spaCy multilingual
        ],
    )
    def test_both_engines_land_on_one_label(self, raw, expected):
        assert canonical_label(raw) == expected

    def test_unknown_labels_become_misc_not_dropped(self):
        """A vocabulary gap must not turn into a privacy hole."""
        assert canonical_label("SOMETHING_NEW_IN_SPACY_6") == "MISC"

    def test_case_is_normalised(self):
        assert canonical_label("organization") == "ORG"

    def test_every_mapping_target_is_canonical(self):
        from .._engines import _LABEL_MAP

        assert set(_LABEL_MAP.values()) <= set(CANONICAL_LABELS)

    def test_defaults_are_canonical(self):
        assert DEFAULT_ENTITY_LABELS <= set(CANONICAL_LABELS)

    def test_prose_categories_stay_out_of_the_defaults(self):
        """A live run labelled part of a phone number as DATE."""
        for label in ("DATE", "TIME", "CARDINAL", "ORDINAL", "MONEY", "PERCENT"):
            assert label not in DEFAULT_ENTITY_LABELS


class TestEngineTable:
    """The static description of each engine."""

    def test_known_engines(self):
        assert set(ENGINES) == {"spacy", "nltk"}

    def test_nltk_is_english_only(self):
        assert ENGINES["nltk"].languages == ("en",)
        assert ENGINES["nltk"].supports("en")
        assert not ENGINES["nltk"].supports("de")

    def test_spacy_is_model_dependent(self):
        assert ENGINES["spacy"].supports("de")
        assert ENGINES["spacy"].supports("anything")

    def test_both_rank_below_the_structural_patterns(self):
        """An address found by both should be labelled EMAIL, not ORG."""
        from .. import get_pattern

        for spec in ENGINES.values():
            assert spec.priority < get_pattern("EMAIL").priority

    def test_spacy_outranks_nltk(self):
        assert ENGINES["spacy"].priority > ENGINES["nltk"].priority

    def test_spacy_is_more_confident_than_nltk(self):
        assert ENGINES["spacy"].confidence > ENGINES["nltk"].confidence


class TestResolveEngine:
    """Turning a mode into the engines that will run."""

    def test_none_selects_nothing(self):
        assert resolve_engine("none") == ()

    def test_both_selects_both(self):
        assert resolve_engine("both") == ("spacy", "nltk")

    @pytest.mark.parametrize("name", ["spacy", "nltk"])
    def test_named_engine_is_selected(self, name):
        assert resolve_engine(name) == (name,)

    def test_named_engine_is_not_dropped_when_unusable(self, monkeypatch):
        """Asked for by name, it must fail loudly rather than vanish."""
        monkeypatch.setattr(caps, "_installed_version", lambda _n: None)
        assert resolve_engine("spacy") == ("spacy",)

    def test_auto_prefers_spacy(self, monkeypatch):
        monkeypatch.setattr(caps, "_installed_version", lambda _n: "3.7.2")
        assert resolve_engine("auto") == ("spacy",)

    def test_auto_falls_back_to_nltk(self, monkeypatch):
        def version(name):
            return None if name == "spacy" else "3.9"

        monkeypatch.setattr(caps, "_installed_version", version)
        assert resolve_engine("auto") == ("nltk",)

    def test_auto_selects_nothing_when_neither_is_usable(self, monkeypatch):
        monkeypatch.setattr(caps, "_installed_version", lambda _n: None)
        assert resolve_engine("auto") == ()

    def test_auto_skips_nltk_for_a_foreign_language(self, monkeypatch):
        def version(name):
            return None if name == "spacy" else "3.9"

        monkeypatch.setattr(caps, "_installed_version", version)
        assert resolve_engine("auto", "de") == ()

    def test_unknown_mode_is_refused(self):
        with pytest.raises(PolicyError, match="unknown NER engine mode"):
            resolve_engine("transformers")

    def test_default_is_auto(self):
        assert DEFAULT_ENGINE == "auto"

    def test_every_mode_resolves(self):
        for mode in ENGINE_MODES:
            assert isinstance(resolve_engine(mode), tuple)


class TestDescribeEngines:
    """The report doctor and the web page render."""

    def test_covers_every_engine(self):
        assert set(describe_engines()["engines"]) == set(ENGINES)

    def test_reports_the_selection(self):
        report = describe_engines("en", "both")
        assert report["selected"] == ["spacy", "nltk"]

    def test_foreign_language_marks_nltk_misconfigured(self):
        """Not ABSENT: it is installed, it simply cannot do this language."""
        entry = describe_engines("de", "auto")["engines"]["nltk"]
        assert entry["status"] == "MISCONFIGURED"
        assert entry["language_supported"] is False
        assert "does not support" in entry["detail"]

    def test_is_json_safe(self):
        import json

        payload = describe_engines()
        assert json.loads(json.dumps(payload)) == payload

    def test_never_imports_an_engine(self):
        import sys

        before = {m for m in ("spacy", "nltk") if m in sys.modules}
        describe_engines()
        assert {m for m in ("spacy", "nltk") if m in sys.modules} == before


class TestBuildDetectors:
    """Construction is free; engines load on first use."""

    def test_none_builds_nothing(self):
        assert build_detectors("none") == []

    def test_builds_one_per_engine(self):
        assert len(build_detectors("both")) == 2

    def test_construction_imports_nothing(self):
        import sys

        before = {m for m in ("spacy", "nltk") if m in sys.modules}
        build_detectors("both")
        assert {m for m in ("spacy", "nltk") if m in sys.modules} == before

    def test_unknown_mode_is_refused(self):
        with pytest.raises(PolicyError):
            build_detectors("nope")

    def test_language_reaches_the_spacy_detector(self):
        detector = build_detectors("spacy", language="de")[0]
        assert detector.model == "de_core_news_sm"
