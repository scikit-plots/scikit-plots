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
from .. import _engines
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
        """With its model present, spaCy wins."""
        monkeypatch.setattr(caps, "_installed_version", lambda _n: "3.7.2")
        monkeypatch.setattr(_engines, "_spacy_model_ready", lambda _m: True)
        assert resolve_engine("auto") == ("spacy",)

    def test_auto_skips_spacy_without_its_model(self, monkeypatch):
        """CP-093: installed without a model is not ready, so NLTK is chosen."""
        monkeypatch.setattr(caps, "_installed_version", lambda _n: "3.7.2")
        monkeypatch.setattr(_engines, "_spacy_model_ready", lambda _m: False)
        assert resolve_engine("auto") == ("nltk",)

    def test_auto_skips_nltk_without_its_data_when_checked(self, monkeypatch):
        """With ``check_assets`` NLTK's missing data rules it out too."""
        monkeypatch.setattr(caps, "_installed_version", lambda _n: "3.7.2")
        monkeypatch.setattr(_engines, "_spacy_model_ready", lambda _m: False)
        monkeypatch.setattr(_engines, "_nltk_missing_corpora", lambda: ("punkt",))
        assert resolve_engine("auto", check_assets=True) == ()
        assert resolve_engine("auto") == ("nltk",), "unchecked data is not judged"

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


class TestReadiness:
    """
    ``CP-093``: an engine is ready when package, language and data all hold.

    Notes
    -----
    **Developer notes.** Every case supplies the installation through the
    three seams the module reads — ``_capabilities._installed_version`` for
    the package, ``_spacy_model_ready`` for the model and
    ``_nltk_missing_corpora`` for NLTK's data — so each answer is checked on
    every machine, whatever it has installed (the lesson of ``CP-088``).
    """

    @staticmethod
    def _machine(monkeypatch, *, spacy=None, nltk=None, model=True, corpora=()):
        """Supply the installation: package versions, model, missing data."""
        versions = {"spacy": spacy, "nltk": nltk}
        monkeypatch.setattr(caps, "_installed_version", lambda name: versions.get(name))
        monkeypatch.setattr(_engines, "_spacy_model_ready", lambda _m: model)

        def missing():
            if isinstance(corpora, BaseException):
                raise corpora
            return tuple(corpora)

        monkeypatch.setattr(_engines, "_nltk_missing_corpora", missing)

    def test_spacy_with_model_is_ready(self, monkeypatch):
        self._machine(monkeypatch, spacy="3.7.2")
        result = _engines.engine_readiness("spacy")
        assert result.ready and result.assets_ready is True
        assert result.status == "AVAILABLE" and result.remedy == ""
        assert result.model == "en_core_web_sm"

    def test_spacy_without_model_names_the_download(self, monkeypatch):
        self._machine(monkeypatch, spacy="3.7.2", model=False)
        result = _engines.engine_readiness("spacy", "de", size="lg")
        assert result.installed and not result.ready
        assert result.assets_ready is False
        assert result.status == "MISCONFIGURED"
        assert result.remedy == "python -m spacy download de_core_news_lg"

    def test_an_explicit_model_is_the_one_checked(self, monkeypatch):
        seen = []
        self._machine(monkeypatch, spacy="3.7.2")
        monkeypatch.setattr(_engines, "_spacy_model_ready", lambda m: seen.append(m) or False)
        result = _engines.engine_readiness("spacy", model="my_pipeline")
        assert seen == ["my_pipeline"]
        assert result.remedy == "python -m spacy download my_pipeline"

    def test_spacy_absent_names_the_install(self, monkeypatch):
        self._machine(monkeypatch)
        result = _engines.engine_readiness("spacy")
        assert not result.installed and result.status == "ABSENT"
        assert result.remedy.startswith("pip install")
        assert result.assets_checked is False

    def test_nltk_data_unchecked_by_default(self, monkeypatch):
        """Checking NLTK's data imports NLTK; by default that is not done."""
        self._machine(monkeypatch, nltk="3.9", corpora=("punkt",))
        result = _engines.engine_readiness("nltk")
        assert result.ready and result.assets_checked is False
        assert result.assets_ready is None
        assert "doctor" in result.reason

    def test_nltk_data_missing_when_checked(self, monkeypatch):
        self._machine(monkeypatch, nltk="3.9", corpora=("punkt", "words"))
        result = _engines.engine_readiness("nltk", check_assets=True)
        assert not result.ready and result.assets_ready is False
        assert result.missing == ("punkt", "words")
        assert "nltk.download('punkt')" in result.remedy
        assert "nltk.download('words')" in result.remedy

    def test_nltk_that_will_not_import_is_broken(self, monkeypatch):
        self._machine(monkeypatch, nltk="3.9", corpora=ImportError("shadowed"))
        result = _engines.engine_readiness("nltk", check_assets=True)
        assert result.status == "BROKEN" and not result.ready
        assert "--force-reinstall" in result.remedy

    def test_language_is_checked_before_the_package(self, monkeypatch):
        """No installation fixes an engine that cannot read the language."""
        self._machine(monkeypatch)
        result = _engines.engine_readiness("nltk", "de")
        assert result.status == "MISCONFIGURED"
        assert "does not support" in result.reason

    def test_unknown_engine_is_refused(self):
        with pytest.raises(PolicyError, match="unknown NER engine"):
            _engines.engine_readiness("transformers")

    def test_ready_is_the_conjunction(self, monkeypatch):
        for spacy, model, expected in (
            ("3.7.2", True, True),
            ("3.7.2", False, False),
            (None, True, False),
        ):
            self._machine(monkeypatch, spacy=spacy, model=model)
            assert _engines.engine_readiness("spacy").ready is expected

    def test_report_is_json_safe(self, monkeypatch):
        import json

        self._machine(monkeypatch, spacy="3.7.2", nltk="3.9", model=False, corpora=("words",))
        payload = describe_engines(check_assets=True)
        assert json.loads(json.dumps(payload)) == payload

    def test_report_names_the_three_conditions(self, monkeypatch):
        self._machine(monkeypatch, spacy="3.7.2", model=False)
        entry = describe_engines("en", "spacy")["engines"]["spacy"]
        for key in ("installed", "language_supported", "assets_ready", "ready", "remedy"):
            assert key in entry
        assert entry["usable"] is entry["ready"] is False

    def test_report_ready_is_false_when_a_selected_engine_cannot_run(self, monkeypatch):
        self._machine(monkeypatch, spacy="3.7.2", model=False)
        assert describe_engines("en", "spacy")["ready"] is False
        self._machine(monkeypatch, spacy="3.7.2", model=True)
        assert describe_engines("en", "spacy")["ready"] is True

    def test_required_named_engine_without_data_is_refused(self, monkeypatch):
        """The doctor / inspect agreement: refused here, before any text."""
        from .._exceptions import CapabilityError

        self._machine(monkeypatch, spacy="3.7.2", model=False)
        with pytest.raises(CapabilityError) as caught:
            build_detectors("spacy", required=True)
        assert caught.value.install_hint == "python -m spacy download en_core_web_sm"
        assert caught.value.status == "MISCONFIGURED"

    def test_required_nltk_without_data_is_refused(self, monkeypatch):
        from .._exceptions import CapabilityError

        self._machine(monkeypatch, nltk="3.9", corpora=("punkt",))
        with pytest.raises(CapabilityError) as caught:
            build_detectors("nltk", required=True)
        assert "nltk.download('punkt')" in caught.value.install_hint

    def test_required_both_needs_both(self, monkeypatch):
        """``both`` is a request for two engines; half of it is not met."""
        from .._exceptions import CapabilityError

        self._machine(monkeypatch, spacy="3.7.2", nltk="3.9", model=True, corpora=("words",))
        with pytest.raises(CapabilityError) as caught:
            build_detectors("both", required=True)
        assert "nltk: MISCONFIGURED" in str(caught.value)
        assert "spacy:" not in str(caught.value)

    def test_required_auto_with_nothing_ready_names_every_engine(self, monkeypatch):
        from .._exceptions import CapabilityError

        self._machine(monkeypatch, spacy="3.7.2", nltk="3.9", model=False, corpora=("words",))
        with pytest.raises(CapabilityError) as caught:
            build_detectors("auto", required=True)
        message = str(caught.value)
        assert "no engine is ready" in message
        assert "python -m spacy download en_core_web_sm" in message
        assert "nltk.download('words')" in message

    def test_required_auto_picks_the_ready_one(self, monkeypatch):
        self._machine(monkeypatch, spacy="3.7.2", nltk="3.9", model=False, corpora=())
        detectors = build_detectors("auto", required=True)
        assert [d.name for d in detectors] == ["nltk"]

    def test_unrequired_construction_still_imports_nothing(self, monkeypatch):
        import sys

        self._machine(monkeypatch, spacy="3.7.2", nltk="3.9")
        before = {m for m in ("spacy", "nltk") if m in sys.modules}
        build_detectors("auto")
        describe_engines()
        assert {m for m in ("spacy", "nltk") if m in sys.modules} == before


class TestSpacyModelLookup:
    """The three ways a spaCy model can be present, none of which imports it."""

    def test_installed_distribution(self, monkeypatch):
        from .. import _languages

        monkeypatch.setattr(_languages, "installed_models", lambda: {"en_core_web_sm": "3.8.0"})
        assert _engines._spacy_model_ready("en_core_web_sm") is True

    def test_importable_package(self, monkeypatch, tmp_path):
        from .. import _languages

        monkeypatch.setattr(_languages, "installed_models", dict)
        package = tmp_path / "my_pipeline_cp093"
        package.mkdir()
        (package / "__init__.py").write_bytes(b"raise RuntimeError('must not be imported')\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        assert _engines._spacy_model_ready("my_pipeline_cp093") is True

    def test_directory(self, monkeypatch, tmp_path):
        from .. import _languages

        monkeypatch.setattr(_languages, "installed_models", dict)
        assert _engines._spacy_model_ready(str(tmp_path)) is True
        assert _engines._spacy_model_ready(str(tmp_path / "missing")) is False

    def test_absent(self, monkeypatch):
        from .. import _languages

        monkeypatch.setattr(_languages, "installed_models", dict)
        assert _engines._spacy_model_ready("xx_no_such_model_cp093") is False
        assert _engines._spacy_model_ready("not-an-identifier") is False
