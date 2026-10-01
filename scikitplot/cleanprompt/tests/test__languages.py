"""Tests for :mod:`scikitplot.cleanprompt._languages`."""

from __future__ import annotations

import pytest

from .._exceptions import PolicyError
from .._languages import (
    LANGUAGES,
    MODEL_SIZES,
    MULTILINGUAL_MODEL,
    installed_models,
    language_report,
    resolve_model,
    supported_languages,
)


class TestTable:
    """The language table itself."""

    def test_covers_the_common_languages(self):
        for code in ("en", "de", "fr", "es", "it", "pt", "nl", "ja", "zh", "ru"):
            assert code in LANGUAGES

    def test_english_uses_the_web_genre(self):
        """spaCy publishes en_core_web_*, not en_core_news_*."""
        assert LANGUAGES["en"].genre == "web"

    def test_other_languages_use_news(self):
        assert LANGUAGES["de"].genre == "news"

    def test_every_spec_declares_sizes(self):
        for spec in LANGUAGES.values():
            assert spec.sizes
            assert set(spec.sizes) <= set(MODEL_SIZES)

    def test_supported_languages_is_sorted(self):
        codes = supported_languages()
        assert list(codes) == sorted(codes)


class TestResolveModel:
    """Language and size to model name."""

    @pytest.mark.parametrize(
        "lang,size,expected",
        [
            ("en", "sm", "en_core_web_sm"),
            ("en", "lg", "en_core_web_lg"),
            ("en", "trf", "en_core_web_trf"),
            ("de", "sm", "de_core_news_sm"),
            ("de", "lg", "de_core_news_lg"),
            ("ja", "trf", "ja_core_news_trf"),
        ],
    )
    def test_known_combinations(self, lang, size, expected):
        assert resolve_model(lang, size)[0] == expected

    def test_unknown_language_falls_back_to_multilingual(self):
        model, note = resolve_model("tr")
        assert model == MULTILINGUAL_MODEL
        assert note["fallback"] is True

    def test_the_fallback_explains_itself(self):
        """Thin results must read as a model limit, not an empty document."""
        _, note = resolve_model("tr")
        assert "no named-entity model" in note["reason"]
        assert "broad categories" in note["reason"]

    def test_size_falls_back_downwards(self):
        """Asking for trf on a language without it gets the largest available."""
        model, note = resolve_model("de", "trf")
        assert model == "de_core_news_lg"
        assert note["resolved_size"] == "lg"
        assert "publishes" in note["reason"]

    def test_explicit_model_wins(self):
        model, note = resolve_model("de", "lg", explicit="my_custom_model")
        assert model == "my_custom_model"
        assert note["fallback"] is False

    def test_unknown_size_is_refused(self):
        with pytest.raises(PolicyError, match="unknown model size"):
            resolve_model("en", "enormous")

    def test_note_is_json_safe(self):
        import json

        for lang in ("en", "de", "tr"):
            json.dumps(resolve_model(lang)[1])

    def test_resolution_is_deterministic(self):
        assert resolve_model("de", "trf") == resolve_model("de", "trf")


class TestInstalledModels:
    """Reading what is actually present."""

    def test_returns_a_mapping(self):
        assert isinstance(installed_models(), dict)

    def test_imports_nothing(self):
        import sys

        before = "spacy" in sys.modules
        installed_models()
        assert ("spacy" in sys.modules) == before

    def test_only_reports_known_models(self):
        known = {MULTILINGUAL_MODEL}
        for spec in LANGUAGES.values():
            for size in spec.sizes:
                known.add("{0}_core_{1}_{2}".format(spec.code, spec.genre, size))
        assert set(installed_models()) <= known


class TestLanguageReport:
    """The report doctor and the web page render."""

    def test_shape(self):
        report = language_report("en")
        for key in ("requested", "supported", "model", "model_installed", "install_hint"):
            assert key in report

    def test_install_hint_is_a_command(self):
        assert language_report("de")["install_hint"].startswith("python -m spacy download")

    def test_unsupported_language_is_marked(self):
        report = language_report("tr")
        assert report["supported"] is False
        assert report["resolution"]["fallback"] is True

    def test_lists_every_supported_language(self):
        assert len(language_report("en")["supported_languages"]) == len(LANGUAGES)

    def test_is_json_safe(self):
        import json

        payload = language_report("de", "lg")
        assert json.loads(json.dumps(payload)) == payload
