"""
Tests for :mod:`scikitplot.cleanprompt._diagnostics`.

Notes
-----
**Developer notes.** The case that motivated this module is
:class:`TestTheReportedDefect`, which runs the exact text that was pasted into
the web interface and got returned unchanged with no explanation. That test is
the gate on the fix; if it ever passes trivially again, the tool has regressed
to silently doing nothing.
"""

from __future__ import annotations

import pytest

from .. import DEFAULT_POLICY, Redactor, default_registry
from .._capabilities import CapabilityStatus
from .._diagnostics import (
    BlindSpot,
    Diagnosis,
    Suggestion,
    describe_outcome,
    diagnose,
    suggest_terms,
)

#: The paragraph a user actually pasted. Contains no structural PII at all:
#: everything sensitive in it is a named entity.
ATATURK = (
    "Mustafa Kemal Atatürk[e] (c. 1881[f] – 10 November 1938) was a Turkish "
    "field marshal and statesperson who was the founder of the Republic of "
    "Turkey and served as its first president from 1923 until his death in "
    "1938. He led sweeping reforms, turning Turkey into an industrialising "
    "nation."
)


class TestDiagnose:
    """Reporting what a configuration can and cannot do."""

    def test_default_configuration(self):
        report = diagnose()
        assert isinstance(report, Diagnosis)
        assert "EMAIL" in report.active_kinds
        assert report.ner_active is False

    def test_never_imports_an_optional_dependency(self):
        import sys

        before = {m for m in ("spacy", "flask", "cryptography") if m in sys.modules}
        diagnose()
        after = {m for m in ("spacy", "flask", "cryptography") if m in sys.modules}
        assert after == before

    def test_title_case_is_not_active_by_default(self):
        assert "TITLE_CASE" not in diagnose().active_kinds

    def test_narrowed_policy_narrows_the_report(self):
        report = diagnose(DEFAULT_POLICY.evolve(kinds=("EMAIL",)))
        assert report.active_kinds == ("EMAIL",)
        assert "URL" in report.inactive_kinds

    def test_registry_decides_when_given(self):
        registry = default_registry(kinds=["EMAIL", "URL"])
        assert diagnose(DEFAULT_POLICY, registry).active_kinds == ("EMAIL", "URL")

    def test_every_tier_is_reported(self):
        assert set(diagnose().tiers) == {"ner", "nltk", "web", "crypto"}

    def test_tier_reports_carry_a_status_and_a_hint(self):
        for tier in diagnose().tiers.values():
            assert tier["status"] in {member.value for member in CapabilityStatus}
            assert tier["install_hint"].startswith("pip install")
            assert tier["purpose"]

    def test_headline_names_the_gap_when_ner_is_off(self):
        assert "NOT being detected" in diagnose().headline()

    def test_as_dict_is_json_safe(self):
        import json

        payload = diagnose().as_dict()
        assert json.loads(json.dumps(payload)) == payload


class TestBlindSpots:
    """The gaps, and how to close them."""

    def test_missing_ner_is_a_high_severity_gap(self):
        report = diagnose()
        high = [spot for spot in report.blind_spots if spot.severity == "high"]
        assert len(high) == 1
        assert "name" in high[0].category.lower()

    def test_every_spot_states_a_remedy(self):
        for spot in diagnose().blind_spots:
            assert spot.remedy
            assert isinstance(spot, BlindSpot)

    def test_the_ner_remedy_is_an_actual_command(self):
        report = diagnose()
        high = [s for s in report.blind_spots if s.severity == "high"][0]
        assert "pip install" in high.remedy or "--ner" in high.remedy

    def test_healthy_tracks_high_severity_spots(self):
        report = diagnose()
        assert report.healthy == (
            not any(spot.severity == "high" for spot in report.blind_spots)
        )

    def test_title_case_gap_is_offered_when_disabled(self):
        categories = [spot.category for spot in diagnose().blind_spots]
        assert any("Capitalised" in category for category in categories)

    def test_enabling_title_case_removes_its_own_gap(self):
        kinds = tuple(sorted(set(diagnose().active_kinds) | {"TITLE_CASE"}))
        report = diagnose(DEFAULT_POLICY.evolve(kinds=kinds))
        assert not any("Capitalised" in spot.category for spot in report.blind_spots)


class TestSuggestTerms:
    """The inspection surface."""

    def test_finds_multi_word_names(self):
        found = [item.text for item in suggest_terms("Ada Lovelace wrote it.")]
        assert "Ada Lovelace" in found

    def test_finds_capitalised_words_after_lowercase_ones(self):
        """The first implementation missed these entirely; see the module note."""
        found = [item.text for item in suggest_terms("she visited Turkey today")]
        assert "Turkey" in found

    def test_ranks_by_frequency(self):
        text = "Turkey and Turkey and Turkey, also Belgium."
        found = suggest_terms(text)
        assert found[0].text == "Turkey"
        assert found[0].count == 3

    def test_min_tokens_narrows_to_multi_word(self):
        found = [item.text for item in suggest_terms(ATATURK, min_tokens=2)]
        assert found == ["Mustafa Kemal Atatürk"]

    def test_sentence_openers_are_dropped(self):
        found = [item.text for item in suggest_terms("The cat sat. It slept.")]
        assert "The" not in found
        assert "It" not in found

    def test_runs_break_on_lowercase(self):
        found = [item.text for item in suggest_terms("Republic of Turkey")]
        assert "Republic" in found and "Turkey" in found
        assert "Republic of Turkey" not in found

    def test_already_redacted_values_are_not_suggested(self):
        text = "Contact Ada Lovelace at ada@example.com"
        result = Redactor().redact(text, extra_terms=["Ada Lovelace"])
        found = [item.text for item in suggest_terms(text, result)]
        assert "Ada Lovelace" not in found

    def test_unicode_names_are_found(self):
        found = [item.text for item in suggest_terms("Über Ätna und Ćevapi")]
        assert "Über Ätna" in found

    def test_max_suggestions_is_honoured(self):
        text = " ".join("Name{0}".format(i) for i in range(100))
        assert len(suggest_terms(text, max_suggestions=5)) <= 5

    def test_empty_text(self):
        assert suggest_terms("") == ()

    def test_rejects_non_string(self):
        with pytest.raises(TypeError):
            suggest_terms(None)

    def test_rejects_zero_min_tokens(self):
        with pytest.raises(ValueError, match="min_tokens"):
            suggest_terms("x", min_tokens=0)

    def test_suggestions_are_json_safe(self):
        import json

        for item in suggest_terms(ATATURK):
            assert isinstance(item, Suggestion)
            json.dumps(item.as_dict())

    def test_offsets_point_at_the_text(self):
        for item in suggest_terms(ATATURK):
            assert ATATURK[item.first_offset :].startswith(item.text)


class TestDescribeOutcome:
    """Turning a result plus a diagnosis into something a human can act on."""

    def test_empty_result_with_a_gap_is_an_alert(self):
        result = Redactor().redact(ATATURK)
        outcome = describe_outcome(result, diagnose())
        assert outcome["level"] == "alert"
        assert "switched off" in outcome["headline"]

    def test_empty_result_without_a_gap_is_ok(self):
        kinds = tuple(sorted(set(diagnose().active_kinds) | {"TITLE_CASE"}))
        policy = DEFAULT_POLICY.evolve(kinds=kinds)
        registry = default_registry(kinds=kinds)
        registry.add(_FakeNer())
        result = Redactor(policy=policy, registry=registry).redact("nothing here")
        outcome = describe_outcome(result, diagnose(policy, registry))
        assert outcome["level"] == "ok"

    def test_non_empty_result_with_a_gap_is_a_warning(self):
        result = Redactor().redact("mail ada@example.com")
        outcome = describe_outcome(result, diagnose())
        assert outcome["level"] == "warning"

    def test_actions_are_actionable(self):
        result = Redactor().redact(ATATURK)
        outcome = describe_outcome(result, diagnose())
        assert outcome["actions"]
        assert all(isinstance(action, str) and action for action in outcome["actions"])

    def test_suggestions_are_carried_through(self):
        result = Redactor().redact(ATATURK)
        suggestions = suggest_terms(ATATURK, result)
        outcome = describe_outcome(result, diagnose(), suggestions)
        assert len(outcome["suggestions"]) == len(suggestions)

    def test_never_names_a_secret(self):
        text = "mail topsecret@example.com"
        result = Redactor().redact(text)
        outcome = describe_outcome(result, diagnose(), suggest_terms(text, result))
        import json

        assert "topsecret@example.com" not in json.dumps(outcome)

    def test_detail_reads_as_a_sentence(self):
        result = Redactor().redact("mail ada@example.com")
        detail = describe_outcome(result, diagnose())["detail"]
        assert detail.endswith(".")
        assert "  " not in detail


class _FakeNer:
    """A detector that claims to be the NER tier, for gap-closure tests."""

    name = "ner:test"
    kind = "NE"
    priority = 30
    confidence = 0.7

    def detect(self, text, policy):  # noqa: D102 - protocol conformance
        return iter(())


class TestTheReportedDefect:
    """The exact scenario that was reported: paste prose, nothing happens."""

    def test_the_engine_is_right_and_that_is_the_problem(self):
        """Zero detections is correct here, which is why silence was dangerous."""
        assert Redactor().redact(ATATURK).stats.entries == 0

    def test_the_outcome_is_never_a_neutral_all_clear(self):
        result = Redactor().redact(ATATURK)
        outcome = describe_outcome(result, diagnose(), suggest_terms(ATATURK, result))
        assert outcome["level"] != "ok"
        assert "Nothing was redacted" in outcome["headline"]
        assert "switched off" in outcome["headline"]

    def test_the_user_is_told_what_to_do(self):
        result = Redactor().redact(ATATURK)
        outcome = describe_outcome(result, diagnose(), suggest_terms(ATATURK, result))
        joined = " ".join(outcome["actions"])
        # Which engine the action names depends on what is installed; that it
        # names the switch that turns name detection on does not.
        assert "--ner" in joined
        assert "suggested term" in joined

    def test_the_name_is_offered_as_a_candidate(self):
        found = [item.text for item in suggest_terms(ATATURK)]
        assert "Mustafa Kemal Atatürk" in found
        assert "Turkey" in found

    def test_enabling_title_case_redacts_the_name_without_spacy(self):
        kinds = tuple(sorted(set(diagnose().active_kinds) | {"TITLE_CASE"}))
        result = Redactor(
            policy=DEFAULT_POLICY.evolve(kinds=kinds),
            registry=default_registry(kinds=kinds),
        ).redact(ATATURK)
        assert result.stats.entries >= 1
        assert "Mustafa Kemal Atatürk" not in result.text

    def test_accepting_suggestions_redacts_the_rest(self):
        result = Redactor().redact(ATATURK, extra_terms=["Turkey", "Turkish"])
        assert "Turkey" not in result.text
        from .. import restore

        assert restore(result.text, result.vault).text == ATATURK


class TestEntityRemedyFollowsWhatIsInstalled:
    """
    CP-088: the remedy for the name-detection gap, in every installation.

    Notes
    -----
    **Developer notes.** ``_entity_remedy`` has three answers and picks one
    from what is installed. A test that reads the answer of the machine it
    runs on covers one of the three and fails on the others: asserting
    ``"spacy"`` held with spaCy installed and with nothing installed, and
    failed where only NLTK was. The installation is supplied here, so every
    answer is checked on every machine.
    """

    @staticmethod
    def _installed(monkeypatch, *present):
        from .. import _diagnostics
        from .._capabilities import probe as real_probe

        def probe(name):
            status = (
                CapabilityStatus.AVAILABLE
                if name in present
                else CapabilityStatus.ABSENT
            )
            return real_probe(name)._replace(status=status)

        monkeypatch.setattr(_diagnostics, "probe", probe)
        return _diagnostics._entity_remedy()

    @pytest.mark.parametrize("present", [("ner",), ("ner", "nltk")])
    def test_spacy_installed_is_a_flag_not_a_download(self, monkeypatch, present):
        remedy = self._installed(monkeypatch, *present)
        assert remedy.startswith("spaCy is installed")
        assert "spacy_detector()" in remedy and "--ner" in remedy
        assert "pip install" not in remedy

    def test_only_nltk_installed_names_nltk(self, monkeypatch):
        remedy = self._installed(monkeypatch, "nltk")
        assert remedy.startswith("NLTK is installed")
        assert "--ner --ner-engine nltk" in remedy and "nltk_detector()" in remedy
        assert "pip install" not in remedy
        assert "spacy" not in remedy.lower()

    def test_nothing_installed_offers_both_engines(self, monkeypatch):
        remedy = self._installed(monkeypatch)
        assert "python -m spacy download" in remedy
        assert remedy.count("pip install") == 2
        assert "--ner --ner-engine nltk" in remedy

    @pytest.mark.parametrize("present", [(), ("ner",), ("nltk",), ("ner", "nltk")])
    def test_every_answer_names_the_switch(self, monkeypatch, present):
        assert "--ner" in self._installed(monkeypatch, *present)
