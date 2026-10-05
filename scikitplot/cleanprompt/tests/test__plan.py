"""
Tests for :mod:`scikitplot.cleanprompt._plan`.

Notes
-----
**Developer notes.** ``TestFingerprint`` is invariant ``I12``: the digest
depends on what a plan *means* — its selection and the definitions that
selection resolves to — and not on the order it was written in.
"""

from __future__ import annotations

import json

import pytest

from .._exceptions import CleanPromptError
from .._plan import KEEPABLE, CleanPlan, FluentCleanPrompt
from .._runtime import Cleaner


class TestImmutability:
    def test_every_setter_returns_a_new_builder(self):
        base = FluentCleanPrompt()
        derived = base.packs("patient")
        assert base.plan().packs == ("auto",)
        assert derived.plan().packs == ("patient",)

    def test_plans_are_hashable(self):
        assert hash(FluentCleanPrompt().packs("patient").plan()) == hash(
            FluentCleanPrompt().packs("patient").plan()
        )


class TestConflicts:
    def test_setting_twice_is_an_error(self):
        with pytest.raises(CleanPromptError, match="already set"):
            FluentCleanPrompt().style("surrogate").style("placeholder")

    def test_replace_and_extend(self):
        builder = FluentCleanPrompt().packs("patient")
        assert builder.packs("finance", conflict="replace").plan().packs == ("finance",)
        assert builder.packs("finance", conflict="extend").plan().packs == (
            "finance",
            "patient",
        )

    def test_a_scalar_cannot_be_extended(self):
        with pytest.raises(CleanPromptError, match="cannot be extended"):
            FluentCleanPrompt().style("surrogate").style(
                "placeholder", conflict="extend"
            )

    def test_unknown_conflict_mode(self):
        with pytest.raises(CleanPromptError, match="conflict"):
            FluentCleanPrompt().packs("patient", conflict="merge")


class TestValidation:
    def test_every_problem_is_listed_at_once(self):
        builder = (
            FluentCleanPrompt()
            .packs("patinet")
            .formats("docz")
            .style("fancy")
            .keep("secrets")
            .roles(churned="label")
        )
        problems = builder.validate()
        joined = "\n".join(problems)
        for fragment in ("packs", "formats", "style", "keep", "roles"):
            assert fragment in joined
        with pytest.raises(CleanPromptError) as caught:
            builder.build()
        assert str(caught.value).count("\n  - ") == len(problems)

    def test_a_missing_custom_file_is_a_problem_not_a_crash(self, tmp_path):
        problems = FluentCleanPrompt().custom(str(tmp_path / "nope.json")).validate()
        assert problems and problems[0].startswith("custom:")

    def test_empty_names_are_refused_immediately(self):
        with pytest.raises(CleanPromptError):
            FluentCleanPrompt().hide("")

    def test_keepable_parts(self):
        assert FluentCleanPrompt().keep(*KEEPABLE).validate() == []


class TestFingerprint:
    def test_order_of_arguments_does_not_matter(self):
        first = (
            FluentCleanPrompt()
            .packs("patient", "finance")
            .formats("csv", ".json")
            .plan()
        )
        second = (
            FluentCleanPrompt()
            .formats(".json", "csv")
            .packs("finance", "patient")
            .plan()
        )
        assert first.fingerprint() == second.fingerprint()

    def test_editing_a_used_definition_changes_it(self, tmp_path):
        pack = {
            "name": "hr",
            "version": 1,
            "summary": "HR.",
            "fields": [{"names": ["badge_id"], "kind": "EMPLOYEE"}],
        }
        path = tmp_path / "hr.json"
        path.write_text(json.dumps(pack), encoding="utf-8")
        plan = FluentCleanPrompt().custom(str(path)).packs("hr").plan()
        before = plan.fingerprint()
        pack["fields"][0]["names"].append("staff_no")
        path.write_text(json.dumps(pack), encoding="utf-8")
        assert plan.fingerprint() != before

    def test_invalid_plan_has_no_fingerprint(self):
        with pytest.raises(CleanPromptError):
            CleanPlan(packs=("nope",)).fingerprint()


def test_as_dict_is_json_and_omits_how_it_was_built():
    plan = FluentCleanPrompt().packs("patient").roles(churned="target").plan()
    document = plan.as_dict()
    json.dumps(document)
    assert "configured" not in document
    assert document["roles"] == [["churned", "target"]]


def test_materialize_returns_a_cleaner():
    assert isinstance(FluentCleanPrompt().packs("patient").materialize(), Cleaner)


def test_repr_names_what_was_configured():
    assert (
        repr(FluentCleanPrompt().packs("patient").style("surrogate"))
        == "FluentCleanPrompt(packs, style)"
    )


class TestPlanFiles:
    """A shared plan file pins what a team's runs mean."""

    def test_save_then_load_is_the_same_plan(self, tmp_path):
        from .._plan import load_plan, save_plan

        plan = (
            FluentCleanPrompt()
            .packs("patient", "finance")
            .roles(churned="target")
            .remember(False)
            .plan()
        )
        fingerprint = save_plan(plan, tmp_path / "team.json")
        loaded = load_plan(tmp_path / "team.json")
        assert loaded == plan and loaded.fingerprint() == fingerprint

    def test_a_changed_definition_is_refused(self, tmp_path):
        from .._plan import load_plan, save_plan

        pack = {
            "name": "hr",
            "version": 1,
            "summary": "HR.",
            "fields": [{"names": ["badge_id"], "kind": "EMPLOYEE"}],
        }
        custom = tmp_path / "hr.json"
        custom.write_text(json.dumps(pack), encoding="utf-8")
        save_plan(
            FluentCleanPrompt().custom(str(custom)).packs("hr").plan(),
            tmp_path / "team.json",
        )
        pack["fields"][0]["names"].append("staff_no")
        custom.write_text(json.dumps(pack), encoding="utf-8")
        with pytest.raises(CleanPromptError, match="changed since it was saved"):
            load_plan(tmp_path / "team.json")
        assert load_plan(tmp_path / "team.json", check=False).packs == ("hr",)

    def test_every_problem_in_a_file_is_listed(self, tmp_path):
        from .._plan import load_plan

        (tmp_path / "bad.json").write_text(
            json.dumps({"packs": "all", "core": "yes", "surprise": 1}), encoding="utf-8"
        )
        with pytest.raises(CleanPromptError) as caught:
            load_plan(tmp_path / "bad.json")
        message = str(caught.value)
        assert "surprise" in message and "packs" in message and "core" in message

    def test_a_plan_without_a_fingerprint_still_loads(self, tmp_path):
        from .._plan import load_plan

        (tmp_path / "p.json").write_text(
            json.dumps({"packs": ["patient"]}), encoding="utf-8"
        )
        assert load_plan(tmp_path / "p.json").packs == ("patient",)
