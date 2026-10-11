r"""
Tests for :mod:`scikitplot.cleanprompt._pattern_risk`.

Notes
-----
**Developer notes.** Three layers are tested separately, because each can be
wrong on its own:

* the *analysis* — which shapes are reported, and that every built-in pattern
  is clean (so a warning is always about something a user wrote);
* the *policy* — ``warn`` / ``ignore`` / ``refuse``, the per-pattern
  acceptance in a pack, and the order in which a mode is chosen (argument or
  plan, then ``CLEANPROMPT_PATTERN_RISK``, then ``warn``);
* the *surfaces* — :func:`load_custom`, a plan, a cleaner, and both
  command-line frontends, where a warning must reach standard error as a
  plain block and leave standard output (a report, JSON) untouched.
"""

from __future__ import annotations

import io
import json
import re
import sys
import warnings

import pytest

from .._catalog import builtin_catalog
from .._custom import load_custom, with_custom
from .._exceptions import CleanPromptError
from .._frontends import is_click_available
from .._packs import PackError, pack_from_document
from .._pattern_risk import (
    DEFAULT_PATTERN_RISK,
    PATTERN_RISK_ENV,
    PATTERN_RISK_MODES,
    PackFinding,
    PatternRisk,
    PatternRiskWarning,
    analyse_pattern,
    enforce,
    pack_findings,
    resolve_pattern_risk,
)
from .._patterns import PATTERNS
from .._plan import CleanPlan, FluentCleanPrompt, plan_from_dict, save_plan
from ._regex_fixtures import regex_fixture

#: A pattern with a nested repetition: ``[A-Z]+`` inside ``(...)+`` with
#: nothing mandatory between repetitions that ``[A-Z]`` cannot match.
_RISKY = regex_fixture(r"\b(?:[A-Z]+\d*)+-\d+\b")


def _pack(pattern=_RISKY, **extra):
    item = {
        "kind": "TICKET",
        "pattern": pattern,
        "intent": "A ticket id.",
        "examples_yes": ["ABC-12"],
        "examples_no": ["abc"],
        **extra,
    }
    return {
        "name": "tickets",
        "version": 1,
        "summary": "Ticket ids.",
        "patterns": [item],
    }


def _write(folder, name, document):
    path = folder / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    """Every test starts without the environment variable."""
    monkeypatch.delenv(PATTERN_RISK_ENV, raising=False)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


class TestAnalysis:
    @pytest.mark.parametrize(
        ("source", "rule"),
        [
            (regex_fixture(r"^(a+)+$"), "nested-quantifier"),
            (regex_fixture(r"(a*)*"), "nested-quantifier"),
            (regex_fixture(r"(\w+\s?)*"), "nested-quantifier"),
            (regex_fixture(r"(?P<value>x+)+"), "nested-quantifier"),
            (regex_fixture(r"(\w+,?)+"), "nested-quantifier"),
            (regex_fixture(r"(?:\.?\w+)+"), "nested-quantifier"),
            (regex_fixture(r"(?:(?:ab)+)+"), "nested-quantifier"),
            (regex_fixture(r"(a|ab)+"), "overlapping-alternation"),
            (regex_fixture(r"(?:\w|\d)+"), "overlapping-alternation"),
            (regex_fixture(r"\d+\d+"), "adjacent-quantifiers"),
            (regex_fixture(r"[a-z]*\w+"), "adjacent-quantifiers"),
        ],
    )
    def test_risky_shapes_are_reported(self, source, rule):
        assert rule in [risk.rule for risk in analyse_pattern(source)]

    @pytest.mark.parametrize(
        "source",
        [
            r"(\w+,)+",
            r"(\d+\.)+\d+",
            r"\bEMP-\d{6}\b",
            r"(?:[A-Z]+-)+\d+",
            r"[a-z]+@[a-z]+",
            r"(?:\.\w+)+",
            r"(?:-[A-Z]+)+",
            r"\d{3}[ -]?\d{3}[ -]?\d{4}",
            r"(?:a|b)+",
            r"(?i)\bMRN\s*:?\s*(?P<value>\d{4,12})\b",
            r"\w+\s+\w+",
            r"\w*\s*:?\s*\w+",  # optional separator: at most quadratic, not reported
            r"\d+(?:,\d+)*",
        ],
    )
    def test_linear_shapes_are_not_reported(self, source):
        assert analyse_pattern(source) == ()

    def test_flags_change_which_characters_overlap(self):
        # [a-z] and X are disjoint, until case is folded.
        assert analyse_pattern(r"(?:[a-z]+X)+") == ()
        folded = analyse_pattern(r"(?:[a-z]+X)+", re.IGNORECASE)
        assert [risk.rule for risk in folded] == ["nested-quantifier"]

    def test_unmodelled_syntax_is_reported_not_passed(self):
        (risk,) = analyse_pattern(r"(a)?(?(1)b|c)+")
        assert risk.rule == "not-analysed"
        assert risk.severity == "info"

    def test_nesting_deeper_than_the_stack_is_not_analysed(self):
        depth = 990
        source = "(?:" * depth + "a" + ")" * depth
        assert [risk.rule for risk in analyse_pattern(source)] == ["not-analysed"]

    def test_analysis_is_deterministic(self):
        assert analyse_pattern(_RISKY) == analyse_pattern(_RISKY)

    def test_every_finding_carries_its_way_out(self):
        (risk,) = analyse_pattern(regex_fixture(r"(a+)+"))
        text = risk.describe("hr.yaml: pack hr, pattern X")
        assert text.startswith("hr.yaml: pack hr, pattern X: nested-quantifier (high)")
        for needle in ("risk: accepted", "--pattern-risk ignore", "--pattern-risk refuse"):
            assert needle in text

    def test_fragment_is_the_repeated_group(self):
        (risk,) = analyse_pattern(_RISKY)
        assert risk.fragment == regex_fixture(r"(?:[A-Z]+\d*)+")


class TestReviewFindings:
    """
    The round-26 independent review's cases.

    Notes
    -----
    **Developer notes.** Each "missed" case is measurably exponential
    (``evidence/probe_round26.py`` times it in a killable child); each
    "flagged" case is measurably linear. The first version of the check got
    every one of them wrong.
    """

    @pytest.mark.parametrize(
        ("source", "flags"),
        [
            (regex_fixture("(?: \\w+ \\s? )+ ;"), re.VERBOSE),  # verbose whitespace is not a separator
            (regex_fixture("(?x) (?: \\w+ \\s? )+ ;  # comment"), 0),
            (regex_fixture("^(?:\\w+,\\w+)+$"), 0),  # equal units: positions, not equality
            (regex_fixture("(?i)^(?:a+A)+$"), 0),  # inline flags widen overlap
            (regex_fixture("(?i:(?:a+A)+)"), 0),
            (regex_fixture("(?s)^(?:.+\\n)+$"), 0),
            (regex_fixture("^(?:[\\x41-\\x5a]+\\x4b)+$"), 0),  # escaped characters are probed
            (regex_fixture("^(?:[\\u0400-\\u04ff]+\\u0430)+$"), 0),
            (regex_fixture("^(?:\\d+[\\uff10-\\uff19])+$"), 0),
            (regex_fixture("^(?:\\w+\\s?){1,40}$"), 0),  # a large bound is still repetition
            (regex_fixture("^(?:\\w{1,50}\\s?)+$"), 0),
            # found by the soundness fuzz (probe_round26_fuzz.py)
            (regex_fixture("^(?:\\w?\\.?)* {2}$"), 0),  # optional parts trade a character
            (regex_fixture("^(?:(?:(?:\\d?){1,9}) {2,})+$"), 0),  # a nullable group is no separator
            (regex_fixture("^(?:(?:\\w{1,5}\\s*)(?:-?){1,9})+$"), 0),
            (regex_fixture("^(?:\\w{1,3}\\s?)+$"), 0),  # any variation counts inside
        ],
    )
    def test_missed_shapes_are_reported(self, source, flags):
        assert "nested-quantifier" in [risk.rule for risk in analyse_pattern(source, flags)]

    @pytest.mark.parametrize(
        "source",
        [
            "(?:(?:\\w+)-)+",  # the separator is one group up
            "(?:([a-z0-9]+)\\.)+[a-z]{2,}",
            "\\b[A-Z]{2}[0-9]{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\\b",  # fixed counts
            "(?x) \\b EMP - \\d{6} \\b  # verbose, linear",
            "[]]+x",
            "[^]]+\\]",
            "(?:\\w+\\s?){1,3}",  # at most a cubic factor
        ],
    )
    def test_linear_shapes_are_not_reported(self, source):
        assert analyse_pattern(source) == ()

    @pytest.mark.skipif(
        sys.version_info < (3, 11), reason="possessive and atomic forms need Python 3.11"
    )
    @pytest.mark.parametrize("source", ["^(?:\\w++\\s?)+$", "^(?:(?>\\w+)\\s?)+$"])
    def test_the_suggested_possessive_and_atomic_rewrites_pass(self, source):
        assert analyse_pattern(source) == ()

    @pytest.mark.parametrize("source", ["(?x:a b)+", "a{3,2}", "(?P<x>a"])
    def test_what_cannot_be_read_or_compiled_is_not_analysed(self, source):
        assert [risk.rule for risk in analyse_pattern(source)] == ["not-analysed"]

    def test_a_valid_but_warning_pattern_is_analysed_quietly(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert analyse_pattern("[[]") == ()

    def test_no_suggestion_recommends_a_bounded_repetition(self):
        (risk,) = analyse_pattern(regex_fixture("(a+)+"))
        assert not any("{1,32}" in item for item in risk.suggestions)


class TestBuiltinsAreClean:
    """A warning is always about something the user wrote."""

    def test_core_patterns(self):
        flagged = {
            spec.kind: analyse_pattern(spec.pattern, spec.flags)
            for spec in PATTERNS.values()
            if analyse_pattern(spec.pattern, spec.flags)
        }
        assert flagged == {}

    def test_builtin_packs(self):
        found = pack_findings(builtin_catalog().packs.values(), include_builtin=True)
        assert found == []

    def test_builtin_packs_are_skipped_by_default(self):
        assert pack_findings(builtin_catalog().packs.values()) == []


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------


class TestResolveMode:
    def test_default_is_warn(self):
        assert resolve_pattern_risk() == DEFAULT_PATTERN_RISK == "warn"

    def test_environment(self, monkeypatch):
        monkeypatch.setenv(PATTERN_RISK_ENV, "refuse")
        assert resolve_pattern_risk() == "refuse"

    def test_explicit_beats_environment(self, monkeypatch):
        monkeypatch.setenv(PATTERN_RISK_ENV, "refuse")
        assert resolve_pattern_risk("ignore") == "ignore"

    def test_empty_environment_is_unset(self, monkeypatch):
        monkeypatch.setenv(PATTERN_RISK_ENV, "")
        assert resolve_pattern_risk() == "warn"

    @pytest.mark.parametrize("bad", ["loud", "WARN", "off"])
    def test_unknown_mode_is_refused(self, monkeypatch, bad):
        with pytest.raises(ValueError, match="pattern_risk"):
            resolve_pattern_risk(bad)
        monkeypatch.setenv(PATTERN_RISK_ENV, bad)
        with pytest.raises(ValueError, match=PATTERN_RISK_ENV):
            resolve_pattern_risk()

    def test_modes(self):
        assert PATTERN_RISK_MODES == ("ignore", "warn", "refuse")


class TestPackAcceptance:
    def test_accepted_with_reason(self):
        pack = pack_from_document(
            _pack(risk="accepted", risk_reason=" inputs are single ids ")
        )
        assert pack.patterns[0].risk_reason == "inputs are single ids"

    def test_unaccepted_has_no_reason(self):
        assert pack_from_document(_pack()).patterns[0].risk_reason is None

    @pytest.mark.parametrize(
        ("extra", "needle"),
        [
            ({"risk": "accepted"}, "needs a non-empty reason"),
            ({"risk": "accepted", "risk_reason": "  "}, "needs a non-empty reason"),
            ({"risk_reason": "why"}, "add `risk: accepted` too"),
            ({"risk": "yes", "risk_reason": "why"}, "must be 'accepted'"),
            ({"risk": True, "risk_reason": "why"}, "must be 'accepted'"),
        ],
    )
    def test_half_an_acceptance_is_refused(self, extra, needle):
        with pytest.raises(PackError, match=re.escape(needle)):
            pack_from_document(_pack(**extra))

    def test_accepted_finding_is_listed_but_not_acted_on(self):
        pack = pack_from_document(
            _pack(risk="accepted", risk_reason="single ids"), source="t.json"
        )
        (found,) = pack_findings([pack])
        assert found.accepted == "single ids"
        assert found.as_dict()["accepted"] == "single ids"
        assert enforce([found], "refuse") == []


class TestEnforce:
    def _finding(self):
        pack = pack_from_document(_pack(), source="t.json")
        return pack_findings([pack])

    def test_warn(self):
        with pytest.warns(PatternRiskWarning, match="t.json: pack tickets, pattern TICKET"):
            open_findings = enforce(self._finding(), "warn")
        assert len(open_findings) == 1

    def test_ignore_is_silent_and_still_reports(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert len(enforce(self._finding(), "ignore")) == 1

    def test_refuse(self):
        with pytest.raises(PackError, match="the mode is 'refuse'"):
            enforce(self._finding(), "refuse", "t.json")

    def test_unknown_mode(self):
        with pytest.raises(ValueError, match="loud"):
            enforce([], "loud")

    def test_finding_is_frozen_data(self):
        (found,) = self._finding()
        assert isinstance(found, PackFinding)
        assert isinstance(found.risk, PatternRisk)
        with pytest.raises(AttributeError):
            found.kind = "OTHER"


# ---------------------------------------------------------------------------
# surfaces: load_custom, plan, cleaner
# ---------------------------------------------------------------------------


class TestLoadCustom:
    def test_warns_by_default(self, tmp_path):
        path = _write(tmp_path, "t.json", _pack())
        with pytest.warns(PatternRiskWarning, match="nested-quantifier"):
            catalog = load_custom(path)
        assert "tickets" in catalog.packs

    def test_ignore(self, tmp_path):
        path = _write(tmp_path, "t.json", _pack())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            load_custom(path, pattern_risk="ignore")

    def test_refuse_names_the_file(self, tmp_path):
        path = _write(tmp_path, "t.json", _pack())
        with pytest.raises(PackError) as caught:
            load_custom(path, pattern_risk="refuse")
        assert caught.value.source == "t.json"

    def test_environment_refuses(self, tmp_path, monkeypatch):
        monkeypatch.setenv(PATTERN_RISK_ENV, "refuse")
        with pytest.raises(PackError):
            with_custom(_write(tmp_path, "t.json", _pack()))

    def test_accepted_pattern_is_silent_even_under_refuse(self, tmp_path):
        path = _write(
            tmp_path, "t.json", _pack(risk="accepted", risk_reason="single ids")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            load_custom(path, pattern_risk="refuse")

    def test_clean_pattern_is_silent(self, tmp_path):
        path = _write(tmp_path, "t.json", _pack(pattern=r"\b[A-Z]+-\d+\b"))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            load_custom(path, pattern_risk="refuse")


class TestPlan:
    def test_unset_mode_keeps_old_fingerprints(self):
        # A plan saved before the field existed has no 'pattern_risk' key;
        # leaving it out while None is what keeps that file's digest valid.
        assert "pattern_risk" not in CleanPlan().as_dict()

    def test_set_mode_is_saved_and_fingerprinted(self, tmp_path):
        plain = FluentCleanPrompt().build()
        pinned = FluentCleanPrompt().pattern_risk("refuse").build()
        assert pinned.as_dict()["pattern_risk"] == "refuse"
        assert plain.fingerprint() != pinned.fingerprint()
        save_plan(pinned, tmp_path / "plan.json")
        saved = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
        assert plan_from_dict(saved).pattern_risk == "refuse"

    @pytest.mark.parametrize("bad", ["loud", 3])
    def test_plan_file_value_is_checked(self, bad):
        with pytest.raises(CleanPromptError, match="pattern_risk"):
            plan_from_dict({"pattern_risk": bad})

    def test_builder_refuses_unknown_mode(self):
        with pytest.raises(CleanPromptError, match="pattern_risk must be one of"):
            FluentCleanPrompt().pattern_risk("loud")

    def test_refuse_is_a_validation_problem(self, tmp_path):
        path = str(_write(tmp_path, "t.json", _pack()))
        problems = (
            FluentCleanPrompt().custom(path).pattern_risk("refuse").validate()
        )
        assert any("nested-quantifier" in problem for problem in problems)

    def test_environment_mode_is_checked_by_validate(self, monkeypatch):
        monkeypatch.setenv(PATTERN_RISK_ENV, "loud")
        problems = CleanPlan().validate()
        assert any(PATTERN_RISK_ENV in problem for problem in problems)

    def test_findings_listed_with_acceptance(self, tmp_path):
        path = str(
            _write(tmp_path, "t.json", _pack(risk="accepted", risk_reason="ids"))
        )
        plan = FluentCleanPrompt().custom(path).build()
        (found,) = plan.pattern_findings()
        assert found.accepted == "ids"
        assert CleanPlan().pattern_findings() == []

    def test_cleaner_warns_and_still_runs(self, tmp_path):
        path = str(_write(tmp_path, "t.json", _pack()))
        builder = FluentCleanPrompt().custom(path).packs("tickets")
        with pytest.warns(PatternRiskWarning) as caught:
            cleaner = builder.materialize()
        assert len([w for w in caught if w.category is PatternRiskWarning]) == 1
        assert "[TICKET-1]" in cleaner.encode_text("see ABC-12").text

    def test_validation_and_fingerprint_do_not_warn(self, tmp_path):
        path = str(_write(tmp_path, "t.json", _pack()))
        plan = FluentCleanPrompt().custom(path).plan()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert plan.validate() == []
            plan.fingerprint()


# ---------------------------------------------------------------------------
# surfaces: command line, both frontends
# ---------------------------------------------------------------------------

_FRONTENDS = [
    "argparse",
    pytest.param(
        "click",
        marks=pytest.mark.skipif(not is_click_available(), reason="click is not installed"),
    ),
]


def _run(argv, frontend):
    from .._cli import main  # ruff: ignore[import-outside-top-level]

    out, err = io.StringIO(), io.StringIO()
    code = main(argv, stdin=io.StringIO(""), stdout=out, stderr=err, frontend=frontend)
    return code, out.getvalue(), err.getvalue()


@pytest.mark.parametrize("frontend", _FRONTENDS)
class TestCommandLine:
    def test_warning_goes_to_stderr_as_a_block(self, tmp_path, frontend):
        path = str(_write(tmp_path, "t.json", _pack()))
        with warnings.catch_warnings():
            warnings.simplefilter("always", PatternRiskWarning)
            code, out, err = _run(
                ["packs", "--pack-file", path, "--pack", "tickets", "--format", "json"],
                frontend,
            )
        assert code == 0
        assert json.loads(out)["status"] == "ok"
        assert err.startswith("warning: t.json: pack tickets, pattern TICKET: ")
        assert "  - accept this pattern in its pack" in err

    def test_ignore_flag(self, tmp_path, frontend):
        path = str(_write(tmp_path, "t.json", _pack()))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            code, _, err = _run(
                ["packs", "--pack-file", path, "--pattern-risk", "ignore"], frontend
            )
        assert (code, err) == (0, "")

    def test_refuse_flag(self, tmp_path, frontend):
        path = str(_write(tmp_path, "t.json", _pack()))
        code, _, err = _run(
            ["packs", "--pack-file", path, "--pattern-risk", "refuse"], frontend
        )
        assert code == 1
        assert "nested-quantifier" in err
        assert "the mode is 'refuse'" in err

    def test_check_lists_findings_with_acceptance(self, tmp_path, frontend):
        risky = str(_write(tmp_path, "t.json", _pack()))
        code, out, _ = _run(
            ["packs", "--pack-file", risky, "--check", "--format", "json"], frontend
        )
        report = json.loads(out)
        assert code == 0
        assert [item["rule"] for item in report["pattern_risk"]] == ["nested-quantifier"]
        assert report["pattern_risk"][0]["accepted"] is None

    def test_unknown_mode_is_a_usage_error(self, tmp_path, frontend):
        path = str(_write(tmp_path, "t.json", _pack()))
        code, _, _ = _run(
            ["packs", "--pack-file", path, "--pattern-risk", "loud"], frontend
        )
        assert code == 2

    def test_abbreviation_is_refused(self, tmp_path, frontend):
        path = str(_write(tmp_path, "t.json", _pack()))
        code, _, _ = _run(
            ["packs", "--pack-file", path, "--pattern-r", "ignore"], frontend
        )
        assert code == 2

    def test_fatal_warnings_are_a_handled_error(self, tmp_path, frontend):
        path = str(_write(tmp_path, "t.json", _pack()))
        with warnings.catch_warnings():
            warnings.simplefilter("error", PatternRiskWarning)
            code, _, err = _run(["packs", "--pack-file", path], frontend)
        assert code == 1
        assert err.startswith("error: t.json: pack tickets, pattern TICKET")
        assert "Traceback" not in err

    def test_plan_fixes_the_mode(self, tmp_path, frontend):
        plan = tmp_path / "plan.json"
        save_plan(FluentCleanPrompt().build(), plan)
        (tmp_path / "src").mkdir()
        code, _, err = _run(
            [
                "batch",
                str(tmp_path / "src"),
                "--out",
                str(tmp_path / "out"),
                "--plan",
                str(plan),
                "--pattern-risk",
                "ignore",
            ],
            frontend,
        )
        assert code == 1
        assert "--pattern-risk" in err
