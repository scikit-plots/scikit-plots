"""
Tests for :mod:`scikitplot.cleanprompt._surrogate_sets`.

Notes
-----
**Developer notes.** Organised by the design's invariants
(``maintenances/cleanprompt/_maintenance/GENERATOR_DESIGN.md``): validation
is total; the safety floor holds for every set and every proposal; the
identity follows the content and only the content; the default grammar keeps
its digest (G6); and the core loop's rules (G2-G5) apply to a set's names
exactly as to the built-in ones.
"""

from __future__ import annotations

import io
import json
import re

import pytest

from .. import FluentCleanPrompt
from .._exceptions import CleanPromptError, PolicyError
from .._frontends import is_click_available
from .._packs import PackError
from .._policy import DEFAULT_POLICY, TagStyle
from .._surrogate_sets import (
    CORE_FORMS,
    MAX_ENTRIES,
    SETTABLE_KINDS,
    SurrogateSet,
    entry_problem,
    load_surrogate_set,
    surrogate_set_from_document,
)
from .._surrogates import SURROGATE_KINDS, surrogate_for

_SET = {
    "name": "nordic",
    "version": 1,
    "summary": "Nordic-sounding invented names.",
    "kinds": {
        "PERSON": {"first": ["Aino", "Eero", "Liv"], "last": ["Halvorsen", "Lindgren", "Virtanen"]},
        "ORG": {"first": ["Fjord", "Norrsken"], "second": ["Data", "Logistik"]},
        "GPE": ["Granvik", "Solberga"],
    },
}

_CSV = "name,company\nAda Lovelace,Acme Corp\nGrace Hopper,Initech\n"


def _set(**changes):
    document = json.loads(json.dumps(_SET))
    document.update(changes)
    return document


def _write(folder, name="nordic.json", document=None):
    path = folder / name
    path.write_text(json.dumps(document or _SET), encoding="utf-8")
    return path


def _problems(document):
    with pytest.raises(PackError) as caught:
        surrogate_set_from_document(document, "s.json")
    return "\n".join(caught.value.problems)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_set(self):
        names = surrogate_set_from_document(_SET, "s.json")
        assert names.kinds() == ("PERSON", "ORG", "GPE")
        assert names.capacity("PERSON") == 9
        assert names.capacity("LOC") == 0
        assert names.source == "s.json"

    def test_every_problem_is_reported_at_once(self):
        problems = _problems(
            {"name": "Bad", "version": 0, "summary": "", "kinds": {}, "extra": 1}
        )
        for needle in ("unknown key 'extra'", "name:", "version:", "summary:", "kinds:"):
            assert needle in problems

    @pytest.mark.parametrize("kind", CORE_FORMS)
    def test_contact_forms_belong_to_the_core(self, kind):
        assert "reserved forms" in _problems(_set(kinds={kind: ["Aino"]}))

    @pytest.mark.parametrize("kind", ["CREDIT_CARD", "IBAN", "NORP", "MRN"])
    def test_other_kinds_keep_placeholders(self, kind):
        assert "keep placeholders" in _problems(_set(kinds={kind: ["Aino"]}))

    def test_two_part_kinds_need_a_mapping(self):
        assert "must be a mapping with first and last" in _problems(
            _set(kinds={"PERSON": ["Aino"]})
        )

    def test_unknown_part(self):
        assert "unknown key 'middle'" in _problems(
            _set(kinds={"PERSON": {"first": ["A"], "last": ["B"], "middle": ["C"]}})
        )

    def test_missing_part(self):
        assert "kinds.ORG.second: must be a non-empty list" in _problems(
            _set(kinds={"ORG": {"first": ["Fjord"]}})
        )

    def test_too_many_entries(self):
        many = [f"Name{chr(0x61 + i % 26)}{chr(0x61 + i // 26)}" for i in range(MAX_ENTRIES + 1)]
        assert f"above {MAX_ENTRIES}" in _problems(_set(kinds={"GPE": many}))

    def test_duplicates_compare_as_written_any_way(self):
        # Case and full-width forms are the same name to a reader.
        problems = _problems(_set(kinds={"GPE": ["Granvik", "granvik", "\uff27ranvik"]}))
        assert problems.count("repeats 'Granvik'") == 1
        assert "wide compatibility form" in problems  # full-width is refused outright

    def test_canonically_equivalent_spellings_are_one_name(self):
        # Precomposed, decomposed, and marks typed in another order (Arabic
        # shadda then fatha) are stored composed, and compare equal.
        problems = _problems(_set(kinds={"GPE": ["Ren\u00e9e", "Rene\u0301e"]}))
        assert "repeats 'Ren\u00e9e'" in problems
        arabic = surrogate_set_from_document(
            _set(kinds={"GPE": ["\u0645\u064f\u062d\u064e\u0645\u0651\u064e\u062f"]})
        )
        (stored,) = arabic.pools[0][1][0]
        import unicodedata

        assert stored == unicodedata.normalize("NFC", stored)

    def test_problems_spell_invisible_characters_out(self):
        problems = _problems(_set(kinds={"GPE": ["Ai\u034fno"]}))
        assert "'Ai\\u034fno'" in problems

    def test_entry_detected_as_a_value_is_refused(self, monkeypatch):
        from .. import _surrogate_sets

        monkeypatch.setattr(
            _surrogate_sets,
            "_detection_rules",
            lambda: (("TICKET", re.compile("Granvik"), None),),
        )
        assert "is detected as TICKET" in _problems(_SET)

    def test_yaml_and_json_give_one_identity(self, tmp_path):
        yaml = pytest.importorskip("yaml")
        (tmp_path / "s.yaml").write_text(yaml.safe_dump(_SET), encoding="utf-8")
        from_json = load_surrogate_set(_write(tmp_path))
        from_yaml = load_surrogate_set(tmp_path / "s.yaml")
        assert from_json.identity == from_yaml.identity
        assert from_json == from_yaml

    def test_missing_file(self, tmp_path):
        with pytest.raises(CleanPromptError, match="does not exist"):
            load_surrogate_set(tmp_path / "absent.json")


class TestDesignExample:
    def test_the_documented_set_loads(self):
        # GENERATOR_DESIGN.md section 5, verbatim. Opt-in core patterns
        # (TITLE_CASE) are not detection rules for entries (round 26 review).
        names = surrogate_set_from_document(
            {
                "name": "nordic",
                "version": 1,
                "summary": "Nordic-sounding invented names.",
                "kinds": {
                    "PERSON": {
                        "first": ["Aino", "Eero", "Liv"],
                        "last": ["Halvorsen", "Lindgren", "Virtanen"],
                    },
                    "ORG": {"first": ["Fjord", "Norrsken"], "second": ["Data", "Logistik"]},
                    "GPE": ["Granvik", "Solberga"],
                    "LOC": ["the Tunturi Fells"],
                    "FAC": ["Granvik Station"],
                },
            }
        )
        assert names.kinds() == ("PERSON", "ORG", "GPE", "LOC", "FAC")

    def test_a_combined_name_that_would_be_detected_is_skipped(self, monkeypatch):
        from .. import _surrogate_sets

        names = surrogate_set_from_document(_SET)
        monkeypatch.setattr(
            _surrogate_sets,
            "_detection_rules",
            lambda: (("TICKET", re.compile("Aino Halvorsen"), None),),
        )
        assert surrogate_for("PERSON", 1, provider=names) == "Eero Virtanen"


class TestEntryFloor:
    @pytest.mark.parametrize(
        "entry",
        ["Aino", "Jean-Luc", "O'Neill", "St. Ives", "the Tunturi Fells", "\u00c5sa",
         "\u0938\u0940\u0924\u093e", "\u5c71\u7530\u3055\u304f\u3089",
         "\u0e19\u0e49\u0e33\u0e1d\u0e19", "\u0130lker", "I\u015f\u0131k", "Stra\u00dfburg",
         "Nguy\u1ec5n", "\uae40\ubbfc\uc900", "\u05e9\u05b8\u05c1\u05dc\u05d5\u05b9\u05dd",
         "\u00de\u00f3r\u00f0ur", "\u039d\u03af\u03ba\u03bf\u03c2",
         "\u0410\u043d\u043d\u0430", "Ren\u00e9e", "D\u2019Arcy"],
    )
    def test_names_pass(self, entry):
        assert entry_problem(entry) is None

    @pytest.mark.parametrize(
        ("entry", "needle"),
        [
            ("Ag3nt", "is not a letter"),
            ("Agent 007", "is not a letter"),
            ("Aino-", "start and end with a letter"),
            ("a@b", "is not a letter"),
            ("https", None),
            ("x/y", "is not a letter"),
            ("[PERSON-1]", "start and end with a letter"),
            (" Aino", "start and end with a letter"),
            ("Aino  Lee", "two spaces"),
            ("Ai\u200bno", "U+200B is an invisible character"),
            ("Ai\u034fno", "U+034F is an invisible character"),  # grapheme joiner (Mn)
            ("Ai\u3164no", "U+3164 is an invisible character"),  # Hangul filler (Lo)
            ("A\ufe0fino", "U+FE0F is an invisible character"),  # variation selector
            ("Ai\U000e0100no", "U+E0100 is an invisible character"),
            ("b\u0336\u0336\u0336x", "combining marks in a row"),
            ("\uff21ino", "wide compatibility form"),  # full-width
            ("Ai\u24dco", "circle compatibility form"),  # circled letter
            ("\u0410ino", "mixes scripts (CYRILLIC, LATIN)"),  # look-alike letter
            ("Ai\nno", "is not a letter"),
            ("", "1 to 64"),
            ("A" * 65, "1 to 64"),
            (7, "must be a string"),
        ],
    )
    def test_non_names_fail(self, entry, needle):
        problem = entry_problem(entry)
        if needle is None:
            assert problem is None  # a word is a word; the core adds no scheme
        else:
            assert needle in problem

    def test_settable_kinds_are_surrogated_kinds(self):
        assert set(SETTABLE_KINDS) | set(CORE_FORMS) == set(SURROGATE_KINDS)


# ---------------------------------------------------------------------------
# identity and the grammar fingerprint (G6)
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_identity_shape(self):
        identity = surrogate_set_from_document(_SET).identity
        assert re.fullmatch(r"nordic@1#[0-9a-f]{16}", identity)

    def test_editing_an_entry_changes_identity(self):
        edited = _set(kinds={**_SET["kinds"], "GPE": ["Granvik", "Solbacka"]})
        assert (
            surrogate_set_from_document(_SET).identity
            != surrogate_set_from_document(edited).identity
        )

    def test_key_order_does_not(self):
        reordered = {"kinds": _SET["kinds"], "summary": _SET["summary"], "version": 1, "name": "nordic"}
        assert surrogate_set_from_document(reordered).identity == surrogate_set_from_document(_SET).identity

    def test_default_digests_are_unchanged(self):
        # Pinned: every vault written before custom sets existed carries these.
        assert TagStyle().fingerprint == "0e7c5e996585e57b"
        assert TagStyle(style="surrogate").fingerprint == "1a85d3ad0a8a8e6f"
        assert "surrogates" not in TagStyle(style="surrogate").as_dict()
        assert DEFAULT_POLICY.fingerprint == "33ad5234bfede14e"

    def test_a_set_changes_the_grammar(self):
        names = surrogate_set_from_document(_SET)
        style = TagStyle(style="surrogate", surrogate_set=names)
        assert style.surrogates == names.identity
        assert style.fingerprint != TagStyle(style="surrogate").fingerprint
        assert TagStyle(**style.as_dict()) == style  # what decode rebuilds

    def test_set_needs_the_surrogate_style(self):
        with pytest.raises(PolicyError, match="needs style='surrogate'"):
            TagStyle(surrogate_set=surrogate_set_from_document(_SET))

    def test_identity_and_set_must_agree(self):
        with pytest.raises(PolicyError, match="different set"):
            TagStyle(
                style="surrogate",
                surrogates="other@1#0000000000000000",
                surrogate_set=surrogate_set_from_document(_SET),
            )

    def test_set_must_be_a_set(self):
        with pytest.raises(PolicyError, match="must be a SurrogateSet"):
            TagStyle(style="surrogate", surrogate_set=object())


# ---------------------------------------------------------------------------
# the core loop applies every rule to a set's names (G1-G5)
# ---------------------------------------------------------------------------


class _Provider:
    identity = "fake@1#0000000000000000"

    def __init__(self, answers, fail=False):
        self.answers = answers
        self.fail = fail
        self.asked = []

    def candidate(self, kind, index):
        self.asked.append(kind)
        if self.fail:
            raise RuntimeError("provider broke")
        return self.answers[index % len(self.answers)]


class TestCoreLoop:
    names = surrogate_set_from_document(_SET)

    def test_names_come_from_the_set(self):
        got = [surrogate_for("PERSON", n, provider=self.names) for n in (1, 2, 3)]
        assert got == ["Aino Halvorsen", "Eero Virtanen", "Liv Lindgren"]

    def test_unlisted_kind_uses_built_in_names(self):
        assert surrogate_for("LOC", 1, provider=self.names) == surrogate_for("LOC", 1)

    @pytest.mark.parametrize("kind", CORE_FORMS)
    def test_contact_forms_never_ask_the_provider(self, kind):
        provider = _Provider(["Aino"])
        assert surrogate_for(kind, 1, provider=provider) == surrogate_for(kind, 1)
        assert provider.asked == []

    def test_credentials_never_ask_the_provider(self):
        provider = _Provider(["Aino"])
        assert surrogate_for("CREDIT_CARD", 1, provider=provider) is None
        assert provider.asked == []

    def test_a_malformed_proposal_is_skipped(self):
        provider = _Provider(["Agent 007", "x@y.example", "Aino Lee"])
        assert surrogate_for("PERSON", 1, provider=provider) == "Aino Lee"

    def test_a_failing_provider_fails_loudly(self):
        with pytest.raises(RuntimeError, match="provider broke"):
            surrogate_for("PERSON", 1, provider=_Provider(["A"], fail=True))

    def test_source_and_issued_and_held_values_are_avoided(self):
        first = surrogate_for("PERSON", 1, provider=self.names, source="Aino Halvorsen")
        assert first == "Eero Virtanen"
        assert surrogate_for("PERSON", 1, provider=self.names, avoid={"Aino Halvorsen"}) == "Eero Virtanen"
        held = surrogate_for("PERSON", 1, provider=self.names, forbidden=lambda c: "Aino" in c)
        assert "Aino" not in held

    def test_a_small_set_falls_back_to_placeholders(self):
        tiny = surrogate_set_from_document(_set(kinds={"GPE": ["Granvik"]}))
        assert surrogate_for("GPE", 1, provider=tiny) == "Granvik"
        assert surrogate_for("GPE", 2, provider=tiny, avoid={"Granvik"}) is None

    def test_deterministic(self):
        run = [surrogate_for("ORG", n, provider=self.names) for n in range(1, 6)]
        assert run == [surrogate_for("ORG", n, provider=self.names) for n in range(1, 6)]


class TestDeterminism:
    def test_same_text_same_stand_ins(self, tmp_path):
        path = str(_write(tmp_path))
        texts = []
        for _ in range(2):
            cleaner = (
                FluentCleanPrompt().packs("all").style("surrogate").surrogates(path).materialize()
            )
            texts.append(cleaner.encode_text(_CSV, "csv", name="p.csv").text)
        assert texts[0] == texts[1]
        assert "Aino Halvorsen" in texts[0]


# ---------------------------------------------------------------------------
# the engine, the plan and the cleaner
# ---------------------------------------------------------------------------


class TestEngine:
    def test_identity_without_the_set_is_refused_when_encoding(self):
        from .._engine import Redactor

        style = TagStyle(style="surrogate", surrogates="nordic@1#0000000000000000")
        redactor = Redactor(DEFAULT_POLICY.evolve(tag_style=style))
        with pytest.raises(PolicyError, match="was not loaded"):
            redactor.redact("mail ada@example.com")

    def test_identity_alone_is_enough_to_restore(self, tmp_path):
        path = str(_write(tmp_path))
        cleaner = FluentCleanPrompt().packs("all").style("surrogate").surrogates(path).materialize()
        encoded = cleaner.encode_text(_CSV, "csv", name="p.csv")
        assert "Ada Lovelace" not in encoded.text
        assert cleaner.decode(encoded.text) == _CSV


class TestPlan:
    def test_plan_round_trip(self, tmp_path):
        from .._plan import plan_from_dict, save_plan

        path = str(_write(tmp_path))
        plan = FluentCleanPrompt().style("surrogate").surrogates(path).build()
        assert plan.as_dict()["surrogates"] == path
        save_plan(plan, tmp_path / "plan.json")
        saved = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
        assert plan_from_dict(saved).surrogates == path

    def test_unset_is_left_out(self):
        assert "surrogates" not in FluentCleanPrompt().build().as_dict()

    def test_needs_surrogate_style(self, tmp_path):
        problems = FluentCleanPrompt().surrogates(str(_write(tmp_path))).validate()
        assert any("needs style 'surrogate'" in p for p in problems)

    def test_invalid_set_is_a_plan_problem(self, tmp_path):
        bad = _write(tmp_path, "bad.json", _set(kinds={"EMAIL": ["Aino"]}))
        problems = FluentCleanPrompt().style("surrogate").surrogates(str(bad)).validate()
        assert any("reserved forms" in p for p in problems)

    def test_editing_the_set_makes_the_plan_stale(self, tmp_path):
        from .._plan import load_plan, save_plan

        path = _write(tmp_path)
        save_plan(
            FluentCleanPrompt().style("surrogate").surrogates(str(path)).build(),
            tmp_path / "plan.json",
        )
        load_plan(tmp_path / "plan.json")
        _write(tmp_path, document=_set(kinds={"GPE": ["Granvik", "Solbacka"]}))
        with pytest.raises(CleanPromptError, match="changed since it was saved"):
            load_plan(tmp_path / "plan.json")


# ---------------------------------------------------------------------------
# command line, both frontends
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
    def test_needs_the_surrogate_style(self, tmp_path, frontend):
        code, _, err = _run(
            ["encode", "--surrogates", str(_write(tmp_path)), "--vault",
             str(tmp_path / "v.json"), "mail ada@example.com"],
            frontend,
        )
        assert code == 1
        assert "--surrogates needs --style surrogate" in err

    def test_batch_uses_the_set_and_decode_needs_nothing(self, tmp_path, frontend):
        source = tmp_path / "src"
        source.mkdir()
        (source / "p.csv").write_text(_CSV, encoding="utf-8")
        vault = tmp_path / "v.json"
        code, _, err = _run(
            ["batch", str(source), "--out", str(tmp_path / "out"), "--pack", "all",
             "--style", "surrogate", "--surrogates", str(_write(tmp_path)),
             "--vault", str(vault)],
            frontend,
        )
        assert code == 0, err
        encoded = (tmp_path / "out" / "p.csv").read_text(encoding="utf-8")
        assert "Aino Halvorsen" in encoded and "Ada Lovelace" not in encoded
        recorded = json.loads(vault.read_text(encoding="utf-8"))["tag_style"]
        assert recorded["surrogates"].startswith("nordic@1#")

    def test_appending_with_another_grammar_is_refused(self, tmp_path, frontend):
        names = str(_write(tmp_path))
        vault = str(tmp_path / "v.json")
        first = ["encode", "--style", "surrogate", "--surrogates", names, "--vault", vault,
                 "--vault-mode", "overwrite", "mail ada@example.com"]
        assert _run(first, frontend)[0] == 0
        before = json.loads((tmp_path / "v.json").read_text(encoding="utf-8"))
        for other in (
            ["encode", "--vault", vault, "mail bob@example.org"],
            ["encode", "--style", "surrogate", "--vault", vault, "mail bob@example.org"],
        ):
            code, _, err = _run(other, frontend)
            assert code == 1
            assert "Appending would mix two kinds of stand-in" in err
            assert "surrogate set nordic@1#" in err
        after = json.loads((tmp_path / "v.json").read_text(encoding="utf-8"))
        assert after["grammar_fingerprint"] == before["grammar_fingerprint"]
        same = ["encode", "--style", "surrogate", "--surrogates", names, "--vault", vault,
                "mail bob@example.org"]
        assert _run(same, frontend)[0] == 0

    def test_abbreviation_is_refused(self, tmp_path, frontend):
        code, _, _ = _run(
            ["encode", "--style", "surrogate", "--surrog", str(_write(tmp_path)), "x"],
            frontend,
        )
        assert code == 2


def test_dataclass_is_frozen():
    names = surrogate_set_from_document(_SET)
    assert isinstance(names, SurrogateSet)
    with pytest.raises(AttributeError):
        names.name = "other"
