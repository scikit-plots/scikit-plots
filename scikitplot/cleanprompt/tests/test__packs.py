"""
Tests for :mod:`scikitplot.cleanprompt._packs`.

Notes
-----
**Developer notes.** The validator's contract is *total*: every problem in a
definition is reported in one error. Most tests therefore assert on the
message listing, not merely that something raised.
"""

from __future__ import annotations

import copy

import pytest

from .. import DEFAULT_POLICY
from .._packs import (
    VALUE_GROUP,
    PackError,
    PackPatternDetector,
    normalise_field,
    pack_detectors,
    pack_from_document,
)

_PACK = {
    "name": "hr",
    "version": 1,
    "summary": "HR identifiers.",
    "requires": ["personal"],
    "fields": [
        {"names": ["badge_id", "Employee Number"], "kind": "EMPLOYEE", "role": "id"}
    ],
    "patterns": [
        {
            "kind": "EMPLOYEE",
            "pattern": r"\bEMP-\d{6}\b",
            "intent": "An employee number.",
            "examples_yes": ["EMP-004121"],
            "examples_no": ["EMP-12"],
        }
    ],
}


def _pack(**changes):
    document = copy.deepcopy(_PACK)
    document.update(changes)
    return document


class TestNormaliseField:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("customerEmail", "customer_email"),
            ("Customer-Email ", "customer_email"),
            ("EMAIL;TYPE=work", "email_type_work"),
            ("__mrn__", "mrn"),
        ],
    )
    def test_folds_case_and_separators(self, raw, expected):
        assert normalise_field(raw) == expected


class TestValidDocument:
    def test_builds_a_frozen_spec(self):
        spec = pack_from_document(_pack(), source="hr.json")
        assert spec.name == "hr"
        assert spec.requires == ("personal",)
        assert spec.source == "hr.json"
        with pytest.raises(AttributeError):
            spec.name = "other"

    def test_field_index_is_keyed_by_normalised_name(self):
        index = pack_from_document(_pack()).field_index()
        assert set(index) == {"badge_id", "employee_number"}
        assert index["badge_id"].kind == "EMPLOYEE"


class TestInvalidDocument:
    def test_every_problem_is_reported_at_once(self):
        document = _pack(name="X", version=0, fields=[])
        with pytest.raises(PackError) as caught:
            pack_from_document(document, source="bad.yaml")
        message = str(caught.value)
        assert "bad.yaml" in message
        assert "name" in message and "version" in message and "fields" in message
        assert len(caught.value.problems) >= 3

    def test_unknown_validator_is_refused(self):
        pattern = dict(_PACK["patterns"][0], validate="os_system")
        with pytest.raises(PackError, match="os_system"):
            pack_from_document(_pack(patterns=[pattern]))

    def test_example_that_does_not_match_is_refused(self):
        """Invariant I11: a pattern cannot disagree with its own examples."""
        pattern = dict(_PACK["patterns"][0], examples_yes=["EMP-12"])
        with pytest.raises(PackError, match="not matched in full"):
            pack_from_document(_pack(patterns=[pattern]))

    def test_negative_example_that_matches_is_refused(self):
        pattern = dict(_PACK["patterns"][0], examples_no=["EMP-004121"])
        with pytest.raises(PackError, match="examples_no"):
            pack_from_document(_pack(patterns=[pattern]))

    def test_pattern_without_positive_examples_is_refused(self):
        pattern = {k: v for k, v in _PACK["patterns"][0].items() if k != "examples_yes"}
        with pytest.raises(PackError, match="examples_yes"):
            pack_from_document(_pack(patterns=[pattern]))

    def test_regex_that_does_not_compile_is_refused(self):
        pattern = dict(_PACK["patterns"][0], pattern="(unclosed")
        with pytest.raises(PackError):
            pack_from_document(_pack(patterns=[pattern]))

    def test_not_a_mapping(self):
        with pytest.raises(PackError):
            pack_from_document(["not", "a", "mapping"])

    @pytest.mark.parametrize("key", ["fields", "patterns", "code"])
    @pytest.mark.parametrize("empty", [None, [], {}])
    def test_a_declared_but_empty_section_is_refused(self, key, empty):
        """A key with nothing under it is an unfinished pack, not an absent section."""
        with pytest.raises(PackError, match=f"{key}: is declared but empty"):
            pack_from_document(_pack(**{key: empty}))

    @pytest.mark.parametrize("key", ["fields", "patterns"])
    def test_an_absent_section_is_fine(self, key):
        document = {k: v for k, v in _PACK.items() if k != key}
        assert pack_from_document(document).name == "hr"


class TestFragmentedExamples:
    """
    An example may be written in pieces so the file holds no whole value.

    This is what lets the ``secrets`` pack carry positive examples for key
    patterns without the file itself being a secret-scanner finding (``I14``).
    """

    def test_fragments_are_joined_and_then_tested(self):
        pattern = dict(_PACK["patterns"][0], examples_yes=[["EMP-", "004", "121"]])
        spec = pack_from_document(_pack(patterns=[pattern]))
        assert spec.patterns[0].examples_yes == ("EMP-004121",)

    def test_strings_and_fragments_mix(self):
        pattern = dict(
            _PACK["patterns"][0],
            examples_yes=["EMP-000001", ["EMP-", "004121"]],
            examples_no=[["EMP", "-12"], "EMP-1"],
        )
        spec = pack_from_document(_pack(patterns=[pattern])).patterns[0]
        assert spec.examples_yes == ("EMP-000001", "EMP-004121")
        assert spec.examples_no == ("EMP-12", "EMP-1")

    def test_a_joined_example_is_still_held_to_the_pattern(self):
        """Invariant I11 applies to the joined value, not to the fragments."""
        pattern = dict(_PACK["patterns"][0], examples_yes=[["EMP-", "12"]])
        with pytest.raises(PackError, match="not matched in full"):
            pack_from_document(_pack(patterns=[pattern]))

    def test_a_joined_negative_example_that_matches_is_refused(self):
        pattern = dict(_PACK["patterns"][0], examples_no=[["EMP-", "004121"]])
        with pytest.raises(PackError, match="examples_no"):
            pack_from_document(_pack(patterns=[pattern]))

    @pytest.mark.parametrize(
        "bad",
        [
            ["EMP-004121"],  # one fragment: a string written as a list
            [],  # nothing
            ["EMP-", ""],  # an empty fragment splits nothing
            ["EMP-", 4121],  # not text
            ["EMP-", ["004121"]],  # nesting has no meaning
        ],
    )
    def test_a_malformed_fragment_list_is_refused(self, bad):
        pattern = dict(_PACK["patterns"][0], examples_yes=[bad])
        with pytest.raises(PackError, match="fragmented example"):
            pack_from_document(_pack(patterns=[pattern]))

    @pytest.mark.parametrize("bad", [7, {"a": "b"}, None])
    def test_an_example_of_another_type_is_refused(self, bad):
        pattern = dict(_PACK["patterns"][0], examples_yes=[bad])
        with pytest.raises(PackError, match="string, or a list of string fragments"):
            pack_from_document(_pack(patterns=[pattern]))

    def test_fragments_do_not_change_the_pack(self):
        """Two spellings of one example are one pack: equality, hence merge, agree."""
        whole = pack_from_document(_pack())
        pattern = dict(_PACK["patterns"][0], examples_yes=[["EMP-00", "4121"]])
        assert pack_from_document(_pack(patterns=[pattern])) == whole


class TestValueGroup:
    _LABELLED = {
        "kind": "EMPLOYEE",
        "pattern": r"(?i)\bbadge\s*:?\s*(?P<value>\d{6})\b",
        "intent": "A badge number with its label.",
        "examples_yes": ["badge: 004121"],
        "examples_no": ["badge"],
    }

    def test_only_the_value_is_replaced(self):
        spec = pack_from_document(_pack(patterns=[self._LABELLED]))
        detector = PackPatternDetector(spec.patterns[0], "hr")
        (span,) = list(detector.detect("Badge: 004121 issued", DEFAULT_POLICY))
        assert span.text == "004121"
        assert "Badge: 004121 issued"[span.start : span.end] == "004121"

    def test_group_name_is_the_documented_one(self):
        assert VALUE_GROUP == "value"


class TestDetectors:
    def test_named_by_pack_and_kind(self):
        spec = pack_from_document(_pack())
        (detector,) = pack_detectors([spec])
        assert detector.name == "pack:hr:EMPLOYEE"

    def test_identical_patterns_across_packs_run_once(self):
        first = pack_from_document(_pack())
        second = pack_from_document(_pack(name="payroll"))
        assert len(pack_detectors([first, second])) == 1
