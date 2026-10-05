"""Tests for :mod:`scikitplot.cleanprompt._policy`."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from .. import (
    DEFAULT_LIMITS,
    DEFAULT_POLICY,
    DEFAULT_TAG_STYLE,
    Limits,
    OverlapStrategy,
    PolicyError,
    RedactionPolicy,
    TagStyle,
)


class TestTagStyle:
    """The placeholder grammar."""

    def test_default_rendering(self):
        assert DEFAULT_TAG_STYLE.render("email", 3) == "[EMAIL-3]"

    def test_custom_rendering(self):
        style = TagStyle(prefix="<<", suffix=">>", separator="_")
        assert style.render("email", 3) == "<<EMAIL_3>>"

    def test_case_preserving_rendering(self):
        assert TagStyle(uppercase_kind=False).render("Email", 1) == "[Email-1]"

    @pytest.mark.parametrize("field", ["prefix", "suffix", "separator"])
    def test_empty_part_is_refused(self, field):
        with pytest.raises(PolicyError, match="non-empty"):
            TagStyle(**{field: ""})

    def test_separator_inside_a_delimiter_is_refused(self):
        with pytest.raises(PolicyError, match="ambiguous"):
            TagStyle(prefix="[-", separator="-")

    def test_empty_kind_is_refused(self):
        with pytest.raises(PolicyError, match="non-empty"):
            DEFAULT_TAG_STYLE.render("", 1)

    def test_non_positive_ordinal_is_refused(self):
        with pytest.raises(PolicyError, match=">= 1"):
            DEFAULT_TAG_STYLE.render("EMAIL", 0)

    @pytest.mark.parametrize("kind", ["A]B", "A[B", "A|B", "A.B", "A*B"])
    def test_reserved_characters_in_kind_are_refused(self, kind):
        with pytest.raises(PolicyError, match="reserved"):
            DEFAULT_TAG_STYLE.render(kind, 1)

    def test_separator_in_kind_is_refused(self):
        with pytest.raises(PolicyError, match="separator"):
            DEFAULT_TAG_STYLE.render("A-B", 1)

    def test_pattern_round_trips_its_own_rendering(self):
        for style in (
            TagStyle(),
            TagStyle(prefix="<<", suffix=">>", separator="_"),
            TagStyle(prefix="{{", suffix="}}", separator=":"),
        ):
            for ordinal in (1, 9, 10, 11, 100, 12345):
                label = style.render("EMAIL", ordinal)
                match = style.pattern().fullmatch(label)
                assert match is not None
                assert match.group("kind") == "EMAIL"
                assert int(match.group("ordinal")) == ordinal

    def test_pattern_does_not_match_a_foreign_grammar(self):
        assert TagStyle().pattern().search("<<EMAIL_1>>") is None

    def test_suffix_prevents_ordinal_prefix_confusion(self):
        pattern = TagStyle().pattern()
        found = [match.group() for match in pattern.finditer("[EMAIL-1] [EMAIL-11]")]
        assert found == ["[EMAIL-1]", "[EMAIL-11]"]

    def test_fingerprint_depends_only_on_the_grammar(self):
        assert TagStyle().fingerprint == TagStyle().fingerprint
        assert TagStyle().fingerprint != TagStyle(prefix="<<").fingerprint

    def test_fingerprint_is_sixteen_hex_characters(self):
        value = TagStyle().fingerprint
        assert len(value) == 16
        assert all(char in "0123456789abcdef" for char in value)

    def test_is_frozen(self):
        with pytest.raises(Exception):
            TagStyle().prefix = "x"


class TestLimits:
    """Hard bounds."""

    def test_defaults_are_positive(self):
        for value in DEFAULT_LIMITS.as_dict().values():
            assert value > 0

    @pytest.mark.parametrize(
        "field",
        [
            "max_input_chars",
            "max_spans",
            "max_entries",
            "max_literal_terms",
            "max_literal_length",
        ],
    )
    @pytest.mark.parametrize("bad", [0, -1, 1.5, "10", True, None])
    def test_invalid_bound_is_refused(self, field, bad):
        with pytest.raises(PolicyError, match="positive int"):
            Limits(**{field: bad})

    def test_ner_ceiling_matches_the_base_ceiling(self):
        """One document ceiling, so the two tiers fail at the same size."""
        assert DEFAULT_LIMITS.max_input_chars == 1_000_000


class TestRedactionPolicy:
    """Validation, evolution and selection."""

    def test_defaults(self):
        assert DEFAULT_POLICY.kinds is None
        assert DEFAULT_POLICY.overlap is OverlapStrategy.LONGEST_WINS
        assert DEFAULT_POLICY.preserve_placeholders is True

    def test_evolve_returns_a_new_policy(self):
        evolved = DEFAULT_POLICY.evolve(case_insensitive=True)
        assert evolved is not DEFAULT_POLICY
        assert evolved.case_insensitive is True
        assert DEFAULT_POLICY.case_insensitive is False

    def test_evolve_normalises_kinds_to_a_tuple(self):
        assert DEFAULT_POLICY.evolve(kinds=["EMAIL", "URL"]).kinds == ("EMAIL", "URL")

    def test_evolve_rejects_a_bare_string(self):
        with pytest.raises(PolicyError, match="not a single string"):
            DEFAULT_POLICY.evolve(kinds="EMAIL")

    def test_evolve_rejects_an_unknown_field(self):
        with pytest.raises(PolicyError, match="unknown RedactionPolicy field"):
            DEFAULT_POLICY.evolve(nope=1)

    def test_evolve_lists_the_known_fields(self):
        with pytest.raises(PolicyError) as caught:
            DEFAULT_POLICY.evolve(nope=1)
        assert "tag_style" in str(caught.value)

    def test_empty_kinds_is_refused(self):
        with pytest.raises(PolicyError, match="must not be empty"):
            RedactionPolicy(kinds=())

    def test_duplicate_kind_is_refused(self):
        with pytest.raises(PolicyError, match="duplicate"):
            RedactionPolicy(kinds=("EMAIL", "EMAIL"))

    def test_non_string_kind_is_refused(self):
        with pytest.raises(PolicyError, match="non-empty strings"):
            RedactionPolicy(kinds=(1,))

    def test_kinds_must_be_a_tuple(self):
        with pytest.raises(PolicyError, match="must be a tuple"):
            RedactionPolicy(kinds=["EMAIL"])

    def test_bad_overlap_is_refused(self):
        with pytest.raises(PolicyError, match="OverlapStrategy"):
            RedactionPolicy(overlap="LONGEST")

    @pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
    def test_confidence_range(self, bad):
        with pytest.raises(PolicyError, match=r"\[0.0, 1.0\]"):
            RedactionPolicy(min_confidence=bad)

    def test_selected_kinds_defaults_to_everything_available(self):
        assert DEFAULT_POLICY.selected_kinds(["A", "B"]) == frozenset({"A", "B"})

    def test_selected_kinds_narrows(self):
        policy = DEFAULT_POLICY.evolve(kinds=("A",))
        assert policy.selected_kinds(["A", "B"]) == frozenset({"A"})

    def test_unknown_kind_raises_rather_than_being_ignored(self):
        """A silently dropped category is a redaction that did not happen."""
        policy = DEFAULT_POLICY.evolve(kinds=("Z",))
        with pytest.raises(PolicyError) as caught:
            policy.selected_kinds(["A", "B"])
        assert "available kinds are A, B" in str(caught.value)


class TestFingerprint:
    """Stability guarantees of the policy digest."""

    def test_identical_policies_agree(self):
        assert RedactionPolicy().fingerprint == RedactionPolicy().fingerprint

    def test_kind_order_does_not_matter(self):
        one = DEFAULT_POLICY.evolve(kinds=("EMAIL", "URL"))
        other = DEFAULT_POLICY.evolve(kinds=("URL", "EMAIL"))
        assert one.fingerprint == other.fingerprint

    @pytest.mark.parametrize(
        "change",
        [
            {"case_insensitive": True},
            {"min_confidence": 0.5},
            {"preserve_placeholders": False},
            {"overlap": OverlapStrategy.STRICT},
            {"tag_style": TagStyle(prefix="<<")},
            {"limits": Limits(max_entries=7)},
            {"kinds": ("EMAIL",)},
        ],
    )
    def test_every_field_changes_the_digest(self, change):
        assert DEFAULT_POLICY.evolve(**change).fingerprint != DEFAULT_POLICY.fingerprint

    def test_stable_across_hash_seeds(self):
        """A hash-derived digest would differ per process; this must not."""
        script = (
            "import sys;sys.path.insert(0,{0!r});"
            "from scikitplot.cleanprompt import DEFAULT_POLICY;"
            "print(DEFAULT_POLICY.fingerprint)"
        ).format(str(_root()))
        seen = set()
        for seed in ("0", "1", "424242"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            out = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            seen.add(out.stdout.strip())
        assert len(seen) == 1
        assert seen.pop() == DEFAULT_POLICY.fingerprint

    def test_as_dict_is_json_serializable(self):
        assert json.loads(json.dumps(DEFAULT_POLICY.as_dict())) == DEFAULT_POLICY.as_dict()


def _root():
    """Return the directory containing the ``scikitplot`` package."""
    import pathlib

    return pathlib.Path(__file__).resolve().parents[3]
