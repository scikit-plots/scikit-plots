"""Tests for :mod:`scikitplot.cleanprompt._types`."""

from __future__ import annotations

import json

import pytest

from .. import Entry, RestorationResult, Span, Stats, as_dict


class TestSpan:
    """Validation and geometry of a detection interval."""

    def test_basic(self):
        span = Span(0, 5, "EMAIL", "a@b.c", "regex:EMAIL")
        assert span.length == 5
        assert span.priority == 0
        assert span.confidence == 1.0

    def test_negative_start_is_refused(self):
        with pytest.raises(ValueError, match="start must be >= 0"):
            Span(-1, 5, "A", "x", "d")

    @pytest.mark.parametrize("end", [0, -1, 5])
    def test_empty_or_inverted_is_refused(self, end):
        with pytest.raises(ValueError, match="non-empty"):
            Span(5, end, "A", "x", "d")

    def test_empty_kind_is_refused(self):
        with pytest.raises(ValueError, match="non-empty string"):
            Span(0, 1, "", "x", "d")

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_confidence_range(self, bad):
        with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
            Span(0, 1, "A", "x", "d", confidence=bad)

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ((0, 5), (5, 10), False),  # touching, half-open
            ((0, 5), (4, 10), True),
            ((0, 10), (2, 4), True),
            ((5, 10), (0, 5), False),
            ((0, 1), (0, 1), True),
        ],
    )
    def test_overlaps(self, a, b, expected):
        left = Span(a[0], a[1], "A", "x" * (a[1] - a[0]), "d")
        right = Span(b[0], b[1], "B", "y" * (b[1] - b[0]), "e")
        assert left.overlaps(right) is expected
        assert right.overlaps(left) is expected

    def test_sorts_by_start_then_end(self):
        spans = [
            Span(5, 9, "A", "xxxx", "d"),
            Span(0, 9, "B", "x" * 9, "d"),
            Span(0, 3, "C", "xxx", "d"),
        ]
        assert [(s.start, s.end) for s in sorted(spans)] == [(0, 3), (0, 9), (5, 9)]

    def test_is_frozen(self):
        with pytest.raises(Exception):
            Span(0, 1, "A", "x", "d").start = 3

    def test_equality_ignores_provenance(self):
        """Two detectors finding the same interval and kind compare equal."""
        assert Span(0, 5, "A", "abcde", "d1") == Span(0, 5, "A", "abcde", "d2")


class TestEntry:
    """The assigned-label record."""

    def test_count_is_the_occurrence_count(self):
        entry = Entry("[A-1]", "A", 1, "secret", ((0, 3), (7, 10)), "d")
        assert entry.count == 2

    def test_repr_hides_the_original(self):
        entry = Entry("[A-1]", "A", 1, "topsecret", ((0, 9),), "d")
        assert "topsecret" not in repr(entry)
        assert "[A-1]" in repr(entry)

    def test_is_frozen(self):
        entry = Entry("[A-1]", "A", 1, "x", ((0, 1),), "d")
        with pytest.raises(Exception):
            entry.label = "[A-2]"


class TestRedactionResult:
    """The safe half of a pass."""

    def test_labels_and_summary(self, redactor):
        result = redactor.redact("mail a@x.com and b@x.com")
        assert result.labels == ("[EMAIL-1]", "[EMAIL-2]")
        assert "EMAIL=2" in result.summary()

    def test_summary_when_nothing_found(self, redactor):
        assert redactor.redact("plain").summary() == "no sensitive values detected"

    def test_truncated_is_always_false(self, redactor):
        assert redactor.redact("a@x.com").truncated is False


class TestRestorationResult:
    """The completeness report."""

    def test_complete_when_nothing_unknown(self):
        assert RestorationResult("x", ("[A-1]",)).complete is True

    def test_incomplete_when_unknown(self):
        assert RestorationResult("x", (), ("[A-9]",)).complete is False


class TestAsDict:
    """The sanctioned serializer."""

    def test_redaction_result_omits_secrets(self, redactor):
        result = redactor.redact("mail topsecret@x.com")
        payload = as_dict(result)
        assert "topsecret@x.com" not in json.dumps(payload)
        assert payload["entries"][0]["label"] == "[EMAIL-1]"

    def test_dataclasses_asdict_would_have_leaked(self, redactor):
        """Documents why a bespoke serializer exists."""
        import dataclasses

        result = redactor.redact("mail topsecret@x.com")
        leaked = dataclasses.asdict(result.entries[0])
        assert leaked["original"] == "topsecret@x.com"

    def test_restoration_result(self):
        payload = as_dict(RestorationResult("text", ("[A-1]",), ("[A-9]",), ("[A-2]",)))
        assert payload["complete"] is False
        assert payload["unknown"] == ["[A-9]"]

    def test_stats(self):
        stats = Stats(10, 8, 3, 2, 1, 2, {"A": 2})
        assert as_dict(stats)["by_kind"] == {"A": 2}

    def test_json_round_trip(self, redactor, sample_text):
        payload = as_dict(redactor.redact(sample_text))
        assert json.loads(json.dumps(payload)) == payload

    def test_unsupported_type_is_refused(self):
        with pytest.raises(TypeError, match="unsupported result type"):
            as_dict(object())

    def test_a_vault_is_not_serializable_through_this_path(self, redactor):
        """The unsafe half has no accidental route into a serializer."""
        result = redactor.redact("mail topsecret@x.com")
        assert "vault" not in as_dict(result)
