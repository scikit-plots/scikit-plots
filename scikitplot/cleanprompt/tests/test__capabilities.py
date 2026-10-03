"""Tests for :mod:`scikitplot.cleanprompt._capabilities`."""

from __future__ import annotations

import sys

import pytest

from .. import TIERS, CapabilityError, CapabilityReport, CapabilityStatus, capabilities, probe
from .._capabilities import _parse_release


class TestVocabulary:
    """The seven-state enum, consumed by value from the project's canon."""

    def test_has_exactly_seven_states(self):
        assert len(list(CapabilityStatus)) == 7

    def test_member_names(self):
        assert {member.value for member in CapabilityStatus} == {
            "AVAILABLE",
            "ABSENT",
            "BROKEN",
            "INCOMPATIBLE",
            "MISCONFIGURED",
            "UNREACHABLE",
            "UNKNOWN",
        }

    def test_members_are_strings(self):
        """So they serialize to JSON and compare to plain strings."""
        assert CapabilityStatus.AVAILABLE == "AVAILABLE"
        assert isinstance(CapabilityStatus.ABSENT, str)

    def test_broken_is_not_absent(self):
        """Installed-and-failing must stay distinguishable from not installed."""
        assert CapabilityStatus.BROKEN is not CapabilityStatus.ABSENT

    def test_the_vocabulary_is_copied_by_value_not_imported(self):
        """
        Independence: no ``scikitplot`` sibling is imported to obtain it.

        Notes
        -----
        **Developer notes.** Checked by parsing the module rather than by
        searching its text: the docstring legitimately names
        ``import scikitplot.cleanprompt``, and a substring search cannot tell
        prose from an import statement.
        """
        import ast
        import pathlib

        path = pathlib.Path(__file__).resolve().parent.parent / "_capabilities.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imported.append(node.module or "")
        assert [name for name in imported if name.split(".")[0] == "scikitplot"] == []


class TestParseRelease:
    """Version prefix parsing."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("3.7.2", (3, 7, 2)),
            ("2.2", (2, 2)),
            ("4", (4,)),
            ("4.0.0rc1", (4, 0, 0)),
            ("1.26.4.post1", (1, 26, 4)),
            ("10.0.0.dev0", (10, 0, 0)),
        ],
    )
    def test_parses(self, text, expected):
        assert _parse_release(text) == expected

    @pytest.mark.parametrize("text", ["", "abc", "v1.2", "unknown"])
    def test_unparseable_returns_none(self, text):
        assert _parse_release(text) is None


class TestTierTable:
    """The static tier description."""

    def test_expected_tiers(self):
        assert set(TIERS) == {"ner", "nltk", "web", "crypto"}

    @pytest.mark.parametrize("tier", sorted(TIERS))
    def test_range_is_ordered(self, tier):
        spec = TIERS[tier]
        assert spec.minimum < spec.below

    @pytest.mark.parametrize("tier", sorted(TIERS))
    def test_no_exact_pin(self, tier):
        """Ranges, never ``==``: a library must span what consumers install."""
        report = probe(tier)
        assert ">=" in report.supported and "<" in report.supported
        assert "==" not in report.supported

    @pytest.mark.parametrize("tier", sorted(TIERS))
    def test_purpose_is_stated(self, tier):
        assert len(TIERS[tier].purpose) > 5


class TestProbe:
    """Per-tier probing."""

    @pytest.mark.parametrize("tier", sorted(TIERS))
    def test_returns_a_report(self, tier):
        report = probe(tier)
        assert isinstance(report, CapabilityReport)
        assert report.tier == tier
        assert report.status in set(CapabilityStatus)

    @pytest.mark.parametrize("tier", sorted(TIERS))
    def test_never_raises_for_an_unusable_tier(self, tier):
        probe(tier)  # must not raise whatever the environment holds

    @pytest.mark.parametrize("tier", sorted(TIERS))
    def test_does_not_import_the_distribution(self, tier):
        """Probing must not pull an optional dependency into the process."""
        distribution = TIERS[tier].distribution
        before = distribution in sys.modules
        probe(tier)
        assert (distribution in sys.modules) == before

    @pytest.mark.parametrize("tier", sorted(TIERS))
    def test_install_hint_is_actionable(self, tier):
        report = probe(tier)
        assert report.install_hint.startswith("pip install")
        assert TIERS[tier].distribution in report.install_hint

    def test_unknown_tier_raises_key_error(self):
        with pytest.raises(KeyError):
            probe("nope")

    def test_available_property_tracks_the_status(self):
        for report in capabilities().values():
            assert report.available == (report.status is CapabilityStatus.AVAILABLE)

    def test_broken_metadata_is_classified_as_broken(self, monkeypatch):
        """A metadata lookup that raises means installed-and-failing."""
        from .. import _capabilities

        def explode(_name):
            raise RuntimeError("corrupt distribution metadata")

        monkeypatch.setattr(_capabilities, "_installed_version", explode)
        report = _capabilities.probe("ner")
        assert report.status is CapabilityStatus.BROKEN
        assert "corrupt distribution metadata" in report.detail

    def test_absent_distribution_is_classified_as_absent(self, monkeypatch):
        from .. import _capabilities

        monkeypatch.setattr(_capabilities, "_installed_version", lambda _n: None)
        assert _capabilities.probe("ner").status is CapabilityStatus.ABSENT

    def test_out_of_range_version_is_incompatible(self, monkeypatch):
        from .. import _capabilities

        monkeypatch.setattr(_capabilities, "_installed_version", lambda _n: "99.0.0")
        report = _capabilities.probe("ner")
        assert report.status is CapabilityStatus.INCOMPATIBLE
        assert "99.0.0" in report.detail

    def test_below_floor_is_incompatible(self, monkeypatch):
        from .. import _capabilities

        monkeypatch.setattr(_capabilities, "_installed_version", lambda _n: "3.0.0")
        assert _capabilities.probe("ner").status is CapabilityStatus.INCOMPATIBLE

    def test_in_range_version_is_available(self, monkeypatch):
        from .. import _capabilities

        monkeypatch.setattr(_capabilities, "_installed_version", lambda _n: "3.7.2")
        assert _capabilities.probe("ner").status is CapabilityStatus.AVAILABLE

    def test_unparseable_version_is_unknown(self, monkeypatch):
        from .. import _capabilities

        monkeypatch.setattr(_capabilities, "_installed_version", lambda _n: "weird")
        assert _capabilities.probe("ner").status is CapabilityStatus.UNKNOWN


class TestCapabilities:
    """The whole-installation report."""

    def test_covers_every_tier(self):
        assert set(capabilities()) == set(TIERS)

    def test_is_repeatable(self):
        assert capabilities() == capabilities()


class TestRequire:
    """The raising wrapper."""

    def test_returns_the_report_when_available(self, monkeypatch):
        from .. import _capabilities

        monkeypatch.setattr(_capabilities, "_installed_version", lambda _n: "3.7.2")
        assert _capabilities.require("ner").available is True

    def test_raises_with_an_actionable_message(self, monkeypatch):
        from .. import _capabilities

        monkeypatch.setattr(_capabilities, "_installed_version", lambda _n: None)
        with pytest.raises(CapabilityError) as caught:
            _capabilities.require("ner")
        error = caught.value
        assert error.tier == "ner"
        assert error.status == "ABSENT"
        assert error.install_hint.startswith("pip install")
        assert "pip install" in str(error)

    def test_preserves_the_broken_status_on_the_error(self, monkeypatch):
        from .. import _capabilities

        def explode(_name):
            raise RuntimeError("bad")

        monkeypatch.setattr(_capabilities, "_installed_version", explode)
        with pytest.raises(CapabilityError) as caught:
            _capabilities.require("ner")
        assert caught.value.status == "BROKEN"

    def test_is_an_import_error_for_legacy_callers(self):
        assert issubclass(CapabilityError, ImportError)
