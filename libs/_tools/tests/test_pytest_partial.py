# libs/_tools/tests/test_pytest_partial.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of ``libs/_tools/pytest_partial.py``: the one skip rule, and its limits."""

from __future__ import annotations

import pytest

from .. import pytest_partial


def _missing(name):
    return ModuleNotFoundError(f"No module named {name!r}", name=name)


@pytest.fixture
def installed(monkeypatch):
    """Decide which parts count as installed; nothing is unless listed."""
    parts = set()
    monkeypatch.setattr(
        pytest_partial, "_is_installed", lambda module: module.split(".")[1] in parts
    )
    return parts


class TestMissingPart:
    def test_an_absent_part_is_named(self, installed):
        assert pytest_partial.missing_part(_missing("scikitplot.corpus")) == "scikitplot.corpus"

    def test_a_deeper_module_of_an_absent_part_is_named(self, installed):
        name = "scikitplot.config.__config__"
        assert pytest_partial.missing_part(_missing(name)) == name

    def test_an_installed_part_is_a_real_failure(self, installed):
        # A module missing from a part that *is* installed is a broken wheel.
        installed.add("corpus")
        assert pytest_partial.missing_part(_missing("scikitplot.corpus._gone")) is None

    @pytest.mark.parametrize("name", ["numpy", "click", "scikitplotx", "scikitplot_extra.x"])
    def test_a_third_party_module_is_a_real_failure(self, installed, name):
        assert pytest_partial.missing_part(_missing(name)) is None

    def test_other_exceptions_are_real_failures(self, installed):
        assert pytest_partial.missing_part(AssertionError("x")) is None
        assert pytest_partial.missing_part(ImportError("x", name="scikitplot.corpus")) is None
        assert pytest_partial.missing_part(ModuleNotFoundError("no name recorded")) is None
        assert pytest_partial.missing_part(None) is None

    def test_the_cause_is_followed(self, installed):
        try:
            try:
                raise _missing("scikitplot.utils")
            except ModuleNotFoundError as exc:
                raise RuntimeError("capability missing") from exc
        except RuntimeError as outer:
            assert pytest_partial.missing_part(outer) == "scikitplot.utils"

    def test_the_implicit_context_is_followed(self, installed):
        try:
            try:
                raise _missing("scikitplot.config")
            except ModuleNotFoundError:
                raise SystemExit("cannot continue")
        except SystemExit as outer:
            assert pytest_partial.missing_part(outer) == "scikitplot.config"

    def test_a_cyclic_chain_terminates(self, installed):
        first, second = RuntimeError("a"), RuntimeError("b")
        first.__cause__, second.__cause__ = second, first
        assert pytest_partial.missing_part(first) is None

    def test_the_first_absent_part_in_the_chain_wins(self, installed):
        inner = _missing("scikitplot.annoy")
        outer = _missing("scikitplot.corpus")
        outer.__cause__ = inner
        assert pytest_partial.missing_part(outer) == "scikitplot.corpus"


class TestIsInstalled:
    def test_the_root_package_is_never_the_absent_part(self):
        assert pytest_partial._is_installed("scikitplot") is True

    def test_only_the_part_is_located_not_the_module(self, monkeypatch):
        asked = []

        def find_spec(name):
            asked.append(name)
            return None

        monkeypatch.setattr(pytest_partial.util, "find_spec", find_spec)
        assert pytest_partial._is_installed("scikitplot.corpus._a._b") is False
        assert asked == ["scikitplot.corpus"]

    def test_a_failing_lookup_means_not_installed(self, monkeypatch):
        def find_spec(name):
            raise ModuleNotFoundError("No module named 'scikitplot'", name="scikitplot")

        monkeypatch.setattr(pytest_partial.util, "find_spec", find_spec)
        assert pytest_partial._is_installed("scikitplot.corpus") is False


class TestPluginContract:
    def test_it_is_a_plain_top_level_module(self):
        """It is loaded with ``-p pytest_partial``, outside any package."""
        import ast
        from pathlib import Path

        tree = ast.parse(Path(pytest_partial.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.level == 0, "a relative import cannot work for a top-level plugin"

    def test_it_defines_the_hooks_it_documents(self):
        for hook in (
            "pytest_configure",
            "pytest_runtest_makereport",
            "pytest_make_collect_report",
            "pytest_terminal_summary",
        ):
            assert callable(getattr(pytest_partial, hook)), hook
