# scikitplot/tests/test__distributions.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Test suite for ``scikitplot/_distributions.py``.

Coverage targets
----------------
- ``canonicalize_name``   : every installer spelling maps to one name; bad input
                            is refused.
- ``DISTRIBUTIONS``       : the ownership statement is well formed, names what
                            exists, and gives no path two owners.
- ``get``                 : lookup by any spelling; unknown names are refused.
- ``provider_of``         : ownership is decided on path components.
- ``install_hint``        : the smallest distribution that ships a module.
- ``installed``/``flavor``: read from installed metadata only.
- ``report``              : the three incoherent states are reported, each with
                            a fix, and a coherent state reports none.
- Module contract         : standard library only, no import-time side effects,
                            loadable by file path (the build tooling does that).

Design decisions
----------------
- Installed metadata is replaced with a fake, so the tests do not depend on
  what happens to be installed where they run.
- Ownership is checked against the real source tree next to this file.

How to run
----------
From the project root::

    pytest scikitplot/tests/test__distributions.py -v --tb=short
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from importlib import metadata
from pathlib import Path

import pytest

from .. import _distributions as dists

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE_DIR / "_distributions.py"


class _FakeEntryPoint:
    def __init__(self, group, name="part", value="scikitplot.part"):
        self.group, self.name, self.value = group, name, value


class _FakeDistribution:
    def __init__(self, groups):
        self.entry_points = [_FakeEntryPoint(group) for group in groups]


def _fake_metadata(monkeypatch, versions, apis=None):
    """
    Make ``importlib.metadata`` answer from the arguments only.

    Parameters
    ----------
    versions : dict of str to str
        Installed project name to version; every other project is absent.
    apis : dict of str to (int, None or list of str), optional
        Project name to the core API it states. An ``int`` is the number, a
        list is the literal entry-point groups (for malformed metadata), and
        ``None`` or a missing name is a distribution that states nothing.
        When ``apis`` itself is omitted, every installed part states
        ``dists.CORE_API``, which is what the build tooling writes.
    """

    def version(name):
        try:
            return versions[name]
        except KeyError:
            raise metadata.PackageNotFoundError(name) from None

    def distribution(name):
        if name not in versions:
            raise metadata.PackageNotFoundError(name)
        if apis is None:
            stated = None if name in {dists.CORE, dists.FULL} else dists.CORE_API
        else:
            stated = apis.get(name)
        if stated is None:
            return _FakeDistribution([])
        if isinstance(stated, int):
            return _FakeDistribution([dists.parts_group(stated)])
        return _FakeDistribution(stated)

    monkeypatch.setattr(metadata, "version", version)
    monkeypatch.setattr(metadata, "distribution", distribution)


# ===========================================================================
# canonicalize_name
# ===========================================================================


class TestCanonicalizeName:
    """One project name, however it is spelled."""

    @pytest.mark.parametrize(
        "spelling",
        [
            "scikit-plots-rank-bm25",
            "scikit_plots_rank_bm25",
            "scikit-plots-rank_bm25",
            "scikit-plots_rank_bm25",
            "scikit.plots.rank.bm25",
            "Scikit-Plots-Rank-BM25",
            "scikit--plots__rank..bm25",
            "  scikit-plots-rank-bm25  ",
        ],
    )
    def test_every_spelling_is_the_same_project(self, spelling):
        assert dists.canonicalize_name(spelling) == "scikit-plots-rank-bm25"

    def test_is_idempotent(self):
        once = dists.canonicalize_name("Scikit_Plots.Corpus")
        assert dists.canonicalize_name(once) == once

    @pytest.mark.parametrize("bad", [None, 3, b"scikit-plots", ["scikit-plots"]])
    def test_non_string_is_a_type_error(self, bad):
        with pytest.raises(TypeError, match="must be a str"):
            dists.canonicalize_name(bad)

    @pytest.mark.parametrize("bad", ["", "   ", "---", "_._"])
    def test_empty_after_normalisation_is_a_value_error(self, bad):
        with pytest.raises(ValueError, match="empty after normalisation"):
            dists.canonicalize_name(bad)


# ===========================================================================
# DISTRIBUTIONS
# ===========================================================================


class TestDistributions:
    """The ownership statement is well formed and matches the source tree."""

    def test_core_is_declared_first(self):
        assert dists.DISTRIBUTIONS[0].name == dists.CORE

    def test_names_are_canonical_and_unique(self):
        names = [dist.name for dist in dists.DISTRIBUTIONS]
        assert names == [dists.canonicalize_name(name) for name in names]
        assert len(names) == len(set(names))

    def test_no_partial_distribution_is_named_like_the_full_one(self):
        assert dists.FULL not in {dist.name for dist in dists.DISTRIBUTIONS}

    def test_every_name_extends_the_full_name(self):
        for dist in dists.DISTRIBUTIONS:
            assert dist.name.startswith(dists.FULL + "-"), dist.name

    def test_core_owns_the_root_package_and_this_module(self):
        core = dists.get(dists.CORE)
        assert "__init__.py" in core.files
        assert "_distributions.py" in core.files

    def test_only_the_core_owns_a_file_directly_under_the_package(self):
        # A root-level file needs the root package to be shipped, and only the
        # core ships the root package.
        for dist in dists.DISTRIBUTIONS[1:]:
            root_files = [path for path in dist.files if "/" not in path]
            assert root_files == [], f"{dist.name} lists root files {root_files}"

    @pytest.mark.parametrize("dist", dists.DISTRIBUTIONS, ids=lambda d: d.name)
    def test_paths_are_relative_posix_and_exist(self, dist):
        for tree in dist.trees:
            assert not tree.startswith("/") and "\\" not in tree and ".." not in tree
            assert (PACKAGE_DIR / tree).is_dir(), f"{dist.name}: no tree {tree}"
        for path in dist.files:
            assert not path.startswith("/") and "\\" not in path and ".." not in path
            assert (PACKAGE_DIR / path).is_file(), f"{dist.name}: no file {path}"

    @pytest.mark.parametrize("dist", dists.DISTRIBUTIONS, ids=lambda d: d.name)
    def test_every_tree_is_a_python_package(self, dist):
        for tree in dist.trees:
            assert (PACKAGE_DIR / tree / "__init__.py").is_file(), tree

    def test_summaries_are_one_non_empty_line(self):
        for dist in dists.DISTRIBUTIONS:
            assert dist.summary.strip() == dist.summary != ""
            assert "\n" not in dist.summary

    def test_no_path_has_two_owners(self):
        """A tree owns everything beneath it; nothing may fall under two."""
        claims = []
        for dist in dists.DISTRIBUTIONS:
            claims += [(tuple(tree.split("/")), True, dist.name) for tree in dist.trees]
            claims += [(tuple(path.split("/")), False, dist.name) for path in dist.files]
        for index, (parts, is_tree, owner) in enumerate(claims):
            for other_parts, other_is_tree, other_owner in claims[index + 1 :]:
                assert parts != other_parts, f"{'/'.join(parts)} is claimed twice"
                if is_tree:
                    assert other_parts[: len(parts)] != parts, (
                        f"{'/'.join(other_parts)} ({other_owner}) is inside tree "
                        f"{'/'.join(parts)} ({owner})"
                    )
                if other_is_tree:
                    assert parts[: len(other_parts)] != other_parts, (
                        f"{'/'.join(parts)} ({owner}) is inside tree "
                        f"{'/'.join(other_parts)} ({other_owner})"
                    )


# ===========================================================================
# get
# ===========================================================================


class TestGet:
    def test_lookup_accepts_any_spelling(self):
        assert dists.get("scikit_plots_rank_bm25").name == "scikit-plots-rank-bm25"
        assert dists.get("Scikit.Plots.Skinny") is dists.DISTRIBUTIONS[0]

    def test_unknown_name_lists_the_known_ones(self):
        with pytest.raises(KeyError) as excinfo:
            dists.get("scikit-plots-nope")
        message = excinfo.value.args[0]
        assert "scikit-plots-nope" in message
        assert dists.CORE in message

    def test_the_full_distribution_is_not_a_partial_one(self):
        with pytest.raises(KeyError):
            dists.get(dists.FULL)


# ===========================================================================
# provider_of / install_hint
# ===========================================================================


class TestProviderOf:
    @pytest.mark.parametrize(
        ("module", "expected"),
        [
            ("scikitplot", "scikit-plots-skinny"),
            ("scikitplot._cli", "scikit-plots-skinny"),
            ("scikitplot._cli._commands.doctor", "scikit-plots-skinny"),
            ("scikitplot.logging", "scikit-plots-skinny"),
            ("scikitplot.exceptions", "scikit-plots-skinny"),
            ("scikitplot._distributions", "scikit-plots-skinny"),
            ("scikitplot.rank_bm25", "scikit-plots-rank-bm25"),
            ("scikitplot.rank_bm25._rank_bm25", "scikit-plots-rank-bm25"),
            ("scikitplot.corpus._similarity._backends", "scikit-plots-corpus"),
            ("scikitplot.annoy", "scikit-plots-annoy"),
            ("scikitplot.annoy._annoy.annoylib", "scikit-plots-annoy"),
            ("scikitplot.cexternals", "scikit-plots-annoy"),
            ("scikitplot.cexternals._annoy", "scikit-plots-annoy"),
            ("scikitplot.cexternals._annoy.annoylib", "scikit-plots-annoy"),
            ("scikitplot.mcp.__main__", "scikit-plots-mcp"),
            ("scikitplot.cleanprompt", "scikit-plots-cleanprompt"),
            ("scikitplot.cython", "scikit-plots-cython"),
            ("scikitplot.mlflow", "scikit-plots-mlflow"),
        ],
    )
    def test_owned_modules(self, module, expected):
        assert dists.provider_of(module) == expected

    @pytest.mark.parametrize(
        "module",
        [
            "scikitplot.utils",
            "scikitplot.config",
            "scikitplot.api",
            "scikitplot.cexternals._astropy",
            "scikitplot.tests",
            "scikitplot._nope",
        ],
    )
    def test_full_only_modules_have_no_partial_provider(self, module):
        assert dists.provider_of(module) is None

    @pytest.mark.parametrize(
        "module", ["scikitplot.corpus_extra", "scikitplot.mcpx", "scikitplot.annoyance"]
    )
    def test_a_name_prefix_is_not_ownership(self, module):
        # Ownership is by path component: "corpus_extra" is not inside "corpus".
        assert dists.provider_of(module) is None

    def test_relative_and_absolute_names_agree(self):
        assert dists.provider_of("corpus._registry") == dists.provider_of(
            "scikitplot.corpus._registry"
        )

    def test_non_string_is_a_type_error(self):
        with pytest.raises(TypeError, match="must be a str"):
            dists.provider_of(None)

    def test_every_owned_tree_resolves_to_its_owner(self):
        for dist in dists.DISTRIBUTIONS:
            for tree in dist.trees:
                assert dists.provider_of(tree.replace("/", ".")) == dist.name


class TestInstallHint:
    def test_partial_distribution_when_one_ships_the_module(self):
        assert dists.install_hint("scikitplot.corpus") == "pip install scikit-plots-corpus"

    def test_full_distribution_otherwise(self):
        assert dists.install_hint("scikitplot.utils") == "pip install scikit-plots"

    def test_a_hint_always_names_a_known_distribution(self):
        known = {dists.FULL} | {dist.name for dist in dists.DISTRIBUTIONS}
        for module in ("scikitplot.mcp", "scikitplot.api", "scikitplot.cexternals._annoy"):
            assert dists.install_hint(module).rsplit(" ", 1)[1] in known


# ===========================================================================
# installed / flavor
# ===========================================================================


class TestInstalledAndFlavor:
    def test_installed_reports_only_what_is_installed(self, monkeypatch):
        _fake_metadata(monkeypatch, {dists.CORE: "0.5.0", "scikit-plots-mcp": "0.5.0", "numpy": "2.0"})
        assert dists.installed() == {dists.CORE: "0.5.0", "scikit-plots-mcp": "0.5.0"}

    def test_installed_is_empty_in_a_source_tree(self, monkeypatch):
        _fake_metadata(monkeypatch, {})
        assert dists.installed() == {}

    @pytest.mark.parametrize(
        ("found", "expected"),
        [
            ({}, "source"),
            ({"scikit-plots": "0.5.0"}, "full"),
            ({"scikit-plots-skinny": "0.5.0"}, "partial"),
            ({"scikit-plots-skinny": "0.5.0", "scikit-plots-corpus": "0.5.0"}, "partial"),
            # The full distribution wins: its files are the ones on disk.
            ({"scikit-plots": "0.5.0", "scikit-plots-skinny": "0.5.0"}, "full"),
            # A part without the core cannot provide the root package.
            ({"scikit-plots-corpus": "0.5.0"}, "source"),
        ],
    )
    def test_flavor(self, found, expected):
        assert dists.flavor(found) == expected
        assert expected in dists.FLAVORS

    def test_flavor_queries_metadata_when_not_given(self, monkeypatch):
        _fake_metadata(monkeypatch, {dists.CORE: "0.5.0"})
        assert dists.flavor() == "partial"


# ===========================================================================
# report
# ===========================================================================


class TestReport:
    @pytest.fixture
    def present(self, monkeypatch):
        """Pretend every part is on disk, so only metadata drives the report."""
        monkeypatch.setattr(dists, "_is_present", lambda module: True)

    def test_keys(self, monkeypatch, present):
        _fake_metadata(monkeypatch, {})
        assert set(dists.report()) == {
            "flavor",
            "installed",
            "available",
            "core_api",
            "problems",
            "notes",
        }
        assert dists.report()["core_api"] == dists.CORE_API

    def test_coherent_partial_install_has_no_problem(self, monkeypatch, present):
        _fake_metadata(monkeypatch, {dists.CORE: "0.5.0", "scikit-plots-mcp": "0.5.0"})
        report = dists.report()
        assert report["flavor"] == "partial"
        assert report["problems"] == []
        assert report["installed"] == {dists.CORE: "0.5.0", "scikit-plots-mcp": "0.5.0"}
        assert "scikit-plots-corpus" in report["available"]
        assert "scikit-plots-mcp" not in report["available"]
        assert report["available"]["scikit-plots-corpus"] == ["corpus"]

    def test_full_install_has_nothing_to_add(self, monkeypatch, present):
        _fake_metadata(monkeypatch, {dists.FULL: "0.5.0"})
        report = dists.report()
        assert report["flavor"] == "full"
        assert report["available"] == {}
        assert report["problems"] == []

    def test_source_tree_lists_every_partial_distribution(self, monkeypatch, present):
        _fake_metadata(monkeypatch, {})
        report = dists.report()
        assert report["flavor"] == "source"
        assert list(report["available"]) == [dist.name for dist in dists.DISTRIBUTIONS]

    def test_full_beside_partial_is_a_problem_with_a_fix(self, monkeypatch, present):
        _fake_metadata(monkeypatch, {dists.FULL: "0.5.0", dists.CORE: "0.5.0", "scikit-plots-mcp": "0.5.0"})
        (problem,) = dists.report()["problems"]
        assert "installed together with" in problem
        assert "pip uninstall scikit-plots-skinny scikit-plots-mcp" in problem
        assert "pip install --force-reinstall --no-deps scikit-plots" in problem

    def test_mixed_versions_with_the_same_core_api_are_a_note(self, monkeypatch, present):
        _fake_metadata(monkeypatch, {dists.CORE: "0.5.1", "scikit-plots-mcp": "0.5.0"})
        report = dists.report()
        assert report["problems"] == []
        (note,) = report["notes"]
        assert "scikit-plots-mcp 0.5.0" in note and "scikit-plots-skinny 0.5.1" in note
        assert f"core API ({dists.CORE_API})" in note
        assert "pip install --upgrade scikit-plots-skinny scikit-plots-mcp" in note

    def test_equal_versions_have_no_note(self, monkeypatch, present):
        _fake_metadata(monkeypatch, {dists.CORE: "0.5.0", "scikit-plots-mcp": "0.5.0"})
        assert dists.report()["notes"] == []

    @pytest.mark.parametrize(
        ("stated", "newer"),
        [(dists.CORE_API + 1, "scikit-plots-mcp"), (dists.CORE_API - 1, dists.CORE)],
    )
    def test_another_core_api_is_a_problem_even_at_equal_versions(
        self, monkeypatch, present, stated, newer
    ):
        _fake_metadata(
            monkeypatch,
            {dists.CORE: "0.5.0", "scikit-plots-mcp": "0.5.0"},
            apis={"scikit-plots-mcp": stated},
        )
        report = dists.report()
        (problem,) = report["problems"]
        assert f"built for core API {stated}" in problem
        assert f"provides core API {dists.CORE_API}" in problem
        assert f"{newer} is the newer side" in problem
        assert "pip install --upgrade scikit-plots-mcp scikit-plots-skinny" in problem
        assert report["notes"] == []

    def test_a_part_that_states_nothing_is_trusted_only_at_the_core_version(
        self, monkeypatch, present
    ):
        versions = {dists.CORE: "0.5.0", "scikit-plots-mcp": "0.5.0"}
        _fake_metadata(monkeypatch, versions, apis={})
        assert dists.report()["problems"] == []
        versions["scikit-plots-mcp"] = "0.4.9"
        (problem,) = dists.report()["problems"]
        assert "does not state which core API" in problem
        assert "cannot be judged" in problem
        assert "pip install --upgrade scikit-plots-mcp scikit-plots-skinny" in problem

    @pytest.mark.parametrize(
        "groups",
        [
            [dists.PARTS_GROUP + "1", dists.PARTS_GROUP + "2"],
            [dists.PARTS_GROUP + "x"],
            [dists.PARTS_GROUP],
        ],
    )
    def test_malformed_statement_is_a_problem_not_a_crash(
        self, monkeypatch, present, groups
    ):
        _fake_metadata(
            monkeypatch,
            {dists.CORE: "0.5.0", "scikit-plots-mcp": "0.5.0"},
            apis={"scikit-plots-mcp": groups},
        )
        (problem,) = dists.report()["problems"]
        assert "expected one entry-point group" in problem
        assert "pip install --force-reinstall --no-deps scikit-plots-mcp" in problem

    def test_a_part_without_the_core_is_not_judged(self, monkeypatch, present):
        _fake_metadata(monkeypatch, {"scikit-plots-mcp": "0.5.0"}, apis={})
        report = dists.report()
        assert report["problems"] == [] and report["notes"] == []

    def test_numpy_two_with_an_older_scikit_learn_is_a_problem_with_both_fixes(
        self, monkeypatch, present
    ):
        _fake_metadata(monkeypatch, {"numpy": "2.0.0", "scikit-learn": "1.3.0rc1"})
        (problem,) = dists.report()["problems"]
        assert "numpy 2.0.0 with scikit-learn 1.3.0rc1" in problem
        assert 'pip install --upgrade "scikit-learn>=1.4.2"' in problem
        assert 'pip install "numpy<2"' in problem

    @pytest.mark.parametrize(
        "versions",
        [
            {"numpy": "2.0.0", "scikit-learn": "1.4.2"},
            {"numpy": "2.3.1", "scikit-learn": "1.7.0"},
            {"numpy": "1.26.4", "scikit-learn": "1.3.0"},
            {"numpy": "2.0.0"},
            {"scikit-learn": "1.3.0"},
        ],
    )
    def test_working_third_party_versions_are_not_a_problem(
        self, monkeypatch, present, versions
    ):
        _fake_metadata(monkeypatch, versions)
        assert dists.report()["problems"] == []

    def test_numpy_one_is_a_note_that_explains_pip_check(self, monkeypatch, present):
        _fake_metadata(monkeypatch, {"numpy": "1.26.4"})
        report = dists.report()
        assert report["problems"] == []
        (note,) = report["notes"]
        assert "numpy 1.26.4" in note and "pip check" in note

    def test_numpy_two_has_no_note(self, monkeypatch, present):
        _fake_metadata(monkeypatch, {"numpy": "2.1.0"})
        assert dists.report()["notes"] == []

    def test_installed_but_missing_files_is_a_problem_with_a_fix(self, monkeypatch):
        _fake_metadata(monkeypatch, {dists.CORE: "0.5.0", "scikit-plots-mcp": "0.5.0"})
        monkeypatch.setattr(dists, "_is_present", lambda module: module != "mcp")
        (problem,) = dists.report()["problems"]
        assert problem.startswith("scikit-plots-mcp is installed but")
        assert "pip install --force-reinstall --no-deps scikit-plots-mcp" in problem

    def test_a_part_that_is_not_installed_is_never_reported_missing(self, monkeypatch):
        _fake_metadata(monkeypatch, {dists.CORE: "0.5.0"})
        monkeypatch.setattr(dists, "_is_present", lambda module: module in {"_cli", "logging", "__main__", "_distributions", "environment_variables", "exceptions"})
        assert dists.report()["problems"] == []

    def test_report_is_json_serialisable(self, monkeypatch, present):
        import json

        _fake_metadata(monkeypatch, {dists.CORE: "0.5.0"})
        assert json.loads(json.dumps(dists.report()))["flavor"] == "partial"


class TestCoreApi:
    def test_number_is_a_positive_integer_with_a_history_line(self):
        assert isinstance(dists.CORE_API, int) and dists.CORE_API >= 1
        numbers = [number for number, _version, _what in dists.CORE_API_HISTORY]
        assert numbers == list(range(1, dists.CORE_API + 1))
        assert all(what.strip() for _number, _version, what in dists.CORE_API_HISTORY)

    def test_group_name(self):
        assert dists.parts_group() == f"{dists.PARTS_GROUP}{dists.CORE_API}"
        assert dists.parts_group(7) == "scikitplot.parts.api7"

    def test_the_core_states_nothing(self):
        assert dists.parts_entry_points(dists.CORE) == {}

    @pytest.mark.parametrize(
        "dist", [d for d in dists.DISTRIBUTIONS if d.name != dists.CORE], ids=lambda d: d.name
    )
    def test_every_part_states_one_entry_per_tree(self, dist):
        entries = dists.parts_entry_points(dist.name)
        assert len(entries) == len(dist.trees) >= 1
        for name, module in entries.items():
            assert module == f"{dists.IMPORT_NAME}.{name}"
            assert dists.provider_of(module) == dist.name

    def test_entry_points_of_an_unknown_distribution(self):
        with pytest.raises(KeyError):
            dists.parts_entry_points("scikit-plots-nope")

    def test_declared_is_none_when_absent_or_silent(self, monkeypatch):
        _fake_metadata(monkeypatch, {dists.CORE: "0.5.0"})
        assert dists.declared_core_api(dists.CORE) is None
        assert dists.declared_core_api("scikit-plots-mcp") is None

    def test_declared_reads_the_number_in_any_spelling(self, monkeypatch):
        _fake_metadata(monkeypatch, {"scikit-plots-mcp": "0.5.0"}, apis={"scikit-plots-mcp": 3})
        assert dists.declared_core_api("Scikit_Plots.MCP") == 3

    def test_other_entry_point_groups_are_ignored(self, monkeypatch):
        _fake_metadata(
            monkeypatch,
            {"scikit-plots-mcp": "0.5.0"},
            apis={"scikit-plots-mcp": ["console_scripts", dists.parts_group(2)]},
        )
        assert dists.declared_core_api("scikit-plots-mcp") == 2

    @pytest.mark.parametrize(
        ("version", "expected"),
        [("1.3.0rc1", (1, 3, 0)), ("2.5.3", (2, 5, 3)), ("2", (2,)), ("dev", ()), (" 1.26.4 ", (1, 26, 4))],
    )
    def test_release(self, version, expected):
        assert dists._release(version) == expected


class TestLogReport:
    @pytest.fixture
    def present(self, monkeypatch):
        monkeypatch.setattr(dists, "_is_present", lambda module: True)

    def test_returns_the_report_and_is_quiet_when_coherent(
        self, monkeypatch, present, caplog
    ):
        _fake_metadata(monkeypatch, {dists.CORE: "0.5.0", "scikit-plots-mcp": "0.5.0"})
        with caplog.at_level("INFO", logger=dists.IMPORT_NAME):
            result = dists.log_report()
        assert result == dists.report()
        assert caplog.records == []

    def test_problems_are_warnings_and_notes_are_info(self, monkeypatch, present, caplog):
        _fake_metadata(
            monkeypatch,
            {
                dists.CORE: "0.5.1",
                "scikit-plots-mcp": "0.5.0",
                "numpy": "2.0.0",
                "scikit-learn": "1.3.0",
            },
        )
        with caplog.at_level("INFO", logger=dists.IMPORT_NAME):
            result = dists.log_report()
        by_level = {record.levelname: record.getMessage() for record in caplog.records}
        assert set(by_level) == {"WARNING", "INFO"}
        assert result["problems"][0] in by_level["WARNING"]
        assert result["notes"][0] in by_level["INFO"]

    def test_writes_to_the_logger_it_is_given(self, monkeypatch, present, caplog):
        import logging

        _fake_metadata(monkeypatch, {"numpy": "1.26.4"})
        with caplog.at_level("INFO", logger="another.logger"):
            dists.log_report(logging.getLogger("another.logger"))
        (record,) = caplog.records
        assert record.name == "another.logger" and "numpy 1.26.4" in record.getMessage()


class TestIsPresent:
    def test_an_existing_part_is_present(self):
        assert dists._is_present("logging") is True

    def test_an_absent_part_is_not(self):
        assert dists._is_present("_no_such_part_xyz") is False

    def test_only_the_first_component_is_probed(self):
        # Locating a deeper name would import its parent package; a report
        # must not run a subpackage's code.
        before = set(sys.modules)
        assert dists._is_present("logging._no_such_child") is True
        assert not {name for name in set(sys.modules) - before if "_no_such_child" in name}


# ===========================================================================
# Module contract
# ===========================================================================


class TestModuleContract:
    """The build tooling loads this file by path, outside the package."""

    def test_imports_only_the_standard_library(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        stdlib = getattr(sys, "stdlib_module_names", None)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.level == 0, "relative import: not loadable by path"
                roots = [(node.module or "").split(".")[0]]
            elif isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            else:
                continue
            for root in roots:
                assert root != "scikitplot"
                if stdlib is not None:  # Python >= 3.10
                    assert root in stdlib or root == "__future__", root

    def test_loads_by_file_path_without_the_package(self):
        spec = importlib.util.spec_from_file_location("_skplt_dists_probe", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.CORE == dists.CORE
        assert [d.name for d in module.DISTRIBUTIONS] == [d.name for d in dists.DISTRIBUTIONS]

    def test_all_names_exist(self):
        for name in dists.__all__:
            assert hasattr(dists, name), name

    def test_metadata_is_not_imported_at_import_time(self):
        # ``installed`` defers the import so that loading the map has no side
        # effects; the module-level namespace therefore holds no ``metadata``.
        assert "metadata" not in vars(dists)
