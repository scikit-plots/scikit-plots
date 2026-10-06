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


def _fake_metadata(monkeypatch, versions):
    """Make ``importlib.metadata.version`` answer from ``versions`` only."""

    def version(name):
        try:
            return versions[name]
        except KeyError:
            raise metadata.PackageNotFoundError(name) from None

    monkeypatch.setattr(metadata, "version", version)


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
        assert set(dists.report()) == {"flavor", "installed", "available", "problems"}

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

    def test_mixed_versions_are_a_problem_with_a_fix(self, monkeypatch, present):
        _fake_metadata(monkeypatch, {dists.CORE: "0.5.1", "scikit-plots-mcp": "0.5.0"})
        (problem,) = dists.report()["problems"]
        assert "different versions" in problem
        assert "scikit-plots-skinny 0.5.1" in problem and "scikit-plots-mcp 0.5.0" in problem
        assert "pip install --upgrade scikit-plots-skinny scikit-plots-mcp" in problem

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
