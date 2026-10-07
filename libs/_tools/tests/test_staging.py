# libs/_tools/tests/test_staging.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of ``libs/_tools/staging.py``: owned files and build-time staging."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from .. import staging


def _owned(repo: Path, name: str):
    dist = staging.load_distributions(repo).get(name)
    return [path.as_posix() for path in staging.iter_owned_files(dist, repo / "scikitplot")]


def _tree(directory: Path):
    """Return every file under ``directory`` as sorted POSIX paths."""
    return sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------
# load_distributions / repo_root
# ---------------------------------------------------------------------------


class TestLoadDistributions:
    def test_loads_the_map_of_the_given_root(self, repo):
        module = staging.load_distributions(repo)
        assert module.CORE == "scikit-plots-skinny"
        assert [d.name for d in module.DISTRIBUTIONS][1:] == ["scikit-plots-alpha", "scikit-plots-beta"]

    def test_writes_no_bytecode_into_the_source_tree(self, repo):
        staging.load_distributions(repo)
        assert not list((repo / "scikitplot").rglob("__pycache__"))

    def test_does_not_import_the_package(self, repo):
        before = set(sys.modules)
        staging.load_distributions(repo)
        assert not {name for name in set(sys.modules) - before if name.startswith("scikitplot")}

    def test_missing_map_is_an_import_error(self, tmp_path):
        with pytest.raises(ImportError, match="cannot load the distribution map"):
            staging.load_distributions(tmp_path)

    def test_repo_root_is_the_repository_this_file_is_in(self):
        root = staging.repo_root()
        assert (root / "libs" / "_tools" / "staging.py").is_file()
        assert (root / "scikitplot" / "_distributions.py").is_file()


# ---------------------------------------------------------------------------
# iter_owned_files
# ---------------------------------------------------------------------------


class TestIterOwnedFiles:
    def test_a_tree_owns_everything_beneath_it(self, repo):
        assert _owned(repo, "scikit-plots-alpha") == [
            "alpha/.hidden/config.json",
            "alpha/__init__.py",
            "alpha/_core.py",
            "alpha/data/table.json",
            "alpha/tests/__init__.py",
            "alpha/tests/test__core.py",
        ]

    def test_files_are_owned_individually(self, repo):
        assert _owned(repo, "scikit-plots-skinny") == [
            "__init__.py",
            "logging/__init__.py",
            "logging/_logging.py",
            "py.typed",
        ]

    def test_a_shared_directory_is_split_by_what_each_owns(self, repo):
        owned = _owned(repo, "scikit-plots-beta")
        assert "shared/__init__.py" in owned
        assert "shared/beta_only/native.cc" in owned
        assert "shared/other/__init__.py" not in owned

    def test_order_is_sorted_and_stable(self, repo):
        first, second = _owned(repo, "scikit-plots-alpha"), _owned(repo, "scikit-plots-alpha")
        assert first == second == sorted(first)

    @pytest.mark.parametrize(
        "relative",
        [
            "alpha/__pycache__/_core.cpython-313.pyc",
            "alpha/stale.pyc",
            "alpha/stale.pyo",
            "alpha/ext.cpython-313-x86_64-linux-gnu.so",
            "alpha/ext.cp313-win_amd64.pyd",
            "alpha/libthing.dylib",
            "alpha/unit.o",
            "alpha/unit.obj",
        ],
    )
    def test_caches_and_native_build_outputs_are_never_owned(self, repo, relative):
        path = repo / "scikitplot" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00")
        assert relative not in _owned(repo, "scikit-plots-alpha")

    def test_missing_tree_stops_with_its_name(self, repo):
        import shutil

        shutil.rmtree(repo / "scikitplot" / "alpha")
        with pytest.raises(FileNotFoundError, match="owned tree scikitplot/alpha does not exist"):
            _owned(repo, "scikit-plots-alpha")

    def test_missing_file_stops_with_its_name(self, repo):
        (repo / "scikitplot" / "py.typed").unlink()
        with pytest.raises(FileNotFoundError, match="owned file scikitplot/py.typed does not exist"):
            _owned(repo, "scikit-plots-skinny")

    def test_a_file_inside_its_own_tree_is_a_mistake(self, repo):
        module = staging.load_distributions(repo)
        bad = module.Distribution("scikit-plots-alpha", ("alpha",), ("alpha/_core.py",), "x")
        with pytest.raises(ValueError, match="already inside one of its trees"):
            list(staging.iter_owned_files(bad, repo / "scikitplot"))


# ---------------------------------------------------------------------------
# check_ownership
# ---------------------------------------------------------------------------


class TestCheckOwnership:
    def test_disjoint_distributions_pass(self, repo):
        module = staging.load_distributions(repo)
        owned = staging.check_ownership(module.DISTRIBUTIONS, repo / "scikitplot")
        assert list(owned) == [d.name for d in module.DISTRIBUTIONS]
        every = [path for files in owned.values() for path in files]
        assert len(every) == len(set(every))

    def test_a_file_with_two_owners_is_refused_and_named(self, repo):
        module = staging.load_distributions(repo)
        thief = module.Distribution("scikit-plots-gamma", ("alpha/data",), (), "x")
        with pytest.raises(ValueError) as excinfo:
            staging.check_ownership([*module.DISTRIBUTIONS, thief], repo / "scikitplot")
        message = str(excinfo.value)
        assert "scikitplot/alpha/data/table.json" in message
        assert "scikit-plots-alpha" in message and "scikit-plots-gamma" in message

    def test_a_distribution_declared_twice_is_refused(self, repo):
        module = staging.load_distributions(repo)
        with pytest.raises(ValueError, match="declared twice"):
            staging.check_ownership(
                [module.DISTRIBUTIONS[1], module.DISTRIBUTIONS[1]], repo / "scikitplot"
            )

    def test_the_real_repository_has_one_owner_per_file(self):
        root = staging.repo_root()
        module = staging.load_distributions(root)
        owned = staging.check_ownership(module.DISTRIBUTIONS, root / "scikitplot")
        assert all(owned[d.name] for d in module.DISTRIBUTIONS)


# ---------------------------------------------------------------------------
# stage / unstage
# ---------------------------------------------------------------------------


class TestStage:
    def test_stages_exactly_the_owned_files_and_the_licence(self, repo):
        lib = repo / "libs" / "alpha"
        staging.stage("scikit-plots-alpha", lib, repo)
        assert _tree(lib / "scikitplot") == _owned(repo, "scikit-plots-alpha")
        assert (lib / "LICENSE.txt").read_text(encoding="utf-8") == "Miniature licence.\n"

    def test_content_is_copied_byte_for_byte(self, repo):
        lib = repo / "libs" / "alpha"
        staging.stage("scikit-plots-alpha", lib, repo)
        for relative in _owned(repo, "scikit-plots-alpha"):
            assert (lib / "scikitplot" / relative).read_bytes() == (
                repo / "scikitplot" / relative
            ).read_bytes()

    def test_a_part_does_not_get_the_root_package(self, repo):
        lib = repo / "libs" / "alpha"
        staging.stage("scikit-plots-alpha", lib, repo)
        assert not (lib / "scikitplot" / "__init__.py").exists()

    def test_any_spelling_of_the_name_is_accepted(self, repo):
        staged = staging.stage("scikit_plots_alpha", repo / "libs" / "alpha", repo)
        assert staged

    def test_staging_twice_gives_the_same_tree(self, repo):
        lib = repo / "libs" / "alpha"
        staging.stage("scikit-plots-alpha", lib, repo)
        first = _tree(lib)
        staging.stage("scikit-plots-alpha", lib, repo)
        assert _tree(lib) == first

    def test_a_file_removed_from_the_source_is_removed_from_the_stage(self, repo):
        lib = repo / "libs" / "alpha"
        staging.stage("scikit-plots-alpha", lib, repo)
        (repo / "scikitplot" / "alpha" / "_core.py").unlink()
        staging.stage("scikit-plots-alpha", lib, repo)
        assert not (lib / "scikitplot" / "alpha" / "_core.py").exists()

    def test_stale_build_directories_are_removed(self, repo):
        lib = repo / "libs" / "alpha"
        (lib / "build" / "lib" / "scikitplot" / "alpha").mkdir(parents=True)
        (lib / "build" / "lib" / "scikitplot" / "alpha" / "deleted_module.py").write_text("", encoding="utf-8")
        (lib / "scikit_plots_alpha.egg-info").mkdir()
        (lib / "scikit_plots_alpha.egg-info" / "SOURCES.txt").write_text("stale\n", encoding="utf-8")
        staging.stage("scikit-plots-alpha", lib, repo)
        assert not (lib / "build").exists()
        assert not (lib / "scikit_plots_alpha.egg-info").exists()

    def test_built_artefacts_are_kept(self, repo):
        lib = repo / "libs" / "alpha"
        (lib / "dist").mkdir()
        (lib / "dist" / "x.whl").write_bytes(b"wheel")
        staging.stage("scikit-plots-alpha", lib, repo)
        staging.unstage(lib, repo)
        assert (lib / "dist" / "x.whl").read_bytes() == b"wheel"

    def test_a_wrong_map_leaves_the_directory_untouched(self, repo):
        lib = repo / "libs" / "alpha"
        staging.stage("scikit-plots-alpha", lib, repo)
        before = _tree(lib)
        import shutil

        shutil.rmtree(repo / "scikitplot" / "alpha")
        with pytest.raises(FileNotFoundError):
            staging.stage("scikit-plots-alpha", lib, repo)
        assert _tree(lib) == before

    def test_missing_licence_stops_the_build(self, repo):
        (repo / "LICENSE.txt").unlink()
        with pytest.raises(FileNotFoundError, match="must ship the licence"):
            staging.stage("scikit-plots-alpha", repo / "libs" / "alpha", repo)

    def test_unknown_distribution_is_refused(self, repo):
        with pytest.raises(KeyError):
            staging.stage("scikit-plots-nope", repo / "libs" / "alpha", repo)

    def test_refuses_a_directory_outside_libs(self, repo, tmp_path):
        elsewhere = repo / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "pyproject.toml").write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="not a direct child"):
            staging.stage("scikit-plots-alpha", elsewhere, repo)

    def test_refuses_a_directory_without_pyproject(self, repo):
        bare = repo / "libs" / "bare"
        bare.mkdir()
        with pytest.raises(ValueError, match="has no pyproject.toml"):
            staging.stage("scikit-plots-alpha", bare, repo)


class TestUnstage:
    def test_removes_everything_staging_and_a_build_left(self, repo):
        lib = repo / "libs" / "alpha"
        staging.stage("scikit-plots-alpha", lib, repo)
        (lib / "build").mkdir()
        (lib / "scikit_plots_alpha.egg-info").mkdir()
        staging.unstage(lib, repo)
        assert sorted(p.name for p in lib.iterdir()) == ["pyproject.toml"]

    def test_is_idempotent(self, repo):
        lib = repo / "libs" / "alpha"
        staging.unstage(lib, repo)
        staging.unstage(lib, repo)
        assert sorted(p.name for p in lib.iterdir()) == ["pyproject.toml"]

    def test_never_touches_the_source_tree(self, repo):
        before = _tree(repo / "scikitplot")
        lib = repo / "libs" / "alpha"
        staging.stage("scikit-plots-alpha", lib, repo)
        staging.unstage(lib, repo)
        assert _tree(repo / "scikitplot") == before

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symbolic links")
    def test_a_link_to_the_source_tree_is_removed_as_a_link(self, repo):
        """
        An earlier layout linked libs/<name>/scikitplot to the source tree.

        Deleting *through* such a link would delete the source. It must be
        unlinked, and the source must survive both unstage and stage.
        """
        lib = repo / "libs" / "alpha"
        try:
            os.symlink(repo / "scikitplot", lib / "scikitplot", target_is_directory=True)
        except OSError:
            pytest.skip("this platform does not allow creating symbolic links here")
        before = _tree(repo / "scikitplot")
        staging.unstage(lib, repo)
        assert not os.path.lexists(lib / "scikitplot")
        assert _tree(repo / "scikitplot") == before

        os.symlink(repo / "scikitplot", lib / "scikitplot", target_is_directory=True)
        staging.stage("scikit-plots-alpha", lib, repo)
        assert not os.path.islink(lib / "scikitplot")
        assert _tree(repo / "scikitplot") == before
        assert _tree(lib / "scikitplot") == _owned(repo, "scikit-plots-alpha")

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symbolic links")
    def test_a_linked_licence_is_removed_as_a_link(self, repo):
        lib = repo / "libs" / "alpha"
        try:
            os.symlink(repo / "LICENSE.txt", lib / "LICENSE.txt")
        except OSError:
            pytest.skip("this platform does not allow creating symbolic links here")
        staging.unstage(lib, repo)
        assert (repo / "LICENSE.txt").is_file()
        assert not os.path.lexists(lib / "LICENSE.txt")
