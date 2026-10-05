# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for ``ci_test_plan.py``.

Run with the standard library only, as the plan job does::

    python -m unittest discover -s .github/scripts/tests -p "test_*.py" -v

Notes
-----
**Developer notes.** Every test builds its own small tree, so the rules are
checked independently of how the real package happens to be laid out. One
class at the end runs against the real repository to hold the shipped
configuration to those rules.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ci_test_plan as plan_module  # noqa: E402
from ci_test_plan import (  # noqa: E402
    PYTEST_DEFAULT_NORECURSEDIRS,
    PlanError,
    Planner,
    load_config,
    load_durations,
    load_norecursedirs,
)


def main(argv):
    """Run the command line with its output captured, returning the exit code."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return plan_module.main(argv)

REPOSITORY = Path(__file__).resolve().parents[3]

CONFIG = {
    "schema": 1,
    "package_root": "pkg",
    "containers": ["pkg", "pkg/_ext", "pkg/_ext/_sphinx"],
    "full_run_globs": ["pyproject.toml", "meson_cpu/**", "tools/plan.py"],
    "ignore_globs": ["pkg/**/*.md"],
    "loose_test_globs": ["test_*.py", "*_test.py"],
    "default_mode": {"pull_request": "auto", "push": "all"},
    "max_shards": 3,
    "shard_timeout_minutes": 350,
    "shard_target_minutes": 1,
    "default_unit_seconds": 60,
    "per_test_seconds": 0.5,
    "dependents": "none",
}

FILES = {
    "pkg/__init__.py": "",
    "pkg/conftest.py": "",
    "pkg/tests/test_root.py": "",
    "pkg/corpus/__init__.py": "from .. import utils\n",
    "pkg/corpus/tests/test_corpus.py": "",
    "pkg/utils/__init__.py": "",
    "pkg/utils/tests/test_utils.py": "",
    "pkg/stats/__init__.py": "import pkg.utils.helpers\n",
    "pkg/stats/README.md": "",
    "pkg/_ext/__init__.py": "",
    "pkg/_ext/tests/test_ext.py": "",
    "pkg/_ext/_sphinx/__init__.py": "",
    "pkg/_ext/_sphinx/tests/test_stack.py": "",
    "pkg/_ext/_sphinx/_learn/__init__.py": "from .._grid import x\n",
    "pkg/_ext/_sphinx/_learn/tests/test_learn.py": "",
    "pkg/_ext/_sphinx/_grid/__init__.py": "",
    "pkg/_ext/_sphinx/_grid/tests/test_grid.py": "",
    "pkg/_ext/_jupyter/__init__.py": "",
    "docs/index.rst": "",
    "pyproject.toml": "",
    # The command line reads this, as pytest does.
    "pytest.ini": "[pytest]\nnorecursedirs = docs\n",
}


class Tree(unittest.TestCase):
    """A scratch repository and a planner over it."""

    config = CONFIG

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        for name, text in FILES.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.planner = Planner(self.root, dict(self.config), {})

    def paths(self, plan):
        return sorted(path for shard in plan["shards"] for path in shard["paths"])


class TestTree(Tree):
    def test_units_are_children_of_containers(self):
        self.assertEqual(
            self.planner.units(),
            [
                "pkg/_ext/_jupyter",
                "pkg/_ext/_sphinx/_grid",
                "pkg/_ext/_sphinx/_learn",
                "pkg/_ext/_sphinx/tests",
                "pkg/_ext/tests",
                "pkg/corpus",
                "pkg/stats",
                "pkg/tests",
                "pkg/utils",
            ],
        )

    def test_owner(self):
        cases = {
            "pkg/corpus/a/b.py": ("unit", "pkg/corpus"),
            "pkg/tests/test_root.py": ("unit", "pkg/tests"),
            "pkg/_ext/_sphinx/_learn/x.py": ("unit", "pkg/_ext/_sphinx/_learn"),
            "pkg/_ext/_sphinx/__init__.py": ("container", "pkg/_ext/_sphinx"),
            "pkg/_ext/conftest.py": ("container", "pkg/_ext"),
            "pkg/__init__.py": ("container", "pkg"),
            "pkg/gone/module.py": ("unit", "pkg/gone"),
            "docs/index.rst": ("outside", ""),
            "pkgs/x.py": ("outside", ""),
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                self.assertEqual(self.planner.owner(path), expected)

    def test_ancestor_tests_stop_at_the_unit_itself(self):
        self.assertEqual(
            self.planner.ancestor_tests("pkg/_ext/_sphinx/_learn"),
            ["pkg/_ext/_sphinx/tests", "pkg/_ext/tests", "pkg/tests"],
        )
        self.assertEqual(self.planner.ancestor_tests("pkg/corpus"), ["pkg/tests"])
        self.assertEqual(self.planner.ancestor_tests("pkg/tests"), [])


class TestAutoMode(Tree):
    def test_a_top_level_submodule_runs_with_the_root_tests(self):
        plan = self.planner.plan("auto", changed=["pkg/corpus/reader.py"])
        self.assertEqual((plan["mode"], plan["run"]), ("auto", True))
        self.assertEqual(self.paths(plan), ["pkg/corpus", "pkg/tests"])

    def test_a_nested_submodule_runs_with_every_tests_directory_above_it(self):
        plan = self.planner.plan("auto", changed=["pkg/_ext/_sphinx/_learn/pages.py"])
        self.assertEqual(
            self.paths(plan),
            ["pkg/_ext/_sphinx/_learn", "pkg/_ext/_sphinx/tests", "pkg/_ext/tests", "pkg/tests"],
        )

    def test_several_changes_are_united(self):
        plan = self.planner.plan(
            "auto", changed=["pkg/corpus/a.py", "pkg/utils/b.py", "pkg/corpus/c.py", "docs/index.rst"]
        )
        self.assertEqual(self.paths(plan), ["pkg/corpus", "pkg/tests", "pkg/utils"])

    def test_nothing_in_the_library_means_no_test_job(self):
        plan = self.planner.plan("auto", changed=["docs/index.rst", "README.md", ".github/x.yml"])
        self.assertEqual((plan["run"], plan["shards"], plan["units"]), (False, [], []))
        self.assertIn("is part of the library", plan["reason"][-1])

    def test_no_changed_files_at_all_means_no_test_job(self):
        self.assertFalse(self.planner.plan("auto", changed=[])["run"])

    def test_build_configuration_runs_everything(self):
        for path in ("pyproject.toml", "meson_cpu/x86/meson.build", "tools/plan.py"):
            with self.subTest(path=path):
                plan = self.planner.plan("auto", changed=["docs/index.rst", path])
                self.assertEqual(plan["mode"], "all")
                self.assertEqual(len(plan["units"]), 9)

    def test_a_file_at_the_package_root_runs_everything(self):
        for path in ("pkg/__init__.py", "pkg/conftest.py", "pkg/removed_module.py"):
            with self.subTest(path=path):
                self.assertEqual(self.planner.plan("auto", changed=[path])["mode"], "all")

    def test_a_file_in_a_nested_container_runs_everything_below_it(self):
        plan = self.planner.plan("auto", changed=["pkg/_ext/_sphinx/__init__.py"])
        self.assertEqual(plan["mode"], "auto")
        self.assertEqual(
            self.paths(plan),
            ["pkg/_ext/_sphinx/_grid", "pkg/_ext/_sphinx/_learn", "pkg/_ext/_sphinx/tests", "pkg/_ext/tests", "pkg/tests"],
        )

    def test_an_ignored_file_selects_nothing(self):
        self.assertFalse(self.planner.plan("auto", changed=["pkg/stats/README.md"])["run"])

    def test_a_deleted_submodule_still_runs_the_tests_above_it(self):
        plan = self.planner.plan("auto", changed=["pkg/_ext/_sphinx/_removed/module.py"])
        self.assertEqual(self.paths(plan), ["pkg/_ext/_sphinx/tests", "pkg/_ext/tests", "pkg/tests"])
        self.assertTrue(any("no longer present" in line for line in plan["reason"]))

    def test_unknown_changes_fail_closed_to_everything(self):
        plan = self.planner.plan("auto", changed=None, diff_error="no base commit to compare with")
        self.assertEqual(plan["mode"], "all")
        self.assertIn("no base commit", plan["reason"][0])

    def test_changing_only_tests_at_the_root_runs_only_them(self):
        self.assertEqual(self.paths(self.planner.plan("auto", changed=["pkg/tests/test_root.py"])), ["pkg/tests"])


class TestAllMode(Tree):
    def test_every_unit_is_planned_exactly_once(self):
        plan = self.planner.plan("all")
        self.assertEqual(self.paths(plan), self.planner.units())
        self.assertEqual(len(plan["shards"]), 3)

    def test_a_loose_test_file_in_a_container_is_planned(self):
        (self.root / "pkg/_ext/test_loose.py").write_text("", encoding="utf-8")
        self.assertIn("pkg/_ext/test_loose.py", self.paths(self.planner.plan("all")))

    def test_a_test_file_no_path_would_collect_stops_the_plan(self):
        config = dict(self.config, loose_test_globs=["test_*.py"])
        planner = Planner(self.root, config, {})
        (self.root / "pkg/_ext/check_test.py").write_text("", encoding="utf-8")
        planner.plan("all")  # not a test file under these globs
        planner.config["loose_test_globs"] = ["test_*.py", "*_test.py"]
        original = planner.loose_tests
        planner.loose_tests = lambda container=None: []
        try:
            with self.assertRaisesRegex(PlanError, "would not collect 1 test file"):
                planner.plan("all")
        finally:
            planner.loose_tests = original


class TestCustomMode(Tree):
    def test_names_resolve_in_every_spelling(self):
        for text, expected in {
            "corpus": ["pkg/corpus"],
            "pkg/corpus": ["pkg/corpus"],
            "_ext/_sphinx/_learn": ["pkg/_ext/_sphinx/_learn"],
            "_learn, corpus\nutils": ["pkg/_ext/_sphinx/_learn", "pkg/corpus", "pkg/utils"],
            "corpus corpus": ["pkg/corpus"],
            "_ext/_sphinx": ["pkg/_ext/_sphinx/_grid", "pkg/_ext/_sphinx/_learn", "pkg/_ext/_sphinx/tests"],
        }.items():
            with self.subTest(text=text):
                self.assertEqual(self.planner.resolve_selection(text), expected)

    def test_a_selected_unit_runs_with_the_tests_above_it(self):
        plan = self.planner.plan("custom", select="_learn")
        self.assertEqual(
            self.paths(plan),
            ["pkg/_ext/_sphinx/_learn", "pkg/_ext/_sphinx/tests", "pkg/_ext/tests", "pkg/tests"],
        )

    def test_an_ambiguous_name_is_refused_with_the_candidates(self):
        (self.root / "pkg/_ext/_learn").mkdir()
        with self.assertRaisesRegex(PlanError, "names several submodules: pkg/_ext/_learn, pkg/_ext/_sphinx/_learn"):
            self.planner.resolve_selection("_learn")

    def test_a_name_that_is_a_path_below_the_root_wins_over_a_bare_name(self):
        self.assertEqual(self.planner.resolve_selection("tests"), ["pkg/tests"])

    def test_an_unknown_or_empty_selection_is_refused(self):
        with self.assertRaisesRegex(PlanError, "is not a submodule"):
            self.planner.resolve_selection("nope")
        with self.assertRaisesRegex(PlanError, "at least one"):
            self.planner.plan("custom", select=" , ")


class TestDependents(Tree):
    def setUp(self):
        super().setUp()
        self.planner.config["dependents"] = "direct"

    def test_importers_are_added_for_relative_and_absolute_imports(self):
        plan = self.planner.plan("auto", changed=["pkg/utils/helpers.py"])
        self.assertEqual(plan["units"], ["pkg/corpus", "pkg/stats", "pkg/utils"])

    def test_a_relative_import_inside_a_nested_container_is_resolved(self):
        plan = self.planner.plan("auto", changed=["pkg/_ext/_sphinx/_grid/x.py"])
        self.assertEqual(plan["units"], ["pkg/_ext/_sphinx/_grid", "pkg/_ext/_sphinx/_learn"])

    def test_importers_of_importers_are_not_followed(self):
        (self.root / "pkg/_ext/_jupyter/__init__.py").write_text("import pkg.stats\n", encoding="utf-8")
        plan = self.planner.plan("auto", changed=["pkg/utils/helpers.py"])
        self.assertNotIn("pkg/_ext/_jupyter", plan["units"])

    def test_off_by_default(self):
        self.planner.config["dependents"] = "none"
        self.assertEqual(self.planner.plan("auto", changed=["pkg/utils/helpers.py"])["units"], ["pkg/utils"])


class TestShards(Tree):
    def test_packing_is_balanced_and_stable(self):
        durations = {"pkg/corpus": (0, 600.0), "pkg/utils": (0, 300.0), "pkg/stats": (0, 290.0), "pkg/tests": (0, 10.0)}
        planner = Planner(self.root, dict(self.config), durations)
        paths = ["pkg/tests", "pkg/stats", "pkg/utils", "pkg/corpus"]
        first = planner.shards(paths, 2)
        self.assertEqual(first, planner.shards(list(reversed(paths)), 2))
        self.assertEqual([shard["paths"] for shard in first], [["pkg/corpus"], ["pkg/stats", "pkg/tests", "pkg/utils"]])
        self.assertEqual([shard["name"] for shard in first], ["shard-1-of-2", "shard-2-of-2"])

    def test_never_more_shards_than_paths_and_never_an_empty_one(self):
        shards = self.planner.shards(["pkg/corpus", "pkg/utils"], 10)
        self.assertEqual(len(shards), 2)
        self.assertTrue(all(shard["paths"] for shard in shards))

    def test_weight_uses_estimates_and_a_default(self):
        planner = Planner(self.root, dict(self.config), {"pkg/corpus": (100, 50.0)})
        self.assertEqual(planner.weight("pkg/corpus"), 100.0)
        self.assertEqual(planner.weight("pkg/unmeasured"), 60.0)

    def test_a_small_selection_runs_in_one_job(self):
        planner = Planner(self.root, dict(self.config, shard_target_minutes=30), {})
        shards = planner.shards(["pkg/corpus", "pkg/utils", "pkg/tests"], 4)
        self.assertEqual(len(shards), 1)
        self.assertEqual(shards[0]["name"], "shard-1-of-1")

    def test_the_number_of_jobs_follows_the_estimate(self):
        durations = {"pkg/corpus": (0, 3000.0), "pkg/utils": (0, 3000.0), "pkg/stats": (0, 3000.0), "pkg/tests": (0, 3000.0)}
        planner = Planner(self.root, dict(self.config, shard_target_minutes=60), durations)
        paths = list(durations)
        self.assertEqual(len(planner.shards(paths, 8)), 4)
        self.assertEqual(len(planner.shards(paths, 2)), 2)
        self.assertEqual(len(planner.shards(paths[:1], 8)), 1)

    def test_duplicates_are_planned_once(self):
        shards = self.planner.shards(["pkg/corpus", "pkg/corpus"], 2)
        self.assertEqual([shard["paths"] for shard in shards], [["pkg/corpus"]])

    def test_a_path_with_white_space_is_refused(self):
        with self.assertRaisesRegex(PlanError, "white space"):
            self.planner.shards(["pkg/has space"], 2)

    def test_a_shard_limit_below_one_is_refused(self):
        with self.assertRaises(PlanError):
            self.planner.shards(["pkg/corpus"], 0)


class TestDefaultMode(Tree):
    def test_the_event_picks_the_mode(self):
        self.assertEqual(self.planner.plan("default", event="push")["mode"], "all")
        plan = self.planner.plan("default", event="pull_request", changed=["pkg/corpus/a.py"])
        self.assertEqual(plan["mode"], "auto")

    def test_an_unlisted_event_runs_everything(self):
        self.assertEqual(self.planner.plan("default", event="release")["mode"], "all")

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaises(PlanError):
            self.planner.plan("some")


class TestConfiguration(unittest.TestCase):
    def _write(self, **changes):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        self.addCleanup(lambda: Path(tmp.name).unlink())
        config = dict(CONFIG, **changes)
        for key, value in changes.items():
            if value is KeyError:
                del config[key]
        json.dump(config, tmp)
        tmp.close()
        return tmp.name

    def test_a_valid_configuration_loads(self):
        self.assertEqual(load_config(self._write())["package_root"], "pkg")

    def test_each_mistake_is_named(self):
        cases = {
            "missing key 'containers'": {"containers": KeyError},
            "'max_shards' has the wrong type": {"max_shards": "4"},
            "'max_shards' must be at least 1": {"max_shards": 0},
            "unsupported schema": {"schema": 2},
            "must include the package root": {"containers": ["pkg/_ext"]},
            "is outside 'pkg'": {"containers": ["pkg", "other/x"]},
            "between 1 and 360": {"shard_timeout_minutes": 361},
            "'shard_target_minutes' must be positive": {"shard_target_minutes": 0},
            "'dependents' must be one of": {"dependents": "all"},
            "must be 'auto' or 'all'": {"default_mode": {"push": "custom"}},
            "'schema' has the wrong type": {"schema": True},
        }
        for message, change in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(PlanError, message):
                    load_config(self._write(**change))

    def test_an_unreadable_configuration_is_reported(self):
        with self.assertRaisesRegex(PlanError, "cannot read the plan configuration"):
            load_config("/nonexistent/plan.json")

    def test_missing_durations_are_not_an_error_but_bad_ones_are(self):
        self.assertEqual(load_durations("/nonexistent/durations.json"), {})
        bad = self._write()
        with self.assertRaisesRegex(PlanError, "cannot read the duration estimates"):
            load_durations(bad)


class TestNorecursedirsSetting(unittest.TestCase):
    """Reading the setting from ``pytest.ini``."""

    def _ini(self, text):
        handle = tempfile.NamedTemporaryFile("w", suffix=".ini", delete=False, encoding="utf-8")
        self.addCleanup(os.unlink, handle.name)
        with handle:
            handle.write(text)
        return handle.name

    def test_patterns_are_read_in_order_across_lines_and_within_one(self):
        ini = self._ini(
            "# a comment\n"
            "[pytest]\n"
            "## another\n"
            "norecursedirs =\n"
            "    .git_clones\n"
            "    # a comment inside the value\n"
            "    galleries examples\n"
            "    pkg/cext/_vendored\n"
            "    'two words'\n"
            "addopts =\n"
            "    -l\n"
            "    \"-p no:cacheprovider\"\n"
            "filterwarnings =\n"
            "    error\n"
            "    ignore:100%% sure:UserWarning\n"
        )
        self.assertEqual(
            load_norecursedirs(ini),
            (".git_clones", "galleries", "examples", "pkg/cext/_vendored", "two words"),
        )

    def test_without_the_setting_pytest_defaults_apply(self):
        for text in ("", "[pytest]\naddopts = -q\n", "[tool:other]\nnorecursedirs = x\n"):
            with self.subTest(text=text):
                self.assertEqual(load_norecursedirs(self._ini(text)), PYTEST_DEFAULT_NORECURSEDIRS)
        self.assertEqual(load_norecursedirs(None), PYTEST_DEFAULT_NORECURSEDIRS)
        self.assertEqual(load_norecursedirs(""), PYTEST_DEFAULT_NORECURSEDIRS)

    def test_an_empty_setting_excludes_nothing(self):
        self.assertEqual(load_norecursedirs(self._ini("[pytest]\nnorecursedirs =\n")), ())

    def test_a_named_file_that_cannot_be_used_is_an_error_not_a_default(self):
        with self.assertRaisesRegex(PlanError, "cannot read the pytest configuration"):
            load_norecursedirs("/nonexistent/pytest.ini")
        with self.assertRaisesRegex(PlanError, "cannot read the pytest configuration"):
            load_norecursedirs(self._ini("norecursedirs = x\n"))  # no section header
        with self.assertRaisesRegex(PlanError, "cannot be split"):
            load_norecursedirs(self._ini("[pytest]\nnorecursedirs = 'unclosed\n"))

    def test_the_configuration_may_name_the_file_but_only_as_a_string(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        self.addCleanup(os.unlink, handle.name)
        with handle:
            json.dump(dict(CONFIG, pytest_ini=["pytest.ini"]), handle)
        with self.assertRaisesRegex(PlanError, "'pytest_ini' must be a string"):
            load_config(handle.name)


class TestExcludedDirectories(Tree):
    """A sharded run must collect what a recursive run collects, no more."""

    patterns = ("docs", "tools", "pkg/vendored", "pkg/_ext/_sphinx/_old*")
    extra = {
        "pkg/vendored/__init__.py": "",
        "pkg/vendored/conftest.py": "",
        "pkg/vendored/_lib/tests/test_lib.py": "",
        "pkg/tools/test_tool.py": "",
        "pkg/corpus/tools/test_nested.py": "",
        "pkg/_ext/_sphinx/_old_stack/tests/test_old.py": "",
        "pkg/_ext/_sphinx/_grid/docs/test_example.py": "",
    }

    def setUp(self):
        super().setUp()
        for name, text in self.extra.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.planner = Planner(self.root, dict(self.config), {}, self.patterns)

    def test_matching_follows_pytest(self):
        cases = {
            "pkg/vendored": True,          # a pattern with a separator matches the end of the path
            "pkg/vendored/_lib": True,     # ... and hides everything below
            "pkg/vendored/_lib/tests/test_lib.py": True,
            "other/pkg/vendored": True,
            "vendored": False,             # ... but not a shorter path
            "pkg/xvendored": False,
            "pkg/tools": True,             # a bare pattern matches a directory name anywhere
            "pkg/corpus/tools": True,
            "pkg/corpus/toolset": False,
            "pkg/_ext/_sphinx/_old_stack": True,
            "pkg/_ext/_sphinx/_grid": False,
            "pkg/corpus": False,
            "pkg/corpus/__pycache__": True,
            "pkg/tests/test_root.py": False,
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                self.assertIs(self.planner.is_excluded(path), expected)

    def test_a_file_is_judged_by_its_directory_not_its_own_name(self):
        (self.root / "pkg" / "corpus" / "tools.py").write_text("", encoding="utf-8")
        (self.root / "pkg" / "corpus" / "docs").write_text("", encoding="utf-8")
        self.assertFalse(self.planner.is_excluded("pkg/corpus/tools.py"))
        self.assertFalse(self.planner.is_excluded("pkg/corpus/docs"))

    def test_an_excluded_directory_is_not_a_unit(self):
        units = self.planner.units()
        for unit in ("pkg/vendored", "pkg/tools", "pkg/_ext/_sphinx/_old_stack"):
            with self.subTest(unit=unit):
                self.assertNotIn(unit, units)
        self.assertEqual(self.planner.excluded_units(), ["pkg/_ext/_sphinx/_old_stack", "pkg/tools", "pkg/vendored"])
        self.assertIn("pkg/corpus", units)

    def test_a_full_run_leaves_excluded_tests_out_and_is_still_complete(self):
        paths = self.paths(self.planner.plan("all"))
        self.assertFalse([path for path in paths if self.planner.is_excluded(path)])
        self.assertEqual(
            self.planner.collectable_test_files(),
            [
                "pkg/_ext/_sphinx/_grid/tests/test_grid.py",
                "pkg/_ext/_sphinx/_learn/tests/test_learn.py",
                "pkg/_ext/_sphinx/tests/test_stack.py",
                "pkg/_ext/tests/test_ext.py",
                "pkg/corpus/tests/test_corpus.py",
                "pkg/tests/test_root.py",
                "pkg/utils/tests/test_utils.py",
            ],
        )

    def test_a_change_inside_an_excluded_directory_runs_only_the_tests_above_it(self):
        plan = self.planner.plan("auto", changed=["pkg/vendored/_lib/core.py"])
        self.assertEqual(self.paths(plan), ["pkg/tests"])
        self.assertTrue(any("norecursedirs" in reason for reason in plan["reason"]))
        nested = self.planner.plan("auto", changed=["pkg/_ext/_sphinx/_old_stack/x.py"])
        self.assertEqual(self.paths(nested), ["pkg/_ext/_sphinx/tests", "pkg/_ext/tests", "pkg/tests"])

    def test_importers_of_an_excluded_directory_are_still_found(self):
        (self.root / "pkg" / "stats" / "io.py").write_text("from ..vendored import _lib\n", encoding="utf-8")
        planner = Planner(self.root, dict(self.config, dependents="direct"), {}, self.patterns)
        plan = planner.plan("auto", changed=["pkg/vendored/_lib/core.py"])
        self.assertEqual(self.paths(plan), ["pkg/stats", "pkg/tests"])

    def test_an_excluded_directory_cannot_be_selected_by_any_spelling(self):
        for name in ("vendored", "pkg/vendored", "pkg/vendored/_lib", "vendored/_lib", "_old_stack", "pkg/tools", "tools"):
            with self.subTest(name=name), self.assertRaisesRegex(PlanError, "excluded from test collection"):
                self.planner.plan("custom", select=name)

    def test_selecting_a_container_skips_the_excluded_directories_below_it(self):
        self.assertEqual(
            self.planner.resolve_selection("_ext/_sphinx"),
            ["pkg/_ext/_sphinx/_grid", "pkg/_ext/_sphinx/_learn", "pkg/_ext/_sphinx/tests"],
        )

    def test_an_excluded_container_contributes_nothing(self):
        planner = Planner(self.root, dict(self.config), {}, ("_ext",))
        self.assertFalse([unit for unit in planner.units() if unit.startswith("pkg/_ext")])
        self.assertFalse(planner.loose_tests("pkg/_ext"))
        self.assertEqual(planner.excluded_units(), ["pkg/_ext"])
        self.assertEqual(self.paths(planner.plan("auto", changed=["pkg/_ext/_sphinx/_grid/a.py"])), ["pkg/tests"])
        planner.plan("all")

    def test_a_planned_path_that_is_excluded_is_refused(self):
        # The invariant itself, independent of how a path got there.
        with self.assertRaisesRegex(PlanError, "excluded by 'norecursedirs': pkg/vendored"):
            self.planner._assert_collectable(["pkg/corpus", "pkg/vendored"])

    def test_every_plan_is_checked_before_it_is_returned(self):
        # A selection rule that went wrong must not reach pytest.
        class Faulty(Planner):
            def ancestor_tests(self, unit):
                return ["pkg/vendored"]

        faulty = Faulty(self.root, dict(self.config), {}, self.patterns)
        with self.assertRaisesRegex(PlanError, "excluded by 'norecursedirs': pkg/vendored"):
            faulty.plan("auto", changed=["pkg/corpus/a.py"])
        with self.assertRaisesRegex(PlanError, "excluded by 'norecursedirs': pkg/vendored"):
            faulty.plan("custom", select="corpus")

        class FaultyAll(Planner):
            def loose_tests(self, container=None):
                return ["pkg/tools/test_tool.py"]

        with self.assertRaisesRegex(PlanError, "excluded by 'norecursedirs'"):
            FaultyAll(self.root, dict(self.config), {}, self.patterns).plan("all")

    def test_without_patterns_nothing_is_excluded(self):
        planner = Planner(self.root, dict(self.config), {}, ())
        self.assertIn("pkg/vendored", planner.units())
        self.assertEqual(planner.excluded_units(), [])

    def test_the_command_line_reads_the_setting(self):
        (self.root / "pytest.ini").write_text("[pytest]\nnorecursedirs = docs tools pkg/vendored\n", encoding="utf-8")
        config = self.root / "plan.json"
        config.write_text(json.dumps(self.config), encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(plan_module.main(["--root", str(self.root), "--config", str(config), "units", "--excluded"]), 0)
        self.assertEqual(out.getvalue().split(), ["pkg/tools", "pkg/vendored"])
        base = ["--root", str(self.root), "--config", str(config)]
        self.assertEqual(main([*base, "plan", "--mode", "custom", "--select", "vendored"]), 2)
        # An explicit empty value means "this project has no pytest.ini".
        self.assertEqual(main([*base, "--pytest-ini", "", "plan", "--mode", "custom", "--select", "vendored"]), 0)
        self.assertEqual(main([*base, "--pytest-ini", "missing.ini", "plan", "--mode", "all"]), 2)
        (self.root / "pytest.ini").unlink()
        self.assertEqual(main([*base, "plan", "--mode", "all"]), 2)

    @unittest.skipUnless(importlib.util.find_spec("pytest"), "pytest is not installed")
    def test_pytest_itself_agrees(self):
        """Named paths collect exactly what a recursive run collects."""
        for path in sorted(self.root.rglob("test_*.py")):
            path.write_text("def test_it():\n    pass\n", encoding="utf-8")
        (self.root / "pytest.ini").write_text(
            "[pytest]\nnorecursedirs = " + " ".join(self.patterns) + "\n", encoding="utf-8",
        )

        def collected(*arguments):
            done = subprocess.run(
                [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *arguments],
                cwd=self.root, capture_output=True, text=True, check=False,
                env=dict(os.environ, PYTEST_ADDOPTS="", PYTHONDONTWRITEBYTECODE="1"),
            )
            return sorted(line for line in done.stdout.splitlines() if "::" in line)

        whole = collected("pkg")
        self.assertTrue(whole)
        self.assertEqual(sorted(line.split("::")[0] for line in whole), self.planner.collectable_test_files())
        self.assertEqual(collected(*self.paths(self.planner.plan("all"))), whole)
        # What the planner prevents: a named excluded path is collected.
        self.assertTrue(collected("pkg/vendored"))


class TestGit(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def git(self, *args):
        done = subprocess.run(
            ["git", "-C", str(self.root), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
            capture_output=True, text=True, check=True,
        )
        return done.stdout.strip()

    def test_changes_are_measured_from_the_merge_base_and_renames_give_both_names(self):
        self.git("init", "-q", "-b", "main")
        (self.root / "a.py").write_text("a\n" * 30, encoding="utf-8")
        (self.root / "kept name.py").write_text("k\n", encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "base")
        base = self.git("rev-parse", "HEAD")
        self.git("checkout", "-q", "-b", "feature")
        self.git("mv", "a.py", "b.py")
        (self.root / "new.py").write_text("n\n", encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "feature")
        head = self.git("rev-parse", "HEAD")
        self.git("checkout", "-q", "main")
        (self.root / "main_only.py").write_text("m\n", encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "main moved on")
        tip = self.git("rev-parse", "HEAD")
        self.assertEqual(plan_module.changed_files_from_git(self.root, tip, head), ["a.py", "b.py", "new.py"])
        self.assertEqual(plan_module.changed_files_from_git(self.root, base, base), [])

    def test_a_missing_or_zero_commit_is_an_error_for_the_caller_to_handle(self):
        self.git("init", "-q")
        for base, head in (("", "abc"), ("0" * 40, "abc"), ("abc", ""), ("deadbeef", "cafebabe")):
            with self.subTest(base=base, head=head):
                with self.assertRaises(PlanError):
                    plan_module.changed_files_from_git(self.root, base, head)


class TestCommandLine(Tree):
    def run_main(self, *args):
        config = self.root / "plan.json"
        config.write_text(json.dumps(self.config), encoding="utf-8")
        return ["--root", str(self.root), "--config", str(config), "--durations", str(self.root / "none.json"), *args]

    def test_outputs_for_a_workflow(self):
        output, summary, changed = self.root / "out.txt", self.root / "summary.md", self.root / "changed.txt"
        changed.write_text("pkg/corpus/a.py\n\n", encoding="utf-8")
        code = main(self.run_main(
            "plan", "--mode", "auto", "--changed-files", str(changed),
            "--github-output", str(output), "--summary", str(summary),
        ))
        self.assertEqual(code, 0)
        lines = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
        self.assertEqual((lines["run"], lines["mode"]), ("true", "auto"))
        matrix = json.loads(lines["matrix"])
        self.assertEqual(sorted(" ".join(row["paths"] for row in matrix["include"]).split()), ["pkg/corpus", "pkg/tests"])
        self.assertTrue(all(row["timeout_minutes"] == 350 for row in matrix["include"]))
        self.assertIn("`pkg/corpus` changed", summary.read_text(encoding="utf-8"))

    def test_a_skip_still_writes_a_valid_empty_matrix(self):
        output, changed = self.root / "out.txt", self.root / "changed.txt"
        changed.write_text("docs/index.rst\n", encoding="utf-8")
        self.assertEqual(main(self.run_main("plan", "--mode", "auto", "--changed-files", str(changed), "--github-output", str(output))), 0)
        lines = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
        self.assertEqual((lines["run"], json.loads(lines["matrix"])), ("false", {"include": []}))

    def test_auto_without_commits_runs_everything(self):
        output = self.root / "out.txt"
        self.assertEqual(main(self.run_main("plan", "--mode", "auto", "--github-output", str(output))), 0)
        self.assertIn("mode=all", output.read_text(encoding="utf-8"))

    def test_bad_input_exits_with_2(self):
        self.assertEqual(main(self.run_main("plan", "--mode", "custom", "--select", "nope")), 2)
        self.assertEqual(main(self.run_main("plan", "--mode", "all", "--max-shards", "0x4")), 2)
        self.assertEqual(main(self.run_main("plan", "--mode", "all", "--max-shards", "0")), 2)

    def test_max_shards_override(self):
        output = self.root / "out.txt"
        main(self.run_main("plan", "--mode", "all", "--max-shards", "2", "--github-output", str(output)))
        lines = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
        self.assertEqual(len(json.loads(lines["matrix"])["include"]), 2)


@unittest.skipUnless((REPOSITORY / "scikitplot").is_dir(), "not a scikit-plots checkout")
class TestShippedConfiguration(unittest.TestCase):
    """The real configuration against the real tree."""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(REPOSITORY / ".github" / "ci" / "test_plan.json")
        cls.norecursedirs = load_norecursedirs(REPOSITORY / cls.config.get("pytest_ini", "pytest.ini"))
        cls.planner = Planner(
            REPOSITORY,
            cls.config,
            load_durations(REPOSITORY / ".github" / "ci" / "test_durations.json"),
            cls.norecursedirs,
        )

    def test_every_configured_container_exists(self):
        for container in self.config["containers"]:
            with self.subTest(container=container):
                self.assertTrue((REPOSITORY / container).is_dir())

    def test_a_full_run_collects_every_test_file(self):
        plan = self.planner.plan("all")
        self.assertTrue(plan["run"])
        self.assertLessEqual(len(plan["shards"]), self.config["max_shards"])

    def test_the_two_examples_from_the_design(self):
        corpus = self.planner.plan("auto", changed=["scikitplot/corpus/_x.py"])
        self.assertEqual(sorted(p for s in corpus["shards"] for p in s["paths"]), ["scikitplot/corpus", "scikitplot/tests"])
        learn = self.planner.plan("auto", changed=["scikitplot/_externals/_sphinx_ext/_sphinx_ai_learn/_pages.py"])
        above = [
            "scikitplot/tests",
            "scikitplot/_externals/tests",
            "scikitplot/_externals/_sphinx_ext/tests",
        ]
        expected = ["scikitplot/_externals/_sphinx_ext/_sphinx_ai_learn"]
        expected += [path for path in above if (REPOSITORY / path).is_dir()]
        self.assertEqual(sorted(p for s in learn["shards"] for p in s["paths"]), sorted(expected))
        self.assertIn("scikitplot/tests", expected)

    def test_the_planner_and_the_workflow_trigger_a_full_run(self):
        for path in (".github/scripts/ci_test_plan.py", ".github/ci/test_plan.json", ".github/workflows/ci_codecov_test_coverage.yml", "pyproject.toml", "meson_cpu/main_config.h.in"):
            with self.subTest(path=path):
                self.assertEqual(self.planner.plan("auto", changed=[path])["mode"], "all")

    def test_documentation_only_changes_run_nothing(self):
        plan = self.planner.plan("auto", changed=["README.md", "docs/source/index.rst", "galleries/examples/a.py", "maintenances/x.md"])
        self.assertFalse(plan["run"])

    def test_the_project_excludes_something_so_the_setting_was_really_read(self):
        # Guards the wiring: a planner that silently fell back to pytest's
        # defaults would pass every other test in this class.
        self.assertNotEqual(self.norecursedirs, PYTEST_DEFAULT_NORECURSEDIRS)
        self.assertTrue(self.planner.excluded_units())

    def test_no_mode_hands_pytest_a_directory_the_project_excludes(self):
        excluded = self.planner.excluded_units()
        plans = [self.planner.plan("all")]
        plans += [self.planner.plan("auto", changed=[f"{unit}/__init__.py"]) for unit in excluded]
        for plan in plans:
            for shard in plan["shards"]:
                for path in shard["paths"]:
                    with self.subTest(path=path):
                        self.assertFalse(self.planner.is_excluded(path))
        for unit in excluded:
            with self.subTest(unit=unit), self.assertRaisesRegex(PlanError, "norecursedirs"):
                self.planner.plan("custom", select=unit)

    def test_a_full_run_is_exactly_what_a_recursive_run_collects(self):
        paths = [path for shard in self.planner.plan("all")["shards"] for path in shard["paths"]]
        for source in self.planner.collectable_test_files():
            covering = [path for path in paths if source == path or source.startswith(path + "/")]
            with self.subTest(source=source):
                self.assertEqual(len(covering), 1)


if __name__ == "__main__":
    unittest.main()
