# SPDX-License-Identifier: BSD-3-Clause
"""Focused tests for tools/maint_tools/generate_towncrier_sections.py."""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "generate_towncrier_sections.py"
SPEC = importlib.util.spec_from_file_location("generate_towncrier_sections", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TowncrierSectionsTests(unittest.TestCase):
    def test_current_repository_passes_check_and_is_already_synchronized(self):
        repo_root = Path(__file__).resolve().parents[3]

        stdout = io.StringIO()
        stderr = io.StringIO()
        status = MODULE.main(
            ["--repo-root", str(repo_root), "check", "--json"],
            stdout=stdout,
            stderr=stderr,
        )
        self.assertEqual(status, 0, stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["counts"]["expected_sections"], 103)

        stdout = io.StringIO()
        status = MODULE.main(
            ["--repo-root", str(repo_root), "sync", "--prune-empty", "--json"],
            stdout=stdout,
            stderr=stderr,
        )
        self.assertEqual(status, 0, stderr.getvalue())
        plan = json.loads(stdout.getvalue())
        self.assertFalse(plan["applied"])
        self.assertFalse(plan["changed"])

    def test_owner_discovery_honors_package_rules_and_explicit_exclusions(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            (root / "scikitplot" / "public").mkdir(parents=True)
            (root / "scikitplot" / "public" / "__init__.py").write_text("")
            (root / "scikitplot" / "not_a_package").mkdir()
            (root / "scikitplot" / "tests").mkdir()
            (root / "scikitplot" / "tests" / "__init__.py").write_text("")
            (root / "libs" / "native").mkdir(parents=True)
            (root / "tools" / "maint_tools").mkdir(parents=True)
            (root / "tools" / "zz_yanked").mkdir()

            policy = MODULE.Policy(
                python_package_roots=("scikitplot",),
                directory_roots=("libs", "tools"),
                root_owner_sections=("scikitplot", "tools"),
                exclude_owner_sections=("scikitplot.tests", "tools.zz_yanked"),
                cross_cutting_sections=("documentation",),
                nested_owner_sections=(),
            )
            paths = {
                section.path
                for section in MODULE._discover_owner_sections(root, policy)
            }

            self.assertEqual(
                paths,
                {
                    "scikitplot",
                    "scikitplot.public",
                    "libs.native",
                    "tools",
                    "tools.maint_tools",
                },
            )

    def test_nested_owner_discovery_keeps_base_owner_and_supports_deep_children(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            packages = [
                "scikitplot",
                "scikitplot/externals",
                "scikitplot/externals/vendor",
                "scikitplot/externals/vendor/deep",
            ]
            for package in packages:
                path = root / package
                path.mkdir(parents=True, exist_ok=True)
                (path / "__init__.py").write_text("")

            policy = MODULE.Policy(
                python_package_roots=("scikitplot",),
                directory_roots=(),
                root_owner_sections=("scikitplot",),
                exclude_owner_sections=(),
                cross_cutting_sections=(),
                nested_owner_sections=(("scikitplot.externals", ("vendor.deep",)),),
            )
            paths = {
                section.path
                for section in MODULE._discover_owner_sections(root, policy)
            }

            self.assertEqual(
                paths,
                {
                    "scikitplot",
                    "scikitplot.externals",
                    "scikitplot.externals.vendor.deep",
                },
            )

    def test_nested_owner_discovery_rejects_missing_or_non_package_paths(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            for package in ("scikitplot", "scikitplot/externals"):
                path = root / package
                path.mkdir(parents=True, exist_ok=True)
                (path / "__init__.py").write_text("")

            policy = MODULE.Policy(
                python_package_roots=("scikitplot",),
                directory_roots=(),
                root_owner_sections=("scikitplot",),
                exclude_owner_sections=(),
                cross_cutting_sections=(),
                nested_owner_sections=(("scikitplot.externals", ("missing",)),),
            )

            with self.assertRaises(SystemExit):
                MODULE._discover_owner_sections(root, policy)

    def test_nested_owner_discovery_rejects_redundant_default_owner(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            for package in ("scikitplot", "scikitplot/externals"):
                path = root / package
                path.mkdir(parents=True, exist_ok=True)
                (path / "__init__.py").write_text("")

            policy = MODULE.Policy(
                python_package_roots=("scikitplot",),
                directory_roots=(),
                root_owner_sections=("scikitplot",),
                exclude_owner_sections=(),
                cross_cutting_sections=(),
                nested_owner_sections=(("scikitplot", ("externals",)),),
            )

            with self.assertRaises(SystemExit):
                MODULE._discover_owner_sections(root, policy)

    def test_prune_guard_accepts_only_empty_or_gitkeep_only_directories(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            empty = root / "empty"
            empty.mkdir()
            self.assertTrue(MODULE._is_prunable_directory(empty))

            gitkeep = root / "gitkeep"
            gitkeep.mkdir()
            (gitkeep / ".gitkeep").write_text("")
            self.assertTrue(MODULE._is_prunable_directory(gitkeep))

            populated = root / "populated"
            populated.mkdir()
            (populated / "123.fix.rst").write_text("- Keep me.\n")
            self.assertFalse(MODULE._is_prunable_directory(populated))

    def test_managed_block_replacement_is_bounded_by_markers(self):
        source = (
            "before\n"
            "  # BEGIN AUTO-GENERATED TOWNCRIER OWNER SECTIONS\n"
            "  old\n"
            "  # END AUTO-GENERATED TOWNCRIER OWNER SECTIONS\n"
            "after\n"
        )
        replacement = (
            "  # BEGIN AUTO-GENERATED TOWNCRIER OWNER SECTIONS\n"
            "  new\n"
            "  # END AUTO-GENERATED TOWNCRIER OWNER SECTIONS"
        )
        result = MODULE._replace_managed_owner_block(source, replacement)
        self.assertEqual(result, "before\n" + replacement + "\nafter\n")


if __name__ == "__main__":
    unittest.main()
