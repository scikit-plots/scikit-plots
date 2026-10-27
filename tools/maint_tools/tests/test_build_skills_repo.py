# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tools/maint_tools/build_skills_repo.py (the released plugin, Rule 166)."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "build_skills_repo.py"
SPEC = importlib.util.spec_from_file_location("build_skills_repo", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
REPO = Path(__file__).resolve().parents[3]


class VersionTests(unittest.TestCase):
    def test_releases_are_accepted(self):
        for version in ("0.5.0", "1.10.2", "0.4.0.post11"):
            self.assertEqual(MODULE.check_version(version), version)

    def test_anything_else_is_refused(self):
        for version in ("0.5.dev0", "v0.5.0", "0.5", "0.5.0rc1", "0.5.0+git", "", None):
            with self.assertRaises(ValueError):
                MODULE.check_version(version)


class BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = MODULE.render(REPO, "0.5.0")

    def _built(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name)
        MODULE.apply(out, self.files)
        return out

    def test_manifest_carries_version_icon_and_stable_repository(self):
        manifest = json.loads(self.files[".claude-plugin/plugin.json"])
        self.assertEqual(manifest["version"], "0.5.0")
        self.assertEqual(manifest["icon"], "./assets/icon.png")
        self.assertEqual(
            manifest["repository"], "https://github.com/scikit-plots/skills"
        )
        self.assertTrue(self.files["assets/icon.png"].startswith(MODULE.PNG_MAGIC))

    def test_marketplace_lists_the_root_plugin_only(self):
        market = json.loads(self.files[".claude-plugin/marketplace.json"])
        self.assertEqual([p["source"] for p in market["plugins"]], ["./"])
        self.assertEqual(market["metadata"]["version"], "0.5.0")

    def test_every_skill_and_agent_of_the_plugin_is_copied(self):
        plugin = REPO / "plugins/scikitplot"
        for base in ("skills", "agents"):
            expected = {
                p.relative_to(plugin).as_posix()
                for p in (plugin / base).rglob("*")
                if p.is_file()
            }
            self.assertEqual(
                {k for k in self.files if k.startswith(base + "/")}, expected
            )

    def test_built_tree_passes_the_static_checks(self):
        self.assertEqual(MODULE.validate(self._built()), [])

    def test_build_is_idempotent_and_removes_stale_files(self):
        out = self._built()
        self.assertEqual(MODULE.diff(out, self.files), [])
        (out / "skills/scikitplot-gone").mkdir()
        (out / "skills/scikitplot-gone/SKILL.md").write_text("x", encoding="utf-8")
        (out / ".gitignore").write_text("x\n", encoding="utf-8")
        self.assertEqual(
            MODULE.apply(out, self.files), ["- skills/scikitplot-gone/SKILL.md"]
        )
        self.assertFalse((out / "skills/scikitplot-gone").exists())
        self.assertTrue(
            (out / ".gitignore").is_file(), "unmanaged paths are left alone"
        )

    def test_validate_reports_directory_blockers(self):
        out = self._built()
        (out / ".gitattributes").write_text("* text=auto\n", encoding="utf-8")
        (out / "README.md").write_text("Too short.\n", encoding="utf-8")
        manifest = json.loads(
            (out / ".claude-plugin/plugin.json").read_text(encoding="utf-8")
        )
        manifest["version"] = "0.5.dev0"
        (out / ".claude-plugin/plugin.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        errors = "\n".join(MODULE.validate(out))
        for needle in (".gitattributes", "README.md", "0.5.dev0"):
            self.assertIn(needle, errors)

    def test_check_command_fails_on_a_stale_target(self):
        out = self._built()
        (out / "README.md").write_text("changed\n", encoding="utf-8")
        self.assertEqual(
            MODULE.main(["check", "--out", str(out), "--version", "0.5.0"]), 1
        )
        self.assertEqual(
            MODULE.main(["build", "--out", str(out), "--version", "0.5.0"]), 0
        )
        self.assertEqual(
            MODULE.main(["check", "--out", str(out), "--version", "0.5.0"]), 0
        )


if __name__ == "__main__":
    unittest.main()


class BranchBuildTests(unittest.TestCase):
    """CI-6: main and maintenance/X.Y.x are built without a version."""

    @classmethod
    def setUpClass(cls):
        cls.files = MODULE.render(REPO, None)

    def test_no_version_anywhere(self):
        manifest = json.loads(self.files[".claude-plugin/plugin.json"])
        market = json.loads(self.files[".claude-plugin/marketplace.json"])
        self.assertNotIn("version", manifest)
        self.assertNotIn("version", market["metadata"])

    def test_same_files_as_a_release_but_the_manifests(self):
        release = MODULE.render(REPO, "0.5.0")
        self.assertEqual(set(release), set(self.files))
        differ = {rel for rel in release if release[rel] != self.files[rel]}
        self.assertEqual(
            differ, {".claude-plugin/plugin.json", ".claude-plugin/marketplace.json"}
        )

    def test_branch_tree_passes_unversioned_and_fails_as_a_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            MODULE.apply(out, self.files)
            self.assertEqual(MODULE.validate(out, versioned=False), [])
            self.assertTrue(
                any("'version' is not set" in e for e in MODULE.validate(out, versioned=True))
            )

    def test_release_tree_fails_as_a_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            MODULE.apply(out, MODULE.render(REPO, "0.5.0"))
            self.assertTrue(
                any("sets no 'version'" in e for e in MODULE.validate(out, versioned=False))
            )

    def test_the_command_needs_exactly_one_of_version_or_unversioned(self):
        with tempfile.TemporaryDirectory() as tmp:
            for argv in (["check", "--out", tmp], ["check", "--out", tmp, "--version", "0.5.0", "--unversioned"]):
                with self.assertRaises(SystemExit):
                    MODULE.main(argv)

    def test_shared_skills_ship_in_the_published_plugin(self):
        shared = REPO / "plugins/_shared/skills"
        for path in shared.rglob("*"):
            if path.is_file():
                rel = "skills/" + path.relative_to(shared).as_posix()
                self.assertEqual(self.files[rel], path.read_bytes(), rel)
