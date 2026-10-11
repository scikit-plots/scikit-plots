# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for tools/maint_tools/check_path_lengths.py.

The last test is the gate: the repository itself must have no failing path.
It runs wherever the tooling tests run, so a too-long file name is caught on
the pull request that adds it, not by a Windows user's ``pip install``.
"""

from __future__ import annotations

import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "check_path_lengths.py"
SPEC = importlib.util.spec_from_file_location("check_path_lengths", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

REPO = Path(__file__).resolve().parents[3]
PLANE = "maintenances/_externals/_sphinx_ext/_sphinx_ai_assistant/_maintenance"


def _tree(root: Path, files: dict) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _libs(root: Path, *names: str) -> None:
    for name in names:
        _tree(root, {f"libs/{name}/pyproject.toml": f'[project]\nname = "{name}"\nversion = "1"\n'})


class BudgetTests(unittest.TestCase):
    def test_budget_follows_the_longest_distribution_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _libs(root, "scikit-plots-skinny", "scikit-plots-cleanprompt")
            budget = MODULE.compute_budget(root)
            self.assertEqual(budget.longest_distribution, "scikit-plots-cleanprompt")
            self.assertEqual(budget.prefix_length, len(budget.prefix_example))
            self.assertEqual(budget.budget, 259 - budget.prefix_length)

    def test_the_reported_failure_is_reproduced_exactly(self):
        # User "devel", scikit-plots-skinny: a 108-character prefix. Paths of
        # 152 and 166 characters failed; 149 checked out.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = ["x" * 149, "y" * 152, "z" * 166]
            report = MODULE.check(root, files, baseline={}, prefix_length=108)
            self.assertEqual(sorted(v.length for v in report.failing), [152, 166])


class ShortenTests(unittest.TestCase):
    def test_shared_affixes_and_the_directory_name_are_dropped(self):
        directory = f"{PLANE}/history/fresh_chat"
        name = MODULE.shorten(directory, "FRESH_CHAT_LIFECYCLE_PRIVACY_CLOSURE_HANDOFF.md", 131, 2, 1)
        self.assertEqual(name, "LIFECYCLE_PRIVACY_CLOSURE.md")

    def test_filler_then_trailing_words_go_and_the_identifier_stays(self):
        directory = f"{PLANE}/checkpoints"
        name = MODULE.shorten(
            directory, "R173T8_WORKING_FILE_BINDING_AND_STALE_RESPONSE_PROTECTION.md", 131
        )
        self.assertTrue(name.startswith("R173T8_WORKING"))
        self.assertNotIn("_AND_", name)
        self.assertLessEqual(len(f"{directory}/{name}"), 131)

    def test_impossible_names_are_refused(self):
        self.assertIsNone(MODULE.shorten("d" * 125, "B1_LONGWORD.md", 131))


class PlanAndApplyTests(unittest.TestCase):
    def _repo(self, root: Path) -> list:
        long_a = "FRESH_CHAT_" + "ALPHA_" * 8 + "HANDOFF.md"
        long_b = "FRESH_CHAT_BETA_HANDOFF.md"
        _tree(
            root,
            {
                f"maintenances/x/history/fresh_chat/{long_a}": "# Alpha\n",
                f"maintenances/x/history/fresh_chat/{long_b}": "# Beta\n",
                "maintenances/x/REGISTRY.md": f"see `{long_a}` and history/fresh_chat/{long_b}\n",
            },
        )
        return MODULE.tracked_files(root)

    def test_a_directory_is_renamed_consistently_and_references_follow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self._repo(root)
            report = MODULE.check(root, files, baseline={}, prefix_length=180)
            renames, refused = MODULE.plan(root, report, files)
            self.assertEqual(refused, [])
            new_names = sorted(r.new.rsplit("/", 1)[-1] for r in renames)
            self.assertIn("BETA.md", new_names)  # the short sibling follows the convention
            MODULE.apply_renames(root, renames, files)
            registry = (root / "maintenances/x/REGISTRY.md").read_text(encoding="utf-8")
            self.assertNotIn("FRESH_CHAT_", registry)
            self.assertIn("history/fresh_chat/BETA.md", registry)
            after = MODULE.check(root, MODULE.tracked_files(root), baseline={}, prefix_length=180)
            self.assertTrue(after.ok)

    def test_code_outside_the_maintenance_planes_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _tree(
                root,
                {
                    "maintenances/x/OLD_NOTE.md": "",
                    "maintenances/x/tool.py": "NAME = 'OLD_NOTE.md'\n",
                    "tools/test_x.py": "FIXTURE = 'OLD_NOTE.md'\n",
                    "docs/index.rst": "OLD_NOTE.md\n",
                },
            )
            files = MODULE.tracked_files(root)
            MODULE.apply_renames(root, [MODULE.Rename("maintenances/x/OLD_NOTE.md", "maintenances/x/N.md")], files)
            read = lambda rel: (root / rel).read_text(encoding="utf-8")  # noqa: E731
            self.assertEqual(read("tools/test_x.py"), "FIXTURE = 'OLD_NOTE.md'\n")
            self.assertEqual(read("maintenances/x/tool.py"), "NAME = 'N.md'\n")
            self.assertEqual(read("docs/index.rst"), "N.md\n")

    def test_a_shared_name_is_not_rewritten_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _tree(root, {"a/NOTE.md": "", "b/NOTE.md": "", "c/README.md": "NOTE.md\n"})
            files = MODULE.tracked_files(root)
            MODULE.apply_renames(root, [MODULE.Rename("a/NOTE.md", "a/N.md")], files)
            self.assertEqual((root / "c/README.md").read_text(encoding="utf-8"), "NOTE.md\n")
            self.assertTrue((root / "a/N.md").is_file())


@unittest.skipIf(shutil.which("git") is None, "git is not installed")
class GitCheckoutTests(unittest.TestCase):
    def test_tracked_files_and_renames_go_through_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = "FRESH_CHAT_" + "DELTA_" * 20 + "HANDOFF.md"
            _tree(root, {f"maintenances/x/fresh_chat/{name}": "", "untracked.md": ""})
            git = ["git", "-C", str(root), "-c", "user.email=t@example.invalid", "-c", "user.name=t"]
            subprocess.run([*git, "init", "-q"], check=True)
            subprocess.run([*git, "add", f"maintenances/x/fresh_chat/{name}"], check=True)
            files = MODULE.tracked_files(root)
            self.assertEqual(files, [f"maintenances/x/fresh_chat/{name}"])  # untracked is not cloned
            report = MODULE.check(root, files, baseline={})
            renames, _ = MODULE.plan(root, report, files)
            MODULE.apply_renames(root, renames, files)
            self.assertEqual(MODULE.tracked_files(root), [renames[0].new])


class BaselineTests(unittest.TestCase):
    def test_known_paths_pass_and_stale_entries_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            long_path = "t" * 200
            report = MODULE.check(root, [long_path], baseline={long_path: "debt"})
            self.assertTrue(report.ok)
            stale = MODULE.check(root, ["short"], baseline={long_path: "debt"})
            self.assertFalse(stale.ok)
            self.assertEqual(stale.stale_baseline, [long_path])


class CommandLineTests(unittest.TestCase):
    def test_bad_prefix_is_a_usage_error(self):
        err = io.StringIO()
        status = MODULE.main(["budget", "--prefix-length", "0"], stdout=io.StringIO(), stderr=err)
        self.assertEqual(status, 2)

    def test_fix_without_apply_changes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = "FRESH_CHAT_" + "GAMMA_" * 20 + "HANDOFF.md"
            _tree(root, {f"maintenances/x/fresh_chat/{name}": ""})
            out = io.StringIO()
            MODULE.main(["fix", "--root", str(root)], stdout=out, stderr=io.StringIO())
            self.assertIn("dry run", out.getvalue())
            self.assertTrue((root / f"maintenances/x/fresh_chat/{name}").is_file())

    def test_repository_has_no_failing_path(self):
        out = io.StringIO()
        status = MODULE.main(
            ["check", "--root", str(REPO), "--format", "json"], stdout=out, stderr=io.StringIO()
        )
        report = json.loads(out.getvalue())
        failing = [v["path"] for v in report["violations"] if not v["known"]]
        self.assertEqual((status, failing, report["stale_baseline"]), (0, [], []))


if __name__ == "__main__":
    unittest.main()
