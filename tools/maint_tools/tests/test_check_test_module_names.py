# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for tools/maint_tools/check_test_module_names.py.

The last test is the gate: ``scikitplot/`` itself must have no clash, so a
second ``test_sphinx_build.py`` fails here in seconds instead of stopping the
test shard at collection after a long install.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "check_test_module_names.py"
SPEC = importlib.util.spec_from_file_location("check_test_module_names", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

REPO = Path(__file__).resolve().parents[3]


def _tree(root: Path, files: list[str]) -> None:
    for rel in files:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


class ModuleNameTests(unittest.TestCase):
    def test_bare_directories_give_bare_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _tree(root, ["a/tests/test_x.py", "b/tests/test_x.py"])
            found = MODULE.clashes(root)
        self.assertEqual(list(found), ["test_x"])
        self.assertEqual(len(found["test_x"]), 2)

    def test_packages_give_dotted_names_and_do_not_clash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _tree(
                root,
                [
                    "a/__init__.py",
                    "a/tests/__init__.py",
                    "a/tests/test_x.py",
                    "b/__init__.py",
                    "b/tests/__init__.py",
                    "b/tests/test_x.py",
                ],
            )
            self.assertEqual(MODULE.clashes(root), {})
            self.assertEqual(
                MODULE.module_name(root / "a/tests/test_x.py"), "a.tests.test_x"
            )

    def test_same_package_path_still_clashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _tree(
                root,
                [
                    "a/tests/__init__.py",
                    "a/tests/test_x.py",
                    "b/tests/__init__.py",
                    "b/tests/test_x.py",
                ],
            )
            # Both resolve to "tests.test_x" because neither a/ nor b/ is a package.
            self.assertEqual(
                MODULE.module_name(root / "b/tests/test_x.py"), "tests.test_x"
            )
            self.assertEqual(list(MODULE.clashes(root)), ["tests.test_x"])

    def test_suffix_form_and_non_tests_are_classified(self):
        self.assertTrue(MODULE.is_test_file("thing_test.py"))
        self.assertTrue(MODULE.is_test_file("test_thing.py"))
        self.assertFalse(MODULE.is_test_file("conftest.py"))
        self.assertFalse(MODULE.is_test_file("test_thing.txt"))

    def test_main_reports_and_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _tree(root, ["a/test_y.py", "b/test_y.py"])
            out = io.StringIO()
            with redirect_stdout(out):
                status = MODULE.main([str(root)])
        self.assertEqual(status, 1)
        self.assertIn("CLASH test_y", out.getvalue())


class RepositoryGate(unittest.TestCase):
    def test_scikitplot_has_no_test_module_name_clash(self):
        root = REPO / "scikitplot"
        if not root.is_dir():
            self.skipTest("no scikitplot/ in this checkout")
        self.assertEqual(MODULE.clashes(root), {})


if __name__ == "__main__":
    unittest.main()
