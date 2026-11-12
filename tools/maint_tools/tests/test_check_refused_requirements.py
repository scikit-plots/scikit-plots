# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for tools/maint_tools/check_refused_requirements.py.

The last test is the gate: no requirement list in this repository names a
refused distribution, so adding ``pdfminer`` next to ``pdfminer.six`` again
fails here in a second instead of breaking an install of the extra.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "check_refused_requirements.py"
SPEC = importlib.util.spec_from_file_location("check_refused_requirements", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

REPO = Path(__file__).resolve().parents[3]

PYPROJECT = """\
[build-system]
requires = ["setuptools>=68"{build}]

[project]
name = "demo"
dependencies = ["numpy>=1.20"{dependencies}]

[project.optional-dependencies]
corpus = ["pypdf", "pdfminer.six"{extra}]
other = ["lxml"]

[dependency-groups]
dev = ["pytest"{group}, {{include-group = "x"}}]
x = []
"""


def _repo(root: Path, text: str, libs=None, requirements=None) -> None:
    (root / "pyproject.toml").write_text(text, encoding="utf-8")
    for name, body in (libs or {}).items():
        path = root / "libs" / name / "pyproject.toml"
        path.parent.mkdir(parents=True)
        path.write_text(body, encoding="utf-8")
    for name, body in (requirements or {}).items():
        path = root / "requirements" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(body, encoding="utf-8")


def _pyproject(build="", dependencies="", extra="", group="") -> str:
    return PYPROJECT.format(
        build=build, dependencies=dependencies, extra=extra, group=group
    )


class RequirementNameTests(unittest.TestCase):
    def test_names_are_normalized(self):
        cases = {
            "pdfminer": "pdfminer",
            "PDFMiner": "pdfminer",
            "pdfminer.six": "pdfminer-six",
            "pdfminer_six": "pdfminer-six",
            "pdfminer-six": "pdfminer-six",
            "  pdfminer>=2019": "pdfminer",
            "pdfminer==20191125": "pdfminer",
            "pdfminer ; python_version < '3.9'": "pdfminer",
            "pdfminer[extra]": "pdfminer",
            "pdfminer @ https://example.invalid/p.tar.gz": "pdfminer",
            "pdfminer  # the import name": "pdfminer",
            "pdfminer.six>=20221105; extra == 'corpus'": "pdfminer-six",
            "numpy>=1.20,!=1.24.0": "numpy",
        }
        for requirement, expected in cases.items():
            with self.subTest(requirement=requirement):
                self.assertEqual(MODULE.requirement_name(requirement), expected)

    def test_lines_that_are_not_requirements_have_no_name(self):
        for line in (
            "",
            "   ",
            "# pdfminer",
            "-r pdfminer.txt",
            "--index-url https://example.invalid/pdfminer",
            "https://example.invalid/pdfminer.tar.gz",
        ):
            with self.subTest(line=line):
                self.assertIsNone(MODULE.requirement_name(line))


class FindingsTests(unittest.TestCase):
    def test_a_clean_tree_has_no_finding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _repo(root, _pyproject(), requirements={"corpus.txt": "pdfminer.six\n"})
            self.assertEqual(MODULE.findings(root), [])

    def test_every_kind_of_list_in_a_pyproject_is_read(self):
        places = {
            "build": "build-system.requires",
            "dependencies": "project.dependencies",
            "extra": "project.optional-dependencies.corpus",
            "group": "dependency-groups.dev",
        }
        for argument, where in places.items():
            with self.subTest(where=where), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _repo(root, _pyproject(**{argument: ', "pdfminer"'}))
                (found,) = MODULE.findings(root)
                self.assertEqual(found.path, "pyproject.toml")
                self.assertEqual(found.where, where)
                self.assertEqual(found.requirement, "pdfminer")
                self.assertEqual(found.name, "pdfminer")

    def test_libs_and_requirement_files_are_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _repo(
                root,
                _pyproject(),
                libs={"corpus": _pyproject(extra=', "PDFMiner>=2019"')},
                requirements={
                    "corpus.txt": "# pdfminer\nlxml\npdfminer\npdfminer.six\n",
                    "notes.md": "pdfminer\n",
                },
            )
            found = MODULE.findings(root)
        self.assertEqual(
            [(f.path, f.where, f.requirement) for f in found],
            [
                (
                    "libs/corpus/pyproject.toml",
                    "project.optional-dependencies.corpus",
                    "PDFMiner>=2019",
                ),
                ("requirements/corpus.txt", "line 3", "pdfminer"),
            ],
        )

    def test_every_refusal_names_another_distribution_and_a_reason(self):
        for name, refusal in MODULE.REFUSED.items():
            with self.subTest(name=name):
                self.assertEqual(name, MODULE.normalize(name))
                self.assertNotEqual(MODULE.normalize(refusal.use), name)
                self.assertNotIn(MODULE.normalize(refusal.use), MODULE.REFUSED)
                self.assertTrue(refusal.reason.strip())


class MainTests(unittest.TestCase):
    def _run(self, root: Path):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = MODULE.main([str(root)])
        return code, out.getvalue(), err.getvalue()

    def test_a_refused_requirement_exits_1_and_says_what_to_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _repo(root, _pyproject(extra=', "pdfminer"'))
            code, out, _ = self._run(root)
        self.assertEqual(code, 1)
        self.assertIn(
            "pyproject.toml: project.optional-dependencies.corpus: 'pdfminer'", out
        )
        self.assertIn("require 'pdfminer.six' instead", out)
        self.assertIn(MODULE.REFUSED["pdfminer"].reason, out)
        self.assertIn("1 refused requirement(s).", out)

    def test_a_clean_tree_exits_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _repo(root, _pyproject(), requirements={"a.txt": "lxml\n"})
            code, out, _ = self._run(root)
        self.assertEqual(code, 0)
        self.assertIn("1 pyproject file(s) and 1 requirement file(s)", out)

    def test_a_missing_or_broken_pyproject_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code, _, err = self._run(root)
            self.assertEqual(code, 2)
            self.assertIn("no pyproject.toml", err)
            (root / "pyproject.toml").write_text("[project\n", encoding="utf-8")
            code, _, err = self._run(root)
            self.assertEqual(code, 2)
            self.assertIn("error:", err)

    def test_this_repository_has_no_refused_requirement(self):
        self.assertEqual(MODULE.report(MODULE.findings(REPO)), [])


if __name__ == "__main__":
    unittest.main()
