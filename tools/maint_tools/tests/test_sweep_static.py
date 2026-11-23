# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tools/maint_tools/sweep_static.py (risk patterns in first-party code)."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "sweep_static.py"
SPEC = importlib.util.spec_from_file_location("sweep_static", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

#: One file holding one example of (nearly) every category, line by line.
RISKY = textwrap.dedent(
    """
    import hashlib, os, pickle, subprocess, tempfile, yaml, requests
    import xml.etree.ElementTree as ET
    from lxml import etree
    from pathlib import Path
    from urllib.request import urlopen


    def everything(s, tar, u, x, a, b):
        open("f")
        open("f", "rb")
        open("f", encoding="utf-8")
        Path("x").read_text()
        pickle.load(s)  # noqa: S301
        yaml.load(s)
        eval("1")
        subprocess.run("ls", shell=True)
        os.system("ls")
        tempfile.mktemp()
        tar.extractall(u)
        requests.get(u)
        urlopen(u)
        hashlib.md5(b"")
        ET.fromstring(s)
        etree.fromstring(s)
        print("x")
        assert x
        s.removeprefix("a")
        zip(a, b, strict=True)
        isinstance(x, int | str)
        try:
            pass
        except Exception:
            pass
        try:
            pass
        except:
            x = 1


    def mutable(a=[]):
        return a


    def annotated(x: list[int]) -> int | None:
        return None
    """
)


class _Repo:
    def __init__(self, files: dict) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        for rel, text in files.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    def close(self) -> None:
        self._tmp.cleanup()


class SweepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = _Repo(
            {
                "scikitplot/__init__.py": "",
                "scikitplot/alpha/mod.py": RISKY,
                "scikitplot/alpha/tests/test_x.py": "eval('1')\n",
                "scikitplot/alpha/tests/helpers.py": "eval('1')\n",
                "scikitplot/alpha/conftest.py": "eval('1')\n",
                "scikitplot/externals/vendored.py": "eval('1')\n",
                "scikitplot/_externals/_sphinx_ext/_myext/x.py": "def f():\n    print(1)\n",
                "scikitplot/top.py": "def f():\n    assert True\n",
            }
        )

    def tearDown(self) -> None:
        self.repo.close()

    def categories(self, **kwargs) -> dict:
        found = {}
        for f in MODULE.sweep(self.repo.root, **kwargs):
            found.setdefault(f["category"], []).append(f)
        return found

    def test_every_category_of_the_risky_file_is_found_once(self) -> None:
        found = self.categories()
        expected = {
            "open-without-encoding", "path-text-without-encoding", "pickle-load",
            "yaml-unsafe-load", "eval-exec", "subprocess-shell", "os-system",
            "tempfile-mktemp", "extractall-without-filter", "requests-without-timeout",
            "urlopen-without-timeout", "weak-hash", "stdlib-xml-parse",
            "print-in-library", "assert-in-library", "floor-str-removeprefix",
            "floor-zip-strict", "floor-isinstance-union", "silent-broad-except",
            "bare-except", "mutable-default", "floor-annotation",
        }
        self.assertEqual(set(found), expected)
        mod = [f for c in found.values() for f in c if f["file"] == "scikitplot/alpha/mod.py"]
        by_category = {}
        for f in mod:
            by_category.setdefault(f["category"], []).append(f)
        self.assertTrue(all(len(v) == 1 for v in by_category.values()), by_category)

    def test_lxml_and_binary_or_encoded_opens_are_not_findings(self) -> None:
        found = self.categories()
        self.assertEqual([f["code"] for f in found["stdlib-xml-parse"]], ["ET.fromstring(s)"])
        self.assertEqual([f["code"] for f in found["open-without-encoding"]], ['open("f")'])

    def test_tests_conftest_and_vendored_code_are_left_out(self) -> None:
        files = {f["file"] for f in MODULE.sweep(self.repo.root)}
        self.assertFalse(any("tests/" in f or "conftest" in f or "externals/vendored" in f for f in files))
        self.assertNotIn("scikitplot/alpha/tests/helpers.py", files)

    def test_areas(self) -> None:
        areas = {f["file"]: f["area"] for f in MODULE.sweep(self.repo.root)}
        self.assertEqual(areas["scikitplot/alpha/mod.py"], "alpha")
        self.assertEqual(areas["scikitplot/_externals/_sphinx_ext/_myext/x.py"], "_sphinx_ext/_myext")
        self.assertEqual(areas["scikitplot/top.py"], "(package)")

    def test_a_matching_noqa_acknowledges_a_finding(self) -> None:
        (pickled,) = self.categories()["pickle-load"]
        self.assertTrue(pickled["acknowledged"])
        self.assertTrue(MODULE._acknowledged("x  # noqa", "eval-exec"))
        self.assertFalse(MODULE._acknowledged("x  # noqa: E501", "eval-exec"))
        self.assertTrue(MODULE._acknowledged("x  # noqa: E501, S307", "eval-exec"))

    def test_the_floor_decides_which_constructs_are_findings(self) -> None:
        floor = "floor-isinstance-union floor-zip-strict floor-str-removeprefix floor-annotation".split()
        at_39 = self.categories(floor=(3, 9))
        self.assertNotIn("floor-str-removeprefix", at_39)
        self.assertIn("floor-isinstance-union", at_39)
        self.assertEqual(at_39["floor-annotation"][0]["detail"], "annotated")
        at_310 = self.categories(floor=(3, 10))
        self.assertFalse(set(floor) & set(at_310))

    def test_syntax_the_floor_cannot_parse(self) -> None:
        source = "def f(x):\n    match x:\n        case 1:\n            return 1\n"
        repo = _Repo({"scikitplot/__init__.py": "", "scikitplot/m.py": source})
        try:
            (finding,) = MODULE.sweep(repo.root)
            self.assertEqual(finding["category"], "floor-syntax")
            self.assertIn(finding["line"], (2, 3, 4))  # where the parser reports it varies
            if sys.version_info >= (3, 10):  # an older parser refuses it at any floor
                self.assertEqual(MODULE.sweep(repo.root, floor=(3, 10)), [])
        finally:
            repo.close()

    def test_a_tree_without_the_package_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.sweep(self.repo.root / "scikitplot")

    def test_two_sweeps_are_identical_and_sorted(self) -> None:
        one, two = MODULE.sweep(self.repo.root), MODULE.sweep(self.repo.root)
        self.assertEqual(one, two)
        self.assertEqual(one, sorted(one, key=lambda f: (f["file"], f["line"], f["category"])))


class CommandLineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = _Repo({"scikitplot/__init__.py": "", "scikitplot/alpha/mod.py": RISKY})

    def tearDown(self) -> None:
        self.repo.close()

    def run_main(self, *args: str) -> tuple:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = MODULE.main(["--root", str(self.repo.root), *args])
        return code, out.getvalue(), err.getvalue()

    def test_summary_list_and_json(self) -> None:
        code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("pickle-load", out)
        self.assertRegex(out, r"pickle-load +new +0 +acknowledged +1")
        code, out, _ = self.run_main("--list", "eval-exec")
        self.assertEqual(out.strip(), 'scikitplot/alpha/mod.py:16: eval("1")')
        target = self.repo.root / "s.json"
        self.run_main("--json", str(target))
        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual((data["format"], data["floor"]), (1, "3.8"))
        self.assertTrue(data["findings"])

    def test_fail_on_counts_only_unacknowledged_findings(self) -> None:
        self.assertEqual(self.run_main("--fail-on", "pickle-load")[0], 0)
        code, _, err = self.run_main("--fail-on", "eval-exec")
        self.assertEqual(code, 1)
        self.assertIn("1 unacknowledged finding(s) in eval-exec", err)

    def test_categories_and_bad_floor(self) -> None:
        code, out, _ = self.run_main("--categories")
        self.assertEqual(code, 0)
        self.assertEqual(len(out.strip().splitlines()), len(MODULE.CATEGORIES))
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                MODULE.main(["--floor", "3"])


if __name__ == "__main__":
    unittest.main()
