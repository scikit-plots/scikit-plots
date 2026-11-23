# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tools/maint_tools/sweep_runtime.py (what importing each submodule does)."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "sweep_runtime.py"
SPEC = importlib.util.spec_from_file_location("sweep_runtime", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

#: A package whose submodules each show one behaviour the sweep reports.
FAKE = {
    "fakepkg/__init__.py": "",
    "fakepkg/good.py": (
        '__all__ = ["f", "C", "CONSTANT"]\n'
        "CONSTANT = 1\n"
        "def f(path, seed=None):\n    '''Documented.'''\n"
        "class C:\n    def __init__(self, random_state=None, n_jobs=1):\n        pass\n"
    ),
    "fakepkg/noisy.py": (
        "import logging, warnings\n"
        "warnings.warn('noisy import')\n"
        "logging.getLogger(__name__).warning('logged on import')\n"
    ),
    "fakepkg/broken.py": "raise ImportError('broken on purpose')\n",
    "fakepkg/lazy.py": (
        '__all__ = ["thing"]\n'
        "def __getattr__(name):\n"
        "    raise ModuleNotFoundError(\"No module named 'missingdep'\")\n"
    ),
    "fakepkg/slow.py": "import time\ntime.sleep(30)\n",
    "fakepkg/_private.py": "raise RuntimeError('never imported')\n",
    "fakepkg/tests/__init__.py": "raise RuntimeError('never imported')\n",
}


class SweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls._tmp.name)
        for rel, text in FAKE.items():
            path = cls.dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        cls.result = MODULE.sweep(
            sys.executable,
            "fakepkg",
            modules=("broken", "good", "lazy", "noisy"),
            pythonpath=[str(cls.dir)],
            api=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_discovery_lists_public_submodules_only(self) -> None:
        found = MODULE._child(
            sys.executable, MODULE._DISCOVER, ["fakepkg"],
            {"PYTHONPATH": str(self.dir), "PATH": ""}, 60,
        )
        self.assertEqual(found["modules"], ["broken", "good", "lazy", "noisy", "slow"])

    def test_a_failed_import_is_recorded(self) -> None:
        self.assertEqual(
            self.result["modules"]["broken"]["error"], "ImportError: broken on purpose"
        )

    def test_warnings_and_logged_records_at_import(self) -> None:
        noisy = self.result["modules"]["noisy"]
        self.assertEqual(noisy["warnings"], ["UserWarning: noisy import"])
        self.assertEqual(noisy["logged"], ["WARNING fakepkg.noisy: logged on import"])
        self.assertIsNone(noisy["all"])

    def test_public_names_documentation_and_lazy_failures(self) -> None:
        good, lazy = self.result["modules"]["good"], self.result["modules"]["lazy"]
        self.assertEqual((good["all"], good["undocumented"], good["error"]), (3, ["C"], None))
        self.assertEqual(lazy["lazy_failures"], ["thing: ModuleNotFoundError"])
        self.assertNotIn("params", good)

    def test_an_idea_spelled_two_ways_is_a_cluster(self) -> None:
        self.assertEqual(
            self.result["clusters"],
            {"random seed": {"random_state": ["good.C"], "seed": ["good.f"]}},
        )

    def test_an_added_heavy_library_is_reported(self) -> None:
        result = MODULE.sweep(
            sys.executable, "fakepkg", modules=("good",), pythonpath=[str(self.dir)],
            heavy=("json", "fakepkg"),
        )
        self.assertEqual(result["modules"]["good"]["heavy"], ["fakepkg"])

    def test_api_status_is_recorded(self) -> None:
        # fakepkg has no __main__, so its "api" command fails: the status is kept.
        self.assertNotEqual(self.result["modules"]["good"]["api_status"], 0)

    def test_a_slow_import_times_out(self) -> None:
        result = MODULE.sweep(
            sys.executable, "fakepkg", modules=("slow",), pythonpath=[str(self.dir)], timeout=2
        )
        self.assertEqual(
            result["modules"]["slow"]["error"], "TimeoutError: no result within 2 s"
        )

    def test_bad_arguments(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.sweep(sys.executable, "no_such_package_sweep_runtime")
        for timeout in (0, -1, True):
            with self.assertRaises(ValueError):
                MODULE.sweep(sys.executable, "fakepkg", timeout=timeout)

    def test_report_and_command_line(self) -> None:
        lines = MODULE.report(self.result)
        self.assertTrue(lines[0].startswith("broken"))
        self.assertTrue(lines[-1].startswith("cluster random seed: "))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "r.json"
            with redirect_stdout(io.StringIO()):
                code = MODULE.main(
                    ["--package", "fakepkg", "--module", "good", "--pythonpath", str(self.dir), "--json", str(out)]
                )
            self.assertEqual(code, 0)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual((data["format"], list(data["modules"])), (1, ["good"]))


if __name__ == "__main__":
    unittest.main()
