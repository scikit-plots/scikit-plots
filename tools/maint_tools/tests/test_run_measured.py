# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tools/maint_tools/run_measured.py (wall time and peak memory of a command)."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "run_measured.py"
SPEC = importlib.util.spec_from_file_location("run_measured", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PY = sys.executable
#: Writes (not only reserves) 80 MiB, so the pages are resident.
BIG = "x = b'\\x01' * (80 * 1024 * 1024)"
POSIX = hasattr(os, "wait4")


class RunOnceTests(unittest.TestCase):
    def test_status_and_time(self) -> None:
        result = MODULE.run_once([PY, "-c", "pass"])
        self.assertEqual(result["status"], 0)
        self.assertFalse(result["timed_out"])
        self.assertGreater(result["seconds"], 0)
        failed = MODULE.run_once([PY, "-c", "import sys; sys.exit(3)"])
        self.assertEqual(failed["status"], 3)

    @unittest.skipUnless(POSIX, "peak memory needs os.wait4")
    def test_the_peak_is_this_runs_own(self) -> None:
        big = MODULE.run_once([PY, "-c", BIG])
        small = MODULE.run_once([PY, "-c", "pass"])
        self.assertGreater(big["peak_rss_mib"], 75)
        self.assertLess(small["peak_rss_mib"], big["peak_rss_mib"] - 40)

    @unittest.skipUnless(POSIX, "peak memory needs os.wait4")
    def test_a_waited_for_descendant_counts(self) -> None:
        code = f"import subprocess, sys; subprocess.run([sys.executable, '-c', {BIG!r}])"
        self.assertGreater(MODULE.run_once([PY, "-c", code])["peak_rss_mib"], 75)

    @unittest.skipUnless(POSIX, "process groups need POSIX")
    def test_a_timeout_ends_the_whole_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "alive"
            child = (
                "import pathlib, sys, time; time.sleep(2);"
                " pathlib.Path(sys.argv[1]).write_text('alive')"
            )
            code = (
                "import subprocess, sys, time;"
                f" subprocess.Popen([sys.executable, '-c', {child!r}, {str(marker)!r}]);"
                " time.sleep(30)"
            )
            start = time.perf_counter()
            result = MODULE.run_once([PY, "-c", code], timeout=0.5)
            self.assertLess(time.perf_counter() - start, 10)
            self.assertEqual(result["status"], MODULE.TIMEOUT_STATUS)
            self.assertTrue(result["timed_out"])
            time.sleep(3)
            self.assertFalse(marker.exists(), "a descendant outlived the timeout")

    def test_bad_arguments(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.run_once([])
        for timeout in (0, -1, True, "1"):
            with self.assertRaises(ValueError):
                MODULE.run_once([PY, "-c", "pass"], timeout=timeout)
        with self.assertRaises(FileNotFoundError):
            MODULE.run_once(["/no/such/program-run-measured"])


class MeasureTests(unittest.TestCase):
    def test_repeat_and_summary(self) -> None:
        result = MODULE.measure([PY, "-c", "pass"], repeat=3, label="noop")
        self.assertEqual(len(result["runs"]), 3)
        self.assertEqual(result["status"], 0)
        self.assertEqual(result["label"], "noop")
        self.assertEqual(result["format"], 1)
        seconds = sorted(r["seconds"] for r in result["runs"])
        self.assertEqual(result["median_seconds"], seconds[1])
        for repeat in (0, True, 1.5):
            with self.assertRaises(ValueError):
                MODULE.measure([PY, "-c", "pass"], repeat=repeat)

    def test_the_first_failing_status_is_kept(self) -> None:
        result = MODULE.measure([PY, "-c", "import sys; sys.exit(4)"], repeat=2)
        self.assertEqual(result["status"], 4)


class CommandLineTests(unittest.TestCase):
    def test_json_and_exit_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "m.json"
            err = io.StringIO()
            with redirect_stderr(err):
                code = MODULE.main(
                    ["--repeat", "2", "--json", str(out), "--", PY, "-c", "pass"]
                )
            self.assertEqual(code, 0)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(data["runs"]), 2)
            self.assertIn("run 2/2: status 0", err.getvalue())
            self.assertIn("median ", err.getvalue())
            with redirect_stderr(io.StringIO()):
                self.assertEqual(
                    MODULE.main(["--", PY, "-c", "import sys; sys.exit(5)"]), 5
                )

    def test_usage_errors(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                MODULE.main([])
            with self.assertRaises(SystemExit):
                MODULE.main(["--", "/no/such/program-run-measured"])


if __name__ == "__main__":
    unittest.main()
