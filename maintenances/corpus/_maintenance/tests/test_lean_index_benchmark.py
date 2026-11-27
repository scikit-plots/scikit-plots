"""
Tests for maintenances/corpus/_maintenance/tools/lean_index_benchmark.py.

The index is a stand-in class, so these tests need no scikit-plots build.
Run: ``python -B maintenances/corpus/_maintenance/tests/test_lean_index_benchmark.py``.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "lean_index_benchmark.py"
SPEC = importlib.util.spec_from_file_location("lean_index_benchmark", TOOL)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


def _listing(count):
    return {"count": count, "items": [{"declaration": "t"}] if count else []}


class StandIn:
    """Answers like LeanIndex, records which stages ran."""

    calls: list[str] = []

    def __init__(self, headline):
        self._name, self._labels, self._skipped = [0, 1, 2], ["A.lean"], []
        self.headline = headline

    options: dict = {}

    @classmethod
    def from_folder(cls, folder, **options):
        cls.calls = ["index"]
        cls.options = options
        return cls(headline=Path(folder, "headline").exists())

    def summary(self, max_items):
        self.calls.append("summary")

    def discover(self, top):
        self.calls.append("discover")
        return {
            "rules": {"load_bearing_theorems": _listing(2)},
            "patterns": {"statement_shapes": _listing(0)},
            "claims": {"axioms": _listing(1)},
            "anomalies": {"generators": _listing(0)},
            "headline_results": _listing(1 if self.headline else 0),
        }

    def dossier(self, name, max_items):
        self.calls.append("dossier")

    def export(self, out, top):
        self.calls.append("export")
        return ["README.md", "records.jsonl"]


def _run(argv):
    out = io.StringIO()
    with redirect_stdout(out):
        status = bench.main(argv, index_class=StandIn)
    return status, out.getvalue()


class LeanIndexBenchmarkTest(unittest.TestCase):
    def test_every_stage_runs_and_is_counted(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "headline").write_text("", encoding="utf-8")
            status, text = _run([root, "--json"])
        result = json.loads(text)
        self.assertEqual(status, 0)
        self.assertEqual(StandIn.calls, ["index", "summary", "discover", "dossier", "export"])
        self.assertEqual(set(result["seconds"]), {"index", "summary", "discover", "dossier", "export"})
        self.assertEqual(result["sections"]["rules.load_bearing_theorems"], 2)
        self.assertEqual(result["nodes"], 3)
        self.assertEqual(result["files_written"], 2)

    def test_no_headline_no_dossier(self):
        with tempfile.TemporaryDirectory() as root:
            status, text = _run([root])
        self.assertEqual(status, 0)
        self.assertNotIn("dossier", StandIn.calls)
        self.assertIn("declarations", text)

    def test_bad_arguments_are_refused(self):
        with tempfile.TemporaryDirectory() as root:
            for argv in ([str(Path(root, "missing"))], [root, "--top", "0"]):
                with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()):
                    import contextlib  # noqa: PLC0415

                    with contextlib.redirect_stderr(io.StringIO()):
                        bench.main(argv, index_class=StandIn)


class CompareTest(unittest.TestCase):
    def test_options_reach_from_folder_and_the_result(self):
        with tempfile.TemporaryDirectory() as root:
            status, text = _run([root, "--graph", "numpy", "--workers", "2", "--json"])
        result = json.loads(text)
        self.assertEqual(status, 0)
        self.assertEqual(StandIn.options, {"graph": "numpy", "workers": 2})
        self.assertEqual(result["options"], {"graph": "numpy", "workers": 2})
        self.assertEqual(len(result["report_sha"]), 16)

    def test_identical_reports_pass_and_unavailable_backends_are_listed(self):
        runs = {
            "python": {"report_sha": "a", "seconds": {}, "peak_mb": 1},
            "numpy": {"report_sha": "a", "seconds": {}, "peak_mb": 1},
            "scipy": {"unavailable": "No module named 'scipy'"},
        }
        status, results = bench.compare(runs.__getitem__)
        self.assertEqual(status, 0)
        self.assertEqual([r["graph"] for r in results], ["python", "numpy", "scipy"])
        self.assertIn("unavailable", bench._compare_text(status, results))

    def test_differing_reports_fail(self):
        runs = {"python": {"report_sha": "a"}, "numpy": {"report_sha": "b"}}
        status, _ = bench.compare(runs.__getitem__, graphs=("python", "numpy"))
        self.assertEqual(status, 1)

    def test_negative_workers_are_refused(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()):
                import contextlib  # noqa: PLC0415

                with contextlib.redirect_stderr(io.StringIO()):
                    bench.main([root, "--workers", "-1"], index_class=StandIn)


if __name__ == "__main__":
    unittest.main()
