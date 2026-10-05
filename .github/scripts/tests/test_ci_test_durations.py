# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for ``ci_test_durations.py``; standard library only."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ci_test_durations as durations_module  # noqa: E402
from ci_test_plan import PlanError, Planner, load_durations  # noqa: E402

CONFIG = {
    "schema": 1,
    "package_root": "pkg",
    "containers": ["pkg", "pkg/_ext"],
    "full_run_globs": [],
    "ignore_globs": [],
    "loose_test_globs": ["test_*.py"],
    "default_mode": {},
    "max_shards": 2,
    "shard_timeout_minutes": 60,
    "shard_target_minutes": 30,
    "default_unit_seconds": 60,
    "per_test_seconds": 0.0,
    "dependents": "none",
}

REPORT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="5">
<testcase classname="pkg.corpus.tests.test_a.TestThing" name="test_one" time="1.25"/>
<testcase classname="pkg.corpus.tests.test_a" name="test_two" time="0.75"/>
<testcase classname="pkg._ext._grid.tests.test_g" name="test_three" time="2.0"><skipped/></testcase>
<testcase classname="pkg.removed.tests.test_r" name="test_four" time="9.0"/>
<testcase classname="other.tests.test_o" name="test_five" time="9.0"/>
</testsuite></testsuites>
"""


class TestDurations(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        for name in ("pkg/corpus/tests", "pkg/_ext/_grid/tests", "pkg/tests"):
            (self.root / name).mkdir(parents=True)
        self.report = self.root / "junit.xml"
        self.report.write_text(REPORT, encoding="utf-8")
        self.config = self.root / "plan.json"
        self.config.write_text(json.dumps(CONFIG), encoding="utf-8")
        self.planner = Planner(self.root, dict(CONFIG))

    def test_time_is_summed_per_unit_and_strays_are_counted(self):
        units, unassigned = durations_module.collect(self.planner, [self.report])
        self.assertEqual(
            units,
            {
                "pkg/_ext/_grid": {"tests": 1, "work_seconds": 2.0},
                "pkg/corpus": {"tests": 2, "work_seconds": 2.0},
            },
        )
        self.assertEqual(unassigned, 2)

    def test_several_reports_add_up(self):
        units, _ = durations_module.collect(self.planner, [self.report, self.report])
        self.assertEqual(units["pkg/corpus"], {"tests": 4, "work_seconds": 4.0})

    def test_an_unreadable_report_is_an_error(self):
        broken = self.root / "broken.xml"
        broken.write_text("<testsuites><testcase", encoding="utf-8")
        with self.assertRaisesRegex(PlanError, "cannot read the JUnit report"):
            durations_module.collect(self.planner, [broken])
        with self.assertRaisesRegex(PlanError, "cannot read the JUnit report"):
            durations_module.collect(self.planner, [self.root / "absent.xml"])

    def test_a_time_that_is_not_a_number_is_an_error(self):
        odd = self.root / "odd.xml"
        odd.write_text('<testsuite><testcase classname="pkg.corpus.t" time="fast"/></testsuite>', encoding="utf-8")
        with self.assertRaisesRegex(PlanError, "not a number"):
            durations_module.collect(self.planner, [odd])

    def test_the_written_file_is_what_the_planner_reads(self):
        output = self.root / "durations.json"
        with contextlib.redirect_stderr(io.StringIO()):
            code = durations_module.main(
                [str(self.report), "--root", str(self.root), "--config", str(self.config), "--output", str(output), "--source", "run 1"]
            )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["source"], "run 1")
        self.assertEqual(load_durations(output), {"pkg/_ext/_grid": (1, 2.0), "pkg/corpus": (2, 2.0)})

    def test_a_bad_report_exits_with_2(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code = durations_module.main(
                [str(self.root / "absent.xml"), "--root", str(self.root), "--config", str(self.config)]
            )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
