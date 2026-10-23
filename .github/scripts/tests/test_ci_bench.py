# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for ``ci_bench.py``; standard library only, as the plan job runs them."""

from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ci_bench as bench  # noqa: E402

SUITE = {
    "__init__.py": "",
    "common.py": "VALUE = 3\n",
    "bench_a.py": (
        """
        import os
        import time

        from .common import VALUE


        def time_sleep():
            time.sleep(float(os.environ.get("BENCH_DELAY", "0.002")))


        def track_value():
            return VALUE


        def helper():  # no prefix: not a benchmark
            return 1


        class Params:
            params = ([1, 2], ["x"])
            param_names = ["n", "s"]
            timeout = 30

            def setup(self, n, s):
                if n == 2:
                    raise NotImplementedError
                self.data = list(range(1000 * n))

            def time_sum(self, n, s):
                sum(self.data)

            def peakmem_list(self, n, s):
                [0] * 100_000


        class Broken:
            def time_fails(self):
                raise ValueError("boom")

            def timeraw_import(self):
                return "import json", "import sys"
        """
    ),
}


def _suite(root):
    package = Path(root) / "suite" / "benchmarks"
    package.mkdir(parents=True)
    for name, text in SUITE.items():
        (package / name).write_text(textwrap.dedent(text), encoding="utf-8")
    return package


def _wrapper(root, name, delay):
    """An interpreter that runs this Python with BENCH_DELAY set."""
    path = Path(root) / name
    path.write_text(
        f'#!/bin/sh\nBENCH_DELAY={delay} exec "{sys.executable}" "$@"\n',
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def _entry(median, rounds=None):
    """A measured result whose rounds had these medians (one round by default)."""
    rounds = list(rounds) if rounds is not None else [median]
    return {
        "params": [],
        "median": median,
        "min": min(rounds),
        "max": max(rounds),
        "q25": min(rounds),
        "q75": max(rounds),
        "samples": list(rounds),
        "round_medians": rounds,
        "number": 1,
    }


def _around(centre, spread=0.01, n=4):
    """``n`` round medians around ``centre``."""
    return [centre * (1 + spread * (i - (n - 1) / 2)) for i in range(n)]


def _doc(results, label="x"):
    return {
        "schema": bench.RESULT_SCHEMA,
        "version": bench.RESULT_VERSION,
        "label": label,
        "machine": {"m": 1},
        "environment": {"python": "3.11.0"},
        "benchmarks": {
            name: {"kind": "time", "unit": "seconds", "param_names": [], "results": [e]}
            for name, e in results.items()
        },
    }


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = bench.main([str(a) for a in argv])
        return status, out.getvalue(), err.getvalue()


class TestDiscoverAndRun(_Tmp):
    def test_the_asv_subset_is_found(self):
        listing, problem = bench._call_worker(
            sys.executable, ["discover", str(_suite(self.root))], timeout=120
        )
        self.assertIsNone(problem)
        found = listing["benchmarks"]
        self.assertEqual(
            sorted(found),
            [
                "bench_a.Broken.time_fails",
                "bench_a.Broken.timeraw_import",
                "bench_a.Params.peakmem_list",
                "bench_a.Params.time_sum",
                "bench_a.time_sleep",
                "bench_a.track_value",
            ],
        )
        self.assertEqual(found["bench_a.Params.time_sum"]["param_names"], ["n", "s"])
        self.assertEqual(
            found["bench_a.Params.time_sum"]["combinations"],
            [["1", "'x'"], ["2", "'x'"]],
        )

    def test_a_run_records_values_skips_and_errors(self):
        result = bench.run(
            _suite(self.root), settings={"repeat": 2, "processes": 2}, label="t"
        )
        b = result["benchmarks"]
        ok, skipped = b["bench_a.Params.time_sum"]["results"]
        self.assertEqual(len(ok["samples"]), 4)  # 2 samples x 2 processes
        self.assertLessEqual(ok["min"], ok["median"])
        self.assertLessEqual(ok["median"], ok["max"])
        self.assertIn("NotImplementedError", skipped["skipped"])
        self.assertIn("boom", b["bench_a.Broken.time_fails"]["results"][0]["error"])
        self.assertEqual(b["bench_a.track_value"]["results"][0]["median"], 3)
        self.assertGreater(b["bench_a.Params.peakmem_list"]["results"][0]["median"], 0)
        self.assertGreater(
            b["bench_a.Broken.timeraw_import"]["results"][0]["median"], 0
        )
        self.assertEqual(result["environment"]["python"].count("."), 2)
        # Written and read back unchanged.
        path = self.root / "r.json"
        bench._write_json(path, result)
        self.assertEqual(
            bench.load_result(path)["benchmarks"], json.loads(json.dumps(b))
        )

    def test_a_worker_starts_outside_any_source_tree(self):
        # A raw sample does not see the suite's directory, so a module there
        # that shadows the standard library is not imported.
        package = _suite(self.root)
        (package.parent / "json.py").write_text("raise SystemExit(9)\n")
        result = bench.run(
            package, patterns=["timeraw"], settings={"repeat": 1, "processes": 1}
        )
        entry = result["benchmarks"]["bench_a.Broken.timeraw_import"]["results"][0]
        self.assertNotIn("error", entry)

    def test_nothing_selected_is_an_error(self):
        status, _, err = self.run_main(
            "run",
            "--dir",
            _suite(self.root),
            "--bench",
            "nomatch",
            "--out",
            self.root / "o",
        )
        self.assertEqual(status, 2)
        self.assertIn("no benchmark matches", err)


class TestAsvSubset(_Tmp):
    """What asv does that the runner must do the same way, or refuse."""

    MODULE = """
        import fractions
        import gc


        def setup():
            global READY
            READY = 1


        class Base:
            def time_inherited(self):
                pass


        class Child(Base):
            def setup(self):
                self.data = READY  # the module setup ran first

            def track_fraction(self):
                return fractions.Fraction(3, 2)  # a numbers.Real, as numpy scalars are

            def track_gc_is_off_while_timing(self):
                return 1

            def time_gc_state(self):
                assert not gc.isenabled()

            def timeraw_indented(self):
                return \"\"\"
                    import json
                    json.dumps([1])
                \"\"\"


        class Cached:
            def setup_cache(self):
                return 1

            def time_cached(self, data):
                pass
        """

    def _suite(self, module):
        package = self.root / "s" / "benchmarks"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "bench_x.py").write_text(textwrap.dedent(module))
        return package

    def test_inheritance_setup_order_values_and_raw_code(self):
        result = bench.run(
            self._suite(self.MODULE), settings={"repeat": 5, "processes": 1}
        )
        b = result["benchmarks"]
        self.assertIn("bench_x.Child.time_inherited", b)
        self.assertEqual(b["bench_x.Child.track_fraction"]["results"][0]["median"], 1.5)
        for name in ("bench_x.Child.time_gc_state", "bench_x.Child.timeraw_indented"):
            self.assertNotIn("error", b[name]["results"][0], name)
        self.assertIn(
            "setup_cache", b["bench_x.Cached.time_cached"]["results"][0]["error"]
        )

    def test_a_parameter_without_values_is_named(self):
        package = self._suite(
            "class P:\n    params = [[]]\n\n    def time_x(self, a):\n        pass\n"
        )
        status, _, err = self.run_main("list", "--dir", package)
        self.assertEqual(status, 2)
        self.assertIn("bench_x.P.time_x: a parameter in params has no values", err)

    def test_a_relative_interpreter_path_works(self):
        package = _suite(self.root)
        link = self.root / "py"
        link.symlink_to(sys.executable)
        previous = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        status, _, err = self.run_main(
            "run",
            "--dir",
            package,
            "--bench",
            "track_value",
            "--quick",
            "--python",
            "./py",
            "--out",
            self.root / "r.json",
        )
        self.assertEqual(status, 0, err)


@unittest.skipIf(os.name != "posix", "the interpreter wrappers are shell scripts")
class TestPair(_Tmp):
    def test_a_real_slowdown_is_found_and_no_change_is_not(self):
        suite = _suite(self.root)
        fast = _wrapper(self.root, "fast", 0.002)
        slow = _wrapper(self.root, "slow", 0.02)
        common = ["--dir", suite, "--bench", "time_sleep$", "--repeat", "3"]
        status, out, err = self.run_main(
            "pair",
            *common,
            "--base-python",
            fast,
            "--head-python",
            slow,
            "--out",
            self.root / "slower",
            "--fail-on-slower",
        )
        self.assertEqual(status, 1, err)
        self.assertIn("1 slower", out)
        status, out, err = self.run_main(
            "pair",
            *common,
            "--base-python",
            fast,
            "--head-python",
            fast,
            "--out",
            self.root / "same",
            "--fail-on-slower",
        )
        self.assertEqual(status, 0, err)
        self.assertIn("0 slower, 0 faster", out)
        for name in ("base.json", "head.json", "compare.md", "compare.json"):
            self.assertTrue((self.root / "same" / name).is_file(), name)


class TestCompare(unittest.TestCase):
    def test_a_change_needs_the_factor_and_every_round_on_one_side(self):
        base = _doc(
            {
                "slower": _entry(1.0, _around(1.0)),
                "outlier": _entry(1.0, [1.0, 1.0, 1.0, 1.0]),
                "faster": _entry(1.0, _around(1.0)),
                "same": _entry(1.0, _around(1.0)),
                "noisy": _entry(1.0, [0.8, 1.0, 1.2, 1.0]),
                "few": _entry(1.0, [1.0, 1.01]),
                "gone": _entry(1.0, _around(1.0)),
            }
        )
        # One slow sample in a base round: its median ignores it.
        base["benchmarks"]["outlier"]["results"][0]["samples"] += [3.0]
        head = _doc(
            {
                "slower": _entry(1.5, _around(1.5)),
                "outlier": _entry(2.0, [2.0, 2.0, 2.0, 2.0]),
                "faster": _entry(0.5, _around(0.5)),
                "same": _entry(1.02, _around(1.02)),
                "noisy": _entry(1.12, [0.95, 1.1, 1.3, 1.14]),
                "few": _entry(2.0, [2.0, 2.01]),
                "new": _entry(1.0, _around(1.0)),
            }
        )
        result = bench.compare(base, head, factor=1.1)
        marks = {row[1]: row[0] for row in result["rows"]}
        self.assertEqual(
            marks,
            {
                "slower": "+",
                "outlier": "+",
                "faster": "-",
                "same": "~",
                "noisy": "x",  # the ratio, but rounds of both sides overlap
                "few": "x",  # two rounds: not enough to mark a change
                "gone": "?",
                "new": "?",
            },
        )
        self.assertEqual(result["rows"][0][0], "+")  # what needs attention first
        text = bench.render_compare(result)
        self.assertIn("2 slower, 1 faster", text)
        self.assertIn("only 2 round(s)", text)

    def test_memory_and_track_values_change_by_the_ratio(self):
        base, head = _doc({"m": _entry(100.0)}), _doc({"m": _entry(150.0)})
        for doc in (base, head):
            doc["benchmarks"]["m"]["kind"] = "peakmem"
        self.assertEqual(bench.compare(base, head)["rows"][0][0], "+")

    def test_failures_are_never_unchanged_and_a_new_one_fails_the_gate(self):
        error = {"params": [], "error": "ImportError: x"}
        skipped = {"params": [], "skipped": "setup raised NotImplementedError"}
        base = _doc(
            {"both": _entry(1), "new": _entry(1.0, _around(1.0)), "skip": _entry(1)}
        )
        head = _doc({"both": _entry(1), "new": _entry(1), "skip": _entry(1)})
        base["benchmarks"]["both"]["results"][0] = dict(error)
        head["benchmarks"]["both"]["results"][0] = dict(error)
        head["benchmarks"]["new"]["results"][0] = dict(error)
        base["benchmarks"]["skip"]["results"][0] = dict(skipped)
        head["benchmarks"]["skip"]["results"][0] = dict(skipped)
        result = bench.compare(base, head)
        marks = {row[1]: row[0] for row in result["rows"]}
        self.assertEqual(marks, {"both": "!", "new": "!", "skip": "~"})
        self.assertEqual(result["new_failures"], 1)
        self.assertTrue(bench._got_worse(result))

    def test_text_from_result_files_cannot_add_markdown(self):
        base = _doc({"a": _entry(1.0, _around(1.0))}, label="x\n# injected | <b>")
        base["benchmarks"]["a"]["results"][0]["params"] = ["1 | <img>\n# also"]
        head = _doc({"a": _entry(1.0, _around(1.0))})
        head["benchmarks"]["a"]["results"][0]["params"] = ["1 | <img>\n# also"]
        text = bench.render_compare(bench.compare(base, head))
        self.assertNotIn("\n# injected", text)
        self.assertNotIn("\n# also", text)
        self.assertEqual(text.count("\n| "), 2)  # header and one row

    def test_runs_on_different_machines_are_flagged(self):
        base, head = _doc({"a": _entry(1)}), _doc({"a": _entry(1)})
        head["machine"] = {"m": 2}
        self.assertIn(
            "differ in machine", bench.render_compare(bench.compare(base, head))
        )

    def test_the_factor_must_be_above_one(self):
        with self.assertRaisesRegex(bench.BenchError, "above 1"):
            bench.compare(_doc({}), _doc({}), factor=1.0)


class TestResultFiles(_Tmp):
    def test_hostile_result_files_are_refused_not_raised(self):
        huge = _doc({"a": _entry(1)})
        huge["benchmarks"]["a"]["results"][0]["median"] = 10**400
        bad_env = {**_doc({}), "environment": [1]}
        for doc in (huge, bad_env):
            with self.assertRaises(bench.BenchError):
                bench.load_result(self._write(doc))
        path = self.root / "deep.json"
        path.write_text("[" * 200_000, encoding="utf-8")
        with self.assertRaisesRegex(bench.BenchError, "not JSON"):
            bench.load_result(path)

    def _write(self, doc):
        path = self.root / "r.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        return path

    def test_malformed_results_are_refused_with_a_reason(self):
        good = _doc({"a": _entry(1)})
        cases = {
            "not a skplt-bench": {"schema": "other"},
            "version": {**good, "version": 99},
            "unexpected shape": {
                **good,
                "benchmarks": {
                    "a": {"kind": "time", "unit": "seconds", "results": [{}]}
                },
            },
        }
        for needle, doc in cases.items():
            with self.assertRaisesRegex(bench.BenchError, needle):
                bench.load_result(self._write(doc))
        nan = _doc({"a": _entry(1)})
        nan["benchmarks"]["a"]["results"][0]["median"] = float("nan")
        with self.assertRaisesRegex(bench.BenchError, "unexpected shape"):
            bench.load_result(self._write(nan))

    def test_compare_on_the_command_line(self):
        base = self._write(_doc({"a": _entry(1.0, _around(1.0))}))
        head = self.root / "h.json"
        head.write_text(
            json.dumps(_doc({"a": _entry(2.0, _around(2.0))})), encoding="utf-8"
        )
        status, out, _ = self.run_main("compare", base, head, "--fail-on-slower")
        self.assertEqual(status, 1)
        self.assertIn("| + | `a`", out)
        status, _, err = self.run_main("compare", base, self.root / "missing.json")
        self.assertEqual(status, 2)
        self.assertIn("no such file", err)


if __name__ == "__main__":
    unittest.main()
