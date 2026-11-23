"""
Tests for maintenances/corpus/_maintenance/tools/reader_snapshot.py.

The reading function is a stand-in, so these tests need no scikit-plots build.
Run: ``python -B maintenances/corpus/_maintenance/tests/test_reader_snapshot.py``.
"""

from __future__ import annotations

import importlib.util
import io
import json
import logging
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "reader_snapshot.py"
SPEC = importlib.util.spec_from_file_location("reader_snapshot", TOOL)
assert SPEC is not None and SPEC.loader is not None
snap = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = snap
SPEC.loader.exec_module(snap)

LOG = logging.getLogger("scikitplot.test_reader_snapshot")


def lines_reader(path: Path):
    """One chunk per line; warns about empty files; fails on '.bad' files."""
    if path.suffix == ".bad":
        raise ValueError(f"cannot read {path}")
    text = path.read_text(encoding="utf-8")
    if not text:
        LOG.warning("empty file %s", path)
    return [{"text": line, "line": i} for i, line in enumerate(text.splitlines())]


class _Folder:
    def __init__(self, files: dict) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "corpus"
        for rel, text in files.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    def close(self) -> None:
        self._tmp.cleanup()


class SnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = _Folder(
            {"b.txt": "one\ntwo\n", "a/c.txt": "x\n", "empty.txt": "", "x.bad": "?"}
        )
        self.root = self.folder.root

    def tearDown(self) -> None:
        self.folder.close()

    def test_every_file_is_recorded_in_sorted_relative_order(self) -> None:
        files = snap.snapshot(self.root, lines_reader)["files"]
        self.assertEqual(list(files), ["a/c.txt", "b.txt", "empty.txt", "x.bad"])
        self.assertEqual(files["b.txt"]["chunks"], 2)
        self.assertEqual(files["b.txt"]["chars"], 6)
        self.assertEqual(files["b.txt"]["result"], "ok")
        self.assertTrue(files["b.txt"]["stable"])

    def test_an_error_is_recorded_with_the_folder_hidden(self) -> None:
        record = snap.snapshot(self.root, lines_reader)["files"]["x.bad"]
        self.assertEqual(record["result"], "error")
        self.assertEqual(record["error"], "ValueError: cannot read <root>/x.bad")
        self.assertIsNone(record["digest"])

    def test_warnings_are_recorded_with_the_folder_hidden(self) -> None:
        record = snap.snapshot(self.root, lines_reader)["files"]["empty.txt"]
        self.assertEqual(record["warnings"], ["empty file <root>/empty.txt"])
        self.assertEqual(record["chunks"], 0)

    def test_every_key_of_a_chunk_is_in_the_digest(self) -> None:
        def shifted(path: Path):
            return [dict(c, line=c["line"] + 1) for c in lines_reader(path)]

        one = snap.snapshot(self.root, lines_reader)["files"]["b.txt"]
        two = snap.snapshot(self.root, shifted)["files"]["b.txt"]
        self.assertEqual(one["chars"], two["chars"])
        self.assertNotEqual(one["digest"], two["digest"])

    def test_a_reader_that_varies_between_reads_is_unstable(self) -> None:
        calls = {"n": 0}

        def varying(path: Path):
            calls["n"] += 1
            return [{"text": str(calls["n"])}]

        files = snap.snapshot(self.root, varying, repeat=3)["files"]
        self.assertFalse(files["b.txt"]["stable"])
        self.assertEqual(calls["n"], 12)
        files = snap.snapshot(self.root, lines_reader, repeat=1)["files"]
        self.assertTrue(all(r["stable"] for r in files.values()))

    def test_a_warning_that_varies_does_not_make_a_file_unstable(self) -> None:
        calls = {"n": 0}

        def temp_paths(path: Path):
            calls["n"] += 1
            LOG.warning("extracted to /tmp/tmp%d", calls["n"])
            return lines_reader(path)

        files = snap.snapshot(self.root, temp_paths)["files"]
        self.assertTrue(files["b.txt"]["stable"])

    def test_the_text_digest_ignores_other_keys(self) -> None:
        def tagged(path: Path):
            return [dict(c, encoding="utf-8") for c in lines_reader(path)]

        one = snap.snapshot(self.root, lines_reader)["files"]["b.txt"]
        two = snap.snapshot(self.root, tagged)["files"]["b.txt"]
        self.assertEqual(one["text_digest"], two["text_digest"])
        self.assertNotEqual(one["digest"], two["digest"])

    def test_bad_arguments(self) -> None:
        with self.assertRaises(ValueError):
            snap.snapshot(self.root / "b.txt", lines_reader)
        for repeat in (0, -1, True, 2.0):
            with self.assertRaises(ValueError):
                snap.snapshot(self.root, lines_reader, repeat=repeat)

    def test_a_copy_of_the_folder_gives_the_same_snapshot(self) -> None:
        other = _Folder(
            {"b.txt": "one\ntwo\n", "a/c.txt": "x\n", "empty.txt": "", "x.bad": "?"}
        )
        try:
            first = snap.snapshot(self.root, lines_reader)
            second = snap.snapshot(other.root, lines_reader)
            self.assertEqual(snap.compare(first, second), [])
        finally:
            other.close()


class CompareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = _Folder({"a.txt": "one\r\ntwo\r\n", "b.txt": "same\n"})
        self.root = self.folder.root
        self.before = snap.snapshot(self.root, lines_reader)

    def tearDown(self) -> None:
        self.folder.close()

    def test_identical_snapshots_have_no_difference(self) -> None:
        after = snap.snapshot(self.root, lines_reader)
        self.assertEqual(snap.compare(self.before, after), [])

    def test_a_changed_reader_is_reported_per_file_and_field(self) -> None:
        def keeps_cr(path: Path):
            text = path.read_bytes().decode("utf-8")
            return [
                {"text": line, "line": i}
                for i, line in enumerate(text.split("\n")[:-1])
            ]

        after = snap.snapshot(self.root, keeps_cr)
        lines = snap.compare(self.before, after)
        self.assertTrue(lines)
        self.assertTrue(all(line.startswith("changed: a.txt: ") for line in lines))
        self.assertIn("changed: a.txt: chars 6 -> 8", lines)
        self.assertTrue(any(" digest " in line for line in lines))

    def test_added_removed_and_unstable_files(self) -> None:
        (self.root / "c.txt").write_text("new\n", encoding="utf-8")
        (self.root / "b.txt").unlink()
        calls = {"n": 0}

        def varying(path: Path):
            calls["n"] += 1
            if path.name == "c.txt":
                return [{"text": str(calls["n"])}]
            return lines_reader(path)

        after = snap.snapshot(self.root, varying)
        self.assertEqual(
            snap.compare(self.before, after),
            [
                "only before: b.txt",
                "only after: c.txt",
                "unstable: c.txt: two reads of the same bytes differ",
            ],
        )

    def test_warnings_can_be_left_out(self) -> None:
        def warns(path: Path):
            LOG.warning("note")
            return lines_reader(path)

        after = snap.snapshot(self.root, warns)
        self.assertEqual(len(snap.compare(self.before, after)), 2)
        self.assertEqual(snap.compare(self.before, after, ignore_warnings=True), [])

    def test_text_only_ignores_added_metadata_but_not_text(self) -> None:
        def tagged(path: Path):
            LOG.warning("tagged")
            return [dict(c, encoding="utf-8") for c in lines_reader(path)]

        after = snap.snapshot(self.root, tagged)
        self.assertEqual(len(snap.compare(self.before, after)), 4)
        self.assertEqual(snap.compare(self.before, after, text_only=True), [])

        def upper(path: Path):
            return [dict(c, text=c["text"].upper()) for c in lines_reader(path)]

        changed = snap.compare(
            self.before, snap.snapshot(self.root, upper), text_only=True
        )
        self.assertEqual(
            sorted({line.split(": ")[1] for line in changed}), ["a.txt", "b.txt"]
        )

    def test_another_format_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            snap.compare(dict(self.before, format=2), self.before)


class SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = _Folder(
            {"b.txt": "one\n", "a/c.TXT": "x\n", "a/d.md": "y\n", "x.bad": "?"}
        )
        self.root = self.folder.root

    def tearDown(self) -> None:
        self.folder.close()

    def names(self, **kwargs) -> list:
        return list(snap.snapshot(self.root, lines_reader, **kwargs)["files"])

    def test_include_takes_matching_files_in_every_subfolder_any_case(self) -> None:
        self.assertEqual(self.names(include=["*.txt"]), ["a/c.TXT", "b.txt"])

    def test_exclude_wins_over_include(self) -> None:
        self.assertEqual(self.names(include=["a/*"], exclude=["*.md"]), ["a/c.TXT"])

    def test_without_patterns_every_file_is_read_and_none_recorded(self) -> None:
        result = snap.snapshot(self.root, lines_reader)
        self.assertEqual(list(result["files"]), ["a/c.TXT", "a/d.md", "b.txt", "x.bad"])
        self.assertNotIn("include", result)
        self.assertNotIn("exclude", result)

    def test_patterns_are_recorded(self) -> None:
        result = snap.snapshot(self.root, lines_reader, exclude=["*.bad"])
        self.assertEqual((result["include"], result["exclude"]), ([], ["*.bad"]))

    def test_a_string_is_refused(self) -> None:
        with self.assertRaises(TypeError):
            snap.snapshot(self.root, lines_reader, include="*.txt")
        with self.assertRaises(TypeError):
            snap.snapshot(self.root, lines_reader, exclude=[1])

    def test_compare_names_a_different_selection(self) -> None:
        a = snap.snapshot(self.root, lines_reader, include=["*.txt"])
        b = snap.snapshot(self.root, lines_reader)
        lines = snap.compare(a, b)
        self.assertEqual(lines[0], "selection: include ['*.txt'] -> []")
        self.assertIn("only after: a/d.md", lines)
        self.assertEqual(snap.compare(a, a), [])

    def test_command_line_patterns_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "s.json"
            real = snap._scikitplot_reader
            snap._scikitplot_reader = lambda kwargs, documents: lines_reader
            try:
                with redirect_stdout(io.StringIO()):
                    code = snap.main(
                        [
                            "snapshot",
                            str(self.root),
                            "--out",
                            str(out_path),
                            "--include",
                            "*.txt",
                            "--include",
                            "*.md",
                            "--exclude",
                            "b.*",
                        ]
                    )
            finally:
                snap._scikitplot_reader = real
            self.assertEqual(code, 0)
            data = json.loads(out_path.read_text(encoding="ascii"))
            self.assertEqual(list(data["files"]), ["a/c.TXT", "a/d.md"])
            self.assertEqual(data["include"], ["*.txt", "*.md"])


class CommandLineTests(unittest.TestCase):
    def test_compare_exits_1_on_a_difference_and_0_without(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = {
                "format": 1,
                "repeat": 1,
                "files": {"a": {"result": "ok", "stable": True}},
            }
            changed = {
                "format": 1,
                "repeat": 1,
                "files": {"a": {"result": "error", "stable": True}},
            }
            paths = []
            for name, data in (("b.json", base), ("c.json", changed)):
                path = Path(tmp) / name
                path.write_text(json.dumps(data), encoding="utf-8")
                paths.append(str(path))
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(snap.main(["compare", paths[0], paths[0]]), 0)
                self.assertEqual(snap.main(["compare", paths[0], paths[1]]), 1)
            self.assertIn("changed: a: result 'ok' -> 'error'", out.getvalue())
            self.assertIn("1 difference(s).", out.getvalue())


#: A reader module for the command line and for child processes: the tests
#: name it with ``--reader FILE.py:read``, as a user names their own reader.
STAND_IN = '''
import logging
import os
import time
from pathlib import Path

LOG = logging.getLogger("scikitplot.stand_in")


def read(path, prefix=""):
    path = Path(path)
    if path.suffix == ".bad":
        raise KeyError(f"cannot read {path}")
    if path.suffix == ".slow":
        time.sleep(60)
    if path.suffix == ".crash":
        os._exit(3)
    if path.suffix == ".noisy":
        print("a library banner on standard output")
    text = path.read_text(encoding="utf-8")
    if not text:
        LOG.warning("empty file %s", path)
    return [
        {"text": prefix + line, "line": i, "raw": line.encode(), "span": (i, i + 1)}
        for i, line in enumerate(text.splitlines())
    ]
'''


class _StandIn:
    """A folder of files and the stand-in reader module, in one temp dir."""

    def __init__(self, files: dict) -> None:
        self.folder = _Folder(files)
        self.root = self.folder.root
        self.module = self.root.parent / "stand_in.py"
        self.module.write_text(STAND_IN, encoding="utf-8")
        self.spec = f"{self.module}:read"

    def close(self) -> None:
        self.folder.close()

    def run(self, *args: str) -> tuple[int, dict, str]:
        """Run ``snapshot`` on the folder; return status, snapshot, output."""
        out_path = self.root.parent / "s.json"
        out = io.StringIO()
        with redirect_stdout(out):
            code = snap.main(
                ["snapshot", str(self.root), "--out", str(out_path)]
                + ["--reader", self.spec, *args]
            )
        return code, json.loads(out_path.read_text(encoding="ascii")), out.getvalue()


class SampleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = _Folder(
            {"a1.txt": "1\n", "a2.txt": "2\n", "a3.TXT": "3\n", "b.md": "b\n", "c": "c\n", "d": "d\n"}
        )
        self.root = self.folder.root

    def tearDown(self) -> None:
        self.folder.close()

    def test_at_most_n_files_of_each_suffix_first_in_sorted_order(self) -> None:
        result = snap.snapshot(self.root, lines_reader, max_per_type=2)
        self.assertEqual(list(result["files"]), ["a1.txt", "a2.txt", "b.md", "c", "d"])
        self.assertEqual(result["max_per_type"], 2)
        one = snap.snapshot(self.root, lines_reader, max_per_type=1)
        self.assertEqual(list(one["files"]), ["a1.txt", "b.md", "c"])

    def test_the_sample_is_taken_after_the_patterns(self) -> None:
        result = snap.snapshot(
            self.root, lines_reader, exclude=["a1.*"], max_per_type=1
        )
        self.assertEqual(list(result["files"]), ["a2.txt", "b.md", "c"])

    def test_without_a_sample_nothing_is_recorded(self) -> None:
        self.assertNotIn("max_per_type", snap.snapshot(self.root, lines_reader))

    def test_bad_sample_sizes_are_refused(self) -> None:
        for value in (0, -1, True, 1.5, "2"):
            with self.assertRaises(ValueError):
                snap.snapshot(self.root, lines_reader, max_per_type=value)

    def test_compare_names_a_different_sample(self) -> None:
        a = snap.snapshot(self.root, lines_reader)
        b = snap.snapshot(self.root, lines_reader, max_per_type=1)
        lines = snap.compare(a, b)
        self.assertEqual(lines[0], "selection: max_per_type None -> 1")
        self.assertIn("only before: a2.txt", lines)


class TimingTests(unittest.TestCase):
    def test_each_file_records_its_fastest_read(self) -> None:
        folder = _Folder({"a.txt": "x\n"})
        try:
            calls = {"n": 0}

            def slow_first(path: Path):
                calls["n"] += 1
                if calls["n"] == 1:
                    import time  # noqa: PLC0415

                    time.sleep(0.2)
                return lines_reader(path)

            record = snap.snapshot(folder.root, slow_first, repeat=2)["files"]["a.txt"]
            self.assertLess(record["seconds"], 0.2)
            self.assertTrue(record["stable"])
        finally:
            folder.close()

    def test_seconds_never_make_a_difference(self) -> None:
        base = {"format": 1, "files": {"a": {"result": "ok", "seconds": 0.1}}}
        slow = {"format": 1, "files": {"a": {"result": "ok", "seconds": 9.0}}}
        self.assertEqual(snap.compare(base, slow), [])

    def test_timing_lines(self) -> None:
        before = {
            "run": {"seconds": 10, "peak_rss_mib": 100.0},
            "files": {
                "a": {"seconds": 0.5},
                "b": {"seconds": 0.1},
                "c": {"seconds": 3.0},
                "d": {"seconds": 1.0},
            },
        }
        after = {
            "run": {"seconds": 20, "peak_rss_mib": 150.5},
            "files": {
                "a": {"seconds": 2.0},
                "b": {"seconds": 0.5},
                "c": {"seconds": 4.0},
                "e": {"seconds": 9.0},
            },
        }
        self.assertEqual(
            snap.timing_lines(before, after),
            [
                "timing: seconds 10 -> 20 (x2.00)",
                "timing: peak_rss_mib 100 -> 150.5 (x1.50)",
                "slower: a: 0.5 s -> 2 s (x4.0)",
            ],
        )
        self.assertEqual(snap.timing_lines({"files": {}}, {"files": {}}), [])

    def test_peak_memory_is_measured_where_the_platform_allows(self) -> None:
        peak = snap.peak_rss_mib()
        if sys.platform.startswith("win"):
            self.assertIsNone(peak)
        else:
            self.assertGreater(peak, 1.0)

    def test_compare_timing_reports_without_changing_the_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name, seconds in (("a.json", 1.0), ("b.json", 5.0)):
                data = {
                    "format": 1,
                    "run": {"seconds": seconds * 10, "peak_rss_mib": 50},
                    "files": {"f": {"result": "ok", "stable": True, "seconds": seconds}},
                }
                path = Path(tmp) / name
                path.write_text(json.dumps(data), encoding="utf-8")
                paths.append(str(path))
            out = io.StringIO()
            with redirect_stdout(out):
                code = snap.main(["compare", paths[0], paths[1], "--timing"])
            self.assertEqual(code, 0)
            self.assertIn("timing: seconds 10 -> 50 (x5.00)", out.getvalue())
            self.assertIn("slower: f: 1 s -> 5 s (x5.0)", out.getvalue())


class ReaderSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stand_in = _StandIn({"b.txt": "one\ntwo\n"})

    def tearDown(self) -> None:
        self.stand_in.close()

    def test_a_reader_file_and_its_keyword_arguments(self) -> None:
        read = snap.make_reader(self.stand_in.spec, {"prefix": ">"})
        self.assertEqual(
            [c["text"] for c in read(self.stand_in.root / "b.txt")], [">one", ">two"]
        )

    def test_bad_reader_specs_are_refused(self) -> None:
        module = self.stand_in.module
        for spec in (
            "stand_in",
            str(module),
            f"{module}:",
            f"{module}:not-a-name",
            f"{module.parent / 'missing.py'}:read",
            f"{module.with_suffix('.txt')}:read",
        ):
            with self.assertRaises(ValueError, msg=spec):
                snap.make_reader(spec)
        with self.assertRaises(ValueError):
            snap.make_reader(f"{module}:absent")
        with self.assertRaises(ValueError):
            snap.make_reader(self.stand_in.spec, documents=True)

    def test_command_line_runs_a_reader_file(self) -> None:
        code, data, output = self.stand_in.run("--reader-kwargs", '{"prefix": ">"}')
        self.assertEqual(code, 0)
        self.assertEqual(data["files"]["b.txt"]["chunks"], 2)
        self.assertEqual(
            set(data["run"]), {"seconds", "peak_rss_mib", "python", "platform", "timeout"}
        )
        self.assertIn("1 file(s): 0 error(s), 0 unstable;", output)

    def test_show_prints_the_chunks(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            code = snap.main(
                ["show", str(self.stand_in.root / "b.txt"), "--reader", self.stand_in.spec]
            )
        self.assertEqual(code, 0)
        self.assertEqual(
            out.getvalue(), "--- chunk 0 ---\none\n--- chunk 1 ---\ntwo\n"
        )
        out = io.StringIO()
        with redirect_stdout(out):
            snap.main(
                ["show", str(self.stand_in.root / "b.txt"), "--json"]
                + ["--reader", self.stand_in.spec]
            )
        first = json.loads(out.getvalue().splitlines()[0])
        self.assertEqual((first["text"], first["line"], first["span"]), ("one", 0, [0, 1]))


class IsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stand_in = _StandIn(
            {
                "b.txt": "one\ntwo\n",
                "a/c.txt": "x\n",
                "empty.txt": "",
                "x.bad": "?",
                "n.noisy": "loud\n",
            }
        )

    def tearDown(self) -> None:
        self.stand_in.close()

    def test_an_isolated_run_records_what_an_in_process_run_records(self) -> None:
        in_process = snap.snapshot(
            self.stand_in.root, snap.make_reader(self.stand_in.spec)
        )
        code, isolated, _ = self.stand_in.run("--timeout", "60")
        self.assertEqual(code, 0)
        self.assertEqual(snap.compare(in_process, isolated), [])
        files = isolated["files"]
        self.assertEqual(files["x.bad"]["error"], "KeyError: 'cannot read <root>/x.bad'")
        self.assertEqual(files["empty.txt"]["warnings"], ["empty file <root>/empty.txt"])
        self.assertEqual(files["n.noisy"]["chunks"], 1)
        self.assertEqual(isolated["run"]["timeout"], 60.0)

    def test_a_slow_file_times_out_and_the_run_goes_on(self) -> None:
        (self.stand_in.root / "z.slow").write_text("never\n", encoding="utf-8")
        start = __import__("time").perf_counter()
        code, data, _ = self.stand_in.run("--timeout", "2", "--repeat", "1")
        self.assertLess(__import__("time").perf_counter() - start, 40)
        self.assertEqual(code, 0)
        self.assertEqual(
            data["files"]["z.slow"]["error"], "TimeoutError: no result within 2 s"
        )
        self.assertEqual(data["files"]["b.txt"]["result"], "ok")

    def test_a_crashing_file_is_recorded(self) -> None:
        (self.stand_in.root / "k.crash").write_text("?\n", encoding="utf-8")
        _, data, _ = self.stand_in.run("--timeout", "60", "--repeat", "1")
        self.assertEqual(
            data["files"]["k.crash"]["error"],
            "ChildProcessError: reader process ended with status 3 and no result",
        )

    def test_bad_isolation_arguments(self) -> None:
        for timeout in (0, -1, True, "1"):
            with self.assertRaises(ValueError):
                snap.isolated([sys.executable], timeout=timeout)
        with self.assertRaises(ValueError):
            snap.isolated([], timeout=1)

    def test_a_child_error_keeps_its_name(self) -> None:
        self.assertEqual(type(snap._exception("KeyError", "x")).__name__, "KeyError")
        self.assertEqual(
            type(snap._exception("not a name!", "x")).__name__, "ReaderError"
        )


if __name__ == "__main__":
    unittest.main()
