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


if __name__ == "__main__":
    unittest.main()
