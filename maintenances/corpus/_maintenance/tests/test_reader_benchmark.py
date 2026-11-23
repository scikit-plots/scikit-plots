"""
Tests for maintenances/corpus/_maintenance/tools/reader_benchmark.py.

The reader is a stand-in module and every reference used here is in the
standard library, so these tests need no scikit-plots build and no optional
package.
Run: ``python -B maintenances/corpus/_maintenance/tests/test_reader_benchmark.py``.
"""

from __future__ import annotations

import collections
import importlib.util
import io
import json
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "reader_benchmark.py"
SPEC = importlib.util.spec_from_file_location("reader_benchmark", TOOL)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)

#: Reads text files as they are, repeats every XML text node once per
#: ancestor (as a naive reader would), refuses ``.bad`` files.
STAND_IN = '''
from pathlib import Path


def read(path):
    path = Path(path)
    if path.suffix == ".bad":
        raise KeyError("refused")
    if path.suffix == ".xml":
        return [{"text": "alpha beta gamma"}, {"text": "alpha beta gamma"}, {"text": "beta"}]
    return [{"text": path.read_text(encoding="utf-8")}]
'''

NESTED_XML = "<doc><sec><p>alpha <b>beta</b> gamma</p></sec></doc>"

ALTO = (
    '<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"><Layout><Page><PrintSpace>'
    '<TextLine><String CONTENT="Geschichts-" SUBS_TYPE="HypPart1"'
    ' SUBS_CONTENT="Geschichtsauffassung"/></TextLine>'
    '<TextLine><String CONTENT="auffassung" SUBS_TYPE="HypPart2"'
    ' SUBS_CONTENT="Geschichtsauffassung"/><String CONTENT="leisten"/></TextLine>'
    "</PrintSpace></Page></Layout></alto>"
)


class _Folder:
    def __init__(self, files: dict) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.root = self.base / "corpus"
        self.root.mkdir()
        for rel, data in files.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(data, bytes):
                path.write_bytes(data)
            else:
                path.write_text(data, encoding="utf-8")
        self.module = self.base / "stand_in.py"
        self.module.write_text(STAND_IN, encoding="utf-8")
        self.spec = f"{self.module}:read"

    def close(self) -> None:
        self._tmp.cleanup()


class WordTests(unittest.TestCase):
    def test_words_are_normalised_case_folded_and_counted(self) -> None:
        self.assertEqual(
            bench.words("Straße \ufb01le FILE file. \uff26\uff29\uff2c\uff25"),
            collections.Counter({"strasse": 1, "file": 4}),
        )

    def test_scores(self) -> None:
        ref = bench.words("a b b c")
        self.assertEqual(
            bench.scores(ref, bench.words("a b x")),
            {"ref_words": 4, "words": 3, "recall": 0.5, "precision": 0.6667},
        )
        self.assertEqual(bench.scores(ref, bench.words(""))["precision"], None)
        self.assertEqual(bench.scores(bench.words(""), ref)["recall"], None)
        repeated = bench.scores(bench.words("a"), bench.words("a a a"))
        self.assertEqual((repeated["recall"], repeated["precision"]), (1.0, 0.3333))


class ReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def file(self, name: str, data) -> Path:
        path = self.dir / name
        if isinstance(data, bytes):
            path.write_bytes(data)
        else:
            path.write_text(data, encoding="utf-8")
        return path

    def test_nested_xml_text_is_read_once(self) -> None:
        text = bench._xml(self.file("n.xml", NESTED_XML))
        self.assertEqual(bench.words(text), bench.words("alpha beta gamma"))

    def test_an_alto_word_broken_across_lines_counts_once(self) -> None:
        text = bench._xml(self.file("a.xml", ALTO))
        self.assertEqual(
            bench.words(text), bench.words("Geschichtsauffassung leisten")
        )

    def test_malformed_xml_has_no_reference(self) -> None:
        with self.assertRaises(bench.NoReference):
            bench._xml(self.file("m.xml", "<a><b></a>"))

    def test_html_text_without_scripts_or_styles(self) -> None:
        page = (
            "<html><head><title>T</title><style>p{}</style>"
            "<script>var x = 1;</script></head>"
            "<body><p>caf&eacute; &amp; more</p><noscript>no</noscript></body></html>"
        )
        self.assertEqual(
            bench.words(bench._html(self.file("p.html", page))),
            bench.words("T café & more"),
        )

    def test_text_needs_utf8(self) -> None:
        self.assertEqual(bench._text(self.file("b.txt", "﻿hello")), "hello")
        with self.assertRaises(bench.NoReference):
            bench._text(self.file("l.txt", "caf\xe9".encode("latin-1")))

    def test_opendocument_content(self) -> None:
        path = self.dir / "d.odt"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("content.xml", "<office><text><p>open document</p></text></office>")
        self.assertEqual(bench.words(bench._opendocument(path)), bench.words("open document"))
        empty = self.dir / "e.odt"
        with zipfile.ZipFile(empty, "w") as archive:
            archive.writestr("other.xml", "<a/>")
        with self.assertRaises(bench.NoReference):
            bench._opendocument(empty)

    def test_zip_members_with_a_reference_and_one_nested_level(self) -> None:
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as archive:
            archive.writestr("deep.txt", "inner words")
        deeper = io.BytesIO()
        with zipfile.ZipFile(deeper, "w") as archive:
            archive.writestr("inner.zip", inner.getvalue())
        path = self.dir / "z.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("a.md", "top words")
            archive.writestr("img.png", b"\x89PNG")
            archive.writestr("n.xml", NESTED_XML)
            archive.writestr("inner.zip", inner.getvalue())
            archive.writestr("deeper.zip", deeper.getvalue())
        self.assertEqual(
            bench.words(bench._zip(path)),
            bench.words("top words alpha beta gamma inner words"),
        )

    def test_an_oversized_archive_has_no_reference(self) -> None:
        path = self.dir / "big.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("a.txt", "x" * 100)
        limit = bench.MAX_MEMBER_BYTES
        bench.MAX_MEMBER_BYTES = 10
        try:
            with self.assertRaises(bench.NoReference):
                bench._zip(path)
        finally:
            bench.MAX_MEMBER_BYTES = limit


class RunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = _Folder(
            {
                "a.txt": "one two two",
                "b.md": "three",
                "n.xml": NESTED_XML,
                "pic.png": b"\x89PNG",
                "x.bad": "?",
                "sub/c.txt": "four",
            }
        )
        self.read = bench.snapshot_tool.make_reader(self.folder.spec)

    def tearDown(self) -> None:
        self.folder.close()

    def test_records_scores_errors_and_skipped_types(self) -> None:
        result = bench.run(self.folder.root, self.read)
        files = result["files"]
        self.assertEqual(list(files), ["a.txt", "b.md", "n.xml", "sub/c.txt"])
        self.assertEqual(result["skipped"], {".bad": 1, ".png": 1})
        self.assertEqual((files["a.txt"]["recall"], files["a.txt"]["precision"]), (1.0, 1.0))
        self.assertEqual(files["n.xml"]["recall"], 1.0)
        self.assertEqual(files["n.xml"]["precision"], 0.4286)
        self.assertEqual(files["a.txt"]["reference"], "text")

    def test_a_reader_error_is_recall_zero_with_the_folder_hidden(self) -> None:
        (self.folder.root / "y.txt").write_text("words here", encoding="utf-8")
        module = self.folder.module
        module.write_text(
            STAND_IN.replace('if path.suffix == ".bad":', 'if path.name == "y.txt":'),
            encoding="utf-8",
        )
        read = bench.snapshot_tool.make_reader(self.folder.spec)
        record = bench.run(self.folder.root, read)["files"]["y.txt"]
        self.assertEqual(record["recall"], 0.0)
        self.assertEqual(record["error"], "KeyError: 'refused'")

    def test_selection_and_sample(self) -> None:
        result = bench.run(
            self.folder.root, self.read, exclude=["*.md"], max_per_type=1
        )
        self.assertEqual(list(result["files"]), ["a.txt", "n.xml"])
        self.assertEqual(result["max_per_type"], 1)
        self.assertEqual(result["exclude"], ["*.md"])
        for bad in (0, True, 1.5):
            with self.assertRaises(ValueError):
                bench.run(self.folder.root, self.read, max_per_type=bad)
        with self.assertRaises(ValueError):
            bench.run(self.folder.root / "a.txt", self.read)


class ReportAndCompareTests(unittest.TestCase):
    def record(self, recall, precision, error=None, suffix=".txt") -> dict:
        return {
            "suffix": suffix,
            "bytes": 1,
            "reference": "text",
            "ref_words": 10,
            "words": 10,
            "recall": recall,
            "precision": precision,
            "seconds": 0.1,
            "ref_seconds": 0.1,
            "error": error,
            "ref_error": None,
        }

    def data(self, **files) -> dict:
        return {"format": 1, "files": files, "skipped": {".png": 2}}

    def test_report_table_and_low_files(self) -> None:
        data = self.data(
            a=self.record(1.0, 1.0),
            b=self.record(0.5, 1.0),
            c=self.record(0.0, None, error="ValueError: no reader"),
        )
        table, low = bench.report(data, min_recall=0.9, min_precision=0.9)
        self.assertTrue(table[0].startswith("type"))
        self.assertTrue(table[1].startswith(".txt"))
        self.assertIn("    3      1", table[1])
        self.assertEqual(table[-1], "no reference for: .png 2")
        self.assertEqual(
            low, ["low: b: recall 0.5", "low: c: recall 0.0, reader error ValueError"]
        )
        self.assertEqual(bench.report(data)[1], [])

    def test_compare_flags_worse_files_only(self) -> None:
        before = self.data(a=self.record(0.5, 1.0), b=self.record(1.0, 1.0), gone=self.record(1, 1))
        better = self.data(a=self.record(0.99, 1.0), b=self.record(1.0, 1.0), new=self.record(1, 1))
        lines, worse = bench.compare(before, better)
        self.assertFalse(worse)
        self.assertEqual(
            lines,
            ["changed: a: recall 0.5 -> 0.99", "only before: gone", "only after: new"],
        )
        lines, worse = bench.compare(better, before)
        self.assertTrue(worse)
        failed = self.data(a=self.record(0.99, 1.0, error="KeyError: x"))
        self.assertTrue(bench.compare(better, failed)[1])
        self.assertEqual(bench.compare(before, before), ([], False))

    def test_other_formats_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            bench.report({"format": 2, "files": {}})
        with self.assertRaises(ValueError):
            bench.compare({"format": 1, "files": {}}, {"files": {}})


class CommandLineTests(unittest.TestCase):
    def test_run_report_and_compare(self) -> None:
        folder = _Folder({"a.txt": "one two", "n.xml": NESTED_XML})
        try:
            out = folder.base / "b.json"
            text = io.StringIO()
            with redirect_stdout(text):
                code = bench.main(
                    ["run", str(folder.root), "--out", str(out), "--reader", folder.spec]
                )
            self.assertEqual(code, 0)
            self.assertIn("2 file(s) benchmarked, 0 reader error(s)", text.getvalue())
            data = json.loads(out.read_text(encoding="ascii"))
            self.assertEqual(set(data["run"]), {"seconds", "peak_rss_mib", "python", "platform"})
            with redirect_stdout(io.StringIO()):
                self.assertEqual(bench.main(["report", str(out)]), 0)
                self.assertEqual(bench.main(["report", str(out), "--min-precision", "0.9"]), 1)
                self.assertEqual(bench.main(["compare", str(out), str(out)]), 0)
        finally:
            folder.close()


if __name__ == "__main__":
    unittest.main()
