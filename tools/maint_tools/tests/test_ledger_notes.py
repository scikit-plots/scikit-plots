# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tools/maint_tools/ledger_notes.py (probed upcoming_changes notes)."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "ledger_notes.py"
SPEC = importlib.util.spec_from_file_location("ledger_notes", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

TEMPLATE = textwrap.dedent(
    """\
    ---
    title: "<short descriptive title>"
    kind: "<bug|security|other>"
    towncrier_type: "<enhancement|fix|unknown>"
    ---
    """
)
SECTIONS = "import sys\nprint('  scikitplot.corpus   Corpus')\nprint('  other   Other')\n"


def _note(**over):
    n = copy.deepcopy(MODULE.EXAMPLE["notes"][0])
    n.update(over)
    return n


class _Repo(unittest.TestCase):
    """A throwaway checkout: template, section lister and one source file."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "upcoming_changes").mkdir()
        (self.root / "upcoming_changes" / "TEMPLATE.md").write_text(TEMPLATE, encoding="utf-8")
        tools = self.root / "tools" / "maint_tools"
        tools.mkdir(parents=True)
        (tools / "generate_towncrier_sections.py").write_text(SECTIONS, encoding="utf-8")
        for rel, text in {
            "scikitplot/corpus/_reader.py": "def Read_Thing(): pass\n",
            "scikitplot/corpus/tests/test_x.py": "only_in_tests = 1\n",
            "scikitplot/corpus/tests/helpers.py": "only_in_test_helpers = 1\n",
            "scikitplot/externals/vendored.py": "only_vendored = 1\n",
            "scikitplot/_externals/docs.py": "only_private = 1\n",
        }.items():
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.root / rel).write_text(text, encoding="utf-8")
        snap = self.root / "snap"
        (snap / "api").mkdir(parents=True)
        (snap / "registry.json").write_text('[".txt", ".pdf"]', encoding="utf-8")
        (snap / "api" / "corpus.txt").write_text(
            "scikitplot.corpus (x)\n\nfunction  to_rows(a) -> list\n          to_rows docs\nclass     Builder()\n",
            encoding="utf-8",
        )
        self.snap = snap
        self.ctx = MODULE.Context(self.root, snap)

    def tearDown(self):
        self._tmp.cleanup()

    def catalog(self, notes, **top):
        data = dict(schema=1, discovered={"@r": "Review on {ref} ({date})", "@review": "Review"}, notes=notes, **top)
        path = self.root / "cat.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = MODULE.main(["--root", str(self.root), *argv])
        return rc, out.getvalue(), err.getvalue()


class ProbeTests(_Repo):
    def test_absent_and_present_ignore_tests_and_vendored_code(self):
        for pattern in ("only_in_tests", "only_in_test_helpers", "only_vendored"):
            self.assertTrue(MODULE.run_probe({"kind": "absent", "pattern": pattern}, self.ctx)[0])
        ok, text = MODULE.run_probe({"kind": "present", "pattern": "read_thing"}, self.ctx)
        self.assertTrue(ok)  # case-insensitive
        self.assertIn("`scikitplot/corpus/_reader.py`", text)
        self.assertFalse(MODULE.run_probe({"kind": "absent", "pattern": "only_private"}, self.ctx)[0])
        ok, text = MODULE.run_probe({"kind": "absent", "pattern": "only_private", "public": True}, self.ctx)
        self.assertTrue(ok)
        self.assertIn("public submodules", text)

    def test_ext_and_api_probes_read_the_snapshot(self):
        self.assertTrue(MODULE.run_probe({"kind": "ext-unregistered", "ext": ".dcm"}, self.ctx)[0])
        ok, text = MODULE.run_probe({"kind": "ext-unregistered", "ext": ".pdf"}, self.ctx)
        self.assertFalse(ok)
        self.assertIn("(2 entries)", text)
        ok, text = MODULE.run_probe({"kind": "api-present", "module": "corpus", "pattern": "to_rows"}, self.ctx)
        self.assertTrue(ok)
        self.assertIn("`to_rows`", text)
        # Indented docstring lines are not names.
        self.assertTrue(MODULE.run_probe({"kind": "api-absent", "module": "corpus", "pattern": "docs"}, self.ctx)[0])

    def test_unusable_probes_raise(self):
        bad = [
            {"kind": "nope"},
            {"kind": "absent"},
            {"kind": "absent", "pattern": "("},
            {"kind": "absent", "pattern": "x", "path": "../.."},
        ]
        for p in bad:
            with self.assertRaises(MODULE.CatalogError, msg=p):
                MODULE.run_probe(p, self.ctx)
        with self.assertRaisesRegex(MODULE.CatalogError, "--snapshot"):
            MODULE.run_probe({"kind": "ext-unregistered", "ext": ".x"}, MODULE.Context(self.root))


class CatalogTests(_Repo):
    def test_validation_names_the_problem(self):
        cases = {
            "missing": _note(summary=None),
            "unknown": dict(_note(), extra=1),
            "relative .md": _note(path="../escape.md"),
            "kind in": _note(kind="feature"),
            "no discovered alias": _note(discovered="@missing"),
            "list of strings": _note(edges="one"),
        }
        for message, n in cases.items():
            if n.get("summary", "") is None:
                del n["summary"]
            with self.assertRaisesRegex(MODULE.CatalogError, message):
                MODULE.load_catalog(self.catalog([n]), self.root)
        with self.assertRaisesRegex(MODULE.CatalogError, "duplicate"):
            MODULE.load_catalog(self.catalog([_note(), _note()]), self.root)
        path = self.root / "bad.json"
        path.write_text('{"schema": 2, "notes": []}', encoding="utf-8")
        with self.assertRaisesRegex(MODULE.CatalogError, "schema"):
            MODULE.load_catalog(path, self.root)

    def test_example_catalog_is_valid(self):
        path = self.root / "example.json"
        path.write_text(json.dumps(MODULE.EXAMPLE), encoding="utf-8")
        notes = MODULE.load_catalog(path, self.root)
        self.assertEqual(notes[0]["discovered"], "Capability review, gap probed on main {ref} ({date})")

    def test_section_is_derived_overridden_or_unknown(self):
        sections = MODULE.towncrier_sections(self.root)
        self.assertEqual(sections, ["other", "scikitplot.corpus"])
        base = MODULE.load_catalog(self.catalog([_note()]), self.root)[0]
        self.assertEqual(MODULE.section_for(base, sections), "scikitplot.corpus")
        self.assertEqual(MODULE.section_for(dict(base, section="other"), sections), "other")
        self.assertEqual(MODULE.section_for(dict(base, release="unknown"), sections), "unknown")
        with self.assertRaisesRegex(MODULE.CatalogError, "not configured"):
            MODULE.section_for(dict(base, section="nope"), sections)


class CommandTests(_Repo):
    def test_render_plans_writes_then_is_idempotent(self):
        cat = str(self.catalog([_note(discovered="@r")]))
        args = ("render", cat, "--snapshot", str(self.snap), "--ref", "abc1234", "--date", "2026-01-02")
        rc, out, _ = self.run_main(*args)
        self.assertEqual((rc, out.strip()), (0, "create 1 (plan only; --apply writes)"))
        target = self.root / "upcoming_changes" / "scikitplot/corpus/example-reader-gap.md"
        self.assertFalse(target.exists())
        self.assertEqual(self.run_main(*args, "--apply")[0], 0)
        text = target.read_text(encoding="utf-8")
        self.assertIn('discovered_during: "Review on abc1234 (2026-01-02)"', text)
        self.assertIn('towncrier_section: "scikitplot.corpus"', text)
        self.assertIn("finds nothing.", text)  # the probe sentence joined the evidence
        self.assertEqual(text.count("\n## "), 12)
        rc, out, _ = self.run_main(*args, "--apply")
        self.assertEqual((rc, out.strip()), (0, "unchanged 1"))

    def test_render_refuses_differing_notes_without_force(self):
        cat = str(self.catalog([_note()]))
        target = self.root / "upcoming_changes" / "scikitplot/corpus/example-reader-gap.md"
        target.parent.mkdir(parents=True)
        target.write_text("edited by hand\n", encoding="utf-8")
        args = ("render", cat, "--snapshot", str(self.snap), "--apply")
        self.assertEqual(self.run_main(*args)[0], 1)
        self.assertEqual(target.read_text(encoding="utf-8"), "edited by hand\n")
        self.assertEqual(self.run_main("render", cat, "--force")[0], 2)  # --force needs --apply
        self.assertEqual(self.run_main(*args, "--force")[0], 0)
        self.assertNotEqual(target.read_text(encoding="utf-8"), "edited by hand\n")

    def test_failing_probe_blocks_render_and_fails_check(self):
        cat = str(self.catalog([_note(probes=[{"kind": "absent", "pattern": "read_thing"}])]))
        rc, out, _ = self.run_main("render", cat, "--apply")
        self.assertEqual(rc, 1)
        self.assertIn("PROBE FAILED", out)
        self.assertFalse((self.root / "upcoming_changes" / "scikitplot").exists())
        report = self.root / "r.json"
        rc, out, _ = self.run_main("check", cat, "--json", str(report))
        self.assertEqual(rc, 1)
        self.assertIn("STALE scikitplot/corpus/example-reader-gap.md [absent]", out)
        self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["stale"][0]["kind"], "absent")

    def test_missing_snapshot_is_a_usage_error(self):
        rc, _, err = self.run_main("check", str(self.catalog([_note()])))
        self.assertEqual(rc, 2)
        self.assertIn("--snapshot", err)

    def test_example_command_prints_valid_json(self):
        rc, out, _ = self.run_main("example")
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out), MODULE.EXAMPLE)


class StoredCatalogTests(unittest.TestCase):
    def test_every_stored_catalog_loads(self):
        root = SCRIPT.parents[2]
        catalogs = sorted((SCRIPT.parent / "catalogs").glob("*.json"))
        self.assertTrue(catalogs)
        for path in catalogs:
            notes = MODULE.load_catalog(path, root)
            for n in notes:
                for p in n["probes"]:
                    self.assertIn(p["kind"], MODULE.PROBES, msg=n["path"])


if __name__ == "__main__":
    unittest.main()
