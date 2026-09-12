from __future__ import annotations
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

LIVE = Path(__file__).resolve().parents[4]
TOOL = Path(__file__).resolve().parents[1] / "tools" / "check_contract.py"
spec = importlib.util.spec_from_file_location("mcp_contract", TOOL)
c = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(c)
MAN = c.load_json(Path(__file__).resolve().parents[2] / "MAINTENANCE.json")


class MCPContractTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name) / "repo"
        for rel in [MAN["runtime_root"], MAN["maintenance_root"], str(Path(MAN["skill"]).parent)]:
            src = LIVE / rel
            dst = self.root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

    def tearDown(self):
        self.td.cleanup()

    def inspect(self):
        return c.inspect(self.root)

    def write(self, rel, text):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def sync_fingerprints(self):
        fp, entries = c.runtime_fingerprint(self.root, MAN["runtime_root"])
        inv = c.inventory(entries)
        tracker = self.root / MAN["maintenance_root"] / "_maintenance" / "TRACKER.json"
        tv = json.loads(tracker.read_text())
        tv["runtime_fingerprint"] = fp
        tv["inventory"] = inv
        tracker.write_text(json.dumps(tv, indent=2) + "\n")
        evidence = self.root / MAN["maintenance_root"] / "_maintenance" / "EVIDENCE.json"
        ev = json.loads(evidence.read_text())
        ev["runtime_fingerprint"] = fp
        evidence.write_text(json.dumps(ev, indent=2) + "\n")

    def test_current_maintenance_plane_passes(self):
        self.assertEqual(self.inspect()["maintenance_status"], "PASS")

    def test_current_runtime_structure_passes(self):
        self.assertEqual(self.inspect()["runtime_status"], "PASS")

    def test_release_is_blocked_without_all_live_evidence(self):
        self.assertEqual(self.inspect()["release_status"], "BLOCKED")

    def test_foreign_cwd_cli(self):
        r = subprocess.run(
            [sys.executable, "-B", str(TOOL), "--repo", str(self.root), "--json"],
            cwd=self.root.parent,
            text=True,
            capture_output=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertEqual(json.loads(r.stdout)["runtime_status"], "PASS")

    def test_missing_required_runtime_file_fails(self):
        target = self.root / MAN["runtime_contract"]["required_files"][0]
        target.unlink()
        self.sync_fingerprints()
        self.assertEqual(self.inspect()["runtime_status"], "FAIL")

    def test_missing_required_symbol_fails(self):
        rel = "scikitplot/mcp/_version.py"
        self.write(rel, "# symbol removed\n")
        self.sync_fingerprints()
        self.assertTrue(self.inspect()["errors"]["contracts"])

    def test_pydantic_module_scope_leak_fails(self):
        self.write("scikitplot/mcp/_demo.py", "import pydantic\n")
        self.sync_fingerprints()
        self.assertTrue(self.inspect()["errors"]["optionality"])

    def test_mcp_sdk_module_scope_leak_fails(self):
        self.write("scikitplot/mcp/_demo.py", "import mcp\n")
        self.sync_fingerprints()
        self.assertTrue(self.inspect()["errors"]["optionality"])

    def test_corpus_module_scope_leak_fails(self):
        self.write("scikitplot/mcp/_demo.py", "from scikitplot.corpus import CorpusBuilder\n")
        self.sync_fingerprints()
        self.assertTrue(self.inspect()["errors"]["optionality"])

    def test_handrolled_protocol_fallback_fails(self):
        self.write(
            "scikitplot/mcp/_fallback.py",
            'x="jsonrpc"\ny="protocolVersion"\nz="tools/list"\n',
        )
        self.sync_fingerprints()
        self.assertTrue(self.inspect()["errors"]["protocol"])

    def test_open_wire_model_fails(self):
        p = self.root / "scikitplot/mcp/_server.py"
        text = p.read_text().replace('ConfigDict(extra="forbid")', 'ConfigDict(extra="ignore")')
        p.write_text(text)
        self.sync_fingerprints()
        self.assertTrue(self.inspect()["errors"]["security"])

    def test_runtime_cannot_import_maintenance(self):
        self.write("scikitplot/mcp/_demo.py", "import maintenances.mcp\n")
        self.sync_fingerprints()
        self.assertTrue(self.inspect()["errors"]["planes"])

    def test_empty_skill_fails_maintenance(self):
        (self.root / MAN["skill"]).write_text("# empty\n")
        self.assertEqual(self.inspect()["maintenance_status"], "FAIL")

    def test_review_metadata_is_not_command_surface(self):
        p = self.root / MAN["maintenance_root"] / "REVIEW.json"
        v = json.loads(p.read_text())
        v["command"] = "echo bad"
        p.write_text(json.dumps(v))
        with self.assertRaises(c.ContractError):
            self.inspect()

    def test_inventory_drift_fails_maintenance(self):
        self.write("scikitplot/mcp/new_runtime_file.py", "# drift\n")
        self.assertTrue(self.inspect()["errors"]["inventory"])
        self.assertEqual(self.inspect()["maintenance_status"], "FAIL")

    def test_refresh_refuses_broken_runtime_contract(self):
        (self.root / "scikitplot/mcp/_version.py").unlink()
        with self.assertRaises(c.ContractError):
            c.inspect(self.root, refresh=True)


if __name__ == "__main__":
    unittest.main()
