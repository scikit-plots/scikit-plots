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
spec = importlib.util.spec_from_file_location("cli_contract", TOOL)
c = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(c)
MAN = c.load_json(Path(__file__).resolve().parents[2] / "MAINTENANCE.json")


class CLIContractTests(unittest.TestCase):
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

    def test_current_runtime_reports_known_contract_findings(self):
        r = self.inspect()
        self.assertEqual(r["runtime_status"], "FAIL")
        joined = "\n".join(r["errors"]["contracts"])
        self.assertIn("spec.capabilities", joined)
        self.assertIn("EXTENDING.md", joined)

    def test_release_is_blocked(self):
        self.assertEqual(self.inspect()["release_status"], "BLOCKED")

    def test_foreign_cwd_cli(self):
        r = subprocess.run(
            [sys.executable, "-B", str(TOOL), "--repo", str(self.root), "--json"],
            cwd=self.root.parent,
            text=True,
            capture_output=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertEqual(json.loads(r.stdout)["maintenance_status"], "PASS")

    def test_synthetic_contract_fixes_can_make_runtime_pass(self):
        self.write("scikitplot/_cli/EXTENDING.md", "# Extending the CLI\n\nOne neutral registry; delegation preserves ownership.\n")
        loader = self.root / "scikitplot/_cli/loader.py"
        text = loader.read_text()
        text += "\n\ndef _capability_contract_reference(spec):\n    return spec.capabilities\n"
        loader.write_text(text)
        self.sync_fingerprints()
        self.assertEqual(self.inspect()["runtime_status"], "PASS")

    def test_missing_required_runtime_file_fails(self):
        (self.root / MAN["runtime_contract"]["required_files"][0]).unlink()
        self.sync_fingerprints()
        self.assertEqual(self.inspect()["runtime_status"], "FAIL")

    def test_exit_code_drift_fails(self):
        p = self.root / "scikitplot/_cli/exit_codes.py"
        p.write_text(p.read_text().replace("UNAVAILABLE: Final = 69", "UNAVAILABLE: Final = 68"))
        self.sync_fingerprints()
        self.assertTrue(self.inspect()["errors"]["contracts"])

    def test_click_module_scope_leak_fails(self):
        self.write("scikitplot/_cli/app.py", "import click\n")
        self.sync_fingerprints()
        self.assertTrue(self.inspect()["errors"]["optionality"])

    def test_backend_module_scope_leak_fails(self):
        self.write("scikitplot/_cli/app.py", "import scikitplot.mcp\n")
        self.sync_fingerprints()
        self.assertTrue(self.inspect()["errors"]["optionality"])

    def test_registry_handler_eager_import_fails(self):
        p = self.root / "scikitplot/_cli/registry.py"
        p.write_text(p.read_text() + "\nfrom ._commands import info\n")
        self.sync_fingerprints()
        self.assertTrue(self.inspect()["errors"]["optionality"])

    def test_bare_print_fails_io_contract(self):
        p = self.root / "scikitplot/_cli/app.py"
        p.write_text(p.read_text() + "\nprint('bad')\n")
        self.sync_fingerprints()
        self.assertTrue(self.inspect()["errors"]["io"])

    def test_frontend_parity_surface_drift_fails(self):
        p = self.root / "scikitplot/_cli/tests/test_cli_frontend_parity.py"
        p.write_text("def test_nothing(): pass\n")
        self.sync_fingerprints()
        self.assertTrue(self.inspect()["errors"]["parity"])

    def test_delegate_target_drift_fails(self):
        p = self.root / "scikitplot/_cli/registry.py"
        p.write_text(p.read_text().replace("scikitplot.mcp.__main__:main", "scikitplot.mcp:main"))
        self.sync_fingerprints()
        self.assertTrue(self.inspect()["errors"]["contracts"])

    def test_runtime_cannot_import_maintenance(self):
        self.write("scikitplot/_cli/app.py", "import maintenances._cli\n")
        self.sync_fingerprints()
        r = self.inspect()
        self.assertTrue(r["errors"]["optionality"] or r["errors"]["planes"])

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
        self.write("scikitplot/_cli/new_runtime_file.py", "# drift\n")
        self.assertTrue(self.inspect()["errors"]["inventory"])
        self.assertEqual(self.inspect()["maintenance_status"], "FAIL")

    def test_refresh_refuses_known_broken_runtime_contract(self):
        with self.assertRaises(c.ContractError):
            c.inspect(self.root, refresh=True)


if __name__ == "__main__":
    unittest.main()
