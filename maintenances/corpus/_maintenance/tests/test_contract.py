from __future__ import annotations
import importlib.util, json, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path

LIVE=Path(__file__).resolve().parents[4]
TOOL=Path(__file__).resolve().parents[1]/'tools'/'check_contract.py'
spec=importlib.util.spec_from_file_location('contract',TOOL); c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
MAN=c.load_json(Path(__file__).resolve().parents[2]/'MAINTENANCE.json')

class CorpusContractTests(unittest.TestCase):
    def setUp(self):
        self.td=tempfile.TemporaryDirectory(); self.root=Path(self.td.name)/'repo'
        for rel in [MAN['runtime_root'],MAN['maintenance_root'],str(Path(MAN['skill']).parent)]:
            src=LIVE/rel; dst=self.root/rel; dst.parent.mkdir(parents=True,exist_ok=True)
            if src.is_dir(): shutil.copytree(src,dst,dirs_exist_ok=True)
            else: shutil.copy2(src,dst)
    def tearDown(self): self.td.cleanup()
    def inspect(self): return c.inspect(self.root)
    def write(self,rel,text):
        p=self.root/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(text,encoding='utf-8'); return p
    def populate_contract_runtime(self):
        for item in MAN['runtime_contract']['required_contracts']:
            body='\n'.join(f'class {s}: pass' for s in item.get('symbols',[])) or '# contract marker\n'
            self.write(item['path'],body+'\n')
        self.write(MAN['runtime_root']+'/tests/test_smoke.py','def test_smoke(): assert True\n')
    def test_current_maintenance_plane_passes(self): self.assertEqual(self.inspect()['maintenance_status'],'PASS')
    def test_current_snapshot_fails_runtime_closed(self): self.assertEqual(self.inspect()['runtime_status'],'FAIL')
    def test_release_is_blocked(self): self.assertEqual(self.inspect()['release_status'],'BLOCKED')
    def test_foreign_cwd_cli(self):
        r=subprocess.run([sys.executable,'-B',str(TOOL),'--repo',str(self.root),'--json'],cwd=self.root.parent,text=True,capture_output=True)
        self.assertEqual(r.returncode,0,r.stderr+r.stdout); self.assertEqual(json.loads(r.stdout)['runtime_status'],'FAIL')
    def test_populated_contract_can_clear_runtime_shape(self):
        self.populate_contract_runtime(); self.assertEqual(self.inspect()['runtime_status'],'PASS')
    def test_missing_required_module_fails(self):
        self.populate_contract_runtime(); Path(self.root/MAN['runtime_contract']['required_contracts'][0]['path']).unlink(); self.assertEqual(self.inspect()['runtime_status'],'FAIL')
    def test_missing_symbol_fails(self):
        self.populate_contract_runtime(); item=next(x for x in MAN['runtime_contract']['required_contracts'] if x.get('symbols')); self.write(item['path'],'# missing symbol\n'); self.assertTrue(self.inspect()['errors']['contracts'])
    def test_runtime_cannot_import_maintenance(self):
        self.populate_contract_runtime(); self.write(MAN['runtime_root']+'/bad_plane.py','import maintenances\n'); self.assertTrue(self.inspect()['errors']['planes'])
    def test_empty_skill_fails_maintenance(self):
        (self.root/MAN['skill']).write_text('# empty\n'); self.assertEqual(self.inspect()['maintenance_status'],'FAIL')
    def test_review_metadata_is_not_command_surface(self):
        p=self.root/MAN['maintenance_root']/'REVIEW.json'; v=json.loads(p.read_text()); v['command']='echo bad'; p.write_text(json.dumps(v))
        with self.assertRaises(c.ContractError): self.inspect()
    def test_inventory_drift_fails_maintenance(self):
        self.write(MAN['runtime_root']+'/new.py','# drift\n'); self.assertTrue(self.inspect()['errors']['inventory']); self.assertEqual(self.inspect()['maintenance_status'],'FAIL')
    def test_refresh_refuses_missing_runtime(self):
        with self.assertRaises(c.ContractError): c.inspect(self.root,refresh=True)

if __name__=='__main__': unittest.main()
