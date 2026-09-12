from __future__ import annotations
import importlib.util, json, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path

LIVE=Path(__file__).resolve().parents[4]
TOOL=Path(__file__).resolve().parents[1]/'tools'/'check_contract.py'
spec=importlib.util.spec_from_file_location('contract',TOOL); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
MAN=m.load_json(Path(__file__).resolve().parents[2]/'MAINTENANCE.json')

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.td=tempfile.TemporaryDirectory(); self.root=Path(self.td.name)/'repo'
        paths=[MAN['runtime_root'],MAN['maintenance_root'],str(Path(MAN['skill']).parent),str(Path(MAN['upstream']['headers'][0]['path']).parent)]
        for rel in paths:
            src=LIVE/rel; dst=self.root/rel; dst.parent.mkdir(parents=True,exist_ok=True)
            if src.is_dir(): shutil.copytree(src,dst,dirs_exist_ok=True)
            else: shutil.copy2(src,dst)
    def tearDown(self): self.td.cleanup()
    def inspect(self): return m.inspect(self.root)
    def write(self,rel,text):
        p=self.root/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(text,encoding='utf-8'); return p
    def test_current_contract_passes(self): self.assertEqual(self.inspect()['maintenance_status'],'PASS')
    def test_foreign_cwd_cli(self):
        r=subprocess.run([sys.executable,'-B',str(TOOL),'--repo',str(self.root),'--json'],cwd=self.root.parent,text=True,capture_output=True)
        self.assertEqual(r.returncode,0,r.stderr+r.stdout)
    def test_wrong_relative_header_fails(self):
        site=MAN['upstream']['headers'][0]['extern_sites'][0]; p=self.root/site; t=p.read_text(); needle='cdef extern from \"../../cexternals/_annoy/src/'; t=t.replace(needle,'cdef extern from \"../../../cexternals/_annoy/src/',1); p.write_text(t)
        self.assertTrue(self.inspect()['errors']['dependencies'])
    def test_missing_header_fails(self):
        (self.root/MAN['upstream']['headers'][0]['path']).unlink();
        with self.assertRaises(m.ContractError): self.inspect()
    def test_runtime_cannot_import_maintenance(self):
        self.write(MAN['runtime_root']+'/bad_plane.py','import maintenances\n'); self.assertTrue(self.inspect()['errors']['planes'])
    def test_required_symbol_drift_fails(self):
        p=self.root/MAN['public_contract']['stub']; t=p.read_text(); name=MAN['public_contract']['required_symbols'][0]; p.write_text(t.replace(name,name+'_REMOVED'))
        self.assertTrue(self.inspect()['errors']['public_surface'])
    def test_build_cpp_contract_fails_closed(self):
        p=self.root/MAN['build']['meson']; p.write_text(p.read_text().replace('cython_language=cpp','cython_language=c'))
        self.assertTrue(self.inspect()['errors']['build'])
    def test_empty_skill_fails(self):
        (self.root/MAN['skill']).write_text('# empty\n'); self.assertTrue(self.inspect()['errors']['hygiene'])
    def test_review_metadata_is_not_command_surface(self):
        p=self.root/MAN['maintenance_root']/'REVIEW.json'; v=json.loads(p.read_text()); v['command']='echo bad'; p.write_text(json.dumps(v))
        with self.assertRaises(m.ContractError): m.inspect(self.root)
    def test_release_is_blocked_without_native_evidence(self): self.assertEqual(self.inspect()['release_status'],'BLOCKED')
    def test_refresh_refuses_architecture_breakage(self):
        site=MAN['upstream']['headers'][0]['extern_sites'][0]; p=self.root/site; needle='cdef extern from \"../../cexternals/_annoy/src/'; p.write_text(p.read_text().replace(needle,'cdef extern from \"../../../cexternals/_annoy/src/',1))
        with self.assertRaises(m.ContractError): m.inspect(self.root,refresh=True)

if __name__=='__main__': unittest.main()
