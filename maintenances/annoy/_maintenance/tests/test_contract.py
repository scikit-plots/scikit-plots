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
        rels=[MAN['runtime_root'],MAN['maintenance_root'],str(Path(MAN['skill']).parent),str(Path(MAN['upstream']['headers'][0]['path']).parent)]
        for rel in rels:
            src=LIVE/rel; dst=self.root/rel; dst.parent.mkdir(parents=True,exist_ok=True)
            if src.is_dir(): shutil.copytree(src,dst,dirs_exist_ok=True)
            elif src.is_file(): shutil.copy2(src,dst)
        # Ensure the three wide-repo marker dirs exist even in this focused fixture.
        for rel in ('scikitplot','maintenances','skills'): (self.root/rel).mkdir(parents=True,exist_ok=True)
        shutil.copy2(LIVE/MAN['build']['project_meson'], self.root/MAN['build']['project_meson'])
    def tearDown(self): self.td.cleanup()
    def inspect(self): return m.inspect(self.root)
    def write(self,rel,text):
        p=self.root/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(text,encoding='utf-8'); return p
    def add_generator(self): self.write(MAN['generation']['template_tool'],'# fixture tempita helper\n')

    def test_current_maintenance_contract_passes(self): self.assertEqual(self.inspect()['maintenance_status'],'PASS')
    def test_current_runtime_fails_on_missing_generator(self):
        r=self.inspect(); self.assertEqual(r['runtime_status'],'FAIL'); self.assertTrue(any('missing template generator' in x for x in r['errors']['generation']))
    def test_foreign_cwd_cli(self):
        r=subprocess.run([sys.executable,'-B',str(TOOL),'--repo',str(self.root),'--json'],cwd=self.root.parent,text=True,capture_output=True)
        self.assertEqual(r.returncode,0,r.stderr+r.stdout)
    def test_runtime_can_pass_when_generator_is_present(self):
        self.add_generator(); r=self.inspect(); self.assertEqual(r['runtime_status'],'PASS',r['errors'])
    def test_wrong_relative_header_fails(self):
        site=MAN['upstream']['headers'][0]['extern_sites'][0]; p=self.root/site; t=p.read_text(); t=t.replace('../../cexternals/_annoy/src/annoylib.h','../../../cexternals/_annoy/src/annoylib.h'); p.write_text(t)
        self.assertTrue(self.inspect()['errors']['dependencies'])
    def test_missing_header_fails_runtime(self):
        (self.root/MAN['upstream']['headers'][0]['path']).unlink(); self.assertTrue(self.inspect()['errors']['dependencies'])
    def test_vendored_header_copy_fails(self):
        self.write(MAN['runtime_root']+'/annoylib.h','// bad local fork\n'); self.assertTrue(self.inspect()['errors']['dependencies'])
    def test_high_level_backend_ownership_drift_fails(self):
        p=self.root/MAN['public_contract']['high_level_index']; p.write_text(p.read_text().replace('from ..cexternals._annoy import Annoy','from somewhere_else import Annoy',1))
        self.assertTrue(self.inspect()['errors']['dependencies'])
    def test_cython_contract_symbol_drift_fails(self):
        p=self.root/MAN['public_contract']['cython_template']; p.write_text(p.read_text().replace('def supported_dtypes(', 'def supported_dtypes_REMOVED(',1))
        self.assertTrue(self.inspect()['errors']['public_surface'])
    def test_checked_in_generated_output_fails(self):
        self.write('scikitplot/annoy/_annoy/annoylib.pyx','# generated output should be build-only\n'); self.assertTrue(self.inspect()['errors']['generation'])
    def test_inactive_legacy_cpp_cannot_silently_become_build_source(self):
        p=self.root/MAN['generation']['meson']; p.write_text(p.read_text()+"\n# accidental activation\nx = ['annoymodule.cpp']\n")
        self.assertTrue(self.inspect()['errors']['generation'])
    def test_build_cpp_contract_fails_closed(self):
        p=self.root/MAN['build']['extension_meson']; p.write_text(p.read_text().replace('cython_language=cpp','cython_language=c'))
        self.assertTrue(self.inspect()['errors']['build'])
    def test_project_cpp17_contract_fails_closed(self):
        p=self.root/MAN['build']['project_meson']; p.write_text(p.read_text().replace('cpp_std=c++17','cpp_std=c++14',1))
        self.assertTrue(self.inspect()['errors']['build'])
    def test_runtime_cannot_import_maintenance(self):
        self.write(MAN['runtime_root']+'/bad_plane.py','import maintenances\n'); self.assertTrue(self.inspect()['errors']['planes'])
    def test_empty_skill_fails_maintenance(self):
        (self.root/MAN['skill']).write_text('# empty\n'); self.assertTrue(self.inspect()['errors']['hygiene'])
    def test_review_metadata_is_not_command_surface(self):
        p=self.root/MAN['maintenance_root']/'REVIEW.json'; v=json.loads(p.read_text()); v['command']='echo bad'; p.write_text(json.dumps(v))
        with self.assertRaises(m.ContractError): m.inspect(self.root)
    def test_release_is_blocked(self): self.assertEqual(self.inspect()['release_status'],'BLOCKED')
    def test_refresh_refuses_current_runtime_failure(self):
        with self.assertRaises(m.ContractError): m.inspect(self.root,refresh=True)

if __name__=='__main__': unittest.main()
