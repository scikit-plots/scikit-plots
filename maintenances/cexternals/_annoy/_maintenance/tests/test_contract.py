"""Isolated maintenance behavior tests. No project runtime is imported or edited."""
from pathlib import Path
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

TOOL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL))
import maintenance as m
LIVE = m.repository_root()
spec = importlib.util.spec_from_file_location('native_probe_tests', TOOL/'tools/verify_native.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)

class ContractTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='annoy-contract-')
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)/'repo with spaces'
        self.root.mkdir()
        shutil.copytree(LIVE/m.MAINT, self.root/m.MAINT,
                        ignore=shutil.ignore_patterns('history', '_backup', '__pycache__', 'evidence'))
        self.write(m.SKILL, (LIVE/m.SKILL).read_text())
        for name in m.KINDS:
            (self.root/'scikitplot'/name).mkdir(parents=True)
        for h in ('annoylib.h','kissrandom.h','annoy_type_support.h','mman.h'):
            self.write(m.RUNTIME+'/src/'+h, '// synthetic header\n')
        self.write(m.RUNTIME+'/__init__.py', 'from . import annoylib\n')
        self.write('scikitplot/annoy/__init__.py', 'from ..cexternals._annoy import Annoy, AnnoyIndex, _plotting\n')
        self.write('scikitplot/annoy/_annoy/annoylib.pxd.in', '\n'.join(
            'cdef extern from "../../cexternals/_annoy/src/'+h+'":\n    pass'
            for h in ('annoylib.h','kissrandom.h','annoy_type_support.h')))
        self.write('scikitplot/random/_kiss/kiss_random.pxd', 'cdef extern from "../../cexternals/_annoy/src/kissrandom.h":\n    pass\n')
        self.write('scikitplot/memmap/_memmap/mem_map.pxd', 'cdef extern from "../../cexternals/_annoy/src/mman.h":\n    pass\n')
        self.write('scikitplot/impute/_ann.py', 'from ..annoy._annoy import Index as AnnoyIndex\n')
        self.write('scikitplot/corpus/_backend.py', 'from scikitplot.annoy import Index\nfrom scikitplot.annoy._annoy import Index as Low\n')
        self.write('scikitplot/mcp/_corpus_annoy.py', 'from scikitplot.corpus import RetrievalIndex\n')
        log = self.write(m.CONTROL+'/evidence/fixture.log', 'Synthetic fixture, not production verification.\n')
        value = dict(schema_version=1,subsystem=m.MODULE,input_digest=m.input_digest(self.root),
            gates={n:dict(status='PASS',detail='synthetic fixture',log=log.relative_to(self.root).as_posix(),
                          sha256=m.digest(log.read_bytes())) for n in m.GATES})
        self.write(m.CONTROL+'/EVIDENCE.json', json.dumps(value))
        self.assertEqual(m.inspect(self.root,refresh=True)['maintenance_status'],'PASS')

    def write(self, rel, text):
        p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text);return p
    def edit_json(self, rel, fn):
        p=self.root/rel;v=json.loads(p.read_text());fn(v);p.write_text(json.dumps(v))
    def findings(self):
        manifest,_=m.load_contract(self.root)
        return m.architecture(self.root,manifest)[1]
    def cli(self,*args,script='review_subsystem.py'):
        return subprocess.run([sys.executable,'-B',str(TOOL/script),'--repo',str(self.root),*args],
            cwd=self.root.parent,text=True,capture_output=True,timeout=20)
    def test_01_read_only_foreign_cwd(self):
        before={p.relative_to(self.root):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        r=self.cli('--json');self.assertEqual(r.returncode,0,r.stderr+r.stdout)
        self.assertEqual(before,{p.relative_to(self.root):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
    def test_02_sample_compatibility(self):
        r=self.cli('--inventory','--json',script='tools/check_contract.py')
        self.assertEqual(r.returncode,0,r.stderr);self.assertIn('runtime_inventory',json.loads(r.stdout))
    def test_03_six_distinct_edges(self):
        manifest,_=m.load_contract(self.root);graph,errors=m.architecture(self.root,manifest)
        self.assertFalse(any(errors.values()));self.assertEqual({x['name'] for x in graph['consumers']},set(m.KINDS))
    def test_04_wrong_relative_path_same_basename(self):
        self.write('scikitplot/random/_kiss/kiss_random.pxd','cdef extern from "../../../cexternals/_annoy/src/kissrandom.h":\n    pass\n')
        self.assertTrue(self.findings()['dependencies'])
    def test_05_missing_header(self):
        (self.root/m.RUNTIME/'src/kissrandom.h').unlink()
        with self.assertRaises(m.ContractError):m.inspect(self.root)
    def test_06_comments_and_docstrings(self):
        self.write('scikitplot/new/example.py','"""from scikitplot.annoy import Index"""\n# import scikitplot.annoy\nimport random\n')
        self.write('scikitplot/new/example.pyx','"""cdef extern from "../cexternals/_annoy/src/ghost.h":\n pass\n"""\n# cdef extern from "../cexternals/_annoy/src/ghost.h":\n')
        self.assertFalse(any(self.findings().values()))
    def test_07_single_quoted_continuation(self):
        self.write('scikitplot/random/_kiss/kiss_random.pxd',"cdef extern from "+chr(92)+"\n    '../../cexternals/_annoy/src/kissrandom.h':\n    pass\n")
        self.assertFalse(self.findings()['dependencies'])
    def test_08_undeclared_consumers(self):
        self.write('scikitplot/new/binding.pxd','cdef extern from "../cexternals/_annoy/src/kissrandom.h":\n pass\n')
        self.write('scikitplot/new/use.py','from scikitplot.annoy import Index\n')
        errors=self.findings()['dependencies'];self.assertTrue(any('shared-source' in x for x in errors));self.assertTrue(any('index consumer' in x for x in errors))
    def test_09_nested_relative_upward_import(self):
        self.write(m.RUNTIME+'/bad.py','def f():\n    from scikitplot import random\n    from ...corpus import RetrievalIndex\n')
        self.assertEqual(len(self.findings()['planes']),2)
    def test_10_upward_cpp_and_fork(self):
        self.write('scikitplot/annoy/consumer.h','// fixture\n');self.write(m.RUNTIME+'/src/bad.cpp','#include "../../../annoy/consumer.h"\n')
        self.write('scikitplot/random/kissrandom.h','// fork\n');errors=self.findings()
        self.assertTrue(errors['planes']);self.assertTrue(any('duplicates' in x for x in errors['dependencies']))
    def test_11_runtime_imports_tooling(self):
        self.write('scikitplot/mcp/bad.py','import maintenances\nimport skills\n');self.assertEqual(len(self.findings()['planes']),2)
    def test_12_mcp_alias_and_dynamic_bypass(self):
        for code in ['from scikitplot import annoy as a','from importlib import import_module as load\nload("scikitplot.annoy")','__import__("scikitplot.cexternals._annoy")','import annoy']:
            with self.subTest(code=code):
                self.write('scikitplot/mcp/bad.py',code+'\n');self.assertTrue(any('through Corpus' in x for x in self.findings()['dependencies']))
    def test_13_compiled_contracts_required(self):
        self.write('scikitplot/impute/_ann.py','AnnoyIndex = object\n');self.write('scikitplot/corpus/_backend.py','from scikitplot.annoy import Index\n');self.write('scikitplot/mcp/_corpus_annoy.py','from scikitplot.corpus import CorpusBuilder\n')
        errors=self.findings()['dependencies']
        for name in ('impute','corpus','mcp'):self.assertTrue(any(x.startswith(name+': missing') for x in errors))
    def test_14_syntax_errors_fail_closed(self):
        self.write('scikitplot/mcp/bad.py','from scikitplot.annoy import\n');r=self.cli('--json')
        self.assertEqual(r.returncode,2);self.assertNotIn('Traceback',r.stderr)
    def test_15_unsafe_paths_and_links(self):
        for p in ['../out','/tmp/out','C:\\out','a/../b','a//b','a\\b','a\x00b']:
            with self.subTest(path=p),self.assertRaises(m.ContractError):m.safe_path(self.root,p,exists=False)
        (self.root/'link').symlink_to(self.root/m.SKILL)
        with self.assertRaises(m.ContractError):m.safe_path(self.root,'link')
    def test_16_required_review_lanes(self):
        self.edit_json(m.MAINT+'/REVIEW.json',lambda v:v['lanes'].pop())
        self.assertEqual(self.cli('--json').returncode,2)
    def test_17_profile_execution_and_release_disabling(self):
        p=self.root/m.MAINT/'REVIEW.json';original=p.read_bytes()
        for fn in [lambda v:v.update(command='unexpected'),lambda v:v['release_gates'].pop(),lambda v:v['lanes'][0]['checks'].append('shell'),lambda v:v.update(schema_version=True)]:
            p.write_bytes(original);self.edit_json(m.MAINT+'/REVIEW.json',fn);self.assertEqual(self.cli('--json').returncode,2)
    def test_18_duplicate_consumer(self):
        self.edit_json(m.MAINT+'/MAINTENANCE.json',lambda v:v['consumers'].append(v['consumers'][0]))
        with self.assertRaises(m.ContractError):m.inspect(self.root)
    def test_19_duplicate_json_keys(self):
        self.write(m.MAINT+'/REVIEW.json','{"schema_version":1,"schema_version":1}')
        with self.assertRaises(m.ContractError):m.inspect(self.root)
    def test_20_drift_invalidates_evidence(self):
        self.write(m.RUNTIME+'/src/kissrandom.h','// changed\n');r=m.inspect(self.root)
        errors={k:v for l in r['lanes'] for k,v in l['checks'].items()};self.assertTrue(errors['inventory']);self.assertTrue(errors['evidence'])
    def test_21_refresh_cannot_bless_architecture(self):
        p=self.root/m.CONTROL/'TRACKER.json';before=p.read_bytes();self.write('scikitplot/mcp/bad.py','import scikitplot.annoy\n')
        self.assertEqual(m.inspect(self.root,refresh=True)['maintenance_status'],'FAIL');self.assertEqual(p.read_bytes(),before)
    def test_22_refresh_preserves_provenance(self):
        p=self.root/m.CONTROL/'STATE.json';before=p.read_bytes();self.write(m.RUNTIME+'/src/kissrandom.h','// deliberate edit\n')
        self.assertEqual(m.inspect(self.root,refresh=True)['maintenance_status'],'FAIL');self.assertEqual(p.read_bytes(),before)
    def test_23_malformed_evidence_prevents_refresh(self):
        p=self.root/m.CONTROL/'TRACKER.json';before=p.read_bytes();self.write(m.RUNTIME+'/src/kissrandom.h','// edit\n')
        self.edit_json(m.CONTROL+'/EVIDENCE.json',lambda v:v.update(command='unexpected'))
        with self.assertRaises(m.ContractError):m.inspect(self.root,refresh=True)
        self.assertEqual(p.read_bytes(),before)
    def test_24_missing_and_tampered_evidence(self):
        self.write(m.CONTROL+'/evidence/fixture.log','tampered\n');self.assertEqual(m.inspect(self.root)['maintenance_status'],'FAIL')
        (self.root/m.CONTROL/'EVIDENCE.json').unlink();self.assertEqual(m.inspect(self.root)['maintenance_status'],'FAIL')
    def test_25_unavailable_is_not_release_pass(self):
        self.edit_json(m.CONTROL+'/EVIDENCE.json',lambda v:v['gates'].update(consumer_build=dict(status='UNAVAILABLE',detail='not run',log=None,sha256=None)))
        self.assertEqual(self.cli('--json').returncode,0);self.assertEqual(self.cli('--release','--json').returncode,1)
    def test_26_skill_and_handoff_and_residue(self):
        self.write(m.SKILL,'# missing frontmatter\n')
        with self.assertRaises(m.ContractError):m.inspect(self.root)
        self.write(m.SKILL,(LIVE/m.SKILL).read_text())
        self.write(m.CONTROL+'/FRESH_CHAT_HANDOFF.md','[broken](missing.md) check_trackers.py review_subsystem.py')
        self.assertTrue(m.handoff_checks(self.root,m.load_contract(self.root)[0]))
        p=self.write(m.RUNTIME+'/tests/__pycache__/old.pyc','untouched');self.assertFalse(m.hygiene(self.root));self.assertEqual(p.read_text(),'untouched')
        self.write(m.CONTROL+'/__pycache__/bad.pyc','bad');self.assertTrue(m.hygiene(self.root))
    def test_27_closed_pipe(self):
        p=subprocess.Popen([sys.executable,'-B',str(TOOL/'review_subsystem.py'),'--repo',str(self.root),'--json'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        p.stdout.close();error=p.stderr.read().decode();p.stderr.close();self.assertEqual(p.wait(timeout=20),1);self.assertNotIn('Traceback',error);self.assertNotIn('Exception ignored',error)

class NativeToolTests(unittest.TestCase):
    def test_missing_or_compile_only_is_not_pass(self):
        for s in ('UNAVAILABLE','GREEN_COMPILE_ONLY'):
            self.assertEqual(native.overall({'a':dict(status='GREEN'),'b':dict(status=s)}),'UNAVAILABLE')
        self.assertEqual(native.overall({}),'UNAVAILABLE')
    def test_timeout_is_failure(self):
        with mock.patch.object(native.subprocess,'run',side_effect=subprocess.TimeoutExpired(['fixture'],60)):
            ok,detail=native.run(['fixture']);self.assertFalse(ok);self.assertIn('timeout',detail)
    def test_missing_executable_has_diagnostic(self):
        with mock.patch.object(native.subprocess,'run',side_effect=FileNotFoundError('fixture')):
            ok,detail=native.run(['fixture']);self.assertFalse(ok);self.assertIn('could not run',detail)
    def test_no_shell_and_failed_probe(self):
        with mock.patch.object(native.subprocess,'run',return_value=subprocess.CompletedProcess([],0)) as run:
            self.assertTrue(native.run(['fixture'])[0]);self.assertFalse(run.call_args.kwargs['shell'])
        self.assertEqual(native.overall({'bad':dict(status='RED')}),'FAIL')

if __name__=='__main__':unittest.main()
