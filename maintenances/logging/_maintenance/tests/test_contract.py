from __future__ import annotations
import importlib.util, json, os, shutil, subprocess, sys
from pathlib import Path
import pytest

THIS=Path(__file__).resolve()
WIDE=THIS
while not (WIDE/"scikitplot").is_dir():
    if WIDE.parent==WIDE: raise RuntimeError("wide root not found")
    WIDE=WIDE.parent
CHECK=WIDE/"maintenances/logging/_maintenance/tools/check_contract.py"
spec=importlib.util.spec_from_file_location("logging_contract",CHECK)
cc=importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)

def make_repo(tmp_path: Path) -> Path:
    r=tmp_path/"repo"
    for rel in ("scikitplot/logging","maintenances/logging","skills/logging"):
        src=WIDE/rel; dst=r/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copytree(src,dst)
    cli=WIDE/"scikitplot/_cli/logging.py"
    (r/"scikitplot/_cli").mkdir(parents=True,exist_ok=True)
    shutil.copy2(cli,r/"scikitplot/_cli/logging.py")
    return r

def write_clean_runtime(r: Path):
    core=r/"scikitplot/logging/_logging.py"
    core.write_text("""
import logging as _logging
_HANDLER_MARKER="_skplt_internal_handler"
def _default_logging_level(): return _logging.WARNING
def get_logger():
    logger=_logging.getLogger("scikitplot")
    logger.setLevel(_default_logging_level())
    _handler=_logging.StreamHandler()
    if not logger.handlers:
        logger.addHandler(_handler)
    return logger
class GoogleLogFormatter(_logging.Formatter): pass
class AlwaysStdErrHandler(_logging.StreamHandler):
    @property
    def stream(self): return self._stream
    @stream.setter
    def stream(self,value): self._stream = value
def setLevel(level): get_logger().setLevel(level)
def set_verbosity(level): setLevel(level)
def log(level,msg,*args,**kwargs): get_logger().log(level,msg,*args,**kwargs)
def error_log(error_msg,*args,level=_logging.ERROR,**kwargs): log(level,error_msg,*args,**kwargs)
def log_every_n(*a,**k): pass
def log_first_n(*a,**k): pass
def sanitize_log_message(x): return x
def __getattr__(name): return getattr(_logging,name)
def __dir__(): return sorted(set(globals())|set(dir(_logging)))
def _make_default_formatter(formatter="GOOGLE_FORMAT"):
    if formatter=="GOOGLE_FORMAT": return GoogleLogFormatter()
    raise ValueError("Unknown formatter")
def _logger_find_caller(stack_info=False, stacklevel=1):
    return _logging.currentframe().f_back.f_code.co_filename, 1, "caller", None
""",encoding="utf-8")
    (r/"scikitplot/logging/__init__.py").write_text(
        "from . import _logging\nfrom ._logging import *\n__getattr__=_logging.__getattr__\n__dir__=_logging.__dir__\n__all__=[]\n",encoding="utf-8")
    t=r/"scikitplot/logging/tests/test__logging.py"
    t.write_text("from .. import _logging as splog\ndef test_smoke(): assert splog.get_logger()\n",encoding="utf-8")
    (r/"scikitplot/_cli/logging.py").write_text('import logging\n_HANDLER=None\ndef configure():\n logger=logging.getLogger("scikitplot")\n if not logger.handlers: logger.addHandler(logging.StreamHandler())\n',encoding="utf-8")
    ev=r/"maintenances/logging/_maintenance/EVIDENCE.json"
    d=json.loads(ev.read_text()); d["runtime_tree_fingerprint"]=cc.tree_fingerprint(r); ev.write_text(json.dumps(d,indent=2)+"\n")

def test_baseline_maintenance_pass_runtime_fail():
    p=cc.payload(WIDE); assert p["maintenance_status"]=="PASS"; assert p["runtime_status"]=="FAIL"

def test_new_layout_is_inventory():
    p=cc.payload(WIDE); assert p["inventory"]["runtime_files"]>=5; assert p["inventory"]["focused_runtime_test_files"]==1

def test_old_single_file_path_is_not_required():
    assert not (WIDE/"scikitplot/logging.py").exists(); assert (WIDE/"scikitplot/logging/_logging.py").is_file()

def test_facade_compat_finding_present():
    assert any("LOG-PKG-001" in x for x in cc.runtime_findings(WIDE))

def test_stale_test_import_finding_present():
    assert any("LOG-TEST-001" in x for x in cc.runtime_findings(WIDE))

def test_behavior_lock_test_finding_present():
    assert any("LOG-TEST-002" in x for x in cc.runtime_findings(WIDE))

def test_previous_runtime_findings_still_present():
    f="\n".join(cc.runtime_findings(WIDE))
    for marker in ("LOG-ERR-001","LOG-ENV-001","LOG-HDL-001","LOG-CLI-001","LOG-CALL-001","LOG-ATTR-001","LOG-FMT-001"):
        assert marker in f

def test_missing_core_is_red(tmp_path):
    r=make_repo(tmp_path); (r/"scikitplot/logging/_logging.py").unlink()
    assert any("_logging.py is missing" in x for x in cc.runtime_findings(r))

def test_missing_facade_is_red(tmp_path):
    r=make_repo(tmp_path); (r/"scikitplot/logging/__init__.py").unlink()
    assert any("__init__.py is missing" in x for x in cc.runtime_findings(r))

def test_runtime_plane_violation_is_red(tmp_path):
    r=make_repo(tmp_path); p=r/"scikitplot/logging/_logging.py"; p.write_text(p.read_text()+"\nimport maintenances.logging\n")
    assert any("maintenance/skill plane" in x for x in cc.runtime_findings(r))

def test_metadata_command_surface_fails_maintenance(tmp_path):
    r=make_repo(tmp_path); p=r/"maintenances/logging/REVIEW.json"; d=json.loads(p.read_text()); d["command"]="echo nope"; p.write_text(json.dumps(d))
    assert cc.payload(r)["maintenance_status"]=="FAIL"

def test_missing_skill_fails_maintenance(tmp_path):
    r=make_repo(tmp_path); (r/"skills/logging/SKILL.md").unlink()
    assert cc.payload(r)["maintenance_status"]=="FAIL"

def test_shallow_skill_fails_maintenance(tmp_path):
    r=make_repo(tmp_path); (r/"skills/logging/SKILL.md").write_text("# short\n")
    assert cc.payload(r)["maintenance_status"]=="FAIL"

def test_fingerprint_drift_fails_maintenance(tmp_path):
    r=make_repo(tmp_path); p=r/"scikitplot/logging/_logging.py"; p.write_text(p.read_text()+"\n# drift\n")
    assert any("runtime_tree_fingerprint" in x for x in cc.maintenance_errors(r))

def test_update_refuses_red_runtime(tmp_path):
    r=make_repo(tmp_path)
    cp=subprocess.run([sys.executable,str(r/"maintenances/logging/_maintenance/tools/check_contract.py"),"--repo",str(r),"--update"],capture_output=True,text=True)
    assert cp.returncode!=0 and "refusing --update" in (cp.stdout+cp.stderr)

def test_foreign_cwd_discovery():
    cp=subprocess.run([sys.executable,str(CHECK),"--json"],cwd="/tmp",capture_output=True,text=True)
    assert cp.returncode==0
    assert json.loads(cp.stdout)["maintenance_status"]=="PASS"

def test_synthetic_repaired_runtime_can_pass(tmp_path):
    r=make_repo(tmp_path); write_clean_runtime(r)
    assert cc.payload(r)["runtime_status"]=="PASS"

def test_synthetic_repaired_maintenance_can_pass(tmp_path):
    r=make_repo(tmp_path); write_clean_runtime(r)
    assert cc.payload(r)["maintenance_status"]=="PASS"

@pytest.mark.parametrize("missing",["MAINTENANCE.json","REVIEW.json","_maintenance/STATE.json","_maintenance/FRESH_CHAT_HANDOFF.md"])
def test_required_maintenance_files_are_guarded(tmp_path,missing):
    r=make_repo(tmp_path); (r/"maintenances/logging"/missing).unlink()
    assert cc.payload(r)["maintenance_status"]=="FAIL"
