#!/usr/bin/env python3
from __future__ import annotations
import io, json, logging as pylog, os, sys, threading
from pathlib import Path

def discover() -> Path:
    here=Path(__file__).resolve()
    for p in (here,*here.parents):
        if (p/"scikitplot/logging/_logging.py").is_file():
            return p
    raise RuntimeError("repo root not found")

ROOT=discover()
sys.path.insert(0,str(ROOT))
import scikitplot.logging as public
from scikitplot.logging import _logging as core
from scikitplot._cli import logging as cli

def reset():
    logger=pylog.getLogger("scikitplot")
    for h in list(logger.handlers): logger.removeHandler(h)
    logger.setLevel(pylog.NOTSET); logger.propagate=True
    logger.findCaller=pylog.Logger.findCaller.__get__(logger,pylog.Logger)
    core._logger=None; core._log_counter_per_token.clear(); cli._HANDLER=None
    for k in ("SKPLT_LOGGING_LEVEL","SKPLT_VERBOSE","SKPLT_LOGGING_AUTO_CONFIG"): os.environ.pop(k,None)

print("PUBLIC", public.__name__, public.__file__)
print("CORE", core.__name__, core.__file__)
print("PUBLIC_ALL_MATCH", set(public.__all__)==set(core.__all__))
try:
    public.getLevelName
    print("PUBLIC_STDLIB_FALLBACK PASS")
except Exception as exc:
    print("PUBLIC_STDLIB_FALLBACK FAIL",type(exc).__name__,str(exc))

reset(); ids=[]; errors=[]
def worker():
    try: ids.append(id(core.get_logger()))
    except BaseException as exc: errors.append(repr(exc))
ts=[threading.Thread(target=worker) for _ in range(32)]
[t.start() for t in ts]; [t.join() for t in ts]
print("THREAD_SINGLETON unique_ids=",len(set(ids)),"errors=",errors,"handlers=",len(pylog.getLogger("scikitplot").handlers))

reset(); buf=io.StringIO(); h=pylog.StreamHandler(buf); lg=pylog.getLogger("scikitplot"); lg.addHandler(h); lg.setLevel(pylog.DEBUG); lg.propagate=False; core._logger=lg
core.error_log("boom")
print("ERROR_LOG",repr(buf.getvalue()))

reset(); os.environ["SKPLT_LOGGING_LEVEL"]="DEBUG"; print("ENV_LEVEL",core.get_logger().level)
reset(); h=core.AlwaysStdErrHandler("stderr"); old=h.setStream(sys.stdout); print("STREAM_SWITCH",old is sys.stderr,h.stream is sys.stdout)

reset()
try: core.__getattr__("definitely_missing_contract_name")
except AttributeError: pass
print("GETATTR_HANDLERS",len(pylog.getLogger("scikitplot").handlers))

reset(); records=[]
class Capture(pylog.Handler):
    def emit(self,r): records.append((r.pathname,r.lineno,r.funcName,r.getMessage()))
lg=core.get_logger()
for existing in list(lg.handlers): lg.removeHandler(existing)
lg.addHandler(Capture()); lg.setLevel(pylog.INFO); lg.propagate=False
def direct(): lg.info("direct")
direct()
print("CALLER",records[-1] if records else None)

reset(); cli.configure(0); before=len(pylog.getLogger("scikitplot").handlers); core.get_logger(); after=len(pylog.getLogger("scikitplot").handlers)
print("CLI_HANDLER_COUNTS",before,after)
print("UNKNOWN_FORMATTER",repr(core._make_default_formatter("NOT_A_FORMAT")))
