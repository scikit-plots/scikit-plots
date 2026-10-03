"""
Fuzz the packs-and-formats layer: round trip, JSON validity, leaks,
determinism and idempotence over randomized documents (round 12, CP-057).

Run from anywhere; the repository root is found from this file's location.
Exits non-zero on any failure.
"""
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[4]))
import warnings as _warnings

_warnings.simplefilter("ignore")
import json, random, csv, io, sys, traceback
from scikitplot.cleanprompt import FluentCleanPrompt
rng = random.Random(7)
KEYS = ["mrn","email","Patient-MRN","patientMrn","db_password","api_key","note","n","x","phone","ssn","diagnosis","billingEmail","id"]
VALS = lambda: rng.choice(["00412345","ann@example.com",'a"b\\c',"é ü","", "x"*300, "+1 555 010 4477","078-05-1120", "[EMAIL-1]", "-9900000001", "line\nbreak", "tab\tt", "password: hunter2hunter2", "Note: MRN: 00412345.\nnext: ok", "k: \u00e9\"q\\\\ \U0001F600"])
NUMS = ["1e5","-0","1.0E+2","12345678","0","-12.5","5550100123"]
def jdoc(depth=0):
    if depth>2 or rng.random()<.3:
        r=rng.random()
        if r<.5: return json.dumps(VALS(), ensure_ascii=rng.random()<.5)
        if r<.8: return rng.choice(NUMS)
        return rng.choice(["true","false","null"])
    if rng.random()<.5:
        return "{"+", ".join(f'{json.dumps(rng.choice(KEYS))}: {jdoc(depth+1)}' for _ in range(rng.randint(0,4)))+"}"
    return "["+", ".join(jdoc(depth+1) for _ in range(rng.randint(0,4)))+"]"
crashes=[]
fails={"rt":0,"json":0,"leak":0,"det":0,"idem":0,"crash":0}; ex={}
def note(k,d):
    fails[k]+=1; ex.setdefault(k,d)
N=4000
for i in range(N):
    fmt = rng.choice(["json","jsonl","csv","env","text","shell","python","ini","yaml"])
    if fmt=="json": t=jdoc()
    elif fmt=="jsonl": t="\n".join(jdoc() for _ in range(3))+("\n" if rng.random()<.5 else "")
    elif fmt=="csv":
        b=io.StringIO(); w=csv.writer(b, lineterminator=rng.choice(["\n","\r\n"]))
        hdr=rng.sample(KEYS,3); w.writerow(hdr)
        for _ in range(3): w.writerow([VALS() for _ in hdr])
        t=b.getvalue()
    elif fmt=="python":
        t="".join(f'{rng.choice(["api_key","db_password","x","email"])} = {json.dumps(VALS())}\n' for _ in range(3))
    else:
        sep={"env":"=","shell":"=","ini":" = ","yaml":": ","text":": "}[fmt]
        t="".join(f'{rng.choice(KEYS)}{sep}{VALS().replace(chr(10)," ")}\n' for _ in range(3))
    mode=rng.choice(["auto","all"])
    try:
        c=FluentCleanPrompt().packs(mode).materialize()
        e=c.encode_text(t, fmt, name="f")
        if c.decode(e.text)!=t: note("rt",(fmt,t,e.text))
        if fmt=="json": json.loads(e.text)
        if fmt=="jsonl":
            for l in e.text.split("\n"):
                if l.strip(): json.loads(l)
        for s in ("ann@example.com","078-05-1120","+1 555 010 4477"):
            if s in e.text: note("leak",(fmt,s,e.text[:200]))
        c2=FluentCleanPrompt().packs(mode).materialize()
        if c2.encode_text(t,fmt,name="f").text!=e.text: note("det",(fmt,t))
        c3=FluentCleanPrompt().packs(mode).materialize()
        if fmt in ("json","jsonl","csv","env","ini","yaml","shell") and c3.encode_text(e.text,fmt,name="f").text!=e.text: note("idem",(fmt,e.text))
    except ValueError as x:
        if fmt in ("json","jsonl"): note("json",(fmt,t,repr(x)))
        else: note("crash",(fmt,t,repr(x)))
    except Exception as x:
        note("crash",(fmt,t,traceback.format_exc().strip().splitlines()[-1][:160])); crashes.append(traceback.format_exc().strip().splitlines()[-1][:120])
print("documents:", N, "failures:", fails)
for k,v in ex.items(): print("==",k, repr(v)[:700])
import collections; print(collections.Counter(crashes).most_common(8))

sys.exit(1 if any(fails.values()) else 0)
