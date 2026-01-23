python - <<'PY'
import importlib
import inspect
import pkgutil
import re
import sys
import types
import scikitplot

SENTINEL = object()

def check_module(modname: str):
    try:
        m = importlib.import_module(modname)
    except Exception as e:
        return ("IMPORT_FAIL", modname, repr(e))

    ga = getattr(m, "__getattr__", None)
    if ga is None or not callable(ga):
        return None  # no module-level __getattr__

    # Core invariants (must hold)
    mro_val = getattr(m, "__mro__", SENTINEL)
    miss_val = getattr(m, "THIS_ATTRIBUTE_SHOULD_NOT_EXIST_12345", SENTINEL)

    ok_mro = (mro_val is SENTINEL)
    ok_missing = (miss_val is SENTINEL)

    # Heuristic-free source checks (pattern detection, not behavior guessing)
    try:
        src = inspect.getsource(ga)
    except OSError:
        src = ""

    bad_default_none = bool(re.search(r"getattr\([^,]+,\s*name\s*,\s*None\s*\)", src))
    bad_hasattr = "hasattr(" in src
    side_effect_logger = "get_logger(" in src or ".debug(" in src

    status = "OK" if (ok_mro and ok_missing and not bad_default_none) else "BAD"
    notes = []
    if not ok_mro: notes.append("__mro__ resolved (should not)")
    if not ok_missing: notes.append("missing attr resolved (should not)")
    if bad_default_none: notes.append("uses getattr(..., None)")
    if bad_hasattr: notes.append("uses hasattr() inside __getattr__")
    if side_effect_logger: notes.append("side effects inside __getattr__")
    return (status, modname, "; ".join(notes) if notes else "")

results = []
for modinfo in pkgutil.walk_packages(scikitplot.__path__, scikitplot.__name__ + "."):
    r = check_module(modinfo.name)
    if r:
        results.append(r)

print("Found scikitplot modules with __getattr__:", len(results))
for status, name, note in sorted(results):
    print(f"{status:4}  {name}  {note}")

bad = [r for r in results if r[0] == "BAD"]
print("\nBAD count:", len(bad))
sys.exit(1 if bad else 0)
PY

######################################################################

python - <<'PY'
import pathlib, re, scikitplot.logging as m

root = pathlib.Path(m.__file__).resolve().parents[1]  # .../site-packages/scikitplot
hits = []
for p in root.rglob("*.py"):
    txt = p.read_text(encoding="utf-8", errors="ignore")
    if re.search(r"^\s*def\s+__getattr__\s*\(", txt, re.M):
        hits.append(str(p))
print("search root:", root)
print("hits:\n" + "\n".join(hits))
PY


python - <<'PY'
import importlib, logging
import scikitplot.logging as m

# importlib.reload(m)

print("m.__file__:", m.__file__)
print("module:", m.__name__)
print("has module __getattr__:", hasattr(m, "__getattr__"))

# 1) The Sphinx killer: __mro__ must NOT resolve to None
print("getattr(m, '__mro__', 'DEFAULT') =", getattr(m, "__mro__", "DEFAULT"))

# 2) Missing attr must raise AttributeError (not return None)
try:
    getattr(m, "THIS_ATTRIBUTE_SHOULD_NOT_EXIST_12345")
    print("ERROR: missing attribute unexpectedly resolved (BAD)")
except AttributeError as e:
    print("missing attribute raises AttributeError (OK):", type(e).__name__)

# 3) A real stdlib logging attribute should resolve and be cached on module
lvl1 = m.INFO
lvl2 = m.INFO
print("m.INFO =", lvl1, "type:", type(lvl1).__name__)
print("m.INFO cached (same object):", lvl1 is lvl2)

# 4) A stdlib function should resolve
print("m.getLogger is logging.getLogger:", m.getLogger is logging.getLogger)

# 5) dir() should include stdlib names
d = dir(m)
print("'INFO' in dir(m):", "INFO" in d)
print("'getLogger' in dir(m):", "getLogger" in d)
# print("'getLogger' in dir(m):", "getLogger" in dir(m))

print("DONE")
PY


python - <<'PY'
import pathlib, re
root = pathlib.Path(".").resolve()
hits = []
for p in root.rglob("*.py"):
    txt = p.read_text(encoding="utf-8", errors="ignore")
    if re.search(r"^\s*def\s+__getattr__\s*\(", txt, re.M):
        hits.append(str(p))
print("\n".join(hits))
PY


######################################################################


python - <<'PY'
import scikitplot
import scikitplot.logging as m

def probe(x, label):
    print(f"\n== {label} ==")
    print("has __getattr__:", hasattr(x, "__getattr__"))
    try:
        print("getattr(__mro__, default):", getattr(x, "__mro__", "DEFAULT"))
    except Exception as e:
        print("getattr(__mro__) raised:", repr(e))

probe(scikitplot, "scikitplot package")
probe(m, "scikitplot.logging module")
PY

python - <<'PY'
import scikitplot.logging as m
print(m.__file__)
PY

python - <<'PY'
import scikitplot.logging as m, inspect
print(inspect.getsource(m.__getattr__))
PY

python - <<'PY'
import scikitplot.logging as m
print("has __getattr__:", hasattr(m, "__getattr__"))
print("getattr(__mro__, default):", getattr(m, "__mro__", "DEFAULT"))
PY
# getattr(__mro__, default): DEFAULT
