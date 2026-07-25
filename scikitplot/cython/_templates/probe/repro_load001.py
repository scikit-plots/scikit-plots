"""CYTHON-LOAD-001 repro: a failed reload must preserve the prior sys.modules
entry. Prints PRESERVED on a fixed tree, DESTROYED on the unpatched tree.
"""
import sys
import types
from pathlib import Path

# Dev/AI probe bootstrap.
try:
    import scikitplot.cython  # noqa: F401
except ImportError:
    _here = Path(__file__).resolve()
    for _cand in _here.parents:
        if (_cand / "scikitplot" / "__init__.py").exists():
            sys.path.insert(0, str(_cand))
            break
from scikitplot.cython._loader import import_extension

# Put a "prior working module" in sys.modules
name = "load001_victim"
prior = types.ModuleType(name)
prior.MARKER = "prior-working-module"
sys.modules[name] = prior

# Attempt to import from a bad artifact path (will fail in exec_module/spec)
bad = Path("/tmp/does_not_exist_load001.so")
try:
    import_extension(name=name, path=bad)
    print("RESULT: unexpectedly succeeded")
except Exception as e:
    still = sys.modules.get(name)
    if still is prior and getattr(still, "MARKER", None) == "prior-working-module":
        print("RESULT: prior module PRESERVED (fixed)")
    else:
        print(f"RESULT: prior module DESTROYED (BUG) — sys.modules[{name!r}]={still!r}")
