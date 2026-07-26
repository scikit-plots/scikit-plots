"""Compile probe (review Appendix C): compile a single Cython module, then
recompile to confirm cache reuse — exercises build_lock on the real build path.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Dev/AI probe bootstrap: make ``scikitplot.cython`` importable whether this
# file is run from its shipped location (_templates/probe/) or copied elsewhere.
# Prefer an already-installed package; otherwise walk up to the repo root that
# contains a ``scikitplot`` package directory.
try:  # already importable (installed or on PYTHONPATH)
    import scikitplot.cython  # noqa: F401
except ImportError:
    _here = Path(__file__).resolve()
    for _cand in _here.parents:
        if (_cand / "scikitplot" / "__init__.py").exists():
            sys.path.insert(0, str(_cand))
            break
import scikitplot.cython as cy  # noqa: E402

SRC = "def add(a, b):\n    return a + b\n"


def main() -> int:
    api = {n: getattr(cy, n) for n in dir(cy) if "compile_and_load" in n}
    print("compile entrypoints:", list(api))
    fn = getattr(cy, "compile_and_load_result", None) or getattr(cy, "compile_and_load", None)
    if fn is None:
        print("no compile_and_load entrypoint found")
        return 2

    with tempfile.TemporaryDirectory() as td:
        cache = Path(td) / "cache"
        # First build (compiles)
        r1 = fn(SRC, module_name="probe_add", cache_dir=cache)
        m1 = r1.module if hasattr(r1, "module") else r1
        used1 = getattr(r1, "used_cache", None)
        # Second build (should hit cache)
        r2 = fn(SRC, module_name="probe_add", cache_dir=cache)
        used2 = getattr(r2, "used_cache", None)
        val = m1.add(2, 3)
        print(f"add(2,3) = {val}")
        print(f"first used_cache={used1}  second used_cache={used2}")
        ok = (val == 5) and (used1 is False) and (used2 is True)
        print("VERDICT:", "OK" if ok else "CHECK")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
