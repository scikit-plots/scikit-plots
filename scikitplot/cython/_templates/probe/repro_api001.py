"""CYTHON-API-001 repro: importing a normal absolute .pyx path under the
default policy. On a FIXED tree this prints import OK (exit 0); on the
unpatched tree it raises SecurityError (exit 1 via the wrapper below).
"""
import sys
import tempfile
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
from scikitplot.cython._public import cython_import_result
from scikitplot.cython._security import SecurityError

with tempfile.TemporaryDirectory() as td:
    pyx = Path(td) / "mymod.pyx"
    pyx.write_text("def add(int a, int b):\n    return a + b\n", encoding="utf-8")
    cache = Path(td) / "cache"
    try:
        r = cython_import_result(pyx, cache_dir=cache, numpy_support=False, verbose=-1)
        print("RESULT: imported OK, add(2,3) =", r.module.add(2,3))
    except SecurityError as e:
        print(f"RESULT: SecurityError -> BUG REPRODUCED: {e}")
    except Exception as e:
        print(f"RESULT: {type(e).__name__}: {e}")
