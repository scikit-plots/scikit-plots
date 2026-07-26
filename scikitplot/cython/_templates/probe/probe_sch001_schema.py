"""CYTHON-SCH-001 probe: versioned cache metadata schema.

Checks (no compiler needed):
1. write_meta stamps CACHE_SCHEMA_VERSION;
2. current entry compatible; legacy (v0) and future versions incompatible;
3. explicit version preserved.

Exit 0 = schema versioning behaves correctly.
"""
from __future__ import annotations

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

from scikitplot.cython._cache import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    is_meta_schema_compatible,
    meta_schema_version,
    read_meta,
    write_meta,
)


def main() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        write_meta(d, {"kind": "module"})
        m = read_meta(d)
        stamped = meta_schema_version(m) == CACHE_SCHEMA_VERSION
        print(f"write stamps version: {'OK' if stamped else 'FAIL'}")
        ok = ok and stamped

        compat = (
            is_meta_schema_compatible(m)
            and not is_meta_schema_compatible({"kind": "module"})
            and not is_meta_schema_compatible({"meta_schema_version": 999})
        )
        print(f"compatibility logic: {'OK' if compat else 'FAIL'}")
        ok = ok and compat

        write_meta(d, {"meta_schema_version": 1, "x": 1})
        preserved = read_meta(d).get("meta_schema_version") == 1
        print(f"explicit version preserved: {'OK' if preserved else 'FAIL'}")
        ok = ok and preserved

    print("VERDICT:", "OK" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
