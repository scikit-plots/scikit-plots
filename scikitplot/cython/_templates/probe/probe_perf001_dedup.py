"""CYTHON-PERF-001 probe: normalized-path dedup + bounded traversal.

Checks:
1. duplicate include dirs collapse by normalized identity;
2. the intrinsic .pyx parent is not duplicated into include_dirs;
3. cache-stats traversal honours a max_files budget.

Exit 0 = the perf hardening behaves correctly without weakening validation.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest import mock

# Dev/AI probe bootstrap.
try:
    import scikitplot.cython  # noqa: F401
except ImportError:
    for _cand in Path(__file__).resolve().parents:
        if (_cand / "scikitplot" / "__init__.py").exists():
            sys.path.insert(0, str(_cand))
            break

import scikitplot.cython._public as P  # noqa: E402
from scikitplot.cython._gc import _dir_size_bytes  # noqa: E402
from scikitplot.cython._public import _dedup_paths  # noqa: E402


def main() -> int:
    ok = True

    with tempfile.TemporaryDirectory() as td:
        a = Path(td) / "a"
        a.mkdir()
        collapsed = len(_dedup_paths([str(a), a, str(a) + "/"])) == 1
        print(f"dedup collapses duplicates: {'OK' if collapsed else 'FAIL'}")
        ok = ok and collapsed

        pyx = Path(td) / "mod.pyx"
        pyx.write_text("def f(): return 1\n")
        cap = {}

        def fake(source, **kw):
            cap["inc"] = [str(x) for x in (kw.get("include_dirs") or [])]

            class R:
                module = None

            return R()

        with mock.patch.object(P, "compile_and_load_result", fake):
            P.cython_import_result(pyx, include_dirs=[str(Path(td))])
        no_dup = str(Path(td)) not in cap["inc"]
        print(f"parent not duplicated: {'OK' if no_dup else 'FAIL'}")
        ok = ok and no_dup

        for i in range(30):
            (Path(td) / f"f{i}.bin").write_bytes(b"x" * 10)
        bounded = _dir_size_bytes(Path(td), max_files=5) < _dir_size_bytes(Path(td))
        print(f"bounded traversal < full: {'OK' if bounded else 'FAIL'}")
        ok = ok and bounded

    print("VERDICT:", "OK" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
