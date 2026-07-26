"""CYTHON-CON-002 probe: global build/import state is thread-safe.

Checks (no compiler needed):
1. concurrent first-call to _import_setuptools returns one identical class pair
   (exactly-once init under contention);
2. concurrent register/list/unregister on CompilerRegistry never corrupts/raises.

Exit 0 = both properties hold.
"""
from __future__ import annotations

import sys
import threading
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

import scikitplot.cython._builder as B  # noqa: E402
from scikitplot.cython._custom_compiler import (  # noqa: E402
    CompilerRegistry,
    CustomCompilerProtocol,
)


class _Fake:
    def __init__(self, name): self.name = name
    def __call__(self, *a, **k): return None
    def compile(self, *a, **k): return None


def main() -> int:
    ok = True

    B._SETUPTOOLS_CACHE = None
    results = []
    barrier = threading.Barrier(16)

    def w():
        barrier.wait()
        ext, dist = B._import_setuptools()
        results.append((id(ext), id(dist)))

    ts = [threading.Thread(target=w) for _ in range(16)]
    for t in ts: t.start()
    for t in ts: t.join()
    once = len(set(results)) == 1
    print(f"setuptools exactly-once init under contention: {'OK' if once else 'FAIL'}")
    ok = ok and once

    reg = CompilerRegistry()
    errors = []

    def rw(i):
        try:
            c = _Fake(f"custom_c{i}")
            if isinstance(c, CustomCompilerProtocol):
                reg.register(c, overwrite=True)
                reg.list()
                reg.unregister(c.name)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=rw, args=(i,)) for i in range(64)]
    for t in ts: t.start()
    for t in ts: t.join()
    clean = not errors
    print(f"registry concurrent ops without corruption: {'OK' if clean else 'FAIL'}")
    ok = ok and clean

    print("VERDICT:", "OK" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
