"""CYTHON-PORT-001 probe: resolved-toolchain contract keys the cache.

Checks:
1. resolved_toolchain() reports the effective compiler (never raises);
2. the cache fingerprint includes the resolved compiler identity;
3. a different resolved compiler changes the fingerprint (distinct cache keys).

Exit 0 = the resolved plan drives cache keying.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

# Dev/AI probe bootstrap.
try:
    import scikitplot.cython  # noqa: F401
except ImportError:
    _here = Path(__file__).resolve()
    for _cand in _here.parents:
        if (_cand / "scikitplot" / "__init__.py").exists():
            sys.path.insert(0, str(_cand))
            break

from scikitplot.cython._cache import runtime_fingerprint  # noqa: E402
from scikitplot.cython._profiles import (  # noqa: E402
    ResolvedToolchain,
    resolved_toolchain,
)


def main() -> int:
    ok = True

    rt = resolved_toolchain()
    reported = isinstance(rt, ResolvedToolchain) and bool(rt.compiler_type)
    print(f"resolved compiler reported: {'OK' if reported else 'FAIL'} "
          f"(type={rt.compiler_type!r}, cc={rt.cc!r})")
    ok = ok and reported

    fp = runtime_fingerprint(cython_version="0", numpy_version=None)
    keyed = "resolved_compiler_type" in fp and "resolved_cc" in fp
    print(f"fingerprint includes resolved compiler: {'OK' if keyed else 'FAIL'}")
    ok = ok and keyed

    with mock.patch(
        "scikitplot.cython._profiles.resolved_toolchain",
        return_value=ResolvedToolchain("msvc", "cl", "cl", "link"),
    ):
        other = runtime_fingerprint(cython_version="0", numpy_version=None)
    differs = other["resolved_compiler_type"] != fp["resolved_compiler_type"]
    print(f"different compiler → different key: {'OK' if differs else 'FAIL'}")
    ok = ok and differs

    print("VERDICT:", "OK" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
