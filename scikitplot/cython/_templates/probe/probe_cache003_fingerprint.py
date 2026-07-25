"""CYTHON-CACHE-003 probe: the cache fingerprint must include toolchain/ABI
inputs so a different compiler/ABI yields a different cache key.

Exit 0 = fingerprint is toolchain/ABI-complete and a compiler change re-keys.
"""
from __future__ import annotations

import sys
import sysconfig
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

from scikitplot.cython._cache import make_cache_key, runtime_fingerprint  # noqa: E402

KEYS = {"cc", "cxx", "ext_suffix", "soabi", "pointer_size", "gil_disabled",
        "sysconfig_platform"}


def main() -> int:
    ok = True
    fp = runtime_fingerprint(cython_version="0", numpy_version=None)
    complete = KEYS <= set(fp)
    print(f"fingerprint toolchain/ABI-complete: {'OK' if complete else 'FAIL'}")
    ok = ok and complete

    base = make_cache_key({"s": "x", "fp": fp})
    cfg = sysconfig.get_config_vars()
    saved = cfg.get("CC")
    try:
        cfg["CC"] = "some-other-compiler"
        fp2 = runtime_fingerprint(cython_version="0", numpy_version=None)
        rekeyed = make_cache_key({"s": "x", "fp": fp2}) != base
    finally:
        cfg["CC"] = saved
    print(f"compiler change re-keys cache: {'OK' if rekeyed else 'FAIL'}")
    ok = ok and rekeyed

    print("VERDICT:", "OK" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
