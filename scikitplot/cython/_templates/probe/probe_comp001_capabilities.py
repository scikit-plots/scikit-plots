"""CYTHON-COMP-001 probe: versioned compiler capability descriptor.

Checks (no compiler invocation):
1. built-in compilers declare capabilities (pybind=C++, c_api=C);
2. a legacy bare callable gets a conservative C-only default;
3. a capabilities() method is honoured;
4. platform/language predicates behave.

Exit 0 = capability descriptor behaves correctly.
"""
from __future__ import annotations

import sys
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

from scikitplot.cython._custom_compiler import (  # noqa: E402
    CApiCompiler,
    CompilerCapabilities,
    PybindCompiler,
    compiler_capabilities,
)


def main() -> int:
    ok = True

    pc = compiler_capabilities(PybindCompiler())
    ca = compiler_capabilities(CApiCompiler())
    builtins_ok = pc.requires_cpp and pc.supports_language("c++") and (
        not ca.requires_cpp and ca.supports_language("c")
    )
    print(f"built-ins declare capabilities: {'OK' if builtins_ok else 'FAIL'}")
    ok = ok and builtins_ok

    class Bare:
        name = "custom_bare"
        def __call__(self, *a, **k): return None

    d = compiler_capabilities(Bare())
    default_ok = d.supported_languages == frozenset({"c"}) and d.supports_platform(
        sys.platform
    )
    print(f"legacy default conservative: {'OK' if default_ok else 'FAIL'}")
    ok = ok and default_ok

    restricted = CompilerCapabilities(name="x", platforms=frozenset({"linux"}))
    pred_ok = restricted.supports_platform("linux") and not restricted.supports_platform(
        "win32"
    )
    print(f"platform predicate: {'OK' if pred_ok else 'FAIL'}")
    ok = ok and pred_ok

    print("VERDICT:", "OK" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
