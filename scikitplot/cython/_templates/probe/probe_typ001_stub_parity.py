"""CYTHON-TYP-001 probe: runtime/stub (.pyi) parity.

Checks that every name in the runtime __all__ is declared in __init__.pyi.
Exit 0 = full parity.
"""
from __future__ import annotations

import ast
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

import scikitplot.cython as skc  # noqa: E402


def main() -> int:
    stub_path = Path(skc.__file__).with_name("__init__.pyi")
    tree = ast.parse(stub_path.read_text(encoding="utf-8"))
    stub = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stub.add(n.name)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            stub.add(n.target.id)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    stub.add(t.id)

    missing = sorted(set(skc.__all__) - stub)
    ok = not missing
    print(f"runtime __all__: {len(skc.__all__)}  stub names: {len(stub)}")
    print(f"parity (0 missing): {'OK' if ok else 'FAIL'} "
          f"({len(missing)} missing: {missing[:8]}{'...' if len(missing) > 8 else ''})")
    print("VERDICT:", "OK" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
