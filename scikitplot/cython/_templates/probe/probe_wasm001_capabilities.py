"""CYTHON-WASM-001 probe: browser/WASM capability contract + template asset guard.

Checks (no compiler needed):
1. platform_capabilities() reports a coherent contract for this process;
2. a simulated Emscripten platform reports prebuilt-only / no runtime compile;
3. verify_template_assets() finds every referenced .pxi/support asset present;
4. the guard detects a dropped asset.

Exit 0 = all properties hold.
"""
from __future__ import annotations

import json
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

from scikitplot.cython._profiles import platform_capabilities  # noqa: E402
from scikitplot.cython import _templates_api as _t  # noqa: E402


def main() -> int:
    ok = True

    caps = platform_capabilities()
    coherent = caps.can_compile_at_runtime is (not caps.prebuilt_only)
    print(f"capability contract coherent: {'OK' if coherent else 'FAIL'} ({caps})")
    ok = ok and coherent

    # Simulate a browser/WASM runtime.
    real = sys.platform
    try:
        sys.platform = "emscripten"
        wcaps = platform_capabilities()
        browser_ok = (
            wcaps.is_browser_wasm
            and not wcaps.can_compile_at_runtime
            and wcaps.prebuilt_only
            and wcaps.wasm_package_suffix == "emscripten-wasm32"
        )
    finally:
        sys.platform = real
    print(f"browser/WASM prebuilt-only: {'OK' if browser_ok else 'FAIL'}")
    ok = ok and browser_ok

    # All shipped template assets present.
    missing = _t.verify_template_assets()
    print(f"template assets present: {'OK' if not missing else 'FAIL'} "
          f"({len(missing)} missing)")
    ok = ok and not missing

    # Guard detects a dropped asset.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "_templates"
        (root / "module_cython").mkdir(parents=True)
        (root / "module_cython" / "x.pyx.meta.json").write_text(
            json.dumps({"support_paths": ["missing.pxi"]}), encoding="utf-8"
        )
        saved = _t._TEMPLATE_ROOT
        try:
            _t._TEMPLATE_ROOT = root
            detected = any("missing.pxi" in m for m in _t.verify_template_assets())
        finally:
            _t._TEMPLATE_ROOT = saved
    print(f"guard detects dropped asset: {'OK' if detected else 'FAIL'}")
    ok = ok and detected

    print("VERDICT:", "OK" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
