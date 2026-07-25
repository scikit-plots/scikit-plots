"""CYTHON-ABI-001 probe: runtime lifecycle contract + safe defaults.

Checks (simulates unsupported configs — no special interpreter needed):
1. runtime_capabilities() declares the contract (supports_unload=False);
2. free-threaded interpreter is rejected by default, allowed on opt-in;
3. non-main subinterpreter is rejected by default, allowed on opt-in.

Exit 0 = the lifecycle contract + guards behave correctly.
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

from scikitplot.cython import _profiles  # noqa: E402
from scikitplot.cython._profiles import (  # noqa: E402
    RuntimeCapabilities,
    UnsupportedRuntimeError,
    check_runtime_supported,
    runtime_capabilities,
)


def _caps(**ov):
    base = RuntimeCapabilities(True, False, True, False, True, "linux")
    return base._replace(**ov)


def main() -> int:
    ok = True

    caps = runtime_capabilities()
    declared = isinstance(caps, RuntimeCapabilities) and caps.supports_unload is False
    print(f"contract declared (unload=False): {'OK' if declared else 'FAIL'}")
    ok = ok and declared

    with mock.patch.object(
        _profiles, "runtime_capabilities", return_value=_caps(free_threaded_build=True)
    ):
        try:
            check_runtime_supported()
            ft = False
        except UnsupportedRuntimeError:
            check_runtime_supported(allow_free_threaded=True)
            ft = True
    print(f"free-threaded rejected + opt-in: {'OK' if ft else 'FAIL'}")
    ok = ok and ft

    with mock.patch.object(
        _profiles, "runtime_capabilities", return_value=_caps(in_main_interpreter=False)
    ):
        try:
            check_runtime_supported()
            si = False
        except UnsupportedRuntimeError:
            check_runtime_supported(allow_subinterpreter=True)
            si = True
    print(f"subinterpreter rejected + opt-in: {'OK' if si else 'FAIL'}")
    ok = ok and si

    print("VERDICT:", "OK" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
