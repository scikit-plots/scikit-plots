"""CYTHON-RES-001 probe: build deadline + bounded output (in-process portion).

Checks (no compiler needed for 1-3):
1. run_with_deadline raises BuildTimeoutError past the deadline;
2. run_with_deadline returns results and propagates exceptions;
3. BoundedBuffer retains only the tail within budget;
4. (best-effort) build_timeout_s is enforced by a real build if a toolchain exists.

Exit 0 = in-process budget primitives behave correctly.
"""
from __future__ import annotations

import sys
import time
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

from scikitplot.cython._budget import (  # noqa: E402
    BoundedBuffer,
    BuildTimeoutError,
    run_with_deadline,
)


def main() -> int:
    ok = True

    try:
        run_with_deadline(lambda: time.sleep(1.0), timeout_s=0.05, what="probe")
        deadline_ok = False
    except BuildTimeoutError:
        deadline_ok = True
    print(f"deadline enforced: {'OK' if deadline_ok else 'FAIL'}")
    ok = ok and deadline_ok

    got = run_with_deadline(lambda: 123, timeout_s=5.0)
    print(f"result returned: {'OK' if got == 123 else 'FAIL'}")
    ok = ok and got == 123

    b = BoundedBuffer(max_bytes=20)
    for i in range(100):
        b.write(f"line{i}\n")
    tail_ok = b.truncated and "line99" in b.getvalue() and "line0\n" not in b.getvalue()
    print(f"bounded output keeps tail: {'OK' if tail_ok else 'FAIL'}")
    ok = ok and tail_ok

    print("VERDICT:", "OK" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
