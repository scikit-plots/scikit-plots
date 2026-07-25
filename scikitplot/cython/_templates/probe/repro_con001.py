"""CYTHON-CON-001 reproduction: two processes overlap inside the same build_lock.

Mirrors the review's Appendix C probe:
  Spawn two processes, synchronize start, enter build_lock(same_path),
  record entry/exit while sleeping inside the context, assert overlap.
Exit code 0 => overlap observed (bug present). Exit code 1 => exclusive (fixed).
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

# Dev/AI probe bootstrap: make ``scikitplot.cython`` importable whether this
# file is run from its shipped location (_templates/probe/) or copied elsewhere.
# Prefer an already-installed package; otherwise walk up to the repo root that
# contains a ``scikitplot`` package directory.
try:  # already importable (installed or on PYTHONPATH)
    import scikitplot.cython  # noqa: F401
except ImportError:
    _here = Path(__file__).resolve()
    for _cand in _here.parents:
        if (_cand / "scikitplot" / "__init__.py").exists():
            sys.path.insert(0, str(_cand))
            break
from scikitplot.cython._lock import build_lock  # noqa: E402

HOLD_S = 0.35


def worker(lock_path: str, start_barrier, q) -> None:
    lock_dir = Path(lock_path)
    start_barrier.wait()  # synchronize start
    try:
        with build_lock(lock_dir, timeout_s=5.0, poll_s=0.01):
            enter = time.monotonic()
            time.sleep(HOLD_S)
            exit_ = time.monotonic()
            q.put((enter, exit_, None))
    except Exception as e:  # noqa: BLE001
        q.put((None, None, f"{type(e).__name__}: {e}"))


def main() -> int:
    ctx = mp.get_context("spawn")
    tmp = Path("/tmp/con001_probe.lock")
    if tmp.exists():
        try:
            tmp.rmdir()
        except OSError:
            pass
    barrier = ctx.Barrier(2)
    q: mp.Queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(str(tmp), barrier, q)) for _ in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(20)

    rows = [q.get() for _ in range(2)]
    ok_rows = [r for r in rows if r[2] is None]
    err_rows = [r for r in rows if r[2] is not None]

    print("results:")
    for r in rows:
        print("  ", r)

    if len(ok_rows) == 2:
        (a_enter, a_exit, _), (b_enter, b_exit, _) = sorted(ok_rows, key=lambda r: r[0])
        overlap = b_enter < a_exit
        overlap_amt = max(0.0, a_exit - b_enter)
        print(f"overlap = {overlap}  (second entered {overlap_amt:.3f}s before first exited)")
        if overlap:
            print("VERDICT: NON-EXCLUSIVE lock reproduced (CYTHON-CON-001 present)")
            return 0
        print("VERDICT: lock held exclusively (no overlap)")
        return 1
    # If exactly one acquired and the other timed out/failed, that's exclusive behaviour.
    if len(ok_rows) == 1 and len(err_rows) == 1:
        print(f"one acquired, one blocked -> {err_rows[0][2]}")
        print("VERDICT: lock held exclusively (contention serialized)")
        return 1
    print("VERDICT: inconclusive")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
