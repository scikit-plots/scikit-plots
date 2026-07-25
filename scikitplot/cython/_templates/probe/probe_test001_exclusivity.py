"""CYTHON-TEST-001 probe: interprocess build-lock exclusivity.

Spawns two processes: while A holds the lock, B (timeout=0) MUST time out;
after A releases, B acquires. Exit 0 = the exclusivity invariant holds.
"""
import sys, time, tempfile, multiprocessing as mp
from pathlib import Path
# Dev/AI probe bootstrap.
try:
    import scikitplot.cython  # noqa: F401
except ImportError:
    for _cand in Path(__file__).resolve().parents:
        if (_cand / "scikitplot" / "__init__.py").exists():
            sys.path.insert(0, str(_cand)); break

def hold_lock(lock_dir_str, hold_ready, hold_secs):
    from scikitplot.cython._lock import build_lock
    from pathlib import Path
    with build_lock(Path(lock_dir_str), timeout_s=10.0):
        hold_ready.set()          # signal: lock acquired
        time.sleep(hold_secs)     # keep holding

def try_acquire(lock_dir_str, result_q):
    from scikitplot.cython._lock import build_lock
    from pathlib import Path
    try:
        with build_lock(Path(lock_dir_str), timeout_s=0.0):
            result_q.put("acquired")
    except TimeoutError:
        result_q.put("timeout")
    except Exception as e:
        result_q.put(f"error:{type(e).__name__}")

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as td:
        lock_dir = str(Path(td) / "build.lock")
        ready = ctx.Event()
        q = ctx.Queue()
        # Process A holds the lock for 1.5s
        pa = ctx.Process(target=hold_lock, args=(lock_dir, ready, 1.5))
        pa.start()
        assert ready.wait(timeout=10), "holder never acquired"
        # Process B tries with timeout=0 while A holds -> must TIMEOUT
        pb = ctx.Process(target=try_acquire, args=(lock_dir, q))
        pb.start(); pb.join(10)
        r1 = q.get()
        print("while-held:", r1, "(want timeout)")
        pa.join(10)
        # Now A released; B tries again -> must ACQUIRE
        pc = ctx.Process(target=try_acquire, args=(lock_dir, q))
        pc.start(); pc.join(10)
        r2 = q.get()
        print("after-release:", r2, "(want acquired)")
        holds = r1=="timeout" and r2=="acquired"
        print("VERDICT:", "OK" if holds else "CHECK")
        raise SystemExit(0 if holds else 1)
