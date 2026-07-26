"""CYTHON-GC-001 probe: GC running while a build holds a per-key lock must not
delete that entry.  A builder process holds KEY's build lock for a while; a GC
process runs max_age_days=0 concurrently.  The entry must survive.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile
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

KEY = "c" * 64


def make_old_entry(root: Path, key: str) -> None:
    d = root / key
    d.mkdir(parents=True, exist_ok=True)
    (d / "artifact.txt").write_text("payload", encoding="utf-8")
    old = os.stat(d).st_mtime - 30 * 86400.0
    os.utime(d, (old, old))


def builder(root_str: str, barrier, hold_s: float) -> None:
    from scikitplot.cython._lock import build_lock  # noqa: PLC0415

    root = Path(root_str)
    lock_dir = root / f"{KEY}.lock"
    barrier.wait()
    with build_lock(lock_dir, timeout_s=30.0):
        time.sleep(hold_s)  # simulate an active build/publish


def gcer(root_str: str, barrier, q) -> None:
    from scikitplot.cython._gc import gc_cache  # noqa: PLC0415

    root = Path(root_str)
    barrier.wait()
    time.sleep(0.05)  # ensure builder holds the lock first
    res = gc_cache(cache_dir=root, max_age_days=0)
    q.put((tuple(res.deleted_keys), tuple(res.skipped_active_keys)))


def main() -> int:
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "cache"
        root.mkdir()
        make_old_entry(root, KEY)

        barrier = ctx.Barrier(2)
        q: mp.Queue = ctx.Queue()
        b = ctx.Process(target=builder, args=(str(root), barrier, 0.6))
        g = ctx.Process(target=gcer, args=(str(root), barrier, q))
        b.start()
        g.start()
        b.join(30)
        g.join(30)

        deleted, skipped_active = q.get()
        survived = (root / KEY).exists()
        print(f"deleted={deleted}")
        print(f"skipped_active={skipped_active}")
        print(f"entry survived = {survived}")

        ok = survived and KEY not in deleted and KEY in skipped_active
        print("VERDICT:", "OK" if ok else "CHECK")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
