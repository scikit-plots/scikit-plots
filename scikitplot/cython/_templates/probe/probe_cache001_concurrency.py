"""CYTHON-CACHE-001 concurrency probe: two processes build the SAME key
concurrently; the published entry must be consistent and importable, with no
leftover staging directories.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import tempfile
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

PYX = "def add(int a, int b):\n    return a + b\n"


def worker(cache_dir: str, barrier, q) -> None:
    from scikitplot.cython._public import compile_and_load_result  # noqa: PLC0415

    barrier.wait()
    try:
        r = compile_and_load_result(
            PYX, cache_dir=Path(cache_dir), numpy_support=False, verbose=-1
        )
        q.put(("ok", r.key, r.used_cache, r.module.add(2, 3)))
    except Exception as e:  # noqa: BLE001
        q.put(("err", type(e).__name__, str(e), None))


def main() -> int:
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td) / "cache"
        cache.mkdir()
        barrier = ctx.Barrier(2)
        q: mp.Queue = ctx.Queue()
        procs = [ctx.Process(target=worker, args=(str(cache), barrier, q)) for _ in range(2)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(120)
        rows = [q.get() for _ in range(2)]
        for r in rows:
            print("  ", r)

        oks = [r for r in rows if r[0] == "ok"]
        if len(oks) != 2:
            print("VERDICT: a worker failed")
            return 1
        keys = {r[1] for r in oks}
        vals = {r[3] for r in oks}
        used = sorted(r[2] for r in oks)
        staging_leftovers = list(cache.glob(".staging-*"))
        entries = [p for p in cache.iterdir() if p.is_dir() and len(p.name) == 64]
        print(f"keys={keys} values={vals} used_cache={used}")
        print(f"published entries={len(entries)} staging_leftovers={len(staging_leftovers)}")

        ok = (
            len(keys) == 1                 # same key
            and vals == {5}                # both correct
            and len(entries) == 1          # exactly one published entry
            and staging_leftovers == []    # no leaked staging dirs
            and used == [False, True]      # one built, one hit the cache
        )
        print("VERDICT:", "OK" if ok else "CHECK")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
