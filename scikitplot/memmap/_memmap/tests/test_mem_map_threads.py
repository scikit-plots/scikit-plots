"""
One ``MemoryMap`` shared by threads (round 33, Run 6: R6-3).

The open/closed state, the export count and the protection flags change
under a lock of the mapping. With the GIL these tests pass without it: the
GIL already serialises those sections. They bite on a free-threaded CPython:
on 3.13t, with the module as it was before the lock, the count test failed
in 6 of 6 runs, the held-view test in 5 of 6 (SIGSEGV), and the
mixed-operation test in 8 of 12 (mostly SIGSEGV); with the lock, none
failed. The close and ``mprotect`` tests
are smoke tests: the base build passed them too. Each test runs in a child
process, so a crash cannot take the session down with it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
import textwrap

import pytest

#: Whether this interpreter is a free-threaded build, read before anything is
#: imported: importing a module not declared free-threading compatible turns
#: the GIL back on, and the tests below then have no power.
_FREE_THREADED_BUILD = bool(sysconfig.get_config_var("Py_GIL_DISABLED")) or (
    # Where the configuration does not say (it may not, on Windows), the
    # interpreter does, as long as nothing has turned the GIL back on yet.
    hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled()
)

memmap = pytest.importorskip("scikitplot.memmap")

#: Threads, rounds per trial and trials: what the measurement in the module
#: docstring used (each test takes 1 to 3 s, with or without the GIL).
THREADS = 8
LOOPS = 20_000
TRIALS = 20

_MAPPED = """
def mapped(address):
    # A region may be merged with a neighbour into one entry: test ranges.
    with open("/proc/self/maps", encoding="ascii") as maps:
        for line in maps:
            low, high = (int(part, 16) for part in line.split()[0].split("-"))
            if low <= address < high:
                return True
    return False
"""


def _child(script: str, cwd) -> subprocess.CompletedProcess:
    """Run ``script`` in a child interpreter in ``cwd``; return its outcome."""
    # ``cwd`` is a temporary directory: ``-c`` puts the working directory
    # first on ``sys.path``, and the source tree must not shadow the build.
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        cwd=str(cwd),
    )


_RACE = f"""
import threading
from scikitplot.memmap import MemoryMap, PY_PROT_READ, PY_PROT_WRITE

THREADS, LOOPS, TRIALS = {THREADS}, {LOOPS}, {TRIALS}


def exports_on_threads(m):
    start = threading.Barrier(THREADS)

    def work():
        start.wait()
        for _ in range(LOOPS):
            memoryview(m).release()

    threads = [threading.Thread(target=work) for _ in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
"""


def test_exports_counted_by_threads_let_close_unmap(tmp_path):
    # Lost updates of the count left the region mapped after close().
    script = _RACE + _MAPPED + """
try:
    open("/proc/self/maps").close()
except OSError:
    print("no-maps")
    raise SystemExit(0)
left = 0
for _ in range(TRIALS):
    m = MemoryMap.create_anonymous(1 << 16, PY_PROT_READ | PY_PROT_WRITE)
    address = m.addr
    exports_on_threads(m)
    m.close()
    left += mapped(address)
print("left", left)
"""
    done = _child(script, tmp_path)
    assert done.returncode == 0, f"exit {done.returncode}; stderr:\n{done.stderr}"
    last = done.stdout.split()
    if last == ["no-maps"]:
        pytest.skip("needs /proc/self/maps")
    assert last == ["left", "0"], done.stdout


def test_a_view_held_across_close_stays_readable_after_racing_exports(tmp_path):
    # A count that drifted down let close() unmap under a live view.
    script = _RACE + """
for _ in range(TRIALS):
    m = MemoryMap.create_anonymous(1 << 16, PY_PROT_READ | PY_PROT_WRITE)
    exports_on_threads(m)
    keep = memoryview(m)
    m.close()
    keep[0], keep[-1]  # SIGSEGV if the region was unmapped
    keep.release()
print("ok")
"""
    done = _child(script, tmp_path)
    assert done.returncode == 0, f"exit {done.returncode}; stderr:\n{done.stderr}"
    assert done.stdout.split() == ["ok"]


def test_close_while_threads_read_and_export_never_crashes(tmp_path):
    # A read or an export either takes its hold before close() (the unmap
    # waits for it) or finds the mapping closed; nothing in between.
    script = """
    import threading
    from scikitplot.memmap import MemoryMap, PY_PROT_READ, PY_PROT_WRITE

    refused = 0
    guard = threading.Lock()
    for _ in range(200):
        m = MemoryMap.create_anonymous(1 << 20, PY_PROT_READ | PY_PROT_WRITE)
        go = threading.Event()

        def reader():
            global refused
            go.wait()
            for _ in range(100):
                try:
                    m.read(1 << 20)
                    view = memoryview(m)
                    view[0], view[-1]
                    view.release()
                except (ValueError, BufferError):
                    with guard:
                        refused += 1
                    return

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for thread in threads:
            thread.start()
        go.set()
        m.close()
        for thread in threads:
            thread.join()
    print("ok", refused > 0)
    """
    done = _child(script, tmp_path)
    assert done.returncode == 0, f"exit {done.returncode}; stderr:\n{done.stderr}"
    assert done.stdout.split()[0] == "ok"


def test_mprotect_on_one_thread_never_strands_a_writable_view(tmp_path):
    # Every write goes through a writable export taken on another thread:
    # mprotect may not take PROT_WRITE away while one is alive, and an
    # export may not start from flags mprotect is changing.
    script = """
    import threading
    from scikitplot.memmap import MemoryMap, PY_PROT_READ, PY_PROT_WRITE

    m = MemoryMap.create_anonymous(1 << 16, PY_PROT_READ | PY_PROT_WRITE)
    stop = threading.Event()
    counts = {"wrote": 0, "refused_export": 0, "refused_mprotect": 0}
    guard = threading.Lock()

    def writer():
        while not stop.is_set():
            try:
                view = memoryview(m)
            except BufferError:
                continue
            try:
                if not view.readonly:
                    view[0] = 1  # SIGSEGV if the pages became read-only
                    with guard:
                        counts["wrote"] += 1
                else:
                    with guard:
                        counts["refused_export"] += 1
            finally:
                view.release()

    threads = [threading.Thread(target=writer) for _ in range(4)]
    for thread in threads:
        thread.start()
    for i in range(20_000):
        try:
            m.mprotect(PY_PROT_READ if i % 2 else PY_PROT_READ | PY_PROT_WRITE)
        except BufferError:
            with guard:
                counts["refused_mprotect"] += 1
    stop.set()
    for thread in threads:
        thread.join()
    m.close()
    print("ok", sorted(counts.items()))  # how the threads interleaved: data
    """
    done = _child(script, tmp_path)
    assert done.returncode == 0, f"exit {done.returncode}; stderr:\n{done.stderr}"
    assert done.stdout.split()[0] == "ok", done.stdout


def test_every_operation_on_threads_while_another_closes(tmp_path):
    # Exports, read, write, mprotect, msync and repr on six threads while
    # the main thread closes: every call either runs on a mapped region or
    # is refused with the documented error, and close() unmaps once, after
    # the last hold. The base free-threaded build died here with SIGSEGV,
    # after mprotect and msync had failed with ENOMEM on an unmapped region.
    script = _MAPPED + """
import random
import threading
from scikitplot.memmap import MemoryMap, PY_PROT_READ as R, PY_PROT_WRITE as W

try:
    open("/proc/self/maps").close()
    has_maps = True
except OSError:
    has_maps = False  # the crash and error checks still run

unexpected = []
threading.excepthook = lambda hook: unexpected.append(repr(hook.exc_value))
left = 0
for trial in range(300):
    m = MemoryMap.create_anonymous(1 << 16, R | W)
    address = m.addr
    go = threading.Barrier(7)

    def work(seed):
        ops = random.Random(seed)  # a fixed sequence per thread
        go.wait()
        for _ in range(300):
            op = ops.randrange(5)
            try:
                if op == 0:
                    view = memoryview(m)
                    view[0], view[-1]
                    if not view.readonly:
                        view[1] = 2
                    view.release()
                elif op == 1:
                    m.read(4096, 100)
                elif op == 2:
                    m.write(b"x" * 100, 5)
                elif op == 3:
                    m.mprotect(R if ops.random() < 0.5 else R | W)
                else:
                    m.msync()
                    repr(m)
            except (ValueError, BufferError):
                pass  # closed, or a right missing or held: documented

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    go.wait()
    m.close()
    for thread in threads:
        thread.join()
    left += mapped(address) if has_maps else 0
print("left", left if has_maps else "unknown", "unexpected", len(unexpected))
for line in unexpected[:5]:
    print(line)
"""
    done = _child(script, tmp_path)
    assert done.returncode == 0, f"exit {done.returncode}; stderr:\n{done.stderr}"
    out = done.stdout.split("\n")[0].split()
    assert out[0::2] == ["left", "unexpected"], done.stdout
    assert out[1] in ("0", "unknown"), done.stdout  # "unknown": no /proc
    assert out[3] == "0", done.stdout


@pytest.mark.skipif(not _FREE_THREADED_BUILD, reason="not a free-threaded build")
def test_a_free_threaded_build_keeps_the_gil_off(tmp_path):
    # Fails, rather than skips, when the module turned the GIL back on (built
    # without ``freethreading_compatible``, Cython before 3.1): the tests
    # above would then have run with the GIL and proved nothing.
    if os.environ.get("PYTHON_GIL") == "1":
        pytest.skip("PYTHON_GIL=1 asks for the GIL")
    done = _child(
        """
        import sys
        import scikitplot.memmap._memmap.mem_map
        print(sys._is_gil_enabled())
        """,
        tmp_path,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["False"], done.stdout + done.stderr
