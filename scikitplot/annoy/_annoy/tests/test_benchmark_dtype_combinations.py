# scikitplot/annoy/_annoy/tests/test_benchmark_dtype_combinations.py
"""
Dtype × Index-dtype Combination Performance Benchmarks
=======================================================

Purpose
-------
Measure and report the *relative* cost of the four runtime type combinations
across the three hot-path operations (add_item, build, query).  This is a
regression baseline — numbers should remain stable across code changes.

Why benchmarks matter here
--------------------------
The ``_w`` bridge introduced to support type-erased ``AnnoyIndexInterfaceBase*``
storage adds one indirection and a temporary ``std::vector`` allocation per
``add_item`` and ``get_item`` call.  These benchmarks verify that the overhead
is negligible compared with the underlying Annoy work.

Design
------
* No external benchmark framework required (``time.perf_counter`` only).
* Each test is also a valid pytest test: it passes when:
    (a) the combination runs without error, and
    (b) the per-call time is below a generous threshold (avoids flakiness on
        slow CI machines while still catching catastrophic regressions).
* A console table is printed via ``capsys`` so results are visible in verbose
  mode (``pytest -v -s``).

Combinations measured (4 × 3 operations = 12 cells)
-----------------------------------------------------
(int32, float32)  add_item  build  query
(int32, float64)  ...
(int64, float32)  ...
(int64, float64)  ...
"""

import math
import random
import time
from typing import NamedTuple

import pytest

from ..annoylib import Index

# ---------------------------------------------------------------------------
# Configuration — kept small so the test suite finishes quickly on CI.
# ---------------------------------------------------------------------------

F = 64           # embedding dimension
N_ITEMS = 2_000  # number of items to add
N_TREES = 5      # trees to build
N_QUERIES = 500  # query calls per benchmark
N_RESULTS = 10   # neighbors per query
SEED = 42  # 0

#: (index_dtype, dtype) pairs to benchmark.
DTYPE_COMBOS = [
    ("int32", "float32"),
    ("int32", "float64"),
    ("int64", "float32"),
    ("int64", "float64"),
]

# Per-operation generous thresholds (seconds per call).
# Set 10× above what we expect on a slow single-core CI box to avoid flakiness.
MAX_ADD_ITEM_US  = 500.0   # µs/call  — add_item (includes Python list iteration)
MAX_BUILD_S      = 30.0    # s total  — build is O(n log n)
MAX_QUERY_US     = 5000.0  # µs/call  — ANN query (search_k=-1, n=10, f=64)


# ---------------------------------------------------------------------------
# Shared fixture: pre-generated random float vectors (avoids contaminating
# timing with Python-level random number generation).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def float_vectors():
    """Return N_ITEMS normalised random vectors of dimension F."""
    rng = random.Random(SEED)
    vecs = []
    for _ in range(N_ITEMS):
        v = [rng.gauss(0.0, 1.0) for _ in range(F)]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs.append([x / norm for x in v])
    return vecs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class BenchResult(NamedTuple):
    """Timing result for one operation × one dtype combo."""
    index_dtype: str
    dtype: str
    operation: str
    n_calls: int
    total_s: float

    @property
    def per_call_us(self) -> float:
        """Microseconds per call."""
        return (self.total_s / self.n_calls) * 1e6

    def __str__(self) -> str:
        return (
            f"{self.index_dtype:6s} {self.dtype:8s} "
            f"{self.operation:12s} "
            f"{self.n_calls:6d} calls  "
            f"{self.per_call_us:9.2f} µs/call  "
            f"({self.total_s * 1000:.1f} ms total)"
        )


def _print_table(results: list, capsys) -> None:
    """Print a formatted results table; requires capsys to un-capture stdout."""
    header = (
        f"\n{'index_dtype':8s} {'dtype':8s} {'operation':12s} "
        f"{'n_calls':>8s} {'µs/call':>12s} {'total ms':>12s}"
    )
    separator = "-" * len(header)
    rows = [header, separator]
    for r in results:
        rows.append(
            f"{r.index_dtype:8s} {r.dtype:8s} {r.operation:12s} "
            f"{r.n_calls:8d} {r.per_call_us:12.2f} {r.total_s*1000:12.1f}"
        )
    rows.append(separator)
    # Compute relative overhead vs (int32, float32) for each operation.
    baseline = {
        op: next((r for r in results if r.index_dtype == "int32"
                  and r.dtype == "float32" and r.operation == op), None)
        for op in ("add_item", "build", "query")
    }
    rows.append("\nRelative overhead vs (int32, float32) baseline:")
    for r in results:
        base = baseline.get(r.operation)
        if base and base.total_s > 0:
            ratio = r.total_s / base.total_s
            rows.append(f"  {r.index_dtype:6s}/{r.dtype:7s} {r.operation:12s}: {ratio:.3f}x")

    with capsys.disabled():
        print("\n".join(rows))


# ---------------------------------------------------------------------------
# Benchmark 1: add_item throughput
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("index_dtype,dtype", DTYPE_COMBOS)
def test_benchmark_add_item(index_dtype: str, dtype: str, float_vectors, capsys):
    """
    Measure per-call cost of add_item for each dtype combination.

    Pass condition: average add_item cost < MAX_ADD_ITEM_US µs.
    """
    index = Index(f=F, metric="angular", index_dtype=index_dtype, dtype=dtype, seed=SEED)

    t0 = time.perf_counter()
    for i, v in enumerate(float_vectors):
        index.add_item(i, v)
    elapsed = time.perf_counter() - t0

    result = BenchResult(
        index_dtype=index_dtype,
        dtype=dtype,
        operation="add_item",
        n_calls=N_ITEMS,
        total_s=elapsed,
    )

    with capsys.disabled():
        print(f"\n  BENCH add_item  {result}")

    assert result.per_call_us < MAX_ADD_ITEM_US, (
        f"add_item too slow: {result.per_call_us:.1f} µs/call "
        f"(threshold {MAX_ADD_ITEM_US} µs/call) "
        f"[{index_dtype}, {dtype}]"
    )
    # Also verify correctness: item count must be exact.
    assert index.get_n_items() == N_ITEMS


# ---------------------------------------------------------------------------
# Benchmark 2: build throughput
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("index_dtype,dtype", DTYPE_COMBOS)
def test_benchmark_build(index_dtype: str, dtype: str, float_vectors, capsys):
    """
    Measure total build time for each dtype combination.

    Pass condition: total build time < MAX_BUILD_S seconds.
    """
    index = Index(f=F, metric="angular", index_dtype=index_dtype, dtype=dtype, seed=SEED)
    for i, v in enumerate(float_vectors):
        index.add_item(i, v)

    t0 = time.perf_counter()
    index.build(n_trees=N_TREES)
    elapsed = time.perf_counter() - t0

    result = BenchResult(
        index_dtype=index_dtype,
        dtype=dtype,
        operation="build",
        n_calls=1,
        total_s=elapsed,
    )

    with capsys.disabled():
        print(f"\n  BENCH build     {result}")

    assert elapsed < MAX_BUILD_S, (
        f"build too slow: {elapsed:.2f}s "
        f"(threshold {MAX_BUILD_S}s) "
        f"[{index_dtype}, {dtype}]"
    )
    assert index.get_n_trees() == N_TREES


# ---------------------------------------------------------------------------
# Benchmark 3: query throughput (get_nns_by_item)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("index_dtype,dtype", DTYPE_COMBOS)
def test_benchmark_query_by_item(index_dtype: str, dtype: str, float_vectors, capsys):
    """
    Measure per-query cost of get_nns_by_item for each dtype combination.

    Pass condition: average query cost < MAX_QUERY_US µs.
    """
    index = Index(f=F, metric="angular", index_dtype=index_dtype, dtype=dtype, seed=SEED)
    for i, v in enumerate(float_vectors):
        index.add_item(i, v)
    index.build(n_trees=N_TREES)

    rng = random.Random(SEED + 1)
    query_ids = [rng.randint(0, N_ITEMS - 1) for _ in range(N_QUERIES)]

    t0 = time.perf_counter()
    for qid in query_ids:
        index.get_nns_by_item(qid, n=N_RESULTS)
    elapsed = time.perf_counter() - t0

    result = BenchResult(
        index_dtype=index_dtype,
        dtype=dtype,
        operation="query",
        n_calls=N_QUERIES,
        total_s=elapsed,
    )

    with capsys.disabled():
        print(f"\n  BENCH query     {result}")

    assert result.per_call_us < MAX_QUERY_US, (
        f"query too slow: {result.per_call_us:.1f} µs/call "
        f"(threshold {MAX_QUERY_US} µs/call) "
        f"[{index_dtype}, {dtype}]"
    )


# ---------------------------------------------------------------------------
# Benchmark 4: query throughput (get_nns_by_vector)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("index_dtype,dtype", DTYPE_COMBOS)
def test_benchmark_query_by_vector(index_dtype: str, dtype: str, float_vectors, capsys):
    """
    Measure per-query cost of get_nns_by_vector for each dtype combination.

    Pass condition: average query cost < MAX_QUERY_US µs.
    """
    index = Index(f=F, metric="angular", index_dtype=index_dtype, dtype=dtype, seed=SEED)
    for i, v in enumerate(float_vectors):
        index.add_item(i, v)
    index.build(n_trees=N_TREES)

    rng = random.Random(SEED + 2)
    query_ids = [rng.randint(0, N_ITEMS - 1) for _ in range(N_QUERIES)]

    t0 = time.perf_counter()
    for qid in query_ids:
        index.get_nns_by_vector(float_vectors[qid], n=N_RESULTS)
    elapsed = time.perf_counter() - t0

    result = BenchResult(
        index_dtype=index_dtype,
        dtype=dtype,
        operation="qbyvec",
        n_calls=N_QUERIES,
        total_s=elapsed,
    )

    with capsys.disabled():
        print(f"\n  BENCH qbyvec    {result}")

    assert result.per_call_us < MAX_QUERY_US, (
        f"get_nns_by_vector too slow: {result.per_call_us:.1f} µs/call "
        f"(threshold {MAX_QUERY_US} µs/call) "
        f"[{index_dtype}, {dtype}]"
    )


# ---------------------------------------------------------------------------
# Benchmark 5: aggregate summary table
#
# This single test exercises all four combinations for all three operations,
# collects results, and prints a comparison table.  It also asserts that no
# combination is more than MAX_RELATIVE_OVERHEAD × slower than the int32/
# float32 baseline — catching accidental O(n) bridges.
# ---------------------------------------------------------------------------

MAX_RELATIVE_OVERHEAD = 10.0   # generous; any real regression stands out clearly


def test_benchmark_summary_table(float_vectors, capsys):
    """
    Full 4-combo × 3-operation benchmark table with relative overhead check.

    Intended to be run with ``pytest -v -s`` to see the table.

    Pass condition:
    * No combination crashes.
    * No combination is > MAX_RELATIVE_OVERHEAD × slower than (int32, float32).
    """
    results: list[BenchResult] = []

    for index_dtype, dtype in DTYPE_COMBOS:
        # --- add_item ---
        index = Index(
            f=F, metric="angular", index_dtype=index_dtype, dtype=dtype, seed=SEED
        )
        t0 = time.perf_counter()
        for i, v in enumerate(float_vectors):
            index.add_item(i, v)
        add_elapsed = time.perf_counter() - t0
        results.append(BenchResult(index_dtype, dtype, "add_item", N_ITEMS, add_elapsed))

        # --- build ---
        t0 = time.perf_counter()
        index.build(n_trees=N_TREES)
        build_elapsed = time.perf_counter() - t0
        results.append(BenchResult(index_dtype, dtype, "build", 1, build_elapsed))

        # --- query ---
        rng = random.Random(SEED + 3)
        qids = [rng.randint(0, N_ITEMS - 1) for _ in range(N_QUERIES)]
        t0 = time.perf_counter()
        for qid in qids:
            index.get_nns_by_item(qid, n=N_RESULTS)
        query_elapsed = time.perf_counter() - t0
        results.append(BenchResult(index_dtype, dtype, "query", N_QUERIES, query_elapsed))

    _print_table(results, capsys)

    # Relative overhead check per operation
    for op in ("add_item", "build", "query"):
        op_results = [r for r in results if r.operation == op]
        baseline = next(
            r for r in op_results if r.index_dtype == "int32" and r.dtype == "float32"
        )
        for r in op_results:
            if baseline.total_s > 0:
                ratio = r.total_s / baseline.total_s
                assert ratio < MAX_RELATIVE_OVERHEAD, (
                    f"({r.index_dtype}, {r.dtype}) {op} is {ratio:.2f}× slower "
                    f"than (int32, float32) baseline — exceeds "
                    f"{MAX_RELATIVE_OVERHEAD}× threshold.\n"
                    f"  Baseline:  {baseline.total_s*1000:.1f} ms\n"
                    f"  This combo:{r.total_s*1000:.1f} ms"
                )
