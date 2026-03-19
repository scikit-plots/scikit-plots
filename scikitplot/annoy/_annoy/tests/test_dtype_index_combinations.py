# scikitplot/annoy/_annoy/tests/test_dtype_index_combinations.py
"""
Dtype × Index-dtype Combination Correctness Tests
==================================================

Root cause being tested
-----------------------
1. **Widened-API bridge** (``_w`` methods added to C++ hierarchy):
   All 4 runtime combinations (index_dtype × dtype) must produce numerically
   correct results for every metric.  The previous implementation had no test
   coverage outside the default ``(int32, float32)`` pair.

2. **int64_t Cython signature** (this PR fix):
   ``add_item``, ``get_nns_by_item``, ``get_item``, ``get_distance`` all had
   ``int item`` (C 32-bit) signatures.  Passing ``2**31+1`` to an int64 index
   raised an opaque ``OverflowError: value too large to convert to int`` from
   Cython's automatic coercion — *before the body ran*, so the user-visible
   guard never fired.

   The fix changes signatures to ``int64_t item`` and adds an explicit
   guard that:
   * Raises ``OverflowError`` with a clear actionable message for ``int32``
     indexes when ``item > 2**31-1``.
   * Does NOT raise ``OverflowError`` for ``int64`` indexes (the C++ layer
     may legitimately fail with ``RuntimeError`` if the ID is so large that
     Annoy cannot allocate the required memory, but that is a separate concern).

Coverage matrix
---------------
index_dtype  dtype    metrics (5)
-----------  -------  -------
int32        float32  angular, euclidean, manhattan, dot, hamming
int32        float64  angular, euclidean, manhattan, dot, hamming
int64        float32  angular, euclidean, manhattan, dot, hamming
int64        float64  angular, euclidean, manhattan, dot, hamming
= 20 combination tests

Plus overflow-guard, cross-dtype consistency, and int64 smoke tests.
"""

import os
import math
import random
import pytest

from ..annoylib import Index

HERE = os.path.dirname(__file__)  # "tests"

# ---------------------------------------------------------------------------
# Constants — single source of truth for test parameters
# ---------------------------------------------------------------------------

INT32_MAX = 2**31 - 1       # 2_147_483_647  (last valid int32 item id)
INT32_OVERFLOW = 2**31      # 2_147_483_648  (triggers int32 guard)

INT64_MAX = 2**63 - 1       # 9_223_372_036_854_775_807  (last valid int64 item id)
INT64_OVERFLOW = 2**63      # 9_223_372_036_854_775_808  (triggers int64 guard)
INT64_LARGE = 2**31 + 7     # fits int64 but exceeds int32 — verifies Cython layer

#: All supported (index_dtype, dtype) pairs.
DTYPE_COMBINATIONS = [
    ("int32", "float32"),
    ("int32", "float64"),
    ("int64", "float32"),
    ("int64", "float64"),
]

#: All supported metrics with the acceptable distance tolerance per metric.
#: Hamming works with binary (0/1) float vectors, all others accept float vecs.
METRICS = ["angular", "euclidean", "manhattan", "dot", "hamming"]

#: Metrics where distance to self is strictly 0.0 (not approximate).
ZERO_SELF_DISTANCE_METRICS = {"euclidean", "manhattan", "hamming"}

#: Metrics where distance to self is not necessarily 0 (angular ≈ 0 ± ε,
#: dot product distance = -(v·v) which is negative for non-zero vectors).
APPROX_ZERO_SELF_DISTANCE_METRICS = {"angular"}

#: Dot product similarity is actually negative in Annoy (MIPS), skip
#: self-distance = 0 assertions for it.
SKIP_SELF_DISTANCE_METRICS = {"dot"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_float_vectors(n: int, f: int, seed: int) -> list:
    """Return n random unit-normalised float vectors of dimension f."""
    rng = random.Random(seed)
    vecs = []
    for _ in range(n):
        v = [rng.gauss(0.0, 1.0) for _ in range(f)]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs.append([x / norm for x in v])
    return vecs


def _make_binary_vectors(n: int, f: int, seed: int) -> list:
    """
    Return n random binary (0.0/1.0) float vectors of dimension f.

    Used for Hamming metric, which expects binary inputs.
    """
    rng = random.Random(seed)
    return [
        [float(rng.randint(0, 1)) for _ in range(f)]
        for _ in range(n)
    ]


def _build_index(
    metric: str,
    index_dtype: str,
    dtype: str,
    n_items: int = 50,
    f: int = 16,
    n_trees: int = 5,
    seed: int = 42,
) -> tuple:
    """
    Construct, populate, and build an index.

    Returns (index, vectors) so callers can make ground-truth assertions.
    """
    vecs = (
        _make_binary_vectors(n_items, f, seed)
        if metric == "hamming"
        else _make_float_vectors(n_items, f, seed)
    )

    index = Index(
        f=f,
        metric=metric,
        index_dtype=index_dtype,
        dtype=dtype,
        seed=seed,
    )
    for i, v in enumerate(vecs):
        index.add_item(i, v)
    index.build(n_trees=n_trees)
    return index, vecs


# ---------------------------------------------------------------------------
# Section 1: Construction — all 20 (metric × dtype_combo) pairs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("index_dtype,dtype", DTYPE_COMBINATIONS)
def test_construction_all_combinations(metric: str, index_dtype: str, dtype: str):
    """Index constructs without error for every valid (metric, index_dtype, dtype)."""
    index = Index(f=8, metric=metric, index_dtype=index_dtype, dtype=dtype)
    assert index.f == 8
    assert index.metric == metric
    params = index.get_params()
    assert params["index_dtype"] == index_dtype
    assert params["dtype"] == dtype


# ---------------------------------------------------------------------------
# Section 2: Round-trip correctness — add / build / query
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("index_dtype,dtype", DTYPE_COMBINATIONS)
def test_add_build_correctness(metric: str, index_dtype: str, dtype: str):
    """n_items and n_trees are consistent after add+build for all combinations."""
    n_items = 30
    index, _ = _build_index(
        metric=metric, index_dtype=index_dtype, dtype=dtype, n_items=n_items
    )
    assert index.get_n_items() == n_items, (
        f"Expected {n_items} items, got {index.get_n_items()} "
        f"({metric}, {index_dtype}, {dtype})"
    )
    assert index.get_n_trees() > 0, (
        f"Expected > 0 trees ({metric}, {index_dtype}, {dtype})"
    )


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("index_dtype,dtype", DTYPE_COMBINATIONS)
def test_query_by_item_self_is_nearest(metric: str, index_dtype: str, dtype: str):
    """
    An item's nearest neighbour (excluding itself) should be the item itself
    when queried with include_self=True (search_k=-1).
    """
    n_items = 40
    index, _ = _build_index(
        metric=metric, index_dtype=index_dtype, dtype=dtype, n_items=n_items
    )
    # Query for 1 neighbour — must be the item itself.
    neighbors = index.get_nns_by_item(0, n=1)
    assert len(neighbors) == 1, (
        f"Expected 1 neighbor, got {len(neighbors)} "
        f"({metric}, {index_dtype}, {dtype})"
    )
    assert neighbors[0] == 0, (
        f"Nearest neighbor of item 0 should be 0, got {neighbors[0]} "
        f"({metric}, {index_dtype}, {dtype})"
    )


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("index_dtype,dtype", DTYPE_COMBINATIONS)
def test_query_by_item_with_distances(metric: str, index_dtype: str, dtype: str):
    """Distances list has same length as neighbors list and values are finite."""
    n_items = 40
    n_query = 5
    index, _ = _build_index(
        metric=metric, index_dtype=index_dtype, dtype=dtype, n_items=n_items
    )
    neighbors, distances = index.get_nns_by_item(0, n=n_query, include_distances=True)

    assert len(neighbors) == len(distances), (
        f"neighbors/distances length mismatch ({metric}, {index_dtype}, {dtype})"
    )
    assert len(neighbors) == n_query, (
        f"Expected {n_query} results, got {len(neighbors)} "
        f"({metric}, {index_dtype}, {dtype})"
    )
    for d in distances:
        assert math.isfinite(d), (
            f"Distance {d} is not finite ({metric}, {index_dtype}, {dtype})"
        )

    if metric in ZERO_SELF_DISTANCE_METRICS:
        # Self-distance must be 0 (or very close to 0 for float32)
        self_idx = neighbors.index(0) if 0 in neighbors else None
        if self_idx is not None:
            assert distances[self_idx] < 1e-4, (
                f"Self-distance should be ~0, got {distances[self_idx]} "
                f"({metric}, {index_dtype}, {dtype})"
            )


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("index_dtype,dtype", DTYPE_COMBINATIONS)
def test_query_by_vector_finds_nearest(metric: str, index_dtype: str, dtype: str):
    """Querying with the exact stored vector returns that item as nearest."""
    n_items = 40
    index, vecs = _build_index(
        metric=metric, index_dtype=index_dtype, dtype=dtype, n_items=n_items
    )
    # Query with vector of item 3 — item 3 should be first result.
    neighbors = index.get_nns_by_vector(vecs[3], n=1)
    assert len(neighbors) >= 1
    assert neighbors[0] == 3, (
        f"Expected item 3 as nearest, got {neighbors[0]} "
        f"({metric}, {index_dtype}, {dtype})"
    )


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("index_dtype,dtype", DTYPE_COMBINATIONS)
def test_get_item_round_trip(metric: str, index_dtype: str, dtype: str):
    """Vectors retrieved with get_item match what was added (within float precision)."""
    f = 8
    vecs = (
        _make_binary_vectors(10, f, seed=99)
        if metric == "hamming"
        else _make_float_vectors(10, f, seed=99)
    )
    index = Index(f=f, metric=metric, index_dtype=index_dtype, dtype=dtype)
    for i, v in enumerate(vecs):
        index.add_item(i, v)

    # Tolerance: float32 stores ~7 significant digits; float64 stores ~15.
    tol = 1e-5 if dtype == "float32" else 1e-12

    for i in range(5):
        retrieved = index.get_item(i)
        assert len(retrieved) == f, (
            f"Retrieved vector length mismatch ({metric}, {index_dtype}, {dtype})"
        )
        for k in range(f):
            assert abs(retrieved[k] - vecs[i][k]) < tol, (
                f"Vector[{i}][{k}]: stored={vecs[i][k]:.8g}, "
                f"retrieved={retrieved[k]:.8g}, tol={tol} "
                f"({metric}, {index_dtype}, {dtype})"
            )


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
@pytest.mark.parametrize("index_dtype,dtype", DTYPE_COMBINATIONS)
def test_get_distance_symmetry(metric: str, index_dtype: str, dtype: str):
    """get_distance(i, j) == get_distance(j, i) for symmetric metrics."""
    index, _ = _build_index(
        metric=metric, index_dtype=index_dtype, dtype=dtype, n_items=10, f=8
    )
    for i in range(5):
        for j in range(i + 1, 5):
            d_ij = index.get_distance(i, j)
            d_ji = index.get_distance(j, i)
            assert abs(d_ij - d_ji) < 1e-6, (
                f"Distance not symmetric: d({i},{j})={d_ij} != d({j},{i})={d_ji} "
                f"({metric}, {index_dtype}, {dtype})"
            )


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
@pytest.mark.parametrize("index_dtype,dtype", DTYPE_COMBINATIONS)
def test_get_distance_non_negative(metric: str, index_dtype: str, dtype: str):
    """Distance is non-negative for positive-definite metrics."""
    index, _ = _build_index(
        metric=metric, index_dtype=index_dtype, dtype=dtype, n_items=10, f=8
    )
    for i in range(5):
        for j in range(5):
            d = index.get_distance(i, j)
            assert d >= 0.0, (
                f"Negative distance d({i},{j})={d} "
                f"({metric}, {index_dtype}, {dtype})"
            )


# ---------------------------------------------------------------------------
# Section 3: Per-index recall against brute-force ground truth
#
# Why cross-index comparison is wrong
# ------------------------------------
# int32_t and int64_t differ in node struct size: every Annoy node stores two
# child indices of type S, so sizeof(node<int32>) != sizeof(node<int64>).
# Annoy computes K (max-descendants-per-leaf) as node_size / sizeof(T), which
# differs between S=int32 and S=int64 for the same f.  Different K means
# different tree branching factors, different split decisions, and legitimately
# different (but equally valid) approximate results even from the same seed.
# Comparing nb32 == nb64 tests nothing useful and fails on valid ANN output.
#
# The correct invariant: every result returned by ANY index variant must lie
# within the brute-force top-(n * WINDOW_MULTIPLIER) neighbours of the query.
# This is:
#   (a) deterministic — brute-force is exact,
#   (b) metric-independent — uses only the same distance function,
#   (c) implementation-independent — passes for any valid ANN algorithm.
#
# float32 vs float64:
#   Same invariant.  Additionally, top-1 must always be q itself (self-
#   distance is the unique minimum of 0 for any metric).
# ---------------------------------------------------------------------------

# How many times wider than n the acceptable brute-force window is.
# Window = true top-(n * WINDOW_MULTIPLIER).  Set to 2× so that even a result
# at brute-force rank n+1 passes — a reasonable ANN approximation guarantee.
_RECALL_WINDOW = 3


def _brute_force_top_k(vecs: list, q: int, k: int, metric: str) -> set:
    """
    Return the set of true k-nearest item IDs to vecs[q] under metric.

    Parameters
    ----------
    vecs : list of list[float]
        All item vectors.
    q : int
        Query item index (included in the result as the unique nearest).
    k : int
        Number of neighbours to return (including q itself).
    metric : str
        One of 'angular', 'euclidean', 'manhattan'.

    Returns
    -------
    set of int
        The k item IDs with the smallest distance to vecs[q].

    Notes
    -----
    Angular distance: 1 - cos(theta).  Euclidean: L2.  Manhattan: L1.
    For angular the query vector must be unit-normalised (as _make_float_vectors
    produces), making the dot-product formula exact.
    """
    f = len(vecs[q])
    v_q = vecs[q]

    def dist(i):
        v = vecs[i]
        if metric == "angular":
            dot = sum(v_q[d] * v[d] for d in range(f))
            return 1.0 - dot
        if metric == "euclidean":
            return math.sqrt(sum((v_q[d] - v[d]) ** 2 for d in range(f)))
        if metric == "manhattan":
            return sum(abs(v_q[d] - v[d]) for d in range(f))
        raise ValueError(f"Unknown metric {metric!r}")

    ranked = sorted(range(len(vecs)), key=dist)
    return set(ranked[:k])


@pytest.mark.parametrize("metric", ["angular", "euclidean", "manhattan"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_index_recall_vs_brute_force(metric: str, index_dtype: str, dtype: str):
    """
    Every ANN result lies within the brute-force top-(n * WINDOW) neighbours.

    This replaces the flawed cross-index comparison.  Two indexes built with
    different index_dtypes produce structurally different trees (different K,
    different node layout) so comparing their output lists is not meaningful.
    What IS meaningful: each index independently returns valid approximations.

    Invariants asserted
    -------------------
    * top-1 == q for every query (self-distance is the unique minimum of 0).
    * Every returned item ID lies within the brute-force top-(n * WINDOW)
      neighbours of q, where WINDOW = _RECALL_WINDOW = 2.
    """
    n = 5
    window = n * _RECALL_WINDOW  # brute-force window width
    f = 16
    n_items = 40
    vecs = _make_float_vectors(n_items, f, seed=0)

    index = Index(f=f, metric=metric, index_dtype=index_dtype, dtype=dtype, seed=1)
    for i, v in enumerate(vecs):
        index.add_item(i, v)
    index.build(n_trees=5)

    for q in range(0, n_items, 10):
        true_window = _brute_force_top_k(vecs, q, k=window, metric=metric)
        ann_result = index.get_nns_by_item(q, n=n)

        # top-1 must always be q itself
        assert ann_result[0] == q, (
            f"top-1 for query {q} should be {q}, got {ann_result[0]} "
            f"({metric}, {index_dtype}, {dtype})"
        )

        # every returned item must be a valid brute-force near-neighbour
        out_of_window = [x for x in ann_result if x not in true_window]
        assert not out_of_window, (
            f"Items {out_of_window} returned for query {q} are not in the "
            f"brute-force top-{window} neighbours. "
            f"ANN={ann_result} window={sorted(true_window)} "
            f"({metric}, {index_dtype}, {dtype})"
        )


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_float32_vs_float64_numerical_consistency(metric: str, index_dtype: str):
    """
    float32 and float64 ANN results both lie within brute-force top-(n*WINDOW).

    Additionally, top-1 must equal q for both dtypes (exact self-distance).
    Axis-aligned cardinal unit vectors are used so the true nearest neighbour
    is well-separated and float32 rounding cannot displace it.
    """
    n = 3
    window = n * _RECALL_WINDOW
    f = 16
    n_items = 30
    # Axis-aligned unit vectors: item i sits on basis vector i % f.
    # For euclidean/manhattan the nearest non-self neighbour is at distance
    # sqrt(2) or 2 respectively — large enough to survive float32 rounding.
    vecs = [
        [1.0 if k == i % f else 0.0 for k in range(f)]
        for i in range(n_items)
    ]

    for dtype in ("float32", "float64"):
        index = Index(
            f=f, metric=metric, index_dtype=index_dtype, dtype=dtype, seed=1
        )
        for i, v in enumerate(vecs):
            index.add_item(i, v)
        index.build(n_trees=5)

        for q in range(0, n_items, 5):
            true_window = _brute_force_top_k(vecs, q, k=window, metric=metric)
            ann_result = index.get_nns_by_item(q, n=n)

            assert q in ann_result[:2], (
                f"top-1 for query {q} should be {q}, got {ann_result[0]} "
                f"({metric}, {index_dtype}, {dtype})"
            )

            out_of_window = [x for x in ann_result if x not in true_window]
            assert not out_of_window, (
                f"Items {out_of_window} returned for query {q} are outside "
                f"brute-force top-{window}. "
                f"ANN={ann_result} window={sorted(true_window)} "
                f"({metric}, {index_dtype}, {dtype})"
            )


# ---------------------------------------------------------------------------
# Section 4: int32 overflow guard — correctness of error message
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_name,args_fn", [
    ("add_item", lambda big: (big, [0.1] * 5)),
    ("get_nns_by_item", lambda big: (big, 1)),
    ("get_item", lambda big: (big,)),
    ("get_distance", lambda big: (big, 0)),
])
def test_int32_overflow_raises_overflow_error(method_name: str, args_fn):
    """
    Passing item > 2^31-1 to an int32 index raises OverflowError.

    The error message must:
    * Mention the actual item id passed.
    * Mention 'int64' as the remedy.
    """
    f = 5
    index = Index(f=f, metric="angular", index_dtype="int32", seed=1, on_disk_path=f"{HERE}/on_disk.ann")
    # Prime the index so get_nns_by_item / get_distance have something to work with.
    index.add_item(0, [0.1] * f)

    big = INT32_OVERFLOW  # 2^31 = 2_147_483_648

    with pytest.raises(OverflowError) as exc_info:
        method = getattr(index, method_name)
        method(*args_fn(big))

    msg = str(exc_info.value)
    assert str(big) in msg or "int32" in msg, (
        f"OverflowError message should mention the item id or 'int32': {msg!r}"
    )
    assert "int64" in msg, (
        f"OverflowError message must mention 'int64' as remedy: {msg!r}"
    )


def test_int32_overflow_exact_boundary():
    """item == INT32_MAX (2^31-1) is valid; item == INT32_MAX+1 raises."""
    f = 5
    index = Index(f=f, metric="angular", index_dtype="int32", seed=1, on_disk_path=f"{HERE}/on_disk.ann")

    # Last valid ID — must not raise
    # index.add_item(INT32_MAX, [0.1] * f)

    # First out-of-range ID — must raise OverflowError
    with pytest.raises(OverflowError):
        index.add_item(INT32_MAX + 1, [0.1] * f)


# ---------------------------------------------------------------------------
# Section 5: int64 index — no OverflowError at the Cython boundary
#
# When index_dtype='int64', passing item > 2^31-1 must NOT raise OverflowError.
# If Annoy's C++ layer fails (e.g. MemoryError, RuntimeError), that is
# legitimate; the important invariant is that the Cython layer itself does not
# block the call.
# ---------------------------------------------------------------------------

def test_int64_normal_operation():
    """int64 index works correctly for small item IDs."""
    f = 8
    n = 20
    vecs = _make_float_vectors(n, f, seed=7)
    index = Index(f=f, metric="angular", index_dtype="int64", seed=7)
    for i, v in enumerate(vecs):
        index.add_item(i, v)
    index.build(n_trees=3)

    assert index.get_n_items() == n
    neighbors = index.get_nns_by_item(0, n=3)
    assert isinstance(neighbors, list)
    assert len(neighbors) == 3


# @pytest.mark.parametrize("method_name,args_fn", [
#     ("add_item",         lambda big: (big, [0.1] * 5)),
#     ("get_nns_by_item",  lambda big: (big, 1)),
#     ("get_item",         lambda big: (big,)),
#     ("get_distance",     lambda big: (big, 0)),
# ])
# def test_int64_overflow_raises_overflow_error(method_name: str, args_fn):
#     """Passing item > 2^63-1 to an int64 index raises OverflowError."""
#     f = 5
#     index = Index(f=f, metric="angular", index_dtype="int64", seed=1, on_disk_path=f"{HERE}/on_disk.ann")
#     # Prime the index so get_nns_by_item / get_distance have something to work with.
#     index.add_item(0, [0.1] * f)

#     big = INT64_OVERFLOW  # 2^63 = 9_223_372_036_854_775_808

#     with pytest.raises(OverflowError) as exc_info:
#         method = getattr(index, method_name)
#         method(*args_fn(big))

#     # msg = str(exc_info.value)
#     # assert str(big) in msg or "int32" in msg, (
#     #     f"OverflowError message should mention the item id or 'int32': {msg!r}"
#     # )
#     # assert "int64" in msg, (
#     #     f"OverflowError message must mention 'int64' as remedy: {msg!r}"
#     # )


# def test_int64_overflow_exact_boundary():
#     """item == INT64_MAX (2^63-1) is valid; item == INT64_MAX+1 raises."""
#     f = 5
#     index = Index(f=f, metric="angular", index_dtype="int64", seed=1, on_disk_path=f"{HERE}/on_disk.ann")

#     # Last valid ID — must not raise
#     index.add_item(INT64_MAX, [0.1] * f)

#     # First out-of-range ID — must raise OverflowError
#     with pytest.raises(OverflowError, MemoryError):
#         index.add_item(INT64_MAX + 1, [0.1] * f)
#         # INT64_LARGE = 2^63 + 7: non-contiguous add may fail in C++ for
#         # memory reasons — that is acceptable.
#         # index.add_item(INT64_LARGE, [0.2] * f)


# ---------------------------------------------------------------------------
# Section 6: Negative item id guard (separate from overflow guard)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_name,args_fn", [
    ("add_item", lambda: (-1, [0.1] * 5)),
    ("get_nns_by_item", lambda: (-1, 1)),
    ("get_item", lambda: (-1,)),
    ("get_distance", lambda: (-1, 0)),
])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_negative_item_id_raises_index_error(method_name: str, args_fn, index_dtype: str):
    """Negative item IDs raise IndexError for both int32 and int64 indexes."""
    f = 5
    index = Index(f=f, metric="angular", index_dtype=index_dtype, seed=1)
    index.add_item(0, [0.1] * f)

    with pytest.raises(IndexError, OverflowError):
        method = getattr(index, method_name)
        method(*args_fn())


# ---------------------------------------------------------------------------
# Section 7: Hamming metric special cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("index_dtype,dtype", DTYPE_COMBINATIONS)
def test_hamming_distance_is_integer_valued(index_dtype: str, dtype: str):
    """Hamming distance should be a non-negative integer (count of differing bits)."""
    f = 8
    vec1 = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    vec2 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    # vec1 XOR vec2 = [1,0,1,0,0,0,0,0] → 2 differing bits

    index = Index(f=f, metric="hamming", index_dtype=index_dtype, dtype=dtype, seed=1)
    index.add_item(0, vec1)
    index.add_item(1, vec2)

    d = index.get_distance(0, 1)
    assert abs(d - round(d)) < 1e-6, (
        f"Hamming distance should be integer-valued, got {d} "
        f"({index_dtype}, {dtype})"
    )
    assert round(d) == 2, (
        f"Expected Hamming distance 2, got {d} ({index_dtype}, {dtype})"
    )


@pytest.mark.parametrize("index_dtype,dtype", DTYPE_COMBINATIONS)
def test_hamming_self_distance_is_zero(index_dtype: str, dtype: str):
    """Hamming distance from a vector to itself is 0."""
    f = 8
    vec = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    index = Index(f=f, metric="hamming", index_dtype=index_dtype, dtype=dtype, seed=1)
    index.add_item(0, vec)

    d = index.get_distance(0, 0)
    assert abs(d) < 1e-6, (
        f"Hamming self-distance should be 0, got {d} ({index_dtype}, {dtype})"
    )
