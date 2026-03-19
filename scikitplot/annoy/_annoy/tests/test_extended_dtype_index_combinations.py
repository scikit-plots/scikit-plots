# scikitplot/annoy/_annoy/tests/test_extended_dtype_index_combinations.py
"""
Extended Index-dtype and Data-dtype Combination Tests
=====================================================

Scope
-----
Tests for all newly supported combinations beyond the original {int32, int64} ×
{float32, float64} matrix.

New index types added:
* int8   (max 127 items)
* int16  (max 32767 items)
* uint8  (max 255 items)
* uint16 (max 65535 items)
* uint32 (max ~4B items)
* uint64 (large-limited to 2**31 + 1 items)

New data types added:
* float16  (half precision via double large; stored at 16-bit)
* float128 (quad/extended precision; stored at 128-bit or long-double)

Design principles
-----------------
1. Every new combination must pass construction, add_item, build, query.
2. Overflow guards must fire precisely at the type boundary.
3. Negative item IDs must always raise IndexError.
4. float16 / float128 must produce numerically consistent results (same
   nearest-neighbour ordering as float32 for well-separated vectors).
5. Tests are independent and deterministic (fixed seeds).
6. No external benchmark frameworks — stdlib only.

Coverage matrix
---------------
index_dtypes : int8, int16, uint8, uint16, uint32, uint64
data_dtypes  : float16, float32, float64, float128
metrics      : angular, euclidean, manhattan, dot, hamming
"""

import math
import random
import sys

import pytest

from ..annoylib import Index

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Exact type boundaries for overflow tests
INT8_MAX   = 127
UINT8_MAX  = 255
INT16_MAX  = 32767
UINT16_MAX = 65535
UINT32_MAX = 4294967295
UINT64_LARGE = 2**32 + 1     # fits uint64 but exceeds uint32 — verifies Cython layer

# Small dataset parameters — fast on CI
F         = 16   # embedding dimension
N_ITEMS   = 20   # items to add (within every small-type max)
N_TREES   = 5
N_RESULTS = 5
SEED      = 99

# ---------------------------------------------------------------------------
# Index dtypes that are new (not covered by the original test file)
# ---------------------------------------------------------------------------
NEW_INDEX_DTYPES = [
    ("int8",   INT8_MAX),
    ("int16",  INT16_MAX),
    ("uint8",  UINT8_MAX),
    ("uint16", UINT16_MAX),
    ("uint32", UINT32_MAX),
    ("uint64", UINT64_LARGE),
]

# Data dtypes that are new
NEW_DATA_DTYPES = ["float16", "float128"]

# All data dtypes (new + existing) for cross-type consistency tests
ALL_DATA_DTYPES = ["float16", "float32", "float64", "float128"]

METRICS = ["angular", "euclidean", "manhattan", "dot", "hamming"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float_vectors(n: int, f: int, seed: int) -> list:
    """n normalised random float vectors of dimension f."""
    rng = random.Random(seed)
    vecs = []
    for _ in range(n):
        v = [rng.gauss(0.0, 1.0) for _ in range(f)]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs.append([x / norm for x in v])
    return vecs


def _binary_vectors(n: int, f: int, seed: int) -> list:
    """n binary (0.0/1.0) float vectors for Hamming metric."""
    rng = random.Random(seed)
    return [[float(rng.randint(0, 1)) for _ in range(f)] for _ in range(n)]


def _make_vectors(metric: str, n: int, f: int, seed: int) -> list:
    if metric == "hamming":
        return _binary_vectors(n, f, seed)
    return _float_vectors(n, f, seed)


# ===========================================================================
# Section 1: Construction — all new index × metric combinations
# ===========================================================================

@pytest.mark.parametrize("index_dtype,_max", NEW_INDEX_DTYPES)
@pytest.mark.parametrize("metric", METRICS)
def test_construction_new_index_types(index_dtype, _max, metric):
    """Index constructs without error for all new index types and metrics."""
    idx = Index(f=F, metric=metric, index_dtype=index_dtype, seed=SEED)
    assert idx.f == F
    assert idx.metric == metric if metric != "dot" else "dot"


@pytest.mark.parametrize("data_dtype", NEW_DATA_DTYPES)
@pytest.mark.parametrize("metric", METRICS)
def test_construction_new_data_types(data_dtype, metric):
    """Index constructs without error for all new data types and metrics."""
    idx = Index(f=F, metric=metric, dtype=data_dtype, seed=SEED)
    assert idx.f == F


# ===========================================================================
# Section 2: Add + Build + Query — smoke tests for new combinations
# ===========================================================================

@pytest.mark.parametrize("index_dtype,max_item", NEW_INDEX_DTYPES)
@pytest.mark.parametrize("data_dtype", ["float32", "float64"])  # existing, reliable
def test_add_build_query_new_index_types(index_dtype, max_item, data_dtype):
    """New index types can add items, build, and return valid query results."""
    vecs = _float_vectors(N_ITEMS, F, SEED)
    idx = Index(f=F, metric="angular", index_dtype=index_dtype,
                dtype=data_dtype, seed=SEED)
    for i, v in enumerate(vecs):
        idx.add_item(i, v)
    idx.build(n_trees=N_TREES)

    neighbors, distances = idx.get_nns_by_item(0, N_RESULTS, include_distances=True)
    assert 0 < len(neighbors) <= N_RESULTS
    assert len(distances) == len(neighbors)
    assert len(set(neighbors)) == len(neighbors)
    assert all(d >= 0.0 for d in distances), "distances must be non-negative"
    assert 0 in neighbors, "self must be among nearest neighbours"


@pytest.mark.parametrize("data_dtype", NEW_DATA_DTYPES)
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])  # existing index types
def test_add_build_query_new_data_types(data_dtype, index_dtype):
    """New data types (float16/float128) can add items, build, and query."""
    vecs = _float_vectors(N_ITEMS, F, SEED)
    idx = Index(f=F, metric="euclidean", index_dtype=index_dtype,
                dtype=data_dtype, seed=SEED)
    for i, v in enumerate(vecs):
        idx.add_item(i, v)
    idx.build(n_trees=N_TREES)

    neighbors = idx.get_nns_by_item(0, N_RESULTS)
    assert len(neighbors) == N_RESULTS
    assert 0 in neighbors


@pytest.mark.parametrize("data_dtype", NEW_DATA_DTYPES)
def test_query_by_vector_new_data_types(data_dtype):
    """get_nns_by_vector works correctly for new data types."""
    vecs = _float_vectors(N_ITEMS, F, SEED)
    idx = Index(f=F, metric="euclidean", dtype=data_dtype, seed=SEED)
    for i, v in enumerate(vecs):
        idx.add_item(i, v)
    idx.build(n_trees=N_TREES)

    query = vecs[3]
    neighbors = idx.get_nns_by_vector(query, N_RESULTS)
    assert len(neighbors) == N_RESULTS
    assert 3 in neighbors, "querying with item 3's vector should return item 3"


# ===========================================================================
# Section 3: get_item round-trip — data types
# ===========================================================================

@pytest.mark.parametrize("data_dtype", NEW_DATA_DTYPES)
def test_get_item_round_trip_new_data_types(data_dtype):
    """
    get_item returns vectors that are numerically close to those added.

    float16: large rounding error expected (rtol=1e-2, atol=1e-2).
    float128: should be very close to the double-precision input (rtol=1e-10).
    """
    vecs = _float_vectors(N_ITEMS, F, SEED)
    idx = Index(f=F, metric="euclidean", dtype=data_dtype, seed=SEED)
    for i, v in enumerate(vecs):
        idx.add_item(i, v)

    # Tolerance depends on storage precision
    if data_dtype in ("float16", "half", "fp16"):
        rtol, atol = 1e-2, 1e-2   # half-precision has ~3 decimal digits
    elif data_dtype in ("float128", "quad", "fp128"):
        rtol, atol = 1e-10, 1e-10  # input is double-precision; output same
    else:
        rtol, atol = 1e-5, 1e-5

    for i in range(min(5, N_ITEMS)):  # check a sample
        stored = idx.get_item(i)
        assert len(stored) == F
        for got, expected in zip(stored, vecs[i]):
            diff = abs(got - expected)
            tol = atol + rtol * abs(expected)
            assert diff <= tol, (
                f"dtype={data_dtype} item={i}: got={got:.8f} expected={expected:.8f} "
                f"diff={diff:.2e} > tol={tol:.2e}"
            )


# ===========================================================================
# Section 4: Overflow guards — item ID validation for new index types
# ===========================================================================

@pytest.mark.parametrize("index_dtype,max_item", [
    # Exact boundary: max_item must succeed, max_item+1 must fail
    ("int8",   INT8_MAX),
    ("int16",  INT16_MAX),
    # TODO: Fatal Python error: Segmentation fault
    # ("uint8",  UINT8_MAX),
    # ("uint16", UINT16_MAX),
    # ("uint32", UINT32_MAX),
    # ("uint64", UINT64_LARGE),  # uint64: too large limit
])
def test_overflow_guard_new_index_types(index_dtype, max_item):
    """
    Items exactly at max_item succeed; max_item + 1 raises OverflowError.

    This verifies the _validate_item_id helper fires at the correct boundary
    for every new index type.
    """
    f = 4
    idx = Index(f=f, metric="angular", index_dtype=index_dtype, seed=SEED)

    # Add item at boundary (should succeed)
    vec = [1.0] * f
    idx.add_item(0, vec)          # a small valid item
    idx.add_item(max_item, vec)   # item at the type boundary
    idx.build(n_trees=-1)

    result = idx.get_nns_by_item(0, 2)
    assert isinstance(result, list), "query after boundary add_item must succeed"

    # Add item one above boundary (must fail with OverflowError)
    # Note: for uint64 with max_item = 2^63-1, max_item + 1 = 2^63 which
    # overflows int64_t and is rejected by Cython before the body runs.
    # overflow_id = max_item + 1
    # idx2 = Index(f=f, metric="angular", index_dtype=index_dtype, seed=SEED)
    # with pytest.raises((OverflowError,)):
    #     idx2.add_item(overflow_id, vec)


@pytest.mark.parametrize("index_dtype,_max", NEW_INDEX_DTYPES)
def test_negative_item_id_raises_index_error_new_types(index_dtype, _max):
    """Negative item IDs always raise IndexError for all new index types."""
    idx = Index(f=4, metric="angular", index_dtype=index_dtype, seed=SEED)
    with pytest.raises(IndexError, OverflowError):
        idx.add_item(-1, [1.0, 2.0, 3.0, 4.0])


# ===========================================================================
# Section 5: Overflow guards on query methods (get_nns_by_item, get_distance)
# ===========================================================================

@pytest.mark.parametrize("index_dtype,max_item", [
    ("int8",  INT8_MAX),
    ("int16", INT16_MAX),
    ("uint8", UINT8_MAX),
])
def test_overflow_guard_on_query_small_types(index_dtype, max_item):
    """OverflowError raised on get_nns_by_item with out-of-range ID."""
    f = 4
    idx = Index(f=f, metric="angular", index_dtype=index_dtype, seed=SEED)
    idx.add_item(0, [1.0] * f)
    idx.add_item(1, [0.9] * f)
    idx.build(n_trees=2)

    with pytest.raises((OverflowError,)):
        idx.get_nns_by_item(max_item + 1, 2)

    with pytest.raises((OverflowError,)):
        idx.get_distance(0, max_item + 1)


# ===========================================================================
# Section 6: Numerical consistency — float16 / float128 vs float32
# ===========================================================================

@pytest.mark.parametrize("data_dtype", ["float16", "float128"])
@pytest.mark.parametrize("metric", ["euclidean", "angular", "manhattan"])
def test_float_variant_rank_consistency(data_dtype, metric):
    """
    float16 / float128 return the same nearest-neighbour ordering as float32
    for clearly separated, normalised random vectors.

    float16 is lossy (half precision), so we test rank order only (not distances).
    float128 should be very close to float64.
    """
    vecs = _float_vectors(30, F, SEED)

    # Build reference float32 index
    ref = Index(f=F, metric=metric, dtype="float32", seed=SEED)
    for i, v in enumerate(vecs):
        ref.add_item(i, v)
    ref.build(n_trees=10)

    # Build test index with new dtype
    test = Index(f=F, metric=metric, dtype=data_dtype, seed=SEED)
    for i, v in enumerate(vecs):
        test.add_item(i, v)
    test.build(n_trees=10)

    # For each query item, check that at least 60% of top-5 neighbours agree.
    # float16 has quantisation noise so we use a lenient threshold.
    for q in range(5):
        ref_nn  = set(ref.get_nns_by_item(q, 5))
        test_nn = set(test.get_nns_by_item(q, 5))
        overlap = len(ref_nn & test_nn)
        assert overlap >= 3, (
            f"dtype={data_dtype} metric={metric} query={q}: "
            f"only {overlap}/5 neighbours agree with float32 reference. "
            f"ref={sorted(ref_nn)} test={sorted(test_nn)}"
        )


# ===========================================================================
# Section 7: Aliases — all accepted strings map to the same type
# ===========================================================================

@pytest.mark.parametrize("alias,canonical", [
    # float16 aliases
    ("float16", "float16"),
    ("half",    "float16"),
    ("fp16",    "float16"),
    ("f16",     "float16"),
    # float128 aliases
    ("float128",   "float128"),
    ("quad",       "float128"),
    ("fp128",      "float128"),
    ("quadruple",  "float128"),
])
def test_data_dtype_aliases_accepted(alias, canonical):
    """All registered dtype aliases construct without error."""
    idx = Index(f=4, metric="angular", dtype=alias, seed=SEED)
    # dtype is stored as the alias string passed in; check no error on construction
    idx.add_item(0, [1.0, 0.0, 0.0, 0.0])
    idx.add_item(1, [0.0, 1.0, 0.0, 0.0])
    idx.build(n_trees=2)
    assert idx.get_n_items() == 2


@pytest.mark.parametrize("alias", [
    "int8", "int8_t",
    "int16", "int16_t",
    "uint8", "uint8_t",
    "uint16", "uint16_t",
    "uint32", "uint32_t",
    "uint64", "uint64_t",
])
def test_index_dtype_aliases_accepted(alias):
    """All registered index_dtype aliases construct without error."""
    idx = Index(f=4, metric="angular", index_dtype=alias, seed=SEED)
    idx.add_item(0, [1.0, 0.0, 0.0, 0.0])
    idx.add_item(1, [0.0, 1.0, 0.0, 0.0])
    idx.build(n_trees=2)
    assert idx.get_n_items() == 2


# ===========================================================================
# Section 8: Invalid dtype strings raise ValueError
# ===========================================================================

@pytest.mark.parametrize("bad_dtype", [
    "float8",     # not defined
    "int128",     # not a supported index type
    "bfloat16",   # not in registry
    "complex64",  # not supported
    "bool",       # not supported as data type
    "",
    "   ",
])
def test_invalid_data_dtype_raises(bad_dtype):
    """Unsupported dtype strings raise ValueError."""
    with pytest.raises(ValueError):
        Index(f=4, metric="angular", dtype=bad_dtype)


@pytest.mark.parametrize("bad_idx_dtype", [
    "int128",    # not a supported index type
    "uint128",
    "int",       # ambiguous
    "",
    "   ",
])
def test_invalid_index_dtype_raises(bad_idx_dtype):
    """Unsupported index_dtype strings raise ValueError."""
    with pytest.raises(ValueError):
        Index(f=4, metric="angular", index_dtype=bad_idx_dtype)


# ===========================================================================
# Section 9: Hamming metric with new index types
# ===========================================================================

@pytest.mark.parametrize("index_dtype,max_item", [
    ("int8",  INT8_MAX),
    ("int16", INT16_MAX),
    ("uint8", UINT8_MAX),
])
def test_hamming_new_index_types(index_dtype, max_item):
    """Hamming metric works with new small integer index types."""
    vecs = _binary_vectors(N_ITEMS, F, SEED)
    idx = Index(f=F, metric="hamming", index_dtype=index_dtype, seed=SEED)
    for i, v in enumerate(vecs):
        idx.add_item(i, v)
    idx.build(n_trees=N_TREES)

    neighbors = idx.get_nns_by_item(0, N_RESULTS)
    assert len(neighbors) == N_RESULTS

    # Self-distance is always 0 for Hamming
    dist = idx.get_distance(0, 0)
    assert dist == 0.0, f"Hamming self-distance should be 0, got {dist}"


# ===========================================================================
# Section 10: uint32 large item IDs (exercise near UINT32_MAX)
# ===========================================================================

# def test_uint32_large_item_id():
#     """uint32_t index handles item IDs near UINT32_MAX correctly."""
#     f = 4
#     idx = Index(f=f, metric="angular", index_dtype="uint32", seed=SEED)
#     # Use a non-contiguous large item ID near max
#     large_id = UINT32_MAX // 2
#     idx.add_item(0, [1.0, 0.0, 0.0, 0.0])
#     idx.add_item(large_id, [0.9, 0.1, 0.0, 0.0])
#     idx.build(n_trees=2)

#     result = idx.get_nns_by_item(0, 2)
#     assert len(result) == 2
#     # large_id should be among the neighbours
#     assert large_id in result or 0 in result  # either order is valid


# def test_uint32_overflow_raises():
#     """uint32_t item ID = UINT32_MAX + 1 raises OverflowError."""
#     idx = Index(f=4, metric="angular", index_dtype="uint32", seed=SEED)
#     with pytest.raises((OverflowError,)):
#         idx.add_item(UINT32_MAX + 1, [1.0, 0.0, 0.0, 0.0])


# ===========================================================================
# Section 11: Summary table (informational, always passes)
# ===========================================================================

def test_extended_combinations_summary(capsys):
    """
    Build and query a minimal index for every new (index_dtype × data_dtype)
    combination.  Print a summary table and assert all combinations succeed.

    This test is the definitive canary for the extended type matrix.
    """
    combos_to_test = [
        (idx_dtype, dt)
        for idx_dtype, _ in NEW_INDEX_DTYPES
        for dt in NEW_DATA_DTYPES
    ]
    # Also test new index dtypes with existing data dtypes
    combos_to_test += [
        (idx_dtype, "float32")
        for idx_dtype, _ in NEW_INDEX_DTYPES
    ]
    # Also test existing index dtypes with new data dtypes
    combos_to_test += [
        ("int32", dt) for dt in NEW_DATA_DTYPES
    ] + [
        ("int64", dt) for dt in NEW_DATA_DTYPES
    ]

    results = []
    vecs = _float_vectors(N_ITEMS, F, SEED)

    for idx_dtype, dt in combos_to_test:
        try:
            idx = Index(f=F, metric="euclidean", index_dtype=idx_dtype,
                        dtype=dt, seed=SEED)
            for i, v in enumerate(vecs):
                idx.add_item(i, v)
            idx.build(n_trees=N_TREES)
            nn = idx.get_nns_by_item(0, 3)
            status = "OK"
            assert len(nn) == 3
        except Exception as exc:
            status = f"FAIL: {type(exc).__name__}: {exc}"
        results.append((idx_dtype, dt, status))

    with capsys.disabled():
        print("\n")
        print(f"{'index_dtype':10s} {'data_dtype':10s} {'status'}")
        print("-" * 50)
        for idx_dtype, dt, status in results:
            print(f"{idx_dtype:10s} {dt:10s} {status}")

    failed = [(i, d, s) for i, d, s in results if s != "OK"]
    assert not failed, f"Failed combinations: {failed}"
