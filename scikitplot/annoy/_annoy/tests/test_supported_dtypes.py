# scikitplot/annoy/_annoy/tests/test_supported_dtypes.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the ``supported_dtypes()`` runtime capability API (CY-016).

The API promotes the C++ ``annoy_support`` registry to Python. These tests lock
its schema, its internal consistency (reported ``usable_as_dtype`` must match what
``Index(dtype=...)`` actually accepts), and the honest reporting of the extended
tiers (float80 available-but-not-a-dtype; float256/512 unavailable without a
multiprecision backend).
"""

import os
import tempfile
import pytest

from scikitplot.annoy._annoy import annoylib as A

CAPS = A.supported_dtypes()
LADDER = ["float16", "float32", "float64", "float80",
          "float128", "float256", "float512"]
FIELDS = {"size_bytes", "mantissa_bits", "tier", "available",
          "usable_as_dtype", "io_precision_capped", "note"}
TIERS = {"native", "runtime-dispatched", "emulated", "unavailable"}


def test_reports_the_full_ladder():
    assert set(CAPS) == set(LADDER)


@pytest.mark.parametrize("name", LADDER)
def test_entry_schema(name):
    entry = CAPS[name]
    assert FIELDS <= set(entry)
    assert entry["tier"] in TIERS
    assert isinstance(entry["size_bytes"], int)
    assert isinstance(entry["mantissa_bits"], int)
    assert isinstance(entry["usable_as_dtype"], bool)


def test_reported_usable_dtypes_match_index_acceptance():
    # the report must not claim a dtype is usable unless Index actually accepts it
    for name, entry in CAPS.items():
        if entry["usable_as_dtype"]:
            idx = A.Index(4, "euclidean", dtype=name)  # must not raise
            idx.add_item(0, [1.0] * 4)
            idx.build(2)
            assert idx.get_n_items() == 1


def test_precision_widths_are_monotone_and_correct():
    assert CAPS["float16"]["mantissa_bits"] == 11
    assert CAPS["float32"]["mantissa_bits"] == 24
    assert CAPS["float64"]["mantissa_bits"] == 53
    # float64 is the widened-bridge width; wider types don't beat it on I/O
    assert CAPS["float128"]["mantissa_bits"] >= CAPS["float64"]["mantissa_bits"]


def test_float80_now_usable_where_distinct_from_float128():
    f80 = CAPS["float80"]
    assert f80["available"] is True
    # float80 is a usable dtype on platforms where long double is distinct from
    # float128_t (native __float128). If usable, Index must actually accept it.
    if f80["usable_as_dtype"]:
        idx = A.Index(4, "euclidean", dtype="float80")
        idx.add_item(0, [1.0] * 4)
        idx.build(2)
        assert idx.get_n_items() == 1


def test_multiprecision_tiers_honest_when_backend_absent():
    for name in ("float256", "float512"):
        e = CAPS[name]
        if not e["available"]:
            # never silently aliased: unavailable => tier says so, zero width
            assert e["tier"] == "unavailable"
            assert e["size_bytes"] == 0
            assert e["usable_as_dtype"] is False


def test_bridge_capping_flagged_for_wide_types():
    # every type wider than the double bridge must flag io_precision_capped
    assert CAPS["float64"]["io_precision_capped"] is False
    assert CAPS["float128"]["io_precision_capped"] is True
    assert CAPS["float16"]["io_precision_capped"] is True


@pytest.mark.skipif(
    not CAPS["float80"]["usable_as_dtype"],
    reason="float80 not a distinct dtype on this platform (long double == float128)",
)
@pytest.mark.parametrize("metric", ["angular", "euclidean", "manhattan", "dot"])
def test_float80_dtype_end_to_end(metric):
    # float80 is a real usable dtype across all metrics: build + query + persist
    #
    # The vectors are chosen so that item 0 is its own nearest neighbour by a
    # margin under every metric: the directions are all different (angular),
    # the points are all different (euclidean, manhattan), and item 0 has the
    # largest inner product with itself (dot: 144 against 132 for item 1).
    #
    # The earlier data, ``[i, i, 0, 0]``, made item 0 the zero vector and every
    # other item a multiple of the same direction. Measured: under "angular"
    # all twelve distances from item 0 are 1.414214, its distance to itself
    # included, and under "dot" all twelve are 0. Which of twelve equal
    # candidates comes first is not defined, so ``[0] == 0`` held only where
    # the tie happened to break that way (it did not on macOS/arm64).
    idx = A.Index(4, metric, dtype="float80")
    for i in range(12):
        idx.add_item(i, [float(12 - i), float(i), 0.0, 0.0])
    idx.build(5)
    ids, distances = idx.get_nns_by_item(0, 3, include_distances=True)
    assert ids[0] == 0
    assert distances[0] != distances[1], "item 0 must win by a margin, not a tie"
    fn = os.path.join(tempfile.mkdtemp(), "f80.ann")
    idx.save(fn)
    r = A.Index(4, metric, dtype="float80")
    r.load(fn)
    assert r.get_item(5) == idx.get_item(5)
