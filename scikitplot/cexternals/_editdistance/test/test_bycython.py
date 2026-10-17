# scikitplot/cexternals/_editdistance/test/test_bycython.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Contract tests for :mod:`scikitplot.cexternals._editdistance.bycython`.

Notes
-----
**Developer.** Three defects survived the original suite, which never used
threshold 0, never compared values with colliding hashes, and never made
``hash`` raise:

* ``eval_criterion`` read the wrong DP row in its early exit, so exact
  matches were rejected at threshold 0;
* elements were compared by ``hash()``, so ``[-1]`` equalled ``[-2]``;
* a raising ``hash`` leaked the ``malloc``-ed buffer.

The differential tests hold every kernel to the plain DP reference and to
an independent pure-Python implementation.
"""

from __future__ import annotations

import ctypes
import random
import sys

import pytest

from ..bycython import eval, eval_criterion, eval_dp


def _reference(a, b):
    """Wagner-Fischer over container equality (identity, then ``==``)."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            same = x is y or x == y
            current.append(
                min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (not same))
            )
        previous = current
    return previous[-1]


# -- threshold criterion -----------------------------------------------------


@pytest.mark.parametrize("length", range(9))
def test_an_exact_match_is_within_threshold_zero(length):
    """Odd and even lengths alike: the bug was confined to odd DP rows."""
    text = "ab" * length
    assert eval_criterion(text, text, 0) is True


@pytest.mark.parametrize("alphabet", ["ab", "abc", "abcdefgh"])
def test_criterion_agrees_with_the_distance_for_every_threshold(alphabet):
    rng = random.Random(20261012)  # noqa: S311 - seeded, reproducible
    for _ in range(1500):
        a = "".join(rng.choices(alphabet, k=rng.randint(0, 9)))
        b = "".join(rng.choices(alphabet, k=rng.randint(0, 9)))
        threshold = rng.randint(0, 6)
        expected = eval_dp(a, b) <= threshold
        assert eval_criterion(a, b, threshold) is expected, (a, b, threshold)


def test_criterion_boundaries():
    assert eval_criterion("kitten", "sitting", 3) is True
    assert eval_criterion("kitten", "sitting", 2) is False
    assert eval_criterion("", "abc", 3) is True
    assert eval_criterion("", "abc", 2) is False
    assert eval_criterion("abc", "", 3) is True


def test_a_negative_threshold_is_refused():
    with pytest.raises(OverflowError):
        eval_criterion("a", "b", -1)


# -- exact element equality --------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([-1], [-2]),  # hash(-1) == hash(-2) == -2
        ([0], [2**61 - 1]),  # hash(2**61 - 1) == 0
        ([(-1,)], [(-2,)]),  # tuple hashes inherit the collision
    ],
)
def test_equal_hashes_do_not_make_elements_equal(a, b):
    assert hash(a[0]) == hash(b[0])  # the premise: CPython collides these
    assert eval(a, b) == 1
    assert eval_dp(a, b) == 1
    assert eval_criterion(a, b, 0) is False


def test_container_equality_is_the_rule():
    nan = float("nan")
    assert eval([1], [1.0]) == 0  # 1 == 1.0
    assert eval([nan], [nan]) == 0  # the same object is equal to itself
    assert eval([float("nan")], [float("nan")]) == 1  # distinct NaNs are not
    assert eval("ab", ["a", "b"]) == 0  # a str and its characters
    assert eval(b"ab", [97, 98]) == 0  # bytes iterate as ints


@pytest.mark.parametrize("text", ["\U0001f600a", "été", "中文"])
def test_strings_compare_by_code_point(text):
    assert eval(text, text) == 0
    assert eval(text, text[:-1] + "#") == 1


def test_all_kernels_match_an_independent_reference():
    rng = random.Random(7)  # noqa: S311 - seeded, reproducible
    pool = [-2, -1, 0, 1, 2**61 - 1, "x", (1, 2), None, 1.5]
    for _ in range(400):
        a = rng.choices(pool, k=rng.randint(0, 12))
        b = rng.choices(pool, k=rng.randint(0, 12))
        expected = _reference(a, b)
        assert eval(a, b) == expected
        assert eval_dp(a, b) == expected


def test_long_inputs_take_the_dp_path_and_agree():
    rng = random.Random(3)  # noqa: S311 - seeded, reproducible
    a = rng.choices(range(5), k=900)  # beyond the 640-element bit-parallel limit
    b = rng.choices(range(5), k=700)
    assert eval(a, b) == eval_dp(a, b)


# -- error paths -------------------------------------------------------------


def test_an_unhashable_element_raises_type_error():
    with pytest.raises(TypeError):
        eval([[1]], [[2]])
    with pytest.raises(TypeError):
        eval_criterion(["a"], [{}], 1)


def _heap_in_use():
    """Bytes the C allocator currently holds, or ``None`` off glibc."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        libc = ctypes.CDLL("libc.so.6")
        mallinfo2 = libc.mallinfo2
    except (OSError, AttributeError):
        return None

    class _MallInfo2(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_size_t)
            for name in (
                "arena",
                "ordblks",
                "smblks",
                "hblks",
                "hblkhd",
                "usmblks",
                "fsmblks",
                "uordblks",
                "fordblks",
                "keepcost",
            )
        ]

    mallinfo2.restype = _MallInfo2
    info = mallinfo2()
    return info.uordblks + info.hblkhd


#: The leak was 16,000 bytes per call (16 MB over the loop); allocator noise
#: is far below this bound.
_LEAK_TOLERANCE_BYTES = 1_000_000


def test_a_raising_hash_does_not_leak():
    """Each failing call used to leak ``len(a) * 8`` bytes (16 KB here)."""
    if _heap_in_use() is None:
        pytest.skip("heap accounting needs glibc's mallinfo2")
    unhashable = [[i] for i in range(2000)]
    for _ in range(50):  # warm up allocator pools
        with pytest.raises(TypeError):
            eval(unhashable, unhashable)
    before = _heap_in_use()
    for _ in range(1000):
        with pytest.raises(TypeError):
            eval(unhashable, unhashable)
    grown = _heap_in_use() - before
    assert grown < _LEAK_TOLERANCE_BYTES, (
        f"heap grew {grown} bytes over 1000 failing calls"
    )
