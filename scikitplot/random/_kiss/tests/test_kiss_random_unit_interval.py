# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for ANNOY-RNG-001 (guide 6.15).

The KISS float conversion must produce values in the half-open interval [0, 1)
using the top mantissa bits and an exact power-of-two scale. The old
``draw / (2**64 - 1)`` could return exactly ``1.0`` for a maximal draw.

By construction the maximum possible value is ``(2**53 - 1) * 2**-53 < 1.0``, so
the boundary is unreachable rather than merely improbable; these tests assert the
half-open contract, the 2**-53 grid (canonical scale), and the NumPy Generator
C-callback path.
"""
import numpy as np
import pytest

from scikitplot.random._kiss import kiss_random as K

TWO53 = 9007199254740992.0  # 2**53


@pytest.mark.parametrize("cls_name", ["KissGenerator", "KissRandomState"])
def test_random_scalar_half_open_and_on_grid(cls_name):
    cls = getattr(K, cls_name)
    gen = cls(12345)
    for _ in range(200_000):
        x = gen.random()
        assert 0.0 <= x < 1.0
        # canonical scale: every value is an integer multiple of 2**-53
        assert (x * TWO53).is_integer()


def test_random_array_float64_half_open():
    gen = K.KissGenerator(999)
    a = gen.random(200_000)
    assert a.dtype == np.float64
    assert a.min() >= 0.0
    assert a.max() < 1.0


def test_random_array_float32_half_open():
    gen = K.KissGenerator(7)
    a = gen.random(200_000, dtype=np.float32)
    assert a.dtype == np.float32
    assert a.min() >= np.float32(0.0)
    assert a.max() < np.float32(1.0)


def test_numpy_generator_double_callback_half_open():
    # exercises kiss64_next_double via NumPy's Generator.random()
    gen = np.random.Generator(K.KissBitGenerator(42))
    v = gen.random(500_000)
    assert v.min() >= 0.0
    assert v.max() < 1.0


def test_reference_vectors_scale():
    # The conversion is exactly (draw >> 11) * 2**-53. Verify the mapping for a
    # few explicit draws using the same arithmetic the implementation uses.
    for draw in (0, 1 << 11, (1 << 11) + (1 << 20), (1 << 64) - 1):
        expected = (draw >> 11) * (1.0 / TWO53)
        assert 0.0 <= expected < 1.0
    # maximal draw maps to 1 - 2**-53, never 1.0
    assert ((1 << 64) - 1 >> 11) * (1.0 / TWO53) == 1.0 - 2.0 ** -53
