# scikitplot/random/_kiss/tests/test_kiss_state_continuation.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for ANNOY-RNG-002 (guide 6.16).

`get_state`/`set_state` (and pickle) must capture the FULL generator state
(`x, y, z, c`), not just the seed. A seed-only restore restarts the stream; the
correct contract resumes it. This encodes the guide's oracle — draw N, serialize,
draw M from the original, restore, require the restored M to match — across
KISS32, KISS64, the bit generator, the generator, and the random-state wrapper.
"""
import pickle

import numpy as np
import pytest

from scikitplot.random._kiss import kiss_random as K


# (factory, draw-callable) for each wrapper
CASES = {
    "Kiss32Random": (lambda: K.Kiss32Random(12345), lambda g: g.kiss()),
    "Kiss64Random": (lambda: K.Kiss64Random(12345), lambda g: g.kiss()),
    "KissBitGenerator": (lambda: K.KissBitGenerator(12345), lambda g: g.random_raw()),
    "KissGenerator": (lambda: K.KissGenerator(12345), lambda g: float(g.random())),
    "KissRandomState": (lambda: K.KissRandomState(12345), lambda g: float(g.random())),
}


@pytest.mark.parametrize("name", list(CASES))
def test_get_set_state_resumes_stream(name):
    make, draw = CASES[name]
    g = make()
    for _ in range(20):          # advance N
        draw(g)
    state = g.get_state()
    expected = [draw(g) for _ in range(10)]   # original continuation (M)
    g2 = make()
    g2.set_state(state)
    actual = [draw(g2) for _ in range(10)]     # restored continuation (M)
    assert actual == expected


@pytest.mark.parametrize("name", list(CASES))
def test_pickle_resumes_stream(name):
    make, draw = CASES[name]
    g = make()
    for _ in range(20):
        draw(g)
    blob = pickle.dumps(g)
    expected = [draw(g) for _ in range(10)]
    g2 = pickle.loads(blob)
    actual = [draw(g2) for _ in range(10)]
    assert actual == expected


def test_low_level_state_contains_words():
    st = K.Kiss64Random(7).get_state()
    for k in ("x", "y", "z", "c"):
        assert k in st


def test_legacy_seed_only_state_still_loads():
    # A pre-fix seed-only state (no words) must load without error, falling back
    # to a re-seed rather than raising.
    g = K.Kiss64Random(7)
    for _ in range(5):
        g.kiss()
    st = g.get_state()
    legacy = {k: v for k, v in st.items() if k not in ("x", "y", "z", "c")}
    g2 = K.Kiss64Random(0)
    g2.set_state(legacy)          # must not raise
    # after a legacy restore the stream restarts from the seed (documented fallback)
    assert isinstance(g2.kiss(), int)
