"""
Unbiased bounded draws and a locked spawn (round 33, Run 6: R6-1, R6-2).

Every draw here is seeded, so every statistic is a fixed number: a test
compares it with the chi-square critical value at significance 0.001.
"""

from __future__ import annotations

import pickle
import sys
import threading

import pytest

from scikitplot.random import (
    Kiss32Random,
    Kiss64Random,
    KissGenerator,
    KissSeedSequence,
)

#: Buckets of the uniformity test, and chi2.ppf(0.999, BUCKETS - 1).
BUCKETS = 16
CRITICAL = 37.69729821835383
DRAWS = 64_000
#: n where the plain modulo is far from uniform for a 32-bit draw.
N_BIASED_32 = 3_000_000_000


def _chi_square(values, n):
    counts = [0] * BUCKETS
    for value in values:
        counts[value * BUCKETS // n] += 1
    expected = len(values) / BUCKETS
    return sum((c - expected) ** 2 / expected for c in counts)


class TestIndex:
    # 2**31 + 1 and 2**63 + 1 are the worst rejection rates (about one draw in
    # two): they test termination, not bias, which the modulo barely has there.
    @pytest.mark.parametrize("n", [2**31 + 1, N_BIASED_32, 2**33, 7 * 2**40 + 3])
    def test_kiss32_is_uniform(self, n):
        rng = Kiss32Random(20261017)
        values = [rng.index(n) for _ in range(DRAWS)]
        assert all(0 <= v < n for v in values)
        assert _chi_square(values, n) < CRITICAL

    @pytest.mark.parametrize("n", [2**63 + 1, 3 * 2**62, 10**19])
    def test_kiss64_is_uniform(self, n):
        rng = Kiss64Random(20261017)
        values = [rng.index(n) for _ in range(DRAWS)]
        assert all(0 <= v < n for v in values)
        assert _chi_square(values, n) < CRITICAL

    @pytest.mark.parametrize(
        ("cls", "n"), [(Kiss32Random, N_BIASED_32), (Kiss64Random, 3 * 2**62)]
    )
    def test_the_plain_modulo_was_not(self, cls, n):
        # The statistic the tests rely on separates the two: the old rule
        # (kiss() % n) fails it at the same n and seed.
        rng = cls(20261017)
        values = [rng.kiss() % n for _ in range(DRAWS)]
        assert _chi_square(values, n) > CRITICAL

    def test_kiss32_reaches_past_2_32(self):
        rng = Kiss32Random(3)
        assert max(rng.index(2**33) for _ in range(20_000)) >= 2**32

    @pytest.mark.parametrize("cls", [Kiss32Random, Kiss64Random])
    @pytest.mark.parametrize("n", [1, 2, 7, 100, 1000, 65537])
    def test_small_n_keeps_the_seeded_stream(self, cls, n):
        # A draw is rejected with probability below n / 2**32: never here,
        # so the output equals the plain modulo of a twin generator.
        rng, twin = cls(99), cls(99)
        assert [rng.index(n) for _ in range(500)] == [
            twin.kiss() % n for _ in range(500)
        ]

    @pytest.mark.parametrize("cls", [Kiss32Random, Kiss64Random])
    def test_zero_returns_zero(self, cls):
        assert cls(1).index(0) == 0

    @pytest.mark.parametrize("cls", [Kiss32Random, Kiss64Random])
    @pytest.mark.parametrize(
        ("n", "error"),
        [
            (-1, ValueError),
            (1.5, TypeError),
            ("3", TypeError),
            (True, TypeError),
            (2**64, OverflowError),
        ],
    )
    def test_bad_bounds_raise_as_documented(self, cls, n, error):
        with pytest.raises(error):
            cls(1).index(n)

    def test_an_integer_like_bound_is_accepted(self):
        np = pytest.importorskip("numpy")
        assert 0 <= Kiss64Random(1).index(np.int64(10)) < 10  # noqa: PLR2004


class TestIntegers:
    def test_a_range_past_2_63_is_uniform(self):
        n = 3 * 2**62  # 2**64 mod n = 2**62: a quarter of the raw draws
        gen = KissGenerator(5)
        values = [int(gen.integers(0, n)) for _ in range(DRAWS)]
        assert _chi_square(values, n) < CRITICAL

    @pytest.mark.parametrize("kind", ["int64", "uint64", "int8"])
    def test_numpy_bounds_are_exact_ints(self, kind):
        np = pytest.importorskip("numpy")
        cast = getattr(np, kind)
        values = KissGenerator(1).integers(cast(0), cast(9), size=50)
        assert values.min() >= 0
        assert values.max() < 9  # noqa: PLR2004
        assert 0 <= KissGenerator(1).integers(cast(5)) < 5  # noqa: PLR2004

    @pytest.mark.parametrize("bounds", [(0.5, 10.2), (True, None), (0, True)])
    def test_non_integer_bounds_are_refused(self, bounds):
        with pytest.raises(TypeError, match="integer bounds"):
            KissGenerator(1).integers(*bounds)

    def test_a_range_wider_than_a_draw_is_refused(self):
        with pytest.raises(ValueError, match="exceeds"):
            KissGenerator(1).integers(0, 2**64 + 1)

    def test_small_ranges_keep_the_seeded_stream(self):
        gen, twin = KissGenerator(7), KissGenerator(7)
        values = gen.integers(-5, 5, size=200).tolist()
        raw = twin.bit_generator.random_raw
        assert values == [-5 + int(raw()) % 10 for _ in range(200)]


class TestSpawn:
    def test_threads_never_share_a_spawn_key(self):
        # A 1 us switch interval makes threads interleave inside spawn: the
        # unlocked code gave about 90 duplicate keys here; at the default
        # interval it gave none, and the test could not tell the two apart.
        interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            self._race()
        finally:
            sys.setswitchinterval(interval)

    def _race(self):
        parent = KissSeedSequence(42)
        keys, guard = [], threading.RLock()

        def work():
            mine = [child.spawn_key for _ in range(3_000) for child in parent.spawn(1)]
            with guard:
                keys.extend(mine)

        threads = [threading.Thread(target=work) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(keys) == len(set(keys)) == 8 * 3_000
        assert parent.n_children_spawned == 8 * 3_000

    def test_children_take_consecutive_keys(self):
        seq = KissSeedSequence(1, spawn_key=(9,))
        assert [c.spawn_key for c in seq.spawn(3)] == [(9, 0), (9, 1), (9, 2)]
        assert [c.spawn_key for c in seq.spawn(2)] == [(9, 3), (9, 4)]

    def test_a_restored_sequence_spawns(self):
        seq = KissSeedSequence(1)
        seq.spawn(2)
        copy = pickle.loads(pickle.dumps(seq))  # noqa: S301 - our own bytes
        assert copy.spawn(1)[0].spawn_key == (2,)
        bare = KissSeedSequence.__new__(KissSeedSequence)
        bare.__setstate__(seq.__getstate__())
        assert bare.spawn(1)[0].spawn_key == (2,)
        assert "_spawn_lock" not in seq.get_state()
