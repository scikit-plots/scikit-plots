#!/usr/bin/env python3
"""
Comprehensive test suite for kissrandom module.

This test suite validates:
- Basic functionality of PyKiss32Random and PyKiss64Random
- Statistical properties of generated random numbers
- Edge cases and error handling
- Property-based seed management
- Reproducibility and determinism
- Performance benchmarks

Test Organization
-----------------
- test_kiss32_*: Tests for PyKiss32Random
- test_kiss64_*: Tests for PyKiss64Random
- test_statistical_*: Statistical tests
- test_benchmark_*: Performance benchmarks

Running Tests
-------------
Run all tests:
    pytest test_kissrandom.py -v

Run specific test:
    pytest test_kissrandom.py::test_kiss32_basic -v

Run with coverage:
    pytest test_kissrandom.py --cov=kissrandom --cov-report=html

Skip slow tests:
    pytest test_kissrandom.py -m "not slow"

Run only benchmarks:
    pytest test_kissrandom.py -m benchmark

Design Principles
-----------------
- Test each public method thoroughly
- Validate edge cases (n=0, overflow, etc.)
- Check statistical properties (within reason)
- Test reproducibility from same seeds
- Validate error handling
- Benchmark performance
- Use fixtures for common setup
- Clear test names and docstrings

References
----------
- pytest: https://docs.pytest.org/
- NumPy testing: https://numpy.org/doc/stable/reference/testing.html
"""

# %%
import sys
from typing import List, Callable

import pytest

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# Import the module to test
try:
    from scikitplot.cexternals._annoy._kissrandom import kissrandom
    from scikitplot.cexternals._annoy._kissrandom.kissrandom import PyKiss32Random, PyKiss64Random
except ImportError as e:
    pytest.skip(f"kissrandom module not built: {e}", allow_module_level=True)


# ===========================================================================
# Test Fixtures
# ===========================================================================

@pytest.fixture
def kiss32_default():
    """Fixture providing PyKiss32Random with default seed."""
    return PyKiss32Random()


@pytest.fixture
def kiss32_fixed():
    """Fixture providing PyKiss32Random with fixed seed for reproducibility."""
    return PyKiss32Random(42)


@pytest.fixture
def kiss64_default():
    """Fixture providing PyKiss64Random with default seed."""
    return PyKiss64Random()


@pytest.fixture
def kiss64_fixed():
    """Fixture providing PyKiss64Random with fixed seed for reproducibility."""
    return PyKiss64Random(42)


# ===========================================================================
# Module-Level Tests
# ===========================================================================

# def test_module_version():
#     """Test that module has version attribute."""
#     assert hasattr(kissrandom, '__version__')
#     assert isinstance(kissrandom.__version__, str)
#     assert len(kissrandom.__version__) > 0


def test_module_exports():
    """Test that module exports expected classes."""
    assert hasattr(kissrandom, 'PyKiss32Random')
    assert hasattr(kissrandom, 'PyKiss64Random')


# ===========================================================================
# PyKiss32Random - Basic Functionality Tests
# ===========================================================================

class TestPyKiss32RandomBasic:
    """Basic functionality tests for PyKiss32Random."""

    def test_initialization_default(self):
        """Test initialization with default seed."""
        rng = PyKiss32Random()
        assert rng is not None
        assert rng.seed == PyKiss32Random.get_default_seed()

    def test_initialization_custom_seed(self):
        """Test initialization with custom seed."""
        seed = 12345
        rng = PyKiss32Random(seed)
        assert rng.seed == seed

    def test_initialization_zero_seed(self):
        """Test that seed=0 is normalized to default_seed."""
        rng = PyKiss32Random(0)
        assert rng.seed == PyKiss32Random.get_default_seed()

    def test_initialization_none_seed(self):
        """Test initialization with None (should use default)."""
        rng = PyKiss32Random(None)
        assert rng.seed == PyKiss32Random.get_default_seed()

    def test_initialization_invalid_seed_negative(self):
        """Test that negative seed raises ValueError."""
        with pytest.raises(ValueError, match="seed must be in"):
            PyKiss32Random(-1)

    def test_initialization_invalid_seed_overflow(self):
        """Test that seed > 2^32-1 raises ValueError."""
        with pytest.raises(ValueError, match="seed must be in"):
            PyKiss32Random(2**32)

    def test_default_seed_constant(self):
        """Test that default_seed is correct constant."""
        assert PyKiss32Random.get_default_seed() == 123456789

    def test_get_default_seed(self):
        """Test get_default_seed static method."""
        assert PyKiss32Random.get_default_seed() == 123456789

    def test_normalize_seed_nonzero(self):
        """Test normalize_seed with non-zero value."""
        assert PyKiss32Random.normalize_seed(42) == 42

    def test_normalize_seed_zero(self):
        """Test normalize_seed with zero."""
        assert PyKiss32Random.normalize_seed(0) == PyKiss32Random.get_default_seed()

    def test_repr(self):
        """Test string representation."""
        rng = PyKiss32Random(42)
        repr_str = repr(rng)
        assert "PyKiss32Random" in repr_str
        assert "42" in repr_str

    def test_str(self):
        """Test string conversion."""
        rng = PyKiss32Random(42)
        str_repr = str(rng)
        assert "PyKiss32Random" in str_repr


class TestPyKiss32RandomSeedProperty:
    """Tests for seed property getter/setter."""

    def test_seed_getter(self):
        """Test seed property getter."""
        rng = PyKiss32Random(42)
        assert rng.seed == 42

    def test_seed_setter(self):
        """Test seed property setter."""
        rng = PyKiss32Random(42)
        rng.seed = 123
        assert rng.seed == 123

    def test_seed_setter_resets_state(self):
        """Test that setting seed resets internal state."""
        rng = PyKiss32Random(42)
        val1 = rng.kiss()

        rng.seed = 42  # Reset to same seed
        val2 = rng.kiss()

        assert val1 == val2  # Should produce same first value

    def test_seed_setter_invalid_negative(self):
        """Test seed setter with negative value."""
        rng = PyKiss32Random()
        with pytest.raises(ValueError, match="seed must be in"):
            rng.seed = -1

    def test_seed_setter_invalid_overflow(self):
        """Test seed setter with overflow value."""
        rng = PyKiss32Random()
        with pytest.raises(ValueError, match="seed must be in"):
            rng.seed = 2**32


class TestPyKiss32RandomMethods:
    """Tests for PyKiss32Random core methods."""

    def test_kiss_returns_uint32(self, kiss32_fixed):
        """Test that kiss() returns valid uint32."""
        val = kiss32_fixed.kiss()
        assert isinstance(val, int)
        assert 0 <= val < 2**32

    def test_kiss_produces_different_values(self, kiss32_fixed):
        """Test that consecutive kiss() calls produce different values."""
        values = [kiss32_fixed.kiss() for _ in range(100)]
        unique_values = len(set(values))
        # Should have at least 95% unique values (probabilistic test)
        assert unique_values >= 95

    def test_flip_returns_binary(self, kiss32_fixed):
        """Test that flip() returns 0 or 1."""
        for _ in range(100):
            val = kiss32_fixed.flip()
            assert val in (0, 1)

    def test_flip_distribution(self, kiss32_fixed):
        """Test that flip() has roughly 50/50 distribution."""
        n_trials = 10000
        ones = sum(kiss32_fixed.flip() for _ in range(n_trials))
        ratio = ones / n_trials
        # Should be close to 0.5 (within 3 standard deviations)
        # std = sqrt(0.5 * 0.5 / n_trials) ≈ 0.005
        assert 0.46 <= ratio <= 0.54

    def test_index_returns_valid_range(self, kiss32_fixed):
        """Test that index(n) returns values in [0, n-1]."""
        n = 100
        for _ in range(1000):
            idx = kiss32_fixed.index(n)
            assert 0 <= idx < n

    def test_index_zero_returns_zero(self, kiss32_fixed):
        """Test that index(0) returns 0."""
        assert kiss32_fixed.index(0) == 0

    def test_index_one_returns_zero(self, kiss32_fixed):
        """Test that index(1) always returns 0."""
        for _ in range(10):
            assert kiss32_fixed.index(1) == 0

    def test_index_coverage(self, kiss32_fixed):
        """Test that index(n) covers all values in range."""
        n = 10
        values = set()
        for _ in range(1000):
            values.add(kiss32_fixed.index(n))
        # Should cover all values (probabilistic, may rarely fail)
        assert len(values) == n


class TestPyKiss32RandomReproducibility:
    """Tests for reproducibility and determinism."""

    def test_same_seed_same_sequence(self):
        """Test that same seed produces identical sequences."""
        seed = 42
        rng1 = PyKiss32Random(seed)
        rng2 = PyKiss32Random(seed)

        seq1 = [rng1.kiss() for _ in range(100)]
        seq2 = [rng2.kiss() for _ in range(100)]

        assert seq1 == seq2

    def test_different_seeds_different_sequences(self):
        """Test that different seeds produce different sequences."""
        rng1 = PyKiss32Random(42)
        rng2 = PyKiss32Random(43)

        seq1 = [rng1.kiss() for _ in range(100)]
        seq2 = [rng2.kiss() for _ in range(100)]

        # Sequences should be mostly different
        differences = sum(a != b for a, b in zip(seq1, seq2))
        assert differences >= 95

    def test_reset_reproduces_sequence(self):
        """Test that reset() reproduces sequence from same seed."""
        rng = PyKiss32Random(42)
        seq1 = [rng.kiss() for _ in range(10)]

        rng.reset(42)
        seq2 = [rng.kiss() for _ in range(10)]

        assert seq1 == seq2

    def test_set_seed_reproduces_sequence(self):
        """Test that set_seed() reproduces sequence."""
        rng = PyKiss32Random(42)
        seq1 = [rng.kiss() for _ in range(10)]

        rng.set_seed(42)
        seq2 = [rng.kiss() for _ in range(10)]

        assert seq1 == seq2

    def test_reset_default_uses_default_seed(self):
        """Test that reset_default() uses default seed."""
        rng = PyKiss32Random(999)
        rng.reset_default()

        rng_default = PyKiss32Random()

        # Should produce same sequence as default seed
        assert rng.kiss() == rng_default.kiss()


# ===========================================================================
# PyKiss64Random - Basic Functionality Tests
# ===========================================================================

class TestPyKiss64RandomBasic:
    """Basic functionality tests for PyKiss64Random."""

    def test_initialization_default(self):
        """Test initialization with default seed."""
        rng = PyKiss64Random()
        assert rng is not None
        assert rng.seed == PyKiss64Random.get_default_seed()

    def test_initialization_custom_seed(self):
        """Test initialization with custom seed."""
        seed = 12345678901234567890
        rng = PyKiss64Random(seed)
        assert rng.seed == seed

    def test_initialization_zero_seed(self):
        """Test that seed=0 is normalized to default_seed."""
        rng = PyKiss64Random(0)
        assert rng.seed == PyKiss64Random.get_default_seed()

    def test_default_seed_constant(self):
        """Test that default_seed is correct constant."""
        assert PyKiss64Random.get_default_seed() == 1234567890987654321

    def test_get_default_seed(self):
        """Test get_default_seed static method."""
        assert PyKiss64Random.get_default_seed() == 1234567890987654321

    def test_normalize_seed_nonzero(self):
        """Test normalize_seed with non-zero value."""
        assert PyKiss64Random.normalize_seed(999) == 999

    def test_normalize_seed_zero(self):
        """Test normalize_seed with zero."""
        assert PyKiss64Random.normalize_seed(0) == PyKiss64Random.get_default_seed()


class TestPyKiss64RandomMethods:
    """Tests for PyKiss64Random core methods."""

    def test_kiss_returns_uint64(self, kiss64_fixed):
        """Test that kiss() returns valid uint64."""
        val = kiss64_fixed.kiss()
        assert isinstance(val, int)
        assert 0 <= val < 2**64

    def test_flip_returns_binary(self, kiss64_fixed):
        """Test that flip() returns 0 or 1."""
        for _ in range(100):
            val = kiss64_fixed.flip()
            assert val in (0, 1)

    def test_index_returns_valid_range(self, kiss64_fixed):
        """Test that index(n) returns values in [0, n-1]."""
        n = 1000000
        for _ in range(1000):
            idx = kiss64_fixed.index(n)
            assert 0 <= idx < n

    def test_index_large_range(self, kiss64_fixed):
        """Test index with very large range."""
        n = 10**15  # 1 quadrillion
        idx = kiss64_fixed.index(n)
        assert 0 <= idx < n


class TestPyKiss64RandomReproducibility:
    """Tests for PyKiss64Random reproducibility."""

    def test_same_seed_same_sequence(self):
        """Test that same seed produces identical sequences."""
        seed = 12345678901234567890
        rng1 = PyKiss64Random(seed)
        rng2 = PyKiss64Random(seed)

        seq1 = [rng1.kiss() for _ in range(100)]
        seq2 = [rng2.kiss() for _ in range(100)]

        assert seq1 == seq2

    def test_seed_property_reset(self):
        """Test seed property reset functionality."""
        rng = PyKiss64Random(42)
        seq1 = [rng.kiss() for _ in range(10)]

        rng.seed = 42
        seq2 = [rng.kiss() for _ in range(10)]

        assert seq1 == seq2


# ===========================================================================
# Statistical Tests
# ===========================================================================

@pytest.mark.slow
@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required for statistical tests")
class TestStatisticalProperties:
    """Statistical tests for random number quality."""

    def test_uniform_distribution_kiss32(self):
        """Test that PyKiss32Random produces uniform distribution."""
        rng = PyKiss32Random(42)
        n_samples = 100000
        n_bins = 100

        # Generate samples and normalize to [0, 1]
        samples = np.array([rng.kiss() for _ in range(n_samples)]) / (2**32 - 1)

        # Chi-square test for uniformity
        hist, _ = np.histogram(samples, bins=n_bins, range=(0, 1))
        expected = n_samples / n_bins
        chi_square = np.sum((hist - expected)**2 / expected)

        # Critical value for chi-square with 99 df at p=0.01 is ~135
        assert chi_square < 135

    def test_mean_kiss32(self):
        """Test that mean of PyKiss32Random is approximately 2^31."""
        rng = PyKiss32Random(42)
        n_samples = 10000

        samples = [rng.kiss() for _ in range(n_samples)]
        mean = sum(samples) / n_samples
        expected_mean = (2**32 - 1) / 2

        # Mean should be within 1% of expected
        assert abs(mean - expected_mean) / expected_mean < 0.01

    def test_flip_probability(self):
        """Test that flip() has p=0.5."""
        rng = PyKiss32Random(123)
        n_trials = 10000

        ones = sum(rng.flip() for _ in range(n_trials))
        ratio = ones / n_trials

        # Within 3 sigma (99.7% confidence)
        std = np.sqrt(0.5 * 0.5 / n_trials)
        assert abs(ratio - 0.5) < 3 * std


# ===========================================================================
# Performance Benchmarks
# ===========================================================================

# @pytest.mark.benchmark
# class TestBenchmarks:
#     """Performance benchmarks."""

#     def test_benchmark_kiss32_generation(self, benchmark):
#         """Benchmark PyKiss32Random.kiss() performance."""
#         rng = PyKiss32Random(42)
#         benchmark(rng.kiss)

#     def test_benchmark_kiss64_generation(self, benchmark):
#         """Benchmark PyKiss64Random.kiss() performance."""
#         rng = PyKiss64Random(42)
#         benchmark(rng.kiss)

#     def test_benchmark_kiss32_flip(self, benchmark):
#         """Benchmark PyKiss32Random.flip() performance."""
#         rng = PyKiss32Random(42)
#         benchmark(rng.flip)

#     def test_benchmark_kiss32_index(self, benchmark):
#         """Benchmark PyKiss32Random.index() performance."""
#         rng = PyKiss32Random(42)
#         benchmark(lambda: rng.index(100))


# ===========================================================================
# Integration Tests
# ===========================================================================

@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required")
class TestIntegration:
    """Integration tests with NumPy and real-world use cases."""

    def test_array_indexing(self):
        """Test random array indexing."""
        rng = PyKiss32Random(42)
        arr = np.arange(1000)

        # Random access should not raise
        for _ in range(100):
            idx = rng.index(len(arr))
            _ = arr[idx]

    def test_fisher_yates_shuffle(self):
        """Test Fisher-Yates shuffle implementation."""
        rng = PyKiss32Random(123)
        arr = np.arange(10)
        original = arr.copy()

        # Fisher-Yates shuffle
        for i in range(len(arr) - 1, 0, -1):
            j = rng.index(i + 1)
            arr[i], arr[j] = arr[j], arr[i]

        # Shuffled array should have same elements
        assert set(arr) == set(original)
        # But different order (probabilistic, may rarely fail)
        assert not np.array_equal(arr, original)

    def test_monte_carlo_pi_estimation(self):
        """Test Monte Carlo estimation of π."""
        rng = PyKiss32Random(42)
        n_samples = 100000
        inside = 0

        for _ in range(n_samples):
            x = rng.kiss() / (2**32 - 1)
            y = rng.kiss() / (2**32 - 1)
            if x*x + y*y <= 1.0:
                inside += 1

        pi_estimate = 4 * inside / n_samples
        # Should be within 1% of π
        assert abs(pi_estimate - np.pi) / np.pi < 0.01


# ===========================================================================
# Edge Cases and Error Handling
# ===========================================================================

class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_kiss32_max_seed(self):
        """Test PyKiss32Random with maximum seed value."""
        rng = PyKiss32Random(2**32 - 1)
        val = rng.kiss()  # Should not crash
        assert 0 <= val < 2**32

    def test_kiss64_max_seed(self):
        """Test PyKiss64Random with maximum seed value."""
        rng = PyKiss64Random(2**64 - 1)
        val = rng.kiss()  # Should not crash
        assert 0 <= val < 2**64

    def test_index_boundary_values(self):
        """Test index() with boundary values."""
        rng = PyKiss32Random(42)

        # n = 0
        assert rng.index(0) == 0

        # n = 1
        assert rng.index(1) == 0

        # n = 2 (should return 0 or 1)
        val = rng.index(2)
        assert val in (0, 1)

    def test_multiple_instances_independent(self):
        """Test that multiple RNG instances are independent."""
        rng1 = PyKiss32Random(42)
        rng2 = PyKiss32Random(43)

        # Should produce different sequences
        val1 = [rng1.kiss() for _ in range(10)]
        val2 = [rng2.kiss() for _ in range(10)]

        assert val1 != val2

# %%

if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])

# %%
