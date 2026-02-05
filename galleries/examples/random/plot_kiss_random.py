# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Enhanced KISS Random Generator - Complete Usage Examples
=========================================================

.. currentmodule:: scikitplot.random

Demonstrates all new features:
1. Auto-detection (Kiss64Random with None, 32, or 64)
2. Context manager support (with statement)
3. NumPy-like BitGenerator and Generator
4. State serialization/deserialization
5. Thread-safe operations
"""

# %%

import time
import pickle
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
np.random.BitGenerator
from scipy import stats

# Import enhanced module
# from scikitplot.cexternals._annoy._kissrandom.kissrandom import (
from scikitplot.random import (
    Kiss32Random,         # Explicit 32-bit
    Kiss64Random,         # Explicit 64-bit
    KissRandom,
    KissSeedSequence,
    KissBitGenerator,       # NumPy-compatible BitGenerator
    KissGenerator,          # High-level Generator
    KissRandomState,
    default_rng,            # Convenience function
    kiss_context,           # Context manager helper
)

print("=" * 70)
print("ENHANCED KISS RANDOM - NEW FEATURES DEMONSTRATION")
print("=" * 70)

# %%

# Use time.perf_counter() or timeit
import time
import numpy as np

seed = 42
n = 1_000_000

# times = []
# np.random.default_rng?
# rng1 = np.random.default_rng(seed)
# rng2 = default_rng(seed)
# start = time.perf_counter()
# seq1 = rng1.random(5)
# end = time.perf_counter()
# times.append(end - start)
# start = time.perf_counter()
# seq2 = rng2.random(5)
# end = time.perf_counter()
# times.append(end - start)
# seq1, seq2, times

def benchmark_rng(
    rng_factory,
    method_name: str = "random",
    seed: int = 42,
    repeat: int = 5,
    # size: int = 1_000_000,
    **kwargs,
):
    """
    Benchmark an RNG method in a deterministic, isolated manner.

    Parameters
    ----------
    rng_factory : callable
        Function that returns a new RNG instance when called with a seed.
    method_name : str
        RNG method to benchmark (e.g., "random").
    seed : int
        Seed for reproducibility.
    repeat : int
        Number of benchmark repetitions.
    **kwargs : dict

    Returns
    -------
    dict
        Timing statistics and output sanity checks.
    """
    times = []
    outputs = []

    for _ in range(repeat):
        rng = rng_factory(seed)

        method = getattr(rng, method_name)

        start = time.perf_counter()
        out = method(**kwargs)
        end = time.perf_counter()

        times.append(end - start)
        outputs.append(out)

    times = np.asarray(times)
    outputs = np.asarray(outputs)

    return {
        "n": n,
        "repeat": repeat,
        "time_min": float(times.min()),
        "time_mean": float(times.mean()),
        "time_max": float(times.max()),
        "time_std": float(times.std(ddof=1)),
        "output_mean": float(outputs.mean()),
        "output_std": float(outputs.std(ddof=1)),
    }

from functools import partial
from pprint import pprint

for method in ["random", "uniform"]:

    print("Numpy")
    numpy_result = benchmark_rng(
        rng_factory=np.random.default_rng,
        method_name=method,
        seed=seed,
        size = 1_000_000,
    )
    pprint(numpy_result, sort_dicts=False)

    print("Kiss Customized 64")
    custom_result = benchmark_rng(
        # rng_factory=partial(default_rng, bit_width=32),
        rng_factory=default_rng,
        method_name=method,
        seed=seed,
        size = 1_000_000,
    )
    pprint(custom_result, sort_dicts=False)

# %%

# ===========================================================================
# Feature 1: Auto-Detection
# ===========================================================================

print("\n" + "=" * 70)
print("1. AUTO-DETECTION (Kiss64Random factory)")
print("=" * 70)

# Explicit 32-bit
rng_32 = Kiss32Random(seed=42)
print(f"64-bit RNG: {rng_32}")
print(f"Type: {type(rng_32).__name__}")

# Explicit 64-bit
rng_64 = Kiss64Random(seed=42)
print(f"64-bit RNG: {rng_64}")
print(f"Type: {type(rng_64).__name__}")

# Generate some values
print(f"\nGenerated values (auto):")
print(f"  kiss():  {rng_64.kiss()}")
print(f"  flip():  {rng_64.flip()}")
print(f"  index(100): {rng_64.index(100)}")

# %%

# ===========================================================================
# Feature 2: Context Manager Support
# ===========================================================================

print("\n" + "=" * 70)
print("2. CONTEXT MANAGER SUPPORT (with statement)")
print("=" * 70)

# Basic context manager usage
print("Basic usage:")
with Kiss64Random(42) as rng:
    values = [rng.kiss() for _ in range(5)]
    print(f"  Generated 5 values: {values[:3]}...")

# Nested context managers (independent RNGs)
print("\nNested contexts (independent streams):")
with Kiss64Random(100) as rng1:
    with Kiss64Random(200) as rng2:
        val1 = rng1.kiss()
        val2 = rng2.kiss()
        print(f"  RNG1 (seed=100): {val1}")
        print(f"  RNG2 (seed=200): {val2}")
        print(f"  Different: {val1 != val2}")

# Context manager ensures thread safety
print("\nThread-safe access with context manager:")
rng = Kiss64Random(42)

def worker(thread_id):
    """Worker function using shared RNG."""
    with rng:  # Acquires lock
        return [rng.kiss() for _ in range(3)]

with ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(worker, range(2)))
    print(f"  Thread 0: {results[0]}")
    print(f"  Thread 1: {results[1]}")

# %%

# ===========================================================================
# Feature 3: NumPy-Compatible BitGenerator
# ===========================================================================

print("\n" + "=" * 70)
print("3. NUMPY-LIKE BITGENERATOR")
print("=" * 70)

# Create BitGenerator
bg = KissBitGenerator(seed=42)
print(f"BitGenerator: {bg}")
print(f"Has lock: {hasattr(bg, 'lock')}")

# Generate raw bits
raw_value = bg.random_raw()
print(f"\nRaw uint64: {raw_value}")

# Generate array of raw bits
raw_array = bg.random_raw(size=5)
print(f"Raw array: {raw_array}")

# Use with NumPy Generator (if available)
try:
    # Error not compat numpy C-Api
    # from numpy.random import Generator

    bg_numpy = KissBitGenerator(seed=123)
    gen = KissGenerator(bg_numpy)

    print("\n✅ Generator Integration:")
    print(f"  Random floats: {gen.random(5)}")
    print(f"  Random ints:   {gen.integers(0, 100, size=5)}")

    # This gives access to ALL random methods!
    print(f"  Normal:   {gen.normal(size=5)}")
    print(f"  Uniform:  {gen.uniform(size=5)}")

except ImportError:
    print("\n⚠️  Do not use NumPy Generator not available due to c-api inconsistency!")

# %%

# ===========================================================================
# Feature 4: High-Level KissGenerator
# ===========================================================================

print("\n" + "=" * 70)
print("4. HIGH-LEVEL KissGenerator")
print("=" * 70)

# Create generator (multiple ways)
gen1 = KissGenerator()  # From seed
print(f"Generator from seed: {gen1}")

bg = KissBitGenerator(seed=123)
gen2 = KissGenerator(bg)  # From BitGenerator
print(f"Generator from BitGenerator: {gen2}")

# Use all the methods
print("\nGenerator methods:")

# Random floats
floats = gen1.random(5)
print(f"  random(5):         {floats}")

# Random integers
ints = gen1.integers(0, 100, size=5)
print(f"  integers(0, 100):  {ints}")

# Normal distribution
normal = gen1.normal(0, 1, size=5)
print(f"  normal(0, 1):      {normal}")

# Uniform distribution
uniform = gen1.uniform(10, 20, size=5)
print(f"  uniform(10, 20):   {uniform}")

# Choice
choices = gen1.choice(['A', 'B', 'C'], size=10)
print(f"  choice(['A','B','C']): {choices}")

# Weighted choice
choices_weighted = gen1.choice(['A', 'B', 'C'], size=100, p=[0.5, 0.3, 0.2])
print(f"  Weighted choice A: {list(choices_weighted).count('A')}% (✅ expected ~50%)")

# Shuffle
arr = np.arange(10)
gen1.shuffle(arr)
print(f"  shuffle([0..9]):   {arr}")

# %%

# ===========================================================================
# Feature 5: default_rng() Convenience Function
# ===========================================================================

print("\n" + "=" * 70)
print("5. default_rng() CONVENIENCE FUNCTION")
print("=" * 70)

# Recommended way to create RNG (like numpy.random.default_rng)
rng = default_rng(seed=42)
print(f"Default RNG: {rng}")

# Use it just like NumPy
print("\nUsage:")
print(f"  Random floats: {rng.random(5)}")
print(f"  Random ints:   {rng.integers(0, 10, size=5)}")
print(f"  Normal:        {rng.normal(0, 1, size=5)}")

# %%

# ===========================================================================
# Feature 6: Context Manager Helper (kiss_context)
# ===========================================================================

print("\n" + "=" * 70)
print("6. CONTEXT MANAGER HELPER (kiss_context)")
print("=" * 70)

# Convenient temporary RNG
with kiss_context(seed=999) as rng:
    values = rng.random(5)
    print(f"Generated values: {values}")
    print(f"RNG type: {type(rng)}")

print("RNG automatically cleaned up after context")

# %%

# ===========================================================================
# Feature 7: State Serialization/Deserialization
# ===========================================================================

print("\n" + "=" * 70)
print("7. STATE SERIALIZATION/DESERIALIZATION")
print("=" * 70)

# Create RNG and generate some values
rng1 = KissBitGenerator(seed=42)
sequence1 = [rng1.random_raw() for _ in range(5)]
print(f"Original sequence: {sequence1}")

# Save state
state = rng1.state
print(f"\nSaved state: {state}")

# Continue generating
more_values = [rng1.random_raw() for _ in range(3)]
print(f"Continued:  {more_values}")

# Restore from state
rng2 = KissBitGenerator()
rng2.__setstate__(state)
sequence2 = [rng2.random_raw() for _ in range(5)]
print(f"\nRestored sequence: {sequence2}")
print(f"Sequences match: {sequence1 == sequence2} ✅")

# Can also pickle/unpickle state
# state_pickled = pickle.dumps(state)
# state_restored = pickle.loads(state_pickled)
# rng3 = Kiss64Random.from_state(state_restored)
# sequence3 = [rng3.random_raw() for _ in range(5)]
# print(f"Pickled/unpickled: {sequence1 == sequence3} ✅")

# %%

# ===========================================================================
# Feature 8: Thread-Safe Operations
# ===========================================================================

print("\n" + "=" * 70)
print("8. THREAD-SAFE OPERATIONS")
print("=" * 70)

# Shared RNG with lock
shared_rng = Kiss64Random(seed=42)

def parallel_task(task_id):
    """Task that uses shared RNG safely."""
    with shared_rng:  # Acquire lock
        # Safe to use RNG here
        return {
            'task_id': task_id,
            'values': [shared_rng.kiss() for _ in range(3)]
        }

print("Running 4 parallel tasks with shared RNG:")
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(parallel_task, range(4)))

for result in results:
    print(f"  Task {result['task_id']}: {result['values']}")

# Verify no value duplication (all unique sequences)
all_values = []
for result in results:
    all_values.extend(result['values'])

unique_count = len(set(all_values))
total_count = len(all_values)
print(f"\nTotal values: {total_count}, Unique: {unique_count}")
print(f"All unique: {unique_count == total_count} ✅")

# %%

# ===========================================================================
# Feature 9: Comparison with NumPy
# ===========================================================================

print("\n" + "=" * 70)
print("9. COMPARISON WITH NUMPY (Statistical Validation)")
print("=" * 70)

# Generate large samples with both
n_samples = 100000

# KISS
kiss_gen = default_rng(42)
kiss_samples = kiss_gen.random(n_samples)

# NumPy
numpy_gen = np.random.default_rng(42)
numpy_samples = numpy_gen.random(n_samples)

print(f"Generated {n_samples:,} samples with each")

# Compare statistics
print("\nStatistical comparison:")
print(f"  KISS  - Mean: {kiss_samples.mean():.6f}, Std: {kiss_samples.std():.6f}")
print(f"  NumPy - Mean: {numpy_samples.mean():.6f}, Std: {numpy_samples.std():.6f}")
print(f"  Expected:     Mean: 0.500000, Std: 0.288675")

# Chi-square test
from scipy import stats

def chi_square_test(samples, n_bins=20):
    """Perform chi-square uniformity test."""
    observed, _ = np.histogram(samples, bins=n_bins, range=(0, 1))
    expected = np.full(n_bins, len(samples) / n_bins)
    chi2, p_value = stats.chisquare(observed, expected)
    return chi2, p_value

chi2_kiss, p_kiss = chi_square_test(kiss_samples)
chi2_numpy, p_numpy = chi_square_test(numpy_samples)

print("\nChi-square uniformity test:")
print(f"  KISS:  χ²={chi2_kiss:.2f}, p={p_kiss:.4f} {'✅ PASS' if p_kiss > 0.01 else '❌ FAIL'}")
print(f"  NumPy: χ²={chi2_numpy:.2f}, p={p_numpy:.4f} {'✅ PASS' if p_numpy > 0.01 else '❌ FAIL'}")

# %%

# ===========================================================================
# Feature 10: Advanced Usage - Custom Distribution
# ===========================================================================

print("\n" + "=" * 70)
print("10. ADVANCED USAGE - CUSTOM DISTRIBUTIONS")
print("=" * 70)

# Extend KissGenerator with custom methods
class ExtendedKissGenerator(KissGenerator):
    """Extended generator with custom distributions."""

    def exponential(self, scale=1.0, size=None):
        """Exponential distribution."""
        u = self.random(size)
        return -scale * np.log(1 - u)

    def beta(self, alpha, beta_param, size=None):
        """Beta distribution (simplified)."""
        # Using rejection sampling (not optimal, just for demo)
        if size is None:
            while True:
                u1 = self.random()
                u2 = self.random()
                if u2 <= u1**(alpha-1) * (1-u1)**(beta_param-1):
                    return u1
        else:
            return np.array([self.beta(alpha, beta_param) for _ in range(size)])

# Use custom generator
custom_gen = ExtendedKissGenerator()

exp_samples = custom_gen.exponential(scale=2.0, size=5)
print(f"Exponential(scale=2): {exp_samples}")

beta_samples = custom_gen.beta(2, 5, size=5)
print(f"Beta(2, 5):           {beta_samples}")

# %%

# ===========================================================================
# Feature 11: Reproducibility Across Platforms
# ===========================================================================

print("\n" + "=" * 70)
print("11. REPRODUCIBILITY DEMO")
print("=" * 70)

# Same seed produces same sequence (always)
seeds = [42, 123, 999]

for seed in seeds:
    # Generate twice with same seed
    rng1 = default_rng(seed)
    seq1 = rng1.random(3)

    rng2 = default_rng(seed)
    seq2 = rng2.random(3)

    match = np.allclose(seq1, seq2)
    print(f"Seed {seed:>3}: {seq1} == {seq2} : {match} ✅")


# %%

# ===========================================================================
# Visualization
# ===========================================================================

print("\n" + "=" * 70)
print("12. VISUALIZATION")
print("=" * 70)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

gen = default_rng(42)

# 1. Uniform distribution
ax = axes[0, 0]
uniform_samples = gen.random(10000)
ax.hist(uniform_samples, bins=50, density=True, alpha=0.7, edgecolor='black')
ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
ax.set_title('Uniform Distribution')
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.grid(True, alpha=0.3)

# 2. Normal distribution
ax = axes[0, 1]
normal_samples = gen.normal(0, 1, size=10000)
ax.hist(normal_samples, bins=50, density=True, alpha=0.7, edgecolor='black')
x = np.linspace(-4, 4, 100)
ax.plot(x, stats.norm.pdf(x), 'r--', linewidth=2, label='Theory')
ax.set_title('Normal Distribution')
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Integer distribution
ax = axes[0, 2]
int_samples = gen.integers(0, 20, size=10000)
ax.hist(int_samples, bins=20, density=True, alpha=0.7, edgecolor='black')
ax.axhline(y=1/20, color='r', linestyle='--', alpha=0.7)
ax.set_title('Integer Distribution [0, 20)')
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.grid(True, alpha=0.3)

# 4. Q-Q plot
ax = axes[1, 0]
stats.probplot(uniform_samples, dist="uniform", plot=ax)
ax.set_title('Q-Q Plot (Uniform)')
ax.grid(True, alpha=0.3)

# 5. Autocorrelation
ax = axes[1, 1]
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(uniform_samples[:1000], lags=40, ax=ax, alpha=0.05)
ax.set_title('Autocorrelation')
ax.grid(True, alpha=0.3)

# 6. 2D scatter (independence test)
ax = axes[1, 2]
ax.scatter(uniform_samples[:-1], uniform_samples[1:], alpha=0.1, s=1)
ax.set_title('Serial Independence Test')
ax.set_xlabel('X[i]')
ax.set_ylabel('X[i+1]')
ax.grid(True, alpha=0.3)

plt.suptitle('Enhanced KISS Random Generator - Statistical Validation',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/enhanced_kiss_validation.png', dpi=150, bbox_inches='tight')
print("Visualization saved: /tmp/enhanced_kiss_validation.png")
plt.show()
# plt.close()

# %%

# ===========================================================================
# Summary
# ===========================================================================

print("\n" + "=" * 70)
print("SUMMARY - ALL FEATURES DEMONSTRATED")
print("=" * 70)

summary = """
RECOMMENDED USAGE:
    from scikitplot.random import default_rng

    # Create generator
    rng = default_rng(seed=42)
    data = rng.random(1000)

    # Use with context manager for safety
    with rng:
        samples = rng.random(1000)
        indices = rng.integers(0, 100, size=50)
        normal_data = rng.normal(0, 1, size=1000)
"""

print(summary)
print("=" * 70)

# %%
