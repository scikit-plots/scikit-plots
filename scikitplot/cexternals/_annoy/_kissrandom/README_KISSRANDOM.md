# KISS Random Number Generator - Updated Source Files

## Overview

This package contains the complete, production-ready source code for a NumPy-compatible KISS random number generator implementation.

## Files Provided

1. **kissrandom.pxd** - Cython declarations (C++ interface)
2. **kissrandom.pyx** - Main Cython implementation (Part 1)
3. **kissrandom_part2.pyx** - Continuation (KissGenerator class)
4. **kissrandom.pyi** - Type stubs for static type checking
5. **KISSRANDOM_NUMPY_COMPATIBLE_FINAL.md** - Complete documentation

## Installation Instructions

### Step 1: Merge the Cython Files

The implementation is split into two parts due to size. Merge them:

```bash
cat kissrandom.pyx kissrandom_part2.pyx > final_kissrandom.pyx
```

Or manually:
1. Open `kissrandom.pyx`
2. Scroll to the end (after `KissBitGenerator` class)
3. Append the entire content of `kissrandom_part2.pyx`
4. Save as `kissrandom.pyx`

### Step 2: File Placement

Place files in your project:

```
your_project/
├── scikitplot/
│   └── cexternals/
│       └── _annoy/
│           └── _kissrandom/
│               ├── src/
│               │   └── kissrandom.h        # Your existing C++ header
│               ├── kissrandom.pxd          # Replace with new version
│               ├── kissrandom.pyx          # Replace with new merged file
│               ├── kissrandom.pyi          # Replace with new version
│               ├── __init__.py
│               └── setup.py
```

### Step 3: Build Configuration

Create or update `setup.py`:

```python
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "scikitplot.cexternals._annoy._kissrandom.kissrandom",
        sources=["kissrandom.pyx"],
        include_dirs=[
            np.get_include(),
            "src",  # For kissrandom.h
        ],
        language="c++",
        extra_compile_args=["-std=c++11", "-O3"],
    )
]

setup(
    name="scikitplot-kissrandom",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'embedsignature': True,
        }
    ),
    zip_safe=False,
)
```

### Step 4: Build the Extension

```bash
# Development build
python setup.py build_ext --inplace

# Or install
pip install -e .
```

## Usage Examples

### Basic Usage (Recommended)

```python
from scikitplot.cexternals._annoy._kissrandom import default_rng

# Create generator
rng = default_rng(42)

# Generate random floats
random_floats = rng.random(1000)

# Generate random integers
random_ints = rng.integers(0, 100, size=50)

# Normal distribution
normal_samples = rng.normal(0, 1, size=1000)

# Sampling
choices = rng.choice(['A', 'B', 'C'], size=100)
```

### NumPy Compatibility

```python
import numpy as np
from scikitplot.cexternals._annoy._kissrandom import KissBitGenerator

# Create KISS bit generator
kiss_bg = KissBitGenerator(42)

# Wrap in NumPy Generator for ALL distributions
gen = np.random.Generator(kiss_bg)

# Now have access to all NumPy distributions
gamma_samples = gen.gamma(2.0, size=100)
beta_samples = gen.beta(2.0, 5.0, size=100)
poisson_samples = gen.poisson(5.0, size=100)
```

### Parallel Computing

```python
from scikitplot.cexternals._annoy._kissrandom import default_rng
from concurrent.futures import ProcessPoolExecutor

def monte_carlo_task(seed):
    """Independent Monte Carlo simulation."""
    rng = default_rng(seed)
    # Perform simulation
    return simulation_result(rng)

# Generate independent seeds
master_rng = default_rng(42)
seeds = [master_rng.integers(0, 2**31) for _ in range(10)]

# Run in parallel
with ProcessPoolExecutor() as executor:
    results = list(executor.map(monte_carlo_task, seeds))
```

### State Persistence

```python
import pickle
from scikitplot.cexternals._annoy._kissrandom import KissBitGenerator

# Create and use generator
bg = KissBitGenerator(42)
_ = bg.random_raw(size=1000)  # Generate some values

# Save state
state = bg.state
with open('rng_state.pkl', 'wb') as f:
    pickle.dump(state, f)

# Later: restore state
with open('rng_state.pkl', 'rb') as f:
    saved_state = pickle.load(f)

bg_restored = KissBitGenerator()
bg_restored.state = saved_state
# Continues from exact same point
```

## Key Features

### 1. Full NumPy API Compatibility

All methods match NumPy's signatures exactly:

- `KissSeedSequence` → `numpy.random.SeedSequence`
- `KissBitGenerator` → `numpy.random.BitGenerator`
- `KissGenerator` → `numpy.random.Generator`

### 2. Comprehensive Documentation

Every function has complete NumPyDoc with:
- **Parameters** - All parameters documented
- **Returns** - Return value specifications
- **Raises** - All possible exceptions
- **See Also** - Related functions
- **Notes** - Implementation details
- **References** - Academic citations
- **Examples** - Working code examples

### 3. Production-Ready Quality

- ✅ Full input validation
- ✅ Comprehensive error messages
- ✅ Thread-safe operations
- ✅ Type hints throughout
- ✅ Zero-copy array operations
- ✅ Efficient Cython implementation

### 4. Advanced Features

- **Spawning** - Independent RNG streams for parallelization
- **State management** - Save/restore for checkpointing
- **Context managers** - Clean resource management
- **Multiple bit widths** - 32-bit or 64-bit generators

## Testing

### Unit Tests

Create `tests/test_kissrandom.py`:

```python
import pytest
import numpy as np
from scikitplot.cexternals._annoy._kissrandom import (
    KissSeedSequence,
    KissBitGenerator,
    KissGenerator,
    default_rng,
)


class TestKissSeedSequence:
    def test_basic_initialization(self):
        seq = KissSeedSequence(42)
        assert seq.entropy == 42
        assert seq.spawn_key == ()

    def test_spawn_independence(self):
        parent = KissSeedSequence(42)
        children = parent.spawn(10)
        states = [c.generate_state(1) for c in children]
        assert len(set(map(tuple, states))) == 10


class TestKissBitGenerator:
    def test_random_raw(self):
        bg = KissBitGenerator(42)
        value = bg.random_raw()
        assert isinstance(value, int)
        assert 0 <= value < 2**64

    def test_determinism(self):
        bg1 = KissBitGenerator(123)
        bg2 = KissBitGenerator(123)
        arr1 = bg1.random_raw(size=100)
        arr2 = bg2.random_raw(size=100)
        np.testing.assert_array_equal(arr1, arr2)


class TestKissGenerator:
    def test_random(self):
        gen = default_rng(42)
        arr = gen.random(100)
        assert np.all((arr >= 0.0) & (arr < 1.0))

    def test_integers(self):
        gen = default_rng(42)
        arr = gen.integers(0, 100, size=100)
        assert np.all((arr >= 0) & (arr < 100))

    def test_normal_distribution(self):
        gen = default_rng(42)
        samples = gen.normal(0, 1, size=10000)
        assert abs(samples.mean()) < 0.1
        assert abs(samples.std() - 1.0) < 0.1


def test_numpy_compatibility():
    """Test that KissBitGenerator works with NumPy's Generator."""
    from scikitplot.cexternals._annoy._kissrandom import KissBitGenerator

    bg = KissBitGenerator(42)
    gen = np.random.Generator(bg)

    # Test various distributions
    gamma = gen.gamma(2.0, size=100)
    assert len(gamma) == 100
    assert np.all(gamma > 0)
```

Run tests:

```bash
pytest tests/test_kissrandom.py -v
```

### Statistical Quality Tests

Create `tests/test_statistical.py`:

```python
import numpy as np
from scipy import stats
from scikitplot.cexternals._annoy._kissrandom import default_rng


def test_uniform_distribution():
    """Chi-square test for uniformity."""
    rng = default_rng(42)
    samples = rng.random(100000)

    # Chi-square test
    observed, _ = np.histogram(samples, bins=20, range=(0, 1))
    expected = np.full(20, len(samples) / 20)
    chi2, p_value = stats.chisquare(observed, expected)

    assert p_value > 0.01, f"Failed uniformity test: p={p_value}"


def test_normal_distribution():
    """Kolmogorov-Smirnov test for normality."""
    rng = default_rng(42)
    samples = rng.normal(0, 1, size=10000)

    # K-S test
    statistic, p_value = stats.kstest(samples, 'norm')

    assert p_value > 0.01, f"Failed normality test: p={p_value}"


def test_independence():
    """Test for serial correlation."""
    rng = default_rng(42)
    samples = rng.random(10000)

    # Lag-1 autocorrelation
    corr = np.corrcoef(samples[:-1], samples[1:])[0, 1]

    # Should be near zero for independent samples
    assert abs(corr) < 0.05, f"Serial correlation too high: {corr}"
```

## Performance Benchmarks

```python
import time
import numpy as np
from scikitplot.cexternals._annoy._kissrandom import default_rng

n = 10_000_000

# KISS
start = time.time()
kiss_rng = default_rng(42)
kiss_values = kiss_rng.random(n)
kiss_time = time.time() - start

# NumPy
start = time.time()
numpy_rng = np.random.default_rng(42)
numpy_values = numpy_rng.random(n)
numpy_time = time.time() - start

print(f"KISS:  {kiss_time:.3f}s ({n/kiss_time:,.0f} values/sec)")
print(f"NumPy: {numpy_time:.3f}s ({n/numpy_time:,.0f} values/sec)")
print(f"Speedup: {numpy_time/kiss_time:.2f}x")
```

## Migration Guide

### From Old KissRandom

**Before:**
```python
from scikitplot.cexternals._annoy._kissrandom import Kiss64Random

rng = Kiss64Random(42)
values = [rng.kiss() / (2**64) for _ in range(1000)]
```

**After:**
```python
from scikitplot.cexternals._annoy._kissrandom import default_rng

rng = default_rng(42)
values = rng.random(1000)  # Vectorized and faster!
```

### From NumPy

The API is identical, so minimal changes needed:

```python
# Just change the import
# from numpy.random import default_rng
from scikitplot.cexternals._annoy._kissrandom import default_rng

# Everything else stays the same
rng = default_rng(42)
data = rng.random(1000)
ints = rng.integers(0, 100, 50)
```

## Type Checking

The `.pyi` file enables static type checking:

```bash
# Install type checker
pip install mypy

# Run type checking
mypy your_code.py --strict
```

Example typed code:

```python
from scikitplot.cexternals._annoy._kissrandom import default_rng
import numpy as np

def monte_carlo_pi(n_samples: int) -> float:
    """Estimate pi using Monte Carlo method."""
    rng = default_rng(42)
    x: np.ndarray = rng.random(n_samples)
    y: np.ndarray = rng.random(n_samples)
    inside: np.ndarray = (x**2 + y**2) < 1
    return 4.0 * inside.sum() / n_samples

# Type checker knows all return types
pi_estimate: float = monte_carlo_pi(1000000)
```

## Troubleshooting

### Build Errors

**Error: `kissrandom.h not found`**
```bash
# Make sure src/kissrandom.h exists
ls src/kissrandom.h

# Check include_dirs in setup.py points to src/
```

**Error: `numpy/arrayobject.h not found`**
```bash
# Install numpy headers
pip install numpy --upgrade

# Verify in setup.py:
# include_dirs=[np.get_include(), ...]
```

### Runtime Errors

**Error: `Cannot import name 'KissGenerator'`**
```bash
# Rebuild the extension
python setup.py build_ext --inplace

# Or reinstall
pip install -e . --force-reinstall
```

**Error: `TypeError: seed must be int`**
```python
# Ensure seed is integer
rng = default_rng(int(42.5))  # Convert to int

# Or use None for random seed
rng = default_rng()  # Uses OS entropy
```

## Contributing

When modifying the code:

1. **Update all three files**:
   - `kissrandom.pyx` - Implementation
   - `kissrandom.pyi` - Type stubs
   - `kissrandom.pxd` - C++ declarations (if needed)

2. **Follow NumPyDoc format** for all docstrings

3. **Add tests** for new functionality

4. **Run type checker**:
   ```bash
   mypy --strict kissrandom.pyi
   ```

5. **Verify compilation**:
   ```bash
   python setup.py build_ext --inplace
   ```

## License

BSD-3-Clause (same as scikit-plot)

## References

1. Marsaglia, G. (1999). "Random Number Generators." Journal of Modern Applied Statistical Methods.
2. NumPy Random Generator: https://numpy.org/doc/stable/reference/random/generator.html
3. Cython Documentation: https://cython.readthedocs.io/

## Support

For issues or questions:
1. Check the documentation in `KISSRANDOM_NUMPY_COMPATIBLE_FINAL.md`
2. Review examples in this README
3. Open an issue on GitHub (if applicable)

---

**Version:** 2.1.0
**Last Updated:** 2026-02-03
**Status:** Production Ready ✅
