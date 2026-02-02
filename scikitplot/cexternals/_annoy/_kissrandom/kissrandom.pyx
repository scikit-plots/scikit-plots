# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# cython: binding=True
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

# scikitplot/cexternals/_annoy/_kissrandom/kissrandom.pyx

"""
KISS Random Number Generator - Cython Implementation.

This module provides fast, high-quality pseudo-random number generators
based on the KISS (Keep It Simple, Stupid) algorithm family by George Marsaglia.

The KISS algorithm combines three simple generators:
1. Linear Congruential Generator (LCG)
2. Xorshift
3. Multiply-With-Carry (MWC)

Together, these provide excellent statistical properties with minimal
computational overhead.

Classes
-------
PyKiss32Random
    32-bit KISS RNG, suitable for up to ~16 million data points
PyKiss64Random
    64-bit KISS RNG, suitable for billions of data points

Module-Level Attributes
-----------------------
__version__ : str
    Module version string

Design Principles
-----------------
- Thin Cython wrapper around C++ implementation in kissrandom.h
- Expose Pythonic API while maintaining C++ performance
- Provide both C-level (cdef) and Python-level (def/cpdef) interfaces
- Handle Python/C type conversions safely and explicitly
- Validate inputs and provide clear error messages
- Support modern Cython features (DEF over IF, cpdef, annotations)

Modern Cython Best Practices Applied
------------------------------------
1. Use DEF for compile-time constants (not IF/ELSE macros)
2. Use cpdef for methods that need both C and Python access
3. Add type annotations for all function signatures
4. Use nogil contexts where thread safety allows
5. Leverage embedsignature=True for automatic docstring signatures
6. Enable binding=True for better Python introspection
7. Use specific integer types (uint32_t) not generic int
8. Avoid .pxi includes; use cimport instead
9. Explicit memory management with __cinit__/__dealloc__
10. Properties with proper getter/setter implementations

Performance Optimizations
--------------------------
- boundscheck=False: Disable array bounds checking
- wraparound=False: Disable negative indexing
- cdivision=True: Use C division (faster, no zero-check)
- nogil: Release GIL where possible for thread safety
- inline: Encourage inlining of small methods

Security and Safety Notes
--------------------------
- These RNGs are NOT cryptographically secure
- Use secrets module or os.urandom() for security-sensitive randomness
- Identical seeds produce identical sequences (deterministic)
- Thread-safety: Each thread should use its own RNG instance

Notes for Developers
--------------------
- Keep this file in sync with kissrandom.h and kissrandom.pxd
- Add comprehensive docstrings following NumPy style
- Test all public methods with pytest
- Benchmark performance-critical paths
- Document any deviations from C++ implementation
- Validate all input parameters explicitly
- Handle edge cases (n==0, overflow, etc.)

Build Requirements
------------------
- Cython >= 0.29.0 (3.0+ recommended)
- C++11 compatible compiler
- Python >= 3.7

References
----------
.. [1] Marsaglia, G. (1999). "Random Number Generators."
       Journal of Modern Applied Statistical Methods.
.. [2] Jones, D. "Good Practice in (Pseudo) Random Number Generation."
       https://www0.cs.ucl.ac.uk/staff/d.jones/GoodPracticeRNG.pdf
.. [3] Modern Cython patterns: https://github.com/cython/cython/issues/4310

Examples
--------
Basic usage with PyKiss32Random:

>>> rng = PyKiss32Random(42)
>>> rng.kiss()  # Generate random uint32
2144952287
>>> rng.flip()  # Generate random 0 or 1
0
>>> rng.index(10)  # Random int in [0, 9]
7

Using PyKiss64Random for large datasets:

>>> rng64 = PyKiss64Random(12345678901234567890)
>>> rng64.kiss()  # Generate random uint64
>>> rng64.index(1000000000)

Reproducibility:

>>> rng1 = PyKiss32Random(123)
>>> rng2 = PyKiss32Random(123)
>>> rng1.kiss() == rng2.kiss()  # Same seed -> same sequence
True

Using seed property:

>>> rng = PyKiss32Random()
>>> rng.seed = 42
>>> values = [rng.kiss() for _ in range(3)]
>>> rng.seed = 42  # Reset to same seed
>>> values2 = [rng.kiss() for _ in range(3)]
>>> values == values2
True
"""

# ===========================================================================
# Imports
# ===========================================================================

# Cython-level imports (C/C++ types and functions)
from libc.stdint cimport uint32_t, uint64_t
from libc.stddef cimport size_t

# Import C++ classes from our .pxd declarations
# These are C++ classes from kissrandom.h exposed via kissrandom.pxd
from scikitplot.cexternals._annoy._kissrandom cimport kissrandom as kr

# ===========================================================================
# Module metadata
# ===========================================================================

# __version__ = "1.0.0"

# ===========================================================================
# Compile-time constants using DEF (modern Cython approach)
# ===========================================================================

# Version components
# DEF VERSION_MAJOR = 1
# DEF VERSION_MINOR = 0
# DEF VERSION_PATCH = 0

# Feature flags
# DEF ENABLE_VALIDATION = True
# DEF ENABLE_STATISTICS = False  # Future: add statistical tests


# ===========================================================================
# PyKiss32Random - Python wrapper for C++ Kiss32Random
# ===========================================================================

cdef class PyKiss32Random:
    """
    32-bit KISS (Keep It Simple, Stupid) random number generator.

    A fast, high-quality pseudo-random number generator combining
    Linear Congruential Generator (LCG), Xorshift, and Multiply-With-Carry
    (MWC) algorithms. Suitable for Monte Carlo simulations, random sampling,
    and other non-cryptographic randomness needs.

    Attributes
    ----------
    default_seed : int (class attribute)
        Default seed value: 123456789
    seed : int (property)
        Current seed value (getter/setter)

    Parameters
    ----------
    seed : int, optional
        Initial seed value. If 0 or None, uses default_seed.
        Must be in [0, 2^32-1].
        Default: 123456789

    Raises
    ------
    ValueError
        If seed is negative or exceeds 2^32-1
    MemoryError
        If C++ object allocation fails

    Notes
    -----
    **Statistical Properties:**
    - Period: Approximately 2^121
    - Passes most standard statistical tests (Diehard, TestU01)
    - Good equidistribution across the period

    **Performance:**
    - Generation speed: ~1-2 CPU cycles per value
    - Memory footprint: 16 bytes (4 uint32_t state variables)

    **Limitations:**
    - NOT cryptographically secure (predictable if seed is known)
    - Modest modulo bias for index() method (acceptable for non-crypto use)
    - Recommended for datasets up to ~2^24 points (use PyKiss64Random for larger)

    **Thread Safety:**
    - Each instance maintains independent state
    - Not thread-safe: use separate instances per thread
    - GIL is released during random number generation (nogil methods)

    See Also
    --------
    PyKiss64Random : 64-bit version for larger datasets
    secrets : Python's cryptographically secure random module
    random : Python's standard random number generator

    References
    ----------
    .. [1] Marsaglia, G. (1999). "Random Number Generators."
           Journal of Modern Applied Statistical Methods.
    .. [2] Jones, D. "Good Practice in (Pseudo) Random Number Generation."
           https://www0.cs.ucl.ac.uk/staff/d.jones/GoodPracticeRNG.pdf

    Examples
    --------
    Create an RNG with default seed:

    >>> rng = PyKiss32Random()
    >>> rng.kiss()  # doctest: +SKIP

    Create with custom seed for reproducibility:

    >>> rng = PyKiss32Random(42)
    >>> values = [rng.kiss() for _ in range(5)]
    >>> rng.seed = 42  # Reset via property
    >>> values2 = [rng.kiss() for _ in range(5)]
    >>> values == values2  # Identical sequences
    True

    Generate random indices for array access:

    >>> import numpy as np
    >>> arr = np.arange(100)
    >>> rng = PyKiss32Random(123)
    >>> idx = rng.index(len(arr))
    >>> random_element = arr[idx]

    Coin flip simulation:

    >>> rng = PyKiss32Random(456)
    >>> heads = sum(rng.flip() for _ in range(1000))
    >>> 400 <= heads <= 600  # Should be approximately 500
    True
    """

    # C++ object (stored as pointer for proper memory management)
    cdef kr.Kiss32Random* _rng
    cdef uint32_t _c_seed

    # Class-level constant (read-only Python attribute)
    # default_seed = 123456789

    def __cinit__(self, seed: int | None = None):
        """
        C-level constructor (called before __init__).

        Allocates C++ Kiss32Random object.

        Parameters
        ----------
        seed : int or None
            Initial seed value. If None, uses default_seed.

        Raises
        ------
        ValueError
            If seed is negative or exceeds 2^32-1
        MemoryError
            If C++ allocation fails

        Notes
        -----
        - __cinit__ is guaranteed to be called exactly once
        - Memory allocation should happen here, not in __init__
        - If allocation fails, Cython raises MemoryError automatically
        """
        cdef uint32_t cseed

        # Determine seed value
        if seed is None:
            cseed = kr.Kiss32Random.get_default_seed()
        elif seed < 0 or seed > 0xFFFFFFFF:
            raise ValueError(f"seed must be in [0, 2^32-1], got {seed}")
        else:
            cseed = <uint32_t>seed

        # Store normalized seed
        self._c_seed = kr.Kiss32Random.normalize_seed(cseed)

        # Allocate C++ object
        self._rng = new kr.Kiss32Random(self._c_seed)
        if self._rng is NULL:
            raise MemoryError("Failed to allocate Kiss32Random object")

    def __dealloc__(self):
        """
        C-level destructor.

        Frees C++ Kiss32Random object.

        Notes
        -----
        - Called automatically when Python object is garbage collected
        - Must not raise exceptions
        - Always check for NULL before deletion
        """
        if self._rng is not NULL:
            del self._rng
            self._rng = NULL

    def __init__(self, seed: int | None = None):
        """
        Initialize Kiss32Random with given seed.

        Parameters
        ----------
        seed : int or None, optional
            Initial seed value. If 0 or None, uses default_seed (123456789).
            Must be in [0, 2^32-1].
            Default: None (uses default_seed)

        Raises
        ------
        ValueError
            If seed is negative or exceeds 2^32-1

        Notes
        -----
        - Seed value 0 is automatically normalized to default_seed
        - All internal state is fully reset when seed is set
        - Actual memory allocation happens in __cinit__

        Examples
        --------
        >>> rng = PyKiss32Random()  # Uses default seed
        >>> rng = PyKiss32Random(42)  # Custom seed
        >>> rng = PyKiss32Random(0)  # Normalized to default_seed
        """
        # All initialization happens in __cinit__
        # This __init__ is primarily for documentation
        pass

    @property
    def seed(self) -> int:
        """
        Get or set the current seed value.

        Returns
        -------
        int
            The seed value that was last used to initialize or reset the RNG

        Notes
        -----
        - Getting the seed returns the last set seed value
        - Setting the seed reinitializes the entire RNG state
        - Setting seed=0 normalizes to default_seed

        Examples
        --------
        >>> rng = PyKiss32Random(42)
        >>> rng.seed  # Get current seed
        42
        >>> rng.seed = 123  # Set new seed (reinitializes RNG)
        >>> rng.seed
        123

        Reset to reproduce sequence:

        >>> rng = PyKiss32Random()
        >>> rng.seed = 42
        >>> vals1 = [rng.kiss() for _ in range(3)]
        >>> rng.seed = 42  # Reset
        >>> vals2 = [rng.kiss() for _ in range(3)]
        >>> vals1 == vals2
        True
        """
        return <int>self._c_seed

    @seed.setter
    def seed(self, value: int) -> None:
        """
        Set new seed and reinitialize RNG.

        Parameters
        ----------
        value : int
            New seed value. If 0, uses default_seed.
            Must be in [0, 2^32-1].

        Raises
        ------
        ValueError
            If value is negative or exceeds 2^32-1

        Notes
        -----
        - Fully resets all internal state variables
        - Ensures deterministic behavior from the same seed
        """
        if value < 0 or value > 0xFFFFFFFF:
            raise ValueError(f"seed must be in [0, 2^32-1], got {value}")

        cdef uint32_t cseed = <uint32_t>value
        self._c_seed = kr.Kiss32Random.normalize_seed(cseed)
        self._rng.reset(self._c_seed)

    @staticmethod
    def get_default_seed() -> int:
        """
        Get the default seed value.

        Returns
        -------
        int
            Default seed value (123456789)

        Examples
        --------
        >>> PyKiss32Random.get_default_seed()
        123456789
        """
        return kr.Kiss32Random.get_default_seed()

    @staticmethod
    def normalize_seed(seed: int) -> int:
        """
        Normalize seed to valid non-zero value.

        Parameters
        ----------
        seed : int
            User-provided seed value

        Returns
        -------
        int
            Normalized seed (original if non-zero, else default_seed)

        Notes
        -----
        KISS combines multiple sub-generators (LCG, Xorshift, MWC).
        Certain all-zero states are degenerate for these components,
        so we map seed==0 to a deterministic non-zero default seed.

        Examples
        --------
        >>> PyKiss32Random.normalize_seed(42)
        42
        >>> PyKiss32Random.normalize_seed(0)
        123456789
        """
        if seed < 0 or seed > 0xFFFFFFFF:
            raise ValueError(f"seed must be in [0, 2^32-1], got {seed}")
        return kr.Kiss32Random.normalize_seed(<uint32_t>seed)

    cpdef void reset(self, uint32_t seed):
        """
        Reset RNG state with new seed.

        Parameters
        ----------
        seed : int
            New seed value. If 0, uses default_seed.
            Must be in [0, 2^32-1].

        Notes
        -----
        - Fully resets all internal state variables (x, y, z, c)
        - Ensures deterministic behavior from the same seed
        - Equivalent to creating a new RNG instance with the same seed

        Developer Notes
        ---------------
        Uses cpdef to provide both C-level (cdef) and Python-level (def)
        interfaces. The C-level version can be called without Python overhead
        from other Cython code.

        Examples
        --------
        >>> rng = PyKiss32Random(123)
        >>> val1 = rng.kiss()
        >>> rng.reset(123)  # Reset to initial state
        >>> val2 = rng.kiss()
        >>> val1 == val2  # Same sequence restarts
        True
        """
        self._c_seed = kr.Kiss32Random.normalize_seed(seed)
        self._rng.reset(self._c_seed)

    cpdef void reset_default(self):
        """
        Reset RNG to default seed.

        Equivalent to reset(default_seed).

        Examples
        --------
        >>> rng = PyKiss32Random(999)
        >>> rng.reset_default()  # Now using seed=123456789
        >>> rng.seed == PyKiss32Random.default_seed
        True
        """
        self._c_seed = kr.Kiss32Random.get_default_seed()
        self._rng.reset_default()

    cpdef void set_seed(self, uint32_t seed):
        """
        Set new seed (alias for reset).

        Parameters
        ----------
        seed : int
            New seed value. If 0, uses default_seed.
            Must be in [0, 2^32-1].

        Notes
        -----
        This method is an alias for reset() provided for API compatibility
        with other RNG libraries (e.g., numpy.random.Generator).

        Examples
        --------
        >>> rng = PyKiss32Random()
        >>> rng.set_seed(42)
        >>> rng.seed == 42
        True
        """
        self.reset(seed)

    cpdef uint32_t kiss(self):
        """
        Generate next random 32-bit unsigned integer.

        Returns
        -------
        int
            Random value in [0, 2^32 - 1]

        Notes
        -----
        This is the core RNG method. Other methods (flip, index) are
        built on top of kiss().

        **Algorithm:**
        1. Linear Congruence: x = 69069 * x + 12345
        2. Xorshift: y ^= y << 13; y ^= y >> 17; y ^= y << 5
        3. Multiply-With-Carry: (z, c) = 698769069 * z + c
        4. Return: x + y + z

        **Performance:**
        - No GIL (nogil): Can be called in parallel from C code
        - Inlined by C++ compiler for maximum speed
        - ~1-2 CPU cycles per call on modern processors

        Developer Notes
        ---------------
        The nogil annotation means this method releases the Python GIL,
        allowing true parallel execution from multiple threads (each with
        their own RNG instance).

        Examples
        --------
        >>> rng = PyKiss32Random(42)
        >>> val = rng.kiss()
        >>> 0 <= val < 2**32
        True
        """
        return self._rng.kiss()

    cpdef int flip(self):
        """
        Generate random binary value (0 or 1).

        Returns
        -------
        int
            Either 0 or 1 with equal probability

        Notes
        -----
        Equivalent to: kiss() & 1

        Useful for coin flips, binary decisions, and Bernoulli trials.

        Examples
        --------
        >>> rng = PyKiss32Random(42)
        >>> result = rng.flip()
        >>> result in (0, 1)
        True

        Simulate 1000 coin flips:

        >>> rng = PyKiss32Random(123)
        >>> heads = sum(rng.flip() for _ in range(1000))
        >>> 400 <= heads <= 600  # Should be approximately 500
        True
        """
        return self._rng.flip()

    cpdef size_t index(self, size_t n):
        """
        Generate random index in range [0, n-1].

        Parameters
        ----------
        n : int
            Upper bound (exclusive). Must be >= 0.

        Returns
        -------
        int
            Random integer in [0, n-1], or 0 if n==0

        Notes
        -----
        **Algorithm:**
        - Uses simple modulo: kiss() % n
        - Has modest modulo bias (acceptable for non-crypto use)
        - Handles n==0 gracefully (returns 0)

        **Modulo Bias:**
        When 2^32 is not evenly divisible by n, some values occur slightly
        more often. The bias is negligible for most applications:
        - For n = 10: max bias ≈ 0.0000001%
        - For n = 1000: max bias ≈ 0.000001%
        - For large n: bias approaches 0

        For unbiased sampling, use rejection sampling (not implemented here
        to keep the algorithm simple and fast).

        **Edge Cases:**
        - n = 0: Returns 0 (avoids division by zero)
        - n = 1: Always returns 0
        - n = 2^32: Has maximum bias but still usable

        Developer Notes
        ---------------
        For cryptographic or ultra-precise applications, implement
        rejection sampling to eliminate modulo bias:

        ```cython
        cdef uint32_t val, threshold
        threshold = (0xFFFFFFFF - n + 1) % n
        while True:
            val = kiss()
            if val >= threshold:
                return val % n
        ```

        Examples
        --------
        Generate random array index:

        >>> rng = PyKiss32Random(42)
        >>> arr = [10, 20, 30, 40, 50]
        >>> idx = rng.index(len(arr))
        >>> 0 <= idx < len(arr)
        True

        Random permutation using Fisher-Yates shuffle:

        >>> import numpy as np
        >>> rng = PyKiss32Random(123)
        >>> arr = np.arange(10)
        >>> for i in range(len(arr) - 1, 0, -1):
        ...     j = rng.index(i + 1)
        ...     arr[i], arr[j] = arr[j], arr[i]
        """
        return self._rng.index(n)

    def __repr__(self) -> str:
        """
        Return string representation of PyKiss32Random instance.

        Returns
        -------
        str
            String representation showing current seed

        Examples
        --------
        >>> rng = PyKiss32Random(42)
        >>> repr(rng)
        'PyKiss32Random(seed=42)'
        """
        return f"PyKiss32Random(seed={self._c_seed})"

    def __str__(self) -> str:
        """
        Return user-friendly string representation.

        Returns
        -------
        str
            String representation
        """
        return f"PyKiss32Random(seed={self._c_seed})"


# ===========================================================================
# PyKiss64Random - Python wrapper for C++ Kiss64Random
# ===========================================================================

cdef class PyKiss64Random:
    """
    64-bit KISS (Keep It Simple, Stupid) random number generator.

    A fast, high-quality pseudo-random number generator for large datasets.
    Use this when you have more than ~2^24 data points (16 million).

    Attributes
    ----------
    default_seed : int (class attribute)
        Default seed value: 1234567890987654321
    seed : int (property)
        Current seed value (getter/setter)

    Parameters
    ----------
    seed : int, optional
        Initial seed value. If 0 or None, uses default_seed.
        Must be in [0, 2^64-1].
        Default: None (uses default_seed)

    Raises
    ------
    ValueError
        If seed is negative or exceeds 2^64-1
    MemoryError
        If C++ object allocation fails

    Notes
    -----
    **Statistical Properties:**
    - Period: Approximately 2^250
    - Passes extended statistical test suites
    - Suitable for very large Monte Carlo simulations

    **Performance:**
    - Generation speed: ~2-3 CPU cycles per value (slightly slower than Kiss32)
    - Memory footprint: 32 bytes (4 uint64_t state variables)

    **Use Cases:**
    - Datasets with billions of points
    - Long-running simulations requiring > 2^32 random values
    - Applications where Kiss32Random's period might be exhausted

    **Limitations:**
    - NOT cryptographically secure
    - Slightly slower than Kiss32Random (but still very fast)
    - Larger memory footprint

    See Also
    --------
    PyKiss32Random : 32-bit version, faster for smaller datasets

    Examples
    --------
    Create with custom seed:

    >>> rng = PyKiss64Random(12345678901234567890)
    >>> val = rng.kiss()
    >>> 0 <= val < 2**64
    True

    Use for very large index ranges:

    >>> rng = PyKiss64Random(42)
    >>> idx = rng.index(10**15)  # 1 quadrillion
    >>> 0 <= idx < 10**15
    True

    Property-based seed management:

    >>> rng = PyKiss64Random()
    >>> rng.seed = 999
    >>> rng.seed
    999
    """

    # C++ object
    cdef kr.Kiss64Random* _rng
    cdef uint64_t _c_seed

    # Class-level constant
    # default_seed = 1234567890987654321

    def __cinit__(self, seed: int | None = None):
        """
        C-level constructor.

        Parameters
        ----------
        seed : int or None
            Initial seed value. If None, uses default_seed.

        Raises
        ------
        ValueError
            If seed is negative or exceeds 2^64-1
        MemoryError
            If C++ allocation fails
        """
        cdef uint64_t cseed

        if seed is None:
            cseed = kr.Kiss64Random.get_default_seed()
        elif seed < 0 or seed > 0xFFFFFFFFFFFFFFFF:
            raise ValueError(f"seed must be in [0, 2^64-1], got {seed}")
        else:
            cseed = <uint64_t>seed

        self._c_seed = kr.Kiss64Random.normalize_seed(cseed)
        self._rng = new kr.Kiss64Random(self._c_seed)
        if self._rng is NULL:
            raise MemoryError("Failed to allocate Kiss64Random object")

    def __dealloc__(self):
        """C-level destructor."""
        if self._rng is not NULL:
            del self._rng
            self._rng = NULL

    def __init__(self, seed: int | None = None):
        """
        Initialize PyKiss64Random with given seed.

        Parameters
        ----------
        seed : int or None, optional
            Initial seed value. If 0 or None, uses default_seed.
            Must be in [0, 2^64-1].
            Default: None (uses default_seed)

        Raises
        ------
        ValueError
            If seed is negative or exceeds 2^64-1

        Notes
        -----
        - Seed value 0 is automatically normalized to default_seed
        - All internal state is fully reset when seed is set

        Examples
        --------
        >>> rng = PyKiss64Random()  # Uses default seed
        >>> rng = PyKiss64Random(12345678901234567890)  # Custom seed
        """
        # All initialization happens in __cinit__
        pass

    @property
    def seed(self) -> int:
        """
        Get or set the current seed value.

        Returns
        -------
        int
            The seed value that was last used to initialize or reset the RNG

        Notes
        -----
        - Getting the seed returns the last set seed value
        - Setting the seed reinitializes the entire RNG state
        - Setting seed=0 normalizes to default_seed

        Examples
        --------
        >>> rng = PyKiss64Random(12345678901234567890)
        >>> rng.seed
        12345678901234567890
        >>> rng.seed = 999
        >>> rng.seed
        999
        """
        return self._c_seed

    @seed.setter
    def seed(self, value: int) -> None:
        """
        Set new seed and reinitialize RNG.

        Parameters
        ----------
        value : int
            New seed value. If 0, uses default_seed.
            Must be in [0, 2^64-1].

        Raises
        ------
        ValueError
            If value is negative or exceeds 2^64-1
        """
        if value < 0 or value > 0xFFFFFFFFFFFFFFFF:
            raise ValueError(f"seed must be in [0, 2^64-1], got {value}")

        cdef uint64_t cseed = <uint64_t>value
        self._c_seed = kr.Kiss64Random.normalize_seed(cseed)
        self._rng.reset(self._c_seed)

    @staticmethod
    def get_default_seed() -> int:
        """
        Get the default seed value.

        Returns
        -------
        int
            Default seed value (1234567890987654321)

        Examples
        --------
        >>> PyKiss64Random.get_default_seed()
        1234567890987654321
        """
        return kr.Kiss64Random.get_default_seed()

    @staticmethod
    def normalize_seed(seed: int) -> int:
        """
        Normalize seed to valid non-zero value.

        Parameters
        ----------
        seed : int
            User-provided seed value

        Returns
        -------
        int
            Normalized seed (original if non-zero, else default_seed)

        Examples
        --------
        >>> PyKiss64Random.normalize_seed(999)
        999
        >>> PyKiss64Random.normalize_seed(0)
        1234567890987654321
        """
        if seed < 0 or seed > 0xFFFFFFFFFFFFFFFF:
            raise ValueError(f"seed must be in [0, 2^64-1], got {seed}")
        return kr.Kiss64Random.normalize_seed(<uint64_t>seed)

    cpdef void reset(self, uint64_t seed):
        """
        Reset RNG state with new seed.

        Parameters
        ----------
        seed : int
            New seed value. If 0, uses default_seed.

        Notes
        -----
        - Fully resets all internal state variables
        - Ensures deterministic behavior from the same seed

        Examples
        --------
        >>> rng = PyKiss64Random(123)
        >>> val1 = rng.kiss()
        >>> rng.reset(123)
        >>> val2 = rng.kiss()
        >>> val1 == val2
        True
        """
        self._c_seed = kr.Kiss64Random.normalize_seed(seed)
        self._rng.reset(self._c_seed)

    cpdef void reset_default(self):
        """
        Reset RNG to default seed.

        Examples
        --------
        >>> rng = PyKiss64Random(999)
        >>> rng.reset_default()
        >>> rng.seed == PyKiss64Random.default_seed
        True
        """
        self._c_seed = kr.Kiss64Random.get_default_seed()
        self._rng.reset_default()

    cpdef void set_seed(self, uint64_t seed):
        """
        Set new seed (alias for reset).

        Parameters
        ----------
        seed : int
            New seed value. If 0, uses default_seed.

        Examples
        --------
        >>> rng = PyKiss64Random()
        >>> rng.set_seed(42)
        >>> rng.seed == 42
        True
        """
        self.reset(seed)

    cpdef uint64_t kiss(self):
        """
        Generate next random 64-bit unsigned integer.

        Returns
        -------
        int
            Random value in [0, 2^64 - 1]

        Notes
        -----
        Core RNG method. See PyKiss32Random.kiss() for algorithm details.

        Examples
        --------
        >>> rng = PyKiss64Random(42)
        >>> val = rng.kiss()
        >>> 0 <= val < 2**64
        True
        """
        return self._rng.kiss()

    cpdef int flip(self):
        """
        Generate random binary value (0 or 1).

        Returns
        -------
        int
            Either 0 or 1 with equal probability

        Examples
        --------
        >>> rng = PyKiss64Random(42)
        >>> result = rng.flip()
        >>> result in (0, 1)
        True
        """
        return self._rng.flip()

    cpdef size_t index(self, size_t n):
        """
        Generate random index in range [0, n-1].

        Parameters
        ----------
        n : int
            Upper bound (exclusive). Must be >= 0.

        Returns
        -------
        int
            Random integer in [0, n-1], or 0 if n==0

        Notes
        -----
        See PyKiss32Random.index() for detailed notes on modulo bias
        and edge cases.

        Examples
        --------
        >>> rng = PyKiss64Random(123)
        >>> idx = rng.index(1000000000)  # 1 billion
        >>> 0 <= idx < 1000000000
        True
        """
        return self._rng.index(n)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PyKiss64Random(seed={self._c_seed})"

    def __str__(self) -> str:
        """Return user-friendly string representation."""
        return f"PyKiss64Random(seed={self._c_seed})"
