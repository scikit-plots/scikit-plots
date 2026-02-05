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
KISS Random Number Generator - NumPy-Compatible Implementation.

.. currentmodule:: scikitplot.random

This module provides a high-performance, NumPy-compatible random number
generator using the KISS (Keep It Simple, Stupid) algorithm. It implements
the full NumPy Generator/BitGenerator protocol while maintaining the speed
and simplicity of the KISS algorithm.

Key Components
--------------
PyKiss32Random, PyKiss64Random
    Low-level Cython wrappers around C++ KISS implementation
    (32-bit: ~2^121 period, 64-bit: ~2^250 period)
PyKissRandom
    Factory function with auto-detection (32/64-bit)
KissSeedSequence
    NumPy SeedSequence compatible entropy management
KissBitGenerator
    NumPy BitGenerator protocol implementation
KissGenerator
    Modern NumPy Generator-style API (recommended)
KissRandomState
    Legacy NumPy RandomState API (backward compatibility)
default_rng()
    Main entry point

Design Philosophy
-----------------
The implementation follows a three-layer architecture:

1. **C++ Core (kissrandom.h)**: Pure KISS algorithm implementation
2. **Cython Wrapper (this file)**: Bridges C++ and Python with zero-copy operations
3. **Python API**: NumPy-compatible high-level interface

This design ensures:
- Maximum performance through C++ implementation
- Zero-copy array operations via Cython
- Full NumPy API compatibility
- Thread-safe operations with explicit locking

References
----------
.. [1] Marsaglia, G. (1999). "Random Number Generators."
       Journal of Modern Applied Statistical Methods, 2(1), 2-13.
.. [2] Jones, D. "Good Practice in (Pseudo) Random Number Generation for
       Bioinformatics Applications."
       https://www0.cs.ucl.ac.uk/staff/d.jones/GoodPracticeRNG.pdf
.. [3] NumPy Development Team. "Random Generator."
       https://numpy.org/doc/stable/reference/random/generator.html

Examples
--------
>>> from scikitplot.cexternals._annoy._kissrandom.kissrandom import default_rng, kiss_context
>>> rng = default_rng(42)
>>> data = rng.random(1000)

Context manager

>>> with default_rng(42) as rng:
...     data = rng.random(1000)
>>>
>>> with kiss_context(42) as rng:
...     data = rng.random(1000)

Serialization

>>> import pickle
>>> state = pickle.dumps(rng)
>>> restored = pickle.loads(state)

JSON export

>>> import json
>>> json_str = json.dumps(rng.serialize())
>>> restored = KissGenerator.deserialize(json.loads(json_str))
"""

from __future__ import annotations

# ===========================================================================
# Imports
# ===========================================================================

# import os
import sys
# import json
import platform
# import struct
import threading
import secrets
import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Optional, Union  # Mapping

# NumPy integration
import numpy as np
cimport numpy as cnp
from numpy cimport ndarray

# NumPy random C API
from numpy.random cimport bitgen_t
from numpy.random import SeedSequence as NumpySeedSequence

# Type hints
from numpy._typing import (
    # ArrayLike,
    DTypeLike,
    NDArray,
    # _ShapeLike,
)

# Initialize NumPy C API
cnp.import_array()

# Cython-level imports
from libc.stdint cimport uint32_t, uint64_t
from libc.stddef cimport size_t
from cpython.pycapsule cimport PyCapsule_New  # PyCapsule_GetPointer

# Import C++ classes from our .pxd declarations
from scikitplot.cexternals._annoy._kissrandom cimport kissrandom as kr

# ===========================================================================
# Module metadata
# ===========================================================================

__version__ = "1.0.0"

__all__ = [
    # Legacy Low-level API RNGs
    "PyKiss32Random",
    "PyKiss64Random",
    "PyKissRandom",

    # NumPy ecosystem
    "KissSeedSequence",
    "KissBitGenerator",
    "KissGenerator",
    "KissRandomState",

    # Convenience functions
    "default_rng",
    "kiss_context",

    # KissGenerator Distribution Methods
    "choice",
    "integers",
    "normal",
    "permutation",
    "random",
    "shuffle",
    "uniform",
]

# ===========================================================================
# Constants
# ===========================================================================

DEF SERIALIZATION_VERSION = "0.0.0"

# Default seeds (from C++ header)
DEF KISS32_DEFAULT_SEED = 123456789
DEF KISS64_DEFAULT_SEED = 1234567890987654321

# Conversion constants
DEF UINT64_MAX = 18446744073709551615  # 2^64 - 1
DEF UINT32_MAX = 4294967295  # 2^32 - 1

# ===========================================================================
# Utility Functions
# ===========================================================================

def _detect_optimal_bit_width() -> int:
    """
    Detect optimal RNG bit width based on system architecture.

    Returns
    -------
    int
        32 or 64

    Notes
    -----
    - 64-bit systems → 64-bit RNG (better period, performance)
    - 32-bit systems → 32-bit RNG (lower memory, still good quality)

    Examples
    --------
    >>> bit_width = _detect_optimal_bit_width()
    >>> print(bit_width)  # 64 on most modern systems
    64
    """
    # sys.maxsize > 2**32 indicates 64-bit Python
    return 64 if sys.maxsize > 2**32 else 32


cdef inline uint64_t _safe_to_uint64(object value) except? 0:
    """
    Safely convert Python int to uint64 without overflow.

    Parameters
    ----------
    value : int or uint64
        Value to convert

    Returns
    -------
    uint64_t
        Masked to 64 bits

    Notes
    -----
    - Handles arbitrarily large Python integers
    - Masks to 64 bits: value & 0xFFFFFFFFFFFFFFFF
    - Prevents OverflowError for large seeds
    """
    if isinstance(value, int):
        # Mask to 64 bits to prevent overflow
        return <uint64_t>(value & 0xFFFFFFFFFFFFFFFF)
    return <uint64_t>value

cdef inline uint32_t _safe_to_uint32(object value) except? 0:
    """
    Safely convert Python int to uint32 without overflow.

    Parameters
    ----------
    value : int or uint32
        Value to convert

    Returns
    -------
    uint32_t
        Masked to 32 bits

    Notes
    -----
    - Handles large Python integers
    - Masks to 32 bits: value & 0xFFFFFFFF
    """
    if isinstance(value, int):
        return <uint32_t>(value & 0xFFFFFFFF)
    return <uint32_t>value

def _validate_dtype(dtype, allowed_dtypes):
    """
    Validate dtype parameter.

    Parameters
    ----------
    dtype : dtype-like
        Data type to validate
    allowed_dtypes : set of dtype
        Allowed data types

    Returns
    -------
    numpy.dtype
        Normalized dtype

    Raises
    ------
    ValueError
        If dtype not in allowed set
    """
    dtype = np.dtype(dtype)
    if dtype not in allowed_dtypes:
        raise ValueError(
            f"dtype must be one of {allowed_dtypes}, got {dtype}"
        )
    return dtype


def _check_version_compatibility(state_version: str,
                                 current_version: str = SERIALIZATION_VERSION) -> bool:
    """
    Check if serialized state version is compatible with current code.

    Parameters
    ----------
    state_version : str
        Version string from serialized state (e.g., "0.0.0")
    current_version : str, default=SERIALIZATION_VERSION
        Current code version

    Returns
    -------
    bool
        True if compatible, False otherwise

    Raises
    ------
    ValueError
        If version string format is invalid

    Notes
    -----
    - Major version must match (X.x.x compatible with X.y.z only)
    - Minor/patch differences are allowed (forward/backward compatible)
    - Breaking changes only in major versions

    Examples
    --------
    >>> _check_version_compatibility("0.0.0", "0.1.0")
    True
    >>> _check_version_compatibility("0.5.0", "1.0.0")
    False
    """
    try:
        state_major = int(state_version.split(".")[0])
        current_major = int(current_version.split(".")[0])
        return state_major == current_major
    except (ValueError, IndexError, AttributeError) as e:
        raise ValueError(f"Invalid version format: {state_version}") from e


def _validate_state_dict(state: dict, required_keys: set) -> None:
    """
    Validate state dictionary has required keys and correct structure.

    Parameters
    ----------
    state : dict
        State dictionary to validate
    required_keys : set of str
        Required keys that must be present

    Raises
    ------
    TypeError
        If state is not a dict
    ValueError
        If required keys are missing or state is invalid

    Notes
    -----
    - Called by all set_state/deserialize methods
    - Ensures data integrity before restoration
    - Provides clear error messages for debugging

    Examples
    --------
    >>> state = {"seed": 42, "__version__": "0.0.0"}
    >>> _validate_state_dict(state, {"seed"})  # OK
    >>> _validate_state_dict(state, {"seed", "missing"})  # Raises ValueError
    """
    if not isinstance(state, dict):
        raise TypeError(f"State must be dict, got {type(state)}")

    missing = required_keys - set(state.keys())
    if missing:
        raise ValueError(f"State dict missing required keys: {missing}")

    # Validate version if present
    if "__version__" in state:
        if not _check_version_compatibility(state["__version__"]):
            warnings.warn(
                f"State version {state['__version__']} may be incompatible "
                f"with current version {SERIALIZATION_VERSION}. "
                f"Proceeding with caution.",
                UserWarning,
                stacklevel=3
            )


def _add_metadata(state: dict) -> dict:
    """
    Add metadata to state dictionary.

    Parameters
    ----------
    state : dict
        State dictionary

    Returns
    -------
    dict
        State with metadata added

    Notes
    -----
    - Adds: module, version, Python version, NumPy version
    - Useful for debugging version compatibility issues
    - Metadata is optional for deserialization

    Examples
    --------
    >>> state = {"seed": 42}
    >>> state_with_meta = _add_metadata(state)
    >>> print("__version__" in state_with_meta)
    True
    """
    state.setdefault("__version__", SERIALIZATION_VERSION)
    state.setdefault("metadata", {})
    state["metadata"].update({
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    })
    return state


# ===========================================================================
# KissSeedSequence - NumPy SeedSequence Compatible
# ===========================================================================

class KissSeedSequence:
    """
    Seed sequence compatible with numpy.random.SeedSequence.

    Parameters
    ----------
    entropy : {None, int, sequence of int}, optional
        Entropy source. If None, uses OS randomness.
    spawn_key : tuple of int, default=()
        Spawn key for child sequences
    pool_size : int, default=4
        Pool size (for NumPy compatibility, not used)
    n_children_spawned : int, default=0
        Number of children spawned

    Attributes
    ----------
    entropy : int or None
        Initial entropy value
    spawn_key : tuple of int
        Spawn key tuple
    pool_size : int
        Pool size
    n_children_spawned : int
        Count of spawned children

    Raises
    ------
    ValueError
        If pool_size < 1 or n_children_spawned < 0
    TypeError
        If entropy has invalid type

    See Also
    --------
    PyKiss32Random : 32-bit version for smaller datasets
    PyKiss64Random : 64-bit version for larger datasets
    PyKissRandom : Factory function for auto-detecting
    KissBitGenerator : NumPy-compatible bit generator
    KissGenerator : High-level generator using this BitGenerator
    KissRandomState inherites from KissGenerator
    default_rng : Convenience function to create generator
    numpy.random.SeedSequence : NumPy's seed sequence implementation

    Notes
    -----
    Simplified implementation of NumPy SeedSequence protocol.
    Compatible but uses simpler mixing algorithm.

    References
    ----------
    .. [1] O'Neill, M.E. (2015). "PCG: A Family of Simple Fast Space-Efficient
           Statistically Good Algorithms for Random Number Generation."
           https://www.pcg-random.org/
    .. [2] NumPy Enhancement Proposal 19: Random Number Generator Policy
           https://numpy.org/neps/nep-0019-rng-policy.html

    Examples
    --------
    >>> seq = KissSeedSequence(42)
    >>> state = seq.generate_state(4, dtype=np.uint32)
    >>> children = seq.spawn(2)
    >>>
    >>> # Serialization
    >>> import pickle
    >>> restored = pickle.loads(pickle.dumps(seq))
    """

    # Class attribute CYTHON_USE_TYPE_SPECS=1
    # __module__ = "scikitplot.cexternals._annoy._kissrandom.kissrandom"
    __module__ = "scikitplot.random"

    def __init__(
        self,
        entropy: int | Sequence[int] | None = None,
        *,
        spawn_key: Sequence[int] = (),
        pool_size: int = 4,
        n_children_spawned: int = 0
    ) -> None:
        """Initialize seed sequence."""
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}")
        if n_children_spawned < 0:
            raise ValueError(
                f"n_children_spawned must be >= 0, got {n_children_spawned}"
            )

        self.spawn_key = tuple(spawn_key)
        self.pool_size = pool_size
        self.n_children_spawned = n_children_spawned

        # Initialize entropy
        if entropy is None:
            # Use OS entropy (64-bit random value)
            # os.urandom(pool_size * 4)y  # (8 bytes = 64 bits)
            self.entropy = secrets.randbits(64)
        elif isinstance(entropy, int):
            if entropy < 0:
                raise ValueError(f"entropy must be non-negative, got {entropy}")
            # Mask to 64 bits to prevent overflow later
            self.entropy = entropy & 0xFFFFFFFFFFFFFFFF
        elif isinstance(entropy, (list, tuple, Sequence)):
            if not entropy:
                raise ValueError("entropy sequence cannot be empty")
            # Mix sequence into single value
            self.entropy = self._mix_sequence(entropy)
        else:
            raise TypeError(f"entropy must be int, sequence, or None; got {type(entropy)}")

    def _mix_sequence(self, seq: Sequence[int]) -> int:
        """
        Mix sequence of integers into single 64-bit value.

        Parameters
        ----------
        seq : sequence of int
            Values to mix

        Returns
        -------
        int
            Mixed entropy value

        Notes
        -----
        Uses multiplicative congruential mixing with rotation. This is
        simpler than SipHash but sufficient for KISS initialization.
        """
        result = 0
        multiplier = 6364136223846793005

        for i, val in enumerate(seq):
            if not isinstance(val, int) or val < 0:
                raise ValueError(f"all entropy values must be non-negative int, got {val}")
            # Mask to 64 bits
            value = val & 0xFFFFFFFFFFFFFFFF
            # Mix with rotation
            result ^= (value << (i % 64))
            result = (result * multiplier + 1) & 0xFFFFFFFFFFFFFFFF

        return result

    def generate_state(
        self,
        n_words: int,
        dtype: DTypeLike = np.uint32
    ) -> NDArray[np.uint32 | np.uint64]:
        """
        Generate state array for RNG initialization.

        Parameters
        ----------
        n_words : int
            Number of words to generate
        dtype : dtype-like, default=np.uint32
            Output data type (uint32 or uint64)

        Returns
        -------
        ndarray
            State array of requested dtype

        Raises
        ------
        ValueError
            If n_words < 1
            If dtype is not uint32 or uint64

        Notes
        -----
        Generates deterministic state from entropy and spawn_key.
        Uses multiplicative congruential generator for mixing.

        Examples
        --------
        >>> seq = KissSeedSequence(42)
        >>> state = seq.generate_state(4, dtype=np.uint32)
        >>> print(state.shape)
        (4,)
        """
        if n_words < 1:
            raise ValueError(f"n_words must be >= 1, got {n_words}")

        # Validate dtype
        # dtype = np.dtype(dtype)
        # if dtype not in (np.dtype(np.uint32), np.dtype(np.uint64)):
        #     raise ValueError(f"dtype must be uint32 or uint64, got {dtype}")
        dtype = _validate_dtype(
            dtype,
            {np.dtype(np.uint32), np.dtype(np.uint64)}
        )

        # Use Python int arithmetic (prevents overflow)
        state = self.entropy
        multiplier = 6364136223846793005

        # Mix spawn key (pure Python to avoid overflow)
        for key in self.spawn_key:
            state ^= (key & 0xFFFFFFFF)
            state = (state * multiplier + 1) & 0xFFFFFFFFFFFFFFFF

        if dtype == np.uint64:
            # Generate uint64 values
            pool_size_needed = n_words * 2
            pool32 = np.zeros(pool_size_needed, dtype=np.uint32)

            for i in range(pool_size_needed):
                state = (state * multiplier + 1) & 0xFFFFFFFFFFFFFFFF
                pool32[i] = (state >> 32) & 0xFFFFFFFF

            # Combine pairs
            result64 = np.zeros(n_words, dtype=np.uint64)
            for i in range(n_words):
                result64[i] = (
                    (np.uint64(pool32[2*i + 1]) << 32) |
                    np.uint64(pool32[2*i])
                )
            return result64
        else:
            # Generate uint32 values
            pool32 = np.zeros(n_words, dtype=np.uint32)
            for i in range(n_words):
                state = (state * multiplier + 1) & 0xFFFFFFFFFFFFFFFF
                pool32[i] = (state >> 32) & 0xFFFFFFFF
            return pool32

    def spawn(self, n_children: int) -> list:
        """
        Create independent child seed sequences.

        Parameters
        ----------
        n_children : int
            Number of children to spawn

        Returns
        -------
        list of KissSeedSequence
            Independent child sequences

        Raises
        ------
        ValueError
            If n_children < 1

        Notes
        -----
        Each child has a unique spawn_key derived from the parent's spawn_key
        and its position in the child list. This ensures statistical
        independence even with identical entropy.

        The spawn tree structure allows for hierarchical parallelization:
        master → workers → tasks, where each level is independent.

        Examples
        --------
        >>> seq = KissSeedSequence(42)
        >>> children = seq.spawn(3)
        >>> print(len(children))
        3
        """
        if n_children < 1:
            raise ValueError(f"n_children must be >= 1, got {n_children}")

        children = []
        for i in range(n_children):
            child_spawn_key = self.spawn_key + (self.n_children_spawned + i,)
            child = KissSeedSequence(
                entropy=self.entropy,
                spawn_key=child_spawn_key,
                pool_size=self.pool_size,
                n_children_spawned=0
            )
            children.append(child)

        self.n_children_spawned += n_children
        return children

    # ===================================================================
    # Serialization Support
    # ===================================================================

    def __getstate__(self) -> dict:
        """
        Return state for pickling.

        Returns
        -------
        dict
            State dictionary

        Examples
        --------
        >>> seq = KissSeedSequence(42)
        >>> state = seq.__getstate__()
        >>> print("entropy" in state)
        True
        """
        return self.get_state()

    def __setstate__(self, state: dict) -> None:
        """
        Restore state from pickle.

        Parameters
        ----------
        state : dict
            State from __getstate__

        Examples
        --------
        >>> seq = KissSeedSequence(42)
        >>> state = seq.__getstate__()
        >>> new_seq = KissSeedSequence.__new__(KissSeedSequence)
        >>> new_seq.__setstate__(state)
        """
        _validate_state_dict(state, {"entropy"})
        self.set_state(state)

    def __reduce__(self) -> tuple:
        """
        Return tuple for pickle reconstruction.

        Returns
        -------
        tuple
            (callable, args, state)

        Examples
        --------
        >>> import pickle
        >>> seq = KissSeedSequence(42)
        >>> restored = pickle.loads(pickle.dumps(seq))
        """
        return (
            self.__class__,
            (),
            self.__getstate__(),
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Protocol-specific pickle reduction."""
        return self.__reduce__()

    def get_state(self) -> dict:
        """
        Get current state as dictionary.

        Returns
        -------
        dict
            Complete state dictionary

        Examples
        --------
        >>> seq = KissSeedSequence(42)
        >>> state = seq.get_state()
        >>> print(state["entropy"])
        42
        """
        return {
            "entropy": int(self.entropy),
            "spawn_key": list(self.spawn_key),
            "pool_size": self.pool_size,
            "n_children_spawned": self.n_children_spawned,
            "__version__": SERIALIZATION_VERSION,
        }

    def set_state(self, state: dict) -> None:
        """
        Set state from dictionary.

        Parameters
        ----------
        state : dict
            State from get_state()

        Raises
        ------
        ValueError
            If state is invalid

        Examples
        --------
        >>> seq1 = KissSeedSequence(42)
        >>> state = seq1.get_state()
        >>> seq2 = KissSeedSequence(0)
        >>> seq2.set_state(state)
        >>> print(seq2.entropy == seq1.entropy)
        True
        """
        _validate_state_dict(state, {"entropy"})
        self.entropy = int(state["entropy"]) & 0xFFFFFFFFFFFFFFFF
        self.spawn_key = tuple(state.get("spawn_key", ()))
        self.pool_size = state.get("pool_size", 4)
        self.n_children_spawned = state.get("n_children_spawned", 0)

    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters (sklearn-style).

        Parameters
        ----------
        deep : bool, default=True
            If True, return params for nested objects

        Returns
        -------
        dict
            Parameter dictionary

        Examples
        --------
        >>> seq = KissSeedSequence(42)
        >>> params = seq.get_params()
        >>> print(params["entropy"])
        42
        """
        return {
            "entropy": int(self.entropy),
            "spawn_key": self.spawn_key,
            "pool_size": self.pool_size,
            "n_children_spawned": self.n_children_spawned,
        }

    def set_params(self, **params):
        """
        Set parameters (sklearn-style).

        Parameters
        ----------
        **params : dict
            Parameters to set

        Returns
        -------
        self
            For method chaining

        Examples
        --------
        >>> seq = KissSeedSequence(42)
        >>> seq.set_params(entropy=123)
        >>> print(seq.entropy)
        123
        """
        for key, value in params.items():
            if key == "entropy":
                self.entropy = int(value) & 0xFFFFFFFFFFFFFFFF
            elif key == "spawn_key":
                self.spawn_key = tuple(value)
            elif key == "pool_size":
                self.pool_size = value
            elif key == "n_children_spawned":
                self.n_children_spawned = value
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self

    def serialize(self) -> dict:
        """
        Serialize to JSON-compatible dict.

        Returns
        -------
        dict
            JSON-serializable state

        Examples
        --------
        >>> import json
        >>> seq = KissSeedSequence(42)
        >>> data = seq.serialize()
        >>> json_str = json.dumps(data)
        """
        state = self.get_state()
        state["__class__"] = self.__class__.__name__
        state["__module__"] = self.__module__
        return _add_metadata(state)

    @classmethod
    def deserialize(cls, data: dict):
        """
        Deserialize from dictionary.

        Parameters
        ----------
        data : dict
            Serialized state from serialize()

        Returns
        -------
        KissSeedSequence
            Restored instance

        Examples
        --------
        >>> import json
        >>> seq = KissSeedSequence(42)
        >>> json_str = json.dumps(seq.serialize())
        >>> data = json.loads(json_str)
        >>> restored = KissSeedSequence.deserialize(data)
        """
        _validate_state_dict(data, {"entropy"})
        instance = cls.__new__(cls)
        instance.set_state(data)
        return instance

    def to_dict(self) -> dict:
        """Alias for serialize()."""
        return self.serialize()

    @classmethod
    def from_dict(cls, data: dict):
        """Alias for deserialize()."""
        return cls.deserialize(data)

    @property
    def state(self) -> dict[str, Any]:
        """
        Get current state as dictionary.

        Returns
        -------
        dict
            State dictionary with keys:
            - 'entropy': int or None
            - 'spawn_key': tuple of int
            - 'pool_size': int
            - 'n_children_spawned': int

        Notes
        -----
        State can be used for serialization and restoration.

        Examples
        --------
        >>> seq = KissSeedSequence(42)
        >>> state = seq.state
        >>> # Restore with: KissSeedSequence(**state)
        """
        return {
            "entropy": self.entropy,
            "spawn_key": self.spawn_key,
            "pool_size": self.pool_size,
            "n_children_spawned": self.n_children_spawned
        }

    def __repr__(self) -> str:
        return f"{self} at 0x{id(self):X}"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"


# ===========================================================================
# Low-Level KISS Random (Legacy API with Context Manager Support)
# ===========================================================================
# ===========================================================================
# PyKiss32Random - Legacy 32-bit wrapper
# ===========================================================================

cdef class PyKiss32Random:
    """
    32-bit KISS RNG with complete serialization support.

    Period: ~2^121 (suitable for <16M data points)

    Parameters
    ----------
    seed : int or None, optional
        Random seed. If None, uses default seed.

    Attributes
    ----------
    default_seed : int
        Default seed value (123456789)
    seed : int
        Current seed (property)
    lock : threading.RLock
        Thread lock (for shared access)

    See Also
    --------
    PyKiss64Random : 64-bit version for larger datasets
    PyKissRandom : Factory function for auto-detecting
    KissSeedSequence : Seed sequence for initialization
    KissBitGenerator : NumPy-compatible bit generator
    KissGenerator : High-level generator using this BitGenerator
    KissRandomState inherites from KissGenerator
    default_rng : Convenience function to create generator

    Notes
    -----
    - Period: approximately 2^121
    - Not cryptographically secure
    - Suitable for up to ~2^24 data points
    - For larger datasets, use PyKiss64Random
    - Thread-safe via context manager
    - Deterministic: same seed → same sequence
    - Complete pickle/JSON support

    The KISS32 algorithm combines:
    - Linear Congruential Generator (LCG)
    - Xorshift generator
    - Multiply-With-Carry (MWC) generator

    References
    ----------
    .. [1] Marsaglia, G. (1999). "Random Number Generators."

    Examples
    --------
    >>> rng = PyKiss32Random(42)
    >>> rng.kiss()  # Random uint32
    >>>
    >>> # Context manager (thread-safe)
    >>> with rng:
    ...     value = rng.kiss()
    >>>
    >>> # Serialization
    >>> import pickle
    >>> restored = pickle.loads(pickle.dumps(rng))
    """

    # Class attribute
    # __module__ = "scikitplot.cexternals._annoy._kissrandom.kissrandom"
    __module__ = "scikitplot.random"
    default_seed = KISS32_DEFAULT_SEED

    # C++ object (stored as pointer)
    cdef kr.Kiss32Random* _rng
    cdef uint32_t _seed
    cdef readonly object lock

    def __cinit__(self, seed: int | None = None):
        """Allocate C++ RNG."""
        cdef uint32_t cseed

        if seed is None:
            cseed = KISS32_DEFAULT_SEED
        else:
            cseed = _safe_to_uint32(seed)

        self._seed = kr.Kiss32Random.normalize_seed(cseed)
        self._rng = new kr.Kiss32Random(self._seed)
        if self._rng is NULL:
            raise MemoryError("Failed to allocate Kiss32Random")

    def __dealloc__(self):
        """C-level destructor (frees C++ object)."""
        if self._rng is not NULL:
            del self._rng
            self._rng = NULL

    def __init__(self, seed: int | None = None):
        """
        Python-level constructor.

        Parameters
        ----------
        seed : int or None
            Initial seed value
        """
        # Initialize lock in __cinit__ to avoid infinite loop
        self.lock = threading.RLock()

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
        return f"{self} at 0x{id(self):X}"

    def __str__(self) -> str:
        """
        Return user-friendly string representation.

        Returns
        -------
        str
            String representation
        """
        return f"{self.__class__.__name__}"

    def __enter__(self):
        """Enter context manager (acquire lock)."""
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager (release lock)."""
        self.lock.release()
        return False

    @property
    def seed(self) -> int:
        """
        Get current seed value.

        Returns
        -------
        int
            Current seed

        Examples
        --------
        >>> rng = PyKiss32Random(42)
        >>> rng.seed
        42
        """
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        """
        Set new seed and reinitialize generator.

        Parameters
        ----------
        value : int
            New seed value

        Raises
        ------
        ValueError
            If seed out of range
        TypeError
            If seed not an integer

        Examples
        --------
        >>> rng = PyKiss32Random()
        >>> rng.seed = 42
        >>> rng.seed
        42
        """
        cdef uint32_t cseed = _safe_to_uint32(value)
        self._seed = kr.Kiss32Random.normalize_seed(cseed)
        with self.lock:
            self._rng.reset(self._seed)

    @staticmethod
    def get_default_seed() -> int:
        """
        Get default seed value.

        Returns
        -------
        int
            Default seed (123456789)

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
            User-provided seed

        Returns
        -------
        int
            Normalized seed (original if non-zero, else default_seed)

        Notes
        -----
        Maps seed==0 to default_seed to avoid degenerate RNG states.

        Examples
        --------
        >>> PyKiss32Random.normalize_seed(42)
        42
        >>> PyKiss32Random.normalize_seed(0)
        123456789
        """
        if not isinstance(seed, int):
            raise TypeError(f"seed must be int, got {type(seed)}")
        if seed < 0 or seed > UINT32_MAX:
            raise ValueError(
                f"seed must be in [0, {UINT32_MAX}], got {seed}"
            )
        return kr.Kiss32Random.normalize_seed(<uint32_t>seed)

    def reset(self, seed: int) -> None:
        """
        Reset RNG state with new seed.

        Parameters
        ----------
        seed : int
            New seed value

        Raises
        ------
        ValueError
            If seed out of range

        Notes
        -----
        Fully resets all internal state variables.

        Examples
        --------
        >>> rng = PyKiss32Random(42)
        >>> values1 = [rng.kiss() for _ in range(5)]
        >>> rng.reset(42)
        >>> values2 = [rng.kiss() for _ in range(5)]
        >>> values1 == values2
        True
        """
        if not isinstance(seed, int):
            raise TypeError(f"seed must be int, got {type(seed)}")
        if seed < 0 or seed > UINT32_MAX:
            raise ValueError(
                f"seed must be in [0, {UINT32_MAX}], got {seed}"
            )

        cdef uint32_t cseed = kr.Kiss32Random.normalize_seed(<uint32_t>seed)
        self._seed = cseed
        with self.lock:
            self._rng.reset(cseed)

    def reset_default(self) -> None:
        """
        Reset to default seed.

        Equivalent to reset(default_seed).

        Examples
        --------
        >>> rng = PyKiss32Random()
        >>> rng.reset_default()
        >>> rng.seed == PyKiss32Random.default_seed
        True
        """
        self.reset(self.default_seed)

    def set_seed(self, seed: int) -> None:
        """
        Set new seed (alias for reset).

        Parameters
        ----------
        seed : int
            New seed value

        Examples
        --------
        >>> rng = PyKiss32Random()
        >>> rng.set_seed(42)
        """
        self.reset(seed)

    # def kiss(self) -> int:
    cpdef uint32_t kiss(self):
        """
        Generate next random 32-bit unsigned integer.

        Returns
        -------
        int
            Random value in [0, 2^32-1]

        Notes
        -----
        This is the core RNG method. Other methods (flip, index) build on it.

        Examples
        --------
        >>> rng = PyKiss32Random(42)
        >>> value = rng.kiss()
        >>> 0 <= value < 2**32
        True
        """
        with self.lock:
            return self._rng.kiss()

    # def flip(self) -> int:
    cpdef int flip(self):
        """
        Generate random binary value (0 or 1).

        Returns
        -------
        int
            Either 0 or 1 with equal probability

        Examples
        --------
        >>> rng = PyKiss32Random(42)
        >>> rng.flip() in {0, 1}
        True

        Coin flip simulation:

        >>> rng = PyKiss32Random(123)
        >>> flips = [rng.flip() for _ in range(1000)]
        >>> abs(sum(flips) - 500) < 50  # Approximately 50% heads
        True
        """
        with self.lock:
            return self._rng.flip()

    # def index(self, n: int) -> int:
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

        Raises
        ------
        ValueError
            If n < 0
        TypeError
            If n not an integer

        Notes
        -----
        - Handles n==0 gracefully (returns 0)
        - Uses modulo for simplicity (suitable for non-crypto use)
        - Slight modulo bias exists but negligible for non-crypto applications

        Examples
        --------
        >>> rng = PyKiss32Random(42)
        >>> idx = rng.index(100)
        >>> 0 <= idx < 100
        True

        Array indexing:

        >>> import numpy as np
        >>> arr = np.arange(100, 200)
        >>> rng = PyKiss32Random(42)
        >>> random_element = arr[rng.index(len(arr))]
        """
        with self.lock:
            return self._rng.index(<size_t>n)

    # ===================================================================
    # Serialization Support
    # ===================================================================

    def __getstate__(self):
        """Return state for pickling."""
        return self.get_state()

    def __setstate__(self, state):
        """Restore state from pickle."""
        _validate_state_dict(state, {"seed"})
        self.set_state(state)

    def __reduce__(self):
        """Custom pickle protocol."""
        return (
            self.__class__,
            (int(self._seed),),  # Constructor args
            None,  # No additional state needed
        )

    def __reduce_ex__(self, protocol):
        """Protocol-specific pickle."""
        return self.__reduce__()

    def get_state(self):
        """
        Get state dictionary.

        Returns
        -------
        dict
            Complete state

        Examples
        --------
        >>> rng = PyKiss32Random(42)
        >>> state = rng.get_state()
        >>> print(state["seed"])
        42
        """
        return {
            "seed": int(self._seed),
            "bit_width": 32,
            "__version__": SERIALIZATION_VERSION,
        }

    def set_state(self, state):
        """
        Set state from dictionary.

        Parameters
        ----------
        state : dict
            State from get_state()

        Examples
        --------
        >>> rng1 = PyKiss32Random(42)
        >>> state = rng1.get_state()
        >>> rng2 = PyKiss32Random(0)
        >>> rng2.set_state(state)
        """
        _validate_state_dict(state, {"seed"})
        self.seed = int(state["seed"])

    def get_params(self, deep=True):
        """
        Get parameters (sklearn-style).

        Parameters
        ----------
        deep : bool, default=True
            Unused, for sklearn compatibility

        Returns
        -------
        dict
            Constructor parameters

        Examples
        --------
        >>> rng = PyKiss32Random(42)
        >>> params = rng.get_params()
        >>> print(params)
        {'seed': 42}
        """
        return {"seed": int(self._seed)}

    def set_params(self, **params):
        """
        Set parameters (sklearn-style).

        Parameters
        ----------
        **params : dict
            Parameters to set

        Returns
        -------
        self
            For chaining

        Examples
        --------
        >>> rng = PyKiss32Random(42)
        >>> rng.set_params(seed=123)
        >>> print(rng.seed)
        123
        """
        if "seed" in params:
            self.seed = int(params["seed"])
        else:
            for key in params:
                raise ValueError(f"Invalid parameter: {key}")
        return self

    def serialize(self):
        """
        Serialize to JSON-compatible dict.

        Returns
        -------
        dict
            JSON-serializable state

        Examples
        --------
        >>> import json
        >>> rng = PyKiss32Random(42)
        >>> data = rng.serialize()
        >>> json_str = json.dumps(data)
        """
        state = self.get_state()
        state["__class__"] = "PyKiss32Random"
        state["__module__"] = self.__module__
        return _add_metadata(state)

    @classmethod
    def deserialize(cls, data):
        """
        Deserialize from dict.

        Parameters
        ----------
        data : dict
            Serialized state

        Returns
        -------
        PyKiss32Random
            Restored instance

        Examples
        --------
        >>> import json
        >>> rng = PyKiss32Random(42)
        >>> json_str = json.dumps(rng.serialize())
        >>> data = json.loads(json_str)
        >>> restored = PyKiss32Random.deserialize(data)
        """
        _validate_state_dict(data, {"seed"})
        return cls(seed=int(data["seed"]))

    def to_dict(self):
        """Alias for serialize()."""
        return self.serialize()

    @classmethod
    def from_dict(cls, data):
        """Alias for deserialize()."""
        return cls.deserialize(data)


# ===========================================================================
# PyKiss64Random - Legacy 64-bit wrapper
# ===========================================================================

cdef class PyKiss64Random:
    """
    Low-level 64-bit KISS RNG with context manager support.

    This class provides direct access to the C++ Kiss64Random implementation.
    For most use cases, prefer KissGenerator or default_rng() instead.
    Period: ~2^250 (suitable for billions of data points)

    Parameters
    ----------
    seed : int or None, default=None
        Initial seed value. If None, uses default seed (1234567890987654321).
        Must be in range [0, 2^64-1].

    Attributes
    ----------
    default_seed : int
        Default seed value (1234567890987654321)
    seed : int
        Current seed value (property, get/set)
    lock : threading.RLock
        Thread lock (for shared access)

    See Also
    --------
    PyKiss32Random : 32-bit version for smaller datasets
    PyKissRandom : Factory function for auto-detecting
    KissSeedSequence : Seed sequence for initialization
    KissBitGenerator : NumPy-compatible bit generator
    KissGenerator : High-level generator using this BitGenerator
    KissRandomState inherites from KissGenerator
    default_rng : Convenience function to create generator

    Notes
    -----
    - Period: approximately 2^250
    - Not cryptographically secure
    - Recommended for datasets larger than ~16 million points
    - Slightly slower than PyKiss32Random but much longer period
    - Thread-safe via context manager
    - Deterministic: same seed → same sequence
    - Complete pickle/JSON support
    - Preferred for large-scale applications

    Examples
    --------
    >>> rng = PyKiss64Random(42)
    >>> rng.kiss()  # Random uint64
    >>>
    >>> # Context manager (thread-safe)
    >>> with rng:
    ...     value = rng.kiss()
    >>>
    >>> # Serialization
    >>> import pickle
    >>> restored = pickle.loads(pickle.dumps(rng))
    """

    # Class attribute
    # __module__ = "scikitplot.cexternals._annoy._kissrandom.kissrandom"
    __module__ = "scikitplot.random"
    default_seed = KISS64_DEFAULT_SEED

    # C++ object (stored as pointer)
    cdef kr.Kiss64Random* _rng
    cdef uint64_t _seed
    cdef readonly object lock

    def __cinit__(self, seed: int | None = None):
        """Allocate C++ RNG. C-level constructor."""
        cdef uint64_t cseed

        if seed is None:
            cseed = KISS32_DEFAULT_SEED
        else:
            cseed = _safe_to_uint64(seed)

        self._seed = kr.Kiss64Random.normalize_seed(cseed)
        self._rng = new kr.Kiss64Random(self._seed)
        if self._rng is NULL:
            raise MemoryError("Failed to allocate Kiss64Random")

    def __dealloc__(self):
        """Free C++ RNG. C-level destructor."""
        if self._rng is not NULL:
            del self._rng
            self._rng = NULL

    def __init__(self, seed: int | None = None):
        """Python-level constructor."""
        # Initialize lock in __cinit__ to avoid infinite loop
        self.lock = threading.RLock()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self} at 0x{id(self):X}"

    def __str__(self) -> str:
        """Return user-friendly string representation."""
        return f"{self.__class__.__name__}"

    def __enter__(self):
        """Enter context manager (acquire lock)."""
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager (release lock)."""
        self.lock.release()
        return False

    @property
    def seed(self) -> int:
        """Get current seed value."""
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        """Set new seed and reinitialize generator."""
        cdef uint64_t cseed = _safe_to_uint64(value)
        self._seed = kr.Kiss64Random.normalize_seed(cseed)
        with self.lock:
            self._rng.reset(self._seed)

    @staticmethod
    def get_default_seed() -> int:
        """Get default seed value."""
        return kr.Kiss64Random.get_default_seed()

    @staticmethod
    def normalize_seed(seed: int) -> int:
        """Normalize seed to valid non-zero value."""
        if not isinstance(seed, int):
            raise TypeError(f"seed must be int, got {type(seed)}")
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        return kr.Kiss64Random.normalize_seed(<uint64_t>(seed & UINT64_MAX))

    def reset(self, seed: int) -> None:
        """Reset RNG state with new seed."""
        if not isinstance(seed, int):
            raise TypeError(f"seed must be int, got {type(seed)}")
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")

        cdef uint64_t cseed = kr.Kiss64Random.normalize_seed(
            <uint64_t>(seed & UINT64_MAX)
        )
        self._seed = cseed
        with self.lock:
            self._rng.reset(cseed)

    def reset_default(self) -> None:
        """Reset to default seed."""
        self.reset(self.default_seed)

    def set_seed(self, seed: int) -> None:
        """Set new seed (alias for reset)."""
        self.reset(seed)

    # def kiss(self) -> int:
    cpdef uint64_t kiss(self):
        """
        Generate next random 64-bit unsigned integer.

        Returns
        -------
        int
            Random value in [0, 2^64-1]
        """
        with self.lock:
            return self._rng.kiss()

    # def flip(self) -> int:
    cpdef int flip(self):
        """Generate random binary value (0 or 1)."""
        with self.lock:
            return self._rng.flip()

    # def index(self, n: int) -> int:
    cpdef size_t index(self, size_t n):
        """Generate random index in range [0, n-1]."""
        with self.lock:
            return self._rng.index(<size_t>n)

    # ===================================================================
    # Serialization Support
    # ===================================================================

    def __getstate__(self):
        """Return state for pickling."""
        return self.get_state()

    def __setstate__(self, state):
        """Restore state from pickle."""
        _validate_state_dict(state, {"seed"})
        self.set_state(state)

    def __reduce__(self):
        """Custom pickle protocol."""
        return (
            self.__class__,
            (int(self._seed),),  # Constructor args
            None,  # No additional state needed
        )

    def __reduce_ex__(self, protocol):
        """Protocol-specific pickle."""
        return self.__reduce__()

    def get_state(self):
        """
        Get state dictionary.

        Returns
        -------
        dict
            Complete state

        Examples
        --------
        >>> rng = PyKiss64Random(42)
        >>> state = rng.get_state()
        >>> print(state["seed"])
        42
        """
        return {
            "seed": int(self._seed),
            "bit_width": 64,
            "__version__": SERIALIZATION_VERSION,
        }

    def set_state(self, state):
        """
        Set state from dictionary.

        Parameters
        ----------
        state : dict
            State from get_state()

        Examples
        --------
        >>> rng1 = PyKiss64Random(42)
        >>> state = rng1.get_state()
        >>> rng2 = PyKiss64Random(0)
        >>> rng2.set_state(state)
        """
        _validate_state_dict(state, {"seed"})
        self.seed = int(state["seed"])

    def get_params(self, deep=True):
        """
        Get parameters (sklearn-style).

        Parameters
        ----------
        deep : bool, default=True
            Unused, for sklearn compatibility

        Returns
        -------
        dict
            Constructor parameters

        Examples
        --------
        >>> rng = PyKiss64Random(42)
        >>> params = rng.get_params()
        >>> print(params)
        {'seed': 42}
        """
        return {"seed": int(self._seed)}

    def set_params(self, **params):
        """
        Set parameters (sklearn-style).

        Parameters
        ----------
        **params : dict
            Parameters to set

        Returns
        -------
        self
            For chaining

        Examples
        --------
        >>> rng = PyKiss64Random(42)
        >>> rng.set_params(seed=123)
        >>> print(rng.seed)
        123
        """
        if "seed" in params:
            self.seed = int(params["seed"])
        else:
            for key in params:
                raise ValueError(f"Invalid parameter: {key}")
        return self

    def serialize(self):
        """
        Serialize to JSON-compatible dict.

        Returns
        -------
        dict
            JSON-serializable state

        Examples
        --------
        >>> import json
        >>> rng = PyKiss64Random(42)
        >>> data = rng.serialize()
        >>> json_str = json.dumps(data)
        """
        state = self.get_state()
        state["__class__"] = "PyKiss64Random"
        state["__module__"] = self.__module__
        return _add_metadata(state)

    @classmethod
    def deserialize(cls, data):
        """
        Deserialize from dict.

        Parameters
        ----------
        data : dict
            Serialized state

        Returns
        -------
        PyKiss64Random
            Restored instance

        Examples
        --------
        >>> import json
        >>> rng = PyKiss64Random(42)
        >>> json_str = json.dumps(rng.serialize())
        >>> data = json.loads(json_str)
        >>> restored = PyKiss64Random.deserialize(data)
        """
        _validate_state_dict(data, {"seed"})
        return cls(seed=int(data["seed"]))

    def to_dict(self):
        """Alias for serialize()."""
        return self.serialize()

    @classmethod
    def from_dict(cls, data):
        """Alias for deserialize()."""
        return cls.deserialize(data)

# ===========================================================================
# Auto-detecting PyKissRandom (NEW!)
# ===========================================================================

def PyKissRandom(seed=None, bit_width=None):
    """
    Factory function for auto-detecting 32-bit vs 64-bit RNG.

    Parameters
    ----------
    seed : int or None, optional
        Random seed
    bit_width : {None, 'auto', 32, 64}, default=None
        Bit width selection:
        - None or 'auto': Auto-detect based on system
        - 32: Force 32-bit
        - 64: Force 64-bit

    Returns
    -------
    PyKiss32Random or PyKiss64Random
        RNG instance

    See Also
    --------
    PyKiss32Random : 32-bit version for smaller datasets
    PyKiss64Random : 64-bit version for larger datasets
    KissSeedSequence : Seed sequence for initialization
    KissBitGenerator : NumPy-compatible bit generator
    KissGenerator : High-level generator using this BitGenerator
    KissRandomState inherites from KissGenerator
    default_rng : Convenience function to create generator

    Examples
    --------
    >>> rng = PyKissRandom(42)  # Auto-detect
    >>> rng = PyKissRandom(42, bit_width=32)  # Force 32-bit
    >>> rng = PyKissRandom(42, bit_width=64)  # Force 64-bit
    >>> rng = PyKissRandom(42, bit_width=None)  # Auto-detect
    """
    if bit_width is None or bit_width == "auto":
        bit_width = _detect_optimal_bit_width()
    elif bit_width not in (32, 64):
        raise ValueError(f"bit_width must be None, 'auto', 32, or 64; got {bit_width}")

    if bit_width == 32:
        return PyKiss32Random(seed)
    else:
        return PyKiss64Random(seed)

# ===========================================================================
# C Type, Callback Functions for NumPy Generator Interface
# ===========================================================================


cdef uint64_t kiss64_next_uint64(void *st) noexcept nogil:
    """C callback for next_uint64 - called by NumPy Generator."""
    cdef kr.Kiss64Random *rng = <kr.Kiss64Random *>st
    return rng.kiss()

cdef uint32_t kiss64_next_uint32(void *st) noexcept nogil:
    """C callback for next_uint32 - called by NumPy Generator."""
    cdef kr.Kiss64Random *rng = <kr.Kiss64Random *>st
    cdef uint64_t val = rng.kiss()
    return <uint32_t>(val >> 32)

cdef double kiss64_next_double(void *st) noexcept nogil:
    """C callback for next_double - called by NumPy Generator."""
    cdef kr.Kiss64Random *rng = <kr.Kiss64Random *>st
    cdef uint64_t val = rng.kiss()
    return (<double>val) / (<double>UINT64_MAX)

cdef uint64_t kiss64_next_raw(void *st) noexcept nogil:
    """C callback for next_raw - called by NumPy Generator."""
    return kiss64_next_uint64(st)

# Similar callbacks for 32-bit
cdef uint64_t kiss32_next_uint64(void *st) noexcept nogil:
    """C callback for next_uint64 from 32-bit RNG."""
    cdef kr.Kiss32Random *rng = <kr.Kiss32Random *>st
    cdef uint64_t high = rng.kiss()
    cdef uint64_t low = rng.kiss()
    return (high << 32) | low

cdef uint32_t kiss32_next_uint32(void *st) noexcept nogil:
    """C callback for next_uint32 from 32-bit RNG."""
    cdef kr.Kiss32Random *rng = <kr.Kiss32Random *>st
    return rng.kiss()

cdef double kiss32_next_double(void *st) noexcept nogil:
    """C callback for next_double from 32-bit RNG."""
    return (<double>kiss32_next_uint32(st)) / (<double>UINT32_MAX)

cdef uint64_t kiss32_next_raw(void *st) noexcept nogil:
    """C callback for next_raw from 32-bit RNG."""
    return kiss32_next_uint64(st)

# ===========================================================================
# KissBitGenerator - NumPy-compatible BitGenerator Protocol
# ===========================================================================

cdef uint64_t kiss_random_raw(void* st) noexcept nogil:
    """
    C callback for NumPy BitGenerator protocol.

    This function is called by NumPy to generate raw random bits.
    """
    cdef kr.Kiss64Random* rng = <kr.Kiss64Random*>st
    return rng.kiss()

cdef class KissBitGenerator:
    """
    NumPy-compatible BitGenerator using KISS algorithm with complete serialization.

    Parameters
    ----------
    seed : {None, int, SeedSequence, KissSeedSequence}, optional, default=None
        Random seed or seed sequence:
        - int: Direct seed value
        - SeedSequence: Use its generated state
        - None: Use OS entropy via secrets.token_bytes()
    bit_width : int, default=None
        Generator bit width (32 or 64)
        - 32: Uses Kiss32Random (faster, period ~2^121)
        - 64: Uses Kiss64Random (slower, period ~2^250)

    Attributes
    ----------
    lock : threading.RLock
        Lock for thread-safe access (NumPy protocol requirement)
    seed_seq : KissSeedSequence
        Underlying seed sequence

    See Also
    --------
    PyKiss32Random : 32-bit version for smaller datasets
    PyKiss64Random : 64-bit version for larger datasets
    PyKissRandom : Factory function for auto-detecting
    KissSeedSequence : Seed sequence for initialization
    KissGenerator : High-level generator using this BitGenerator
    KissRandomState inherites from KissGenerator
    default_rng : Convenience function to create generator
    numpy.random.BitGenerator : NumPy's BitGenerator base class

    Notes
    -----
    The bit_width parameter determines internal generator:
    - 32: Uses Kiss32Random (faster, period ~2^121)
    - 64: Uses Kiss64Random (slower, period ~2^250)
    - NumPy BitGenerator protocol compatible
    - Thread-safe via lock
    - Complete pickle/JSON support

    For NumPy compatibility, random_raw() always returns uint64 values
    regardless of internal bit width.

    Examples
    --------
    >>> bg = KissBitGenerator(42)
    >>> bg.random_raw()
    >>>
    >>> # NOTE: Due to C API differences, use KissGenerator instead
    >>> # of wrapping in numpy.random.Generator
    >>> gen = KissGenerator(bg)
    >>> gen.random(10)
    >>>
    >>> # Serialization
    >>> import pickle
    >>> restored = pickle.loads(pickle.dumps(bg))
    """

    # Class attribute
    # __module__ = "scikitplot.cexternals._annoy._kissrandom.kissrandom"
    __module__ = "scikitplot.random"

    cdef kr.Kiss32Random* _rng32
    cdef kr.Kiss64Random* _rng
    cdef uint64_t _seed
    cdef readonly object lock
    cdef bitgen_t _bitgen
    cdef int _bit_width

    cdef public object seed_seq  # readonly

    def __cinit__(self):
        """Allocate C++ RNG."""
        self._rng = NULL  # Initialize to NULL first
        self._rng32 = NULL

    def __init__(
        self,
        seed: Optional[Union[int, KissSeedSequence]] = None,
        bit_width: int = None
    ):
        """Initialize the bit generator."""
        # Handle seed types (NumPy-compatible)
        if seed is None:
            # Use OS entropy
            self.seed_seq = KissSeedSequence()
        elif isinstance(seed, int):
            # Create KissSeedSequence from int
            self.seed_seq = KissSeedSequence(seed)
        elif isinstance(seed, KissSeedSequence):
            # Use directly
            self.seed_seq = seed
        elif hasattr(seed, "generate_state"):
            # NumPy SeedSequence or compatible
            self.seed_seq = KissSeedSequence(
                int(seed.entropy) if hasattr(seed, "entropy") else 0
            )
        else:
            raise TypeError(
                f"seed must be None, int, or SeedSequence; got {type(seed)}"
            )

        # Extract seed value (safely)
        cdef uint64_t cseed = _safe_to_uint64(self.seed_seq.entropy)
        self._seed = kr.Kiss64Random.normalize_seed(cseed)

        # Allocate C++ RNG
        self._rng = new kr.Kiss64Random(self._seed)
        if self._rng is NULL:
            raise MemoryError("Failed to allocate Kiss64Random")

        # Create lock
        self.lock = threading.RLock()

        # Setup bitgen_t for NumPy
        self._bitgen.state = <void*>self._rng
        self._bitgen.next_uint64 = &kiss_random_raw
        self._bitgen.next_uint32 = NULL
        self._bitgen.next_double = NULL
        self._bitgen.next_raw = &kiss_random_raw

        # Initialize RNG based on bit width
        # if self._bit_width == 32:
        #     state = self.seed_seq.generate_state(1, dtype=np.uint32)
        #     self._rng32 = new kr.Kiss32Random(<uint32_t>state[0])
        #     if self._rng32 == NULL:
        #         raise MemoryError("Failed to allocate Kiss32Random")
        #     # Setup bitgen_t structure for NumPy
        #     self._bitgen.state = <void *>self._rng32
        #     self._bitgen.next_uint64 = &kiss32_next_uint64
        #     self._bitgen.next_uint32 = &kiss32_next_uint32
        #     self._bitgen.next_double = &kiss32_next_double
        #     self._bitgen.next_raw = &kiss32_next_raw

    def __dealloc__(self):
        """Clean up C++ objects."""
        if self._rng != NULL:
            del self._rng
            self._rng = NULL
        if self._rng32 != NULL:
            del self._rng32
            self._rng32 = NULL

    def __repr__(self) -> str:
        return f"{self} at 0x{id(self):X}"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def __enter__(self):
        """Context manager entry."""
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.lock.release()
        return False

    def random_raw(self, size=None, output=True):
        """
        Generate random uint64 values.

        Parameters
        ----------
        size : int, tuple of ints, or None
            Output shape
        output : bool, default=True
            If True, return values. If False, just advance state.

        Returns
        -------
        int or ndarray
            Random uint64 values

        Examples
        --------
        >>> bg = KissBitGenerator(seed=42)
        >>> bg.random_raw()  # Single value
        <class 'int'>

        >>> bg.random_raw(10)  # 10 values
        >>> bg.random_raw((3, 4))  # 3x4 array
        """
        if size is None:
            if not output:
                with self.lock:
                    self._rng.kiss()
                return None

            with self.lock:
                return self._rng.kiss()

        # Normalize size
        if isinstance(size, int):
            size = (size,)
        elif not isinstance(size, tuple):
            size = tuple(size)

        # Calculate total
        n_total = int(np.prod(size))

        if not output:
            with self.lock:
                for _ in range(n_total):
                    self._rng.kiss()
            return None

        # Allocate and fill
        cdef cnp.ndarray[uint64_t, ndim=1] result = np.empty(n_total, dtype=np.uint64)
        cdef size_t i

        with self.lock:
            for i in range(n_total):
                result[i] = self._rng.kiss()

        return result.reshape(size)

    def spawn(self, n_children: int) -> list["KissBitGenerator"]:
        """
        Create independent child BitGenerators (NumPy protocol).

        Parameters
        ----------
        n_children : int
            Number of children to spawn

        Returns
        -------
        list of KissBitGenerator
            Independent bit generators

        Raises
        ------
        ValueError
            If n_children < 1

        Notes
        -----
        This is REQUIRED by NumPy's BitGenerator protocol.
        Each child is statistically independent.

        Examples
        --------
        >>> bg = KissBitGenerator(42)
        >>> children = bg.spawn(3)
        >>> print(len(children))
        3
        >>> # Use in parallel workers
        """
        child_seed_seqs = self.seed_seq.spawn(n_children)
        return [
            KissBitGenerator(seed=child_seq)
            for child_seq in child_seed_seqs
        ]

    @property
    def capsule(self):
        """Get PyCapsule for NumPy C API (protocol requirement)."""
        # Create capsule for NumPy - THIS IS CRITICAL
        # Without this, np.random.Generator(bg) will fail with AttributeError
        return PyCapsule_New(
            <void *>&self._bitgen,
            "BitGenerator",
            NULL
        )

    # @property
    # def seed_seq(self):
    #     """
    #     Seed sequence used for initialization.
    #
    #     Returns
    #     -------
    #     KissSeedSequence
    #         The seed sequence instance
    #
    #     Notes
    #     -----
    #     This is REQUIRED by NumPy's BitGenerator protocol.
    #     """
    #     return self.seed_seq

    @property
    def state(self) -> dict:
        """Get state dict (NumPy protocol)."""
        return self.get_state()

    @state.setter
    def state(self, value: dict):
        """Set state from dict (NumPy protocol)."""
        self.set_state(value)

    # ===================================================================
    # Serialization Support
    # ===================================================================

    def __getstate__(self):
        """Return state for pickling."""
        return self.get_state()

    def __setstate__(self, state):
        """Restore state from pickle."""
        _validate_state_dict(state, {"seed_sequence", "seed_sequence_state"})
        self.set_state(state)

    def __reduce__(self):
        """Custom pickle protocol."""
        return (
            self.__class__,
            (self.seed_seq,),  # Constructor args
            None,  # No additional state
        )

    def __reduce_ex__(self, protocol):
        """Protocol-specific pickle."""
        return self.__reduce__()

    def get_state(self):
        """
        Get state dictionary.

        Returns
        -------
        dict
            Complete state

        Examples
        --------
        >>> bg = KissBitGenerator(42)
        >>> state = bg.get_state()
        >>> print("seed_sequence" in state)
        True
        """
        return {
            "seed_sequence": f"{self.seed_seq.__class__.__name__}",
            "seed_sequence_state": {
                "seed": int(self._seed),
                "seed_seq": self.seed_seq.get_state(),
            },
            "__version__": SERIALIZATION_VERSION,
        }

    def set_state(self, state):
        """
        Set state from dictionary.

        Parameters
        ----------
        state : dict
            State from get_state()

        Examples
        --------
        >>> bg1 = KissBitGenerator(42)
        >>> state = bg1.get_state()
        >>> bg2 = KissBitGenerator(0)
        >>> bg2.set_state(state)
        """
        _validate_state_dict(state, {"seed_sequence", "seed_sequence_state"})

        inner_state = state["seed_sequence_state"]
        seed_value = int(inner_state["seed"])

        # Update seed_seq if present
        if "seed_seq" in inner_state:
            self.seed_seq.set_state(inner_state["seed_seq"])

        # Reset RNG
        cdef uint64_t cseed = _safe_to_uint64(seed_value)
        self._seed = kr.Kiss64Random.normalize_seed(cseed)
        with self.lock:
            self._rng.reset(self._seed)

    def get_params(self, deep=True):
        """
        Get parameters (sklearn-style).

        Parameters
        ----------
        deep : bool, default=True
            If True, include nested params

        Returns
        -------
        dict
            Parameters

        Examples
        --------
        >>> bg = KissBitGenerator(42)
        >>> params = bg.get_params()
        """
        params = {"seed": int(self._seed)}
        if deep and self.seed_seq is not None:
            params["seed_seq"] = self.seed_seq.get_params(deep=True)
        return params

    def set_params(self, **params):
        """
        Set parameters (sklearn-style).

        Parameters
        ----------
        **params : dict
            Parameters to set

        Returns
        -------
        self

        Examples
        --------
        >>> bg = KissBitGenerator(42)
        >>> bg.set_params(seed=123)
        """
        cdef uint64_t cseed

        if "seed" in params:
            seed_value = int(params["seed"])
            cseed = _safe_to_uint64(seed_value)
            self._seed = kr.Kiss64Random.normalize_seed(cseed)
            with self.lock:
                self._rng.reset(self._seed)

        if "seed_seq" in params and self.seed_seq is not None:
            self.seed_seq.set_params(**params["seed_seq"])

        return self

    def serialize(self):
        """
        Serialize to JSON-compatible dict.

        Returns
        -------
        dict
            JSON-serializable state

        Examples
        --------
        >>> import json
        >>> bg = KissBitGenerator(42)
        >>> data = bg.serialize()
        >>> json_str = json.dumps(data)
        """
        state = self.get_state()
        state["__class__"] = "KissBitGenerator"
        state["__module__"] = self.__module__
        return _add_metadata(state)

    @classmethod
    def deserialize(cls, data):
        """
        Deserialize from dict.

        Parameters
        ----------
        data : dict
            Serialized state

        Returns
        -------
        KissBitGenerator
            Restored instance

        Examples
        --------
        >>> import json
        >>> bg = KissBitGenerator(42)
        >>> json_str = json.dumps(bg.serialize())
        >>> data = json.loads(json_str)
        >>> restored = KissBitGenerator.deserialize(data)
        """
        _validate_state_dict(data, {"seed_sequence", "seed_sequence_state"})

        inner_state = data["seed_sequence_state"]
        seed_value = int(inner_state["seed"])

        # Reconstruct seed_seq if present
        seed_seq = None
        if "seed_seq" in inner_state:
            seed_seq = KissSeedSequence.deserialize(inner_state["seed_seq"])

        instance = cls(seed=seed_value)
        if seed_seq is not None:
            instance.seed_seq = seed_seq

        return instance

    def to_dict(self):
        """Alias for serialize()."""
        return self.serialize()

    @classmethod
    def from_dict(cls, data):
        """Alias for deserialize()."""
        return cls.deserialize(data)

    def _benchmark(self, cnt: int, method: str = "uint64") -> None:
        """
        Run performance benchmark.

        Parameters
        ----------
        cnt : int
            Number of values to generate
        method : {'uint64', 'uint32', 'double'}
            Method to benchmark
        """
        import time

        start = time.perf_counter()

        if method == "uint64":
            for _ in range(cnt):
                self.random_raw()
        elif method == "uint32":
            for _ in range(cnt):
                self._rng.kiss()
        elif method == "double":
            for _ in range(cnt):
                self.random_raw() / (2**64)
        else:
            raise ValueError(f"Invalid method: {method}")

        elapsed = time.perf_counter() - start
        print(f"Generated {cnt:,} {method} values in {elapsed:.3f}s")
        print(f"Rate: {cnt/elapsed:,.0f} values/sec")


# ===========================================================================
# High-Level Generator (NumPy-Compatible)
# ===========================================================================

cdef class KissGenerator:
    """
    High-level random number generator using KISS algorithm.

    Provides NumPy-compatible interface for common distributions
    with complete serialization support.
    For advanced distributions, wrap KissBitGenerator in numpy.random.Generator.

    Parameters
    ----------
    bit_generator : {None, int, KissBitGenerator}, optional
        BitGenerator or seed value

    Attributes
    ----------
    bit_generator : KissBitGenerator
        Underlying bit generator

    See Also
    --------
    PyKiss32Random : 32-bit version for smaller datasets
    PyKiss64Random : 64-bit version for larger datasets
    PyKissRandom : Factory function for auto-detecting
    KissSeedSequence : Seed sequence for initialization
    KissBitGenerator : NumPy-compatible bit generator
    KissRandomState inherites from KissGenerator
    default_rng : Convenience function to create generator
    numpy.random.Generator : NumPy's Generator class

    Notes
    -----
    This is the recommended high-level API for most users.
    Provides methods similar to numpy.random.Generator.

    References
    ----------
    .. [1] NumPy Random Generator API
           https://numpy.org/doc/stable/reference/random/generator.html

    Examples
    --------
    >>> gen = KissGenerator(42)
    >>> gen.random(5)
    >>> gen.integers(0, 100, 10)
    >>> gen.normal(0, 1, 1000)
    >>>
    >>> # Context manager
    >>> with KissGenerator(42) as gen:
    ...     data = gen.random(1000)
    >>>
    >>> # Serialization
    >>> import pickle
    >>> restored = pickle.loads(pickle.dumps(gen))
    """

    # Class attribute
    # __module__ = "scikitplot.cexternals._annoy._kissrandom.kissrandom"
    __module__ = "scikitplot.random"

    cdef public object _bit_generator
    cdef bitgen_t _bitgen
    cdef object lock

    def __init__(self, bit_generator=None):
        """Initialize generator."""
        if bit_generator is None:
            self._bit_generator = KissBitGenerator()
        elif isinstance(bit_generator, int):
            self._bit_generator = KissBitGenerator(seed=bit_generator)
        elif isinstance(bit_generator, KissBitGenerator):
            self._bit_generator = bit_generator
        else:
            raise TypeError(
                f"bit_generator must be None, int, or KissBitGenerator; got {type(bit_generator)}"
            )

        # Create lock
        self.lock = threading.RLock()

    def __repr__(self) -> str:
        return f"{self} at 0x{id(self):X}"

    def __str__(self) -> str:
        # return f'{self.__class__.__name__}({self.bit_generator.__class__.__name__})'
        return f"{self.__class__.__name__}"

    def __enter__(self):
        """Enter context manager (acquire lock)."""
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager (release lock)."""
        self.lock.release()
        return False

    def spawn(self, n_children):
        """
        Create independent child Generators.

        Parameters
        ----------
        n_children : int
            Number of children to spawn

        Returns
        -------
        list of KissGenerator
            Independent generators

        Raises
        ------
        ValueError
            If n_children < 1

        Notes
        -----
        Uses seed sequence spawning to ensure statistical independence.
        Each child generator has its own independent random stream.

        Useful for parallel computation where each worker needs an
        independent RNG.

        See Also
        --------
        KissSeedSequence.spawn : Low-level seed sequence spawning

        Examples
        --------
        >>> gen = KissGenerator()
        >>> children = gen.spawn(4)
        >>> print(len(children))
        3

        Use in parallel computation

        >>> from concurrent.futures import ThreadPoolExecutor
        >>> def worker(gen):
        ...     return gen.random(100).mean()
        >>>
        >>> with ThreadPoolExecutor() as executor:
        ...     results = list(executor.map(worker, children))
        """
        child_bgs = self.bit_generator.spawn(n_children)
        return [KissGenerator(bg) for bg in child_bgs]

    # ===================================================================
    # BitGenerator Management
    # ===================================================================

    @property
    def bit_generator(self):
        """
        Gets the bit generator instance used by the generator

        Returns
        -------
        bit_generator : KissBitGenerator
            The bit generator instance used by the generator

        Examples
        --------
        >>> gen = KissGenerator(42)
        >>> bg = gen.get_bit_generator()
        >>> print(type(bg))
        <class 'KissBitGenerator'>
        """
        return self._bit_generator

    @bit_generator.setter
    def bit_generator(self, bit_generator):
        """
        Set new bit generator.

        Parameters
        ----------
        bit_generator : KissBitGenerator
            New bit generator

        Raises
        ------
        TypeError
            If not KissBitGenerator

        Examples
        --------
        >>> gen = KissGenerator(42)
        >>> new_bg = KissBitGenerator(123)
        >>> gen.set_bit_generator(new_bg)
        """
        if not isinstance(bit_generator, KissBitGenerator):
            raise TypeError("bit_generator must be KissBitGenerator")
        self._bit_generator = bit_generator

    def get_bit_generator(self):
        """
        Get underlying bit generator.

        Returns
        -------
        KissBitGenerator
            The bit generator

        Examples
        --------
        >>> gen = KissGenerator(42)
        >>> bg = gen.get_bit_generator()
        >>> print(type(bg))
        <class 'KissBitGenerator'>
        """
        return self._bit_generator

    def set_bit_generator(self, bit_generator):
        """
        Set new bit generator.

        Parameters
        ----------
        bit_generator : KissBitGenerator
            New bit generator

        Raises
        ------
        TypeError
            If not KissBitGenerator

        Examples
        --------
        >>> gen = KissGenerator(42)
        >>> new_bg = KissBitGenerator(123)
        >>> gen.set_bit_generator(new_bg)
        """
        if not isinstance(bit_generator, KissBitGenerator):
            raise TypeError("bit_generator must be KissBitGenerator")
        self._bit_generator = bit_generator

    # ===================================================================
    # Serialization Support
    # ===================================================================

    def __getstate__(self):
        """Return state for pickling."""
        return self.get_state()

    def __setstate__(self, state):
        """Restore state from pickle."""
        _validate_state_dict(state, {"bit_generator_state"})
        self.set_state(state)

    def __reduce__(self):
        """Custom pickle protocol."""
        return (
            self.__class__,
            (),  # No args
            self.__getstate__(),
        )

    def __reduce_ex__(self, protocol):
        """Protocol-specific pickle."""
        return self.__reduce__()

    def get_state(self):
        """
        Get state dictionary.

        Returns
        -------
        dict
            Complete state

        Examples
        --------
        >>> gen = KissGenerator(42)
        >>> state = gen.get_state()
        >>> print("bit_generator_state" in state)
        True
        """
        return {
            "bit_generator": f"{self.bit_generator.__class__.__name__}",
            "bit_generator_state": self.bit_generator.get_state(),
            "__version__": SERIALIZATION_VERSION,
        }

    def set_state(self, state):
        """
        Set state from dictionary.

        Parameters
        ----------
        state : dict
            State from get_state()

        Examples
        --------
        >>> gen1 = KissGenerator(42)
        >>> state = gen1.get_state()
        >>> gen2 = KissGenerator(0)
        >>> gen2.set_state(state)
        """
        _validate_state_dict(state, {"bit_generator_state"})
        self.bit_generator.set_state(state["bit_generator_state"])

    def get_params(self, deep=True):
        """
        Get parameters (sklearn-style).

        Parameters
        ----------
        deep : bool, default=True
            If True, include nested params

        Returns
        -------
        dict
            Parameters

        Examples
        --------
        >>> gen = KissGenerator(42)
        >>> params = gen.get_params()
        """
        params = {}
        if deep:
            params["bit_generator"] = self.bit_generator.get_params(deep=True)
        return params

    def set_params(self, **params):
        """
        Set parameters (sklearn-style).

        Parameters
        ----------
        **params : dict
            Parameters to set

        Returns
        -------
        self

        Examples
        --------
        >>> gen = KissGenerator(42)
        >>> gen.set_params(bit_generator={"seed": 123})
        """
        if "bit_generator" in params:
            self.bit_generator.set_params(**params["bit_generator"])
        return self

    def serialize(self):
        """
        Serialize to JSON-compatible dict.

        Returns
        -------
        dict
            JSON-serializable state

        Examples
        --------
        >>> import json
        >>> gen = KissGenerator(42)
        >>> data = gen.serialize()
        >>> json_str = json.dumps(data)
        """
        state = self.get_state()
        state["__class__"] = "KissGenerator"
        state["__module__"] = self.__module__
        return _add_metadata(state)

    @classmethod
    def deserialize(cls, data):
        """
        Deserialize a KissGenerator from a JSON-compatible dict.

        Parameters
        ----------
        data : dict
            Serialized generator state.

        Returns
        -------
        KissGenerator
            Fully restored generator instance.

        Raises
        ------
        TypeError
            If types are invalid.
        KeyError
            If required fields are missing.
        ValueError
            If state is incompatible or unsupported.

        Examples
        --------
        >>> import json
        >>> gen = KissGenerator(42)
        >>> json_str = json.dumps(gen.serialize())
        >>> data = json.loads(json_str)
        >>> restored = KissGenerator.deserialize(data)
        """
        # ---------- Top-level validation ----------
        if not isinstance(data, dict):
            raise TypeError("Serialized state must be a dict")

        _validate_state_dict(data, {"bit_generator", "bit_generator_state"})

        if data["bit_generator"] != "KissBitGenerator":
            raise ValueError(
                f"Unsupported bit generator '{data['bit_generator']}'"
            )

        bitgen_state = data["bit_generator_state"]
        if not isinstance(bitgen_state, dict):
            raise TypeError("'bit_generator_state' must be a dict")

        _validate_state_dict(
            bitgen_state,
            {"seed_sequence", "seed_sequence_state"}
        )

        # ---------- SeedSequence validation ----------
        if bitgen_state["seed_sequence"] != "KissSeedSequence":
            raise ValueError(
                f"Unsupported seed sequence '{bitgen_state['seed_sequence']}'"
            )

        seed_seq_state = bitgen_state["seed_sequence_state"]
        if not isinstance(seed_seq_state, dict):
            raise TypeError("'seed_sequence_state' must be a dict")

        # ---------- Reconstruct SeedSequence ----------
        # Reconstruct seed_seq if present
        seed_seq = None
        if "seed_seq" in seed_seq_state:
            seed_seq = KissSeedSequence.deserialize(seed_seq_state["seed_seq"])

        # ---------- Reconstruct BitGenerator ----------
        bit_generator = KissBitGenerator(seed_seq)

        # ---------- Reconstruct Generator ----------
        instance = cls(bit_generator)

        return instance

    def to_dict(self):
        """Alias for serialize()."""
        return self.serialize()

    @classmethod
    def from_dict(cls, data):
        """Alias for deserialize()."""
        return cls.deserialize(data)

    # ===================================================================
    # Distribution Methods
    # ===================================================================

    def choice(self, a, size=None, replace=True, p=None):
        """
        Random sample from array.

        Parameters
        ----------
        a : int or array_like
            If int, random sample from np.arange(a)
        size : int or tuple, optional
            Output shape
        replace : bool, default=True
            Whether to sample with replacement
        p : array_like, optional
            Probabilities for each element

        Returns
        -------
        scalar or ndarray
            Random samples

        Examples
        --------
        >>> gen = KissGenerator(42)
        >>> gen.choice(10, size=5)
        array([...])
        """
        if isinstance(a, int):
            a = np.arange(a)

        a = np.asarray(a)

        if p is not None:
            p = np.asarray(p) / np.sum(p)
            cum_p = np.cumsum(p)

            if size is None:
                r = self.random()
                return a[np.searchsorted(cum_p, r)]

            if isinstance(size, int):
                size = (size,)

            r_values = self.random(int(np.prod(size)))
            indices = np.searchsorted(cum_p, r_values)
            return a[indices].reshape(size)

        if size is None:
            with self.lock:
                raw = self.bit_generator.random_raw()
                return a[(raw % len(a))]

        if isinstance(size, int):
            size = (size,)

        n = int(np.prod(size))

        if not replace and n > len(a):
            raise ValueError("Cannot sample without replacement: size > population")

        if replace:
            indices = self.integers(0, len(a), size=n)
            return a[indices].reshape(size)
        else:
            indices = np.arange(len(a))
            self.shuffle(indices)
            return a[indices[:n]].reshape(size)

    def integers(self, low, high=None, size=None, dtype=np.int64, endpoint=False):
        """
        Random integers in [low, high) or [low, high].

        Parameters
        ----------
        low : int
            Lowest value (inclusive)
        high : int, optional
            Highest value (exclusive unless endpoint=True)
        size : int or tuple, optional
            Output shape
        dtype : dtype, default=np.int64
            Data type
        endpoint : bool, default=False
            If True, sample from [low, high] instead of [low, high)

        Returns
        -------
        int or ndarray
            Random integers

        Examples
        --------
        >>> gen = KissGenerator(42)
        >>> gen.integers(0, 10, size=5)
        array([...])
        """
        if high is None:
            low, high = 0, low

        if endpoint:
            high += 1

        if low >= high:
            raise ValueError(f"low >= high: {low} >= {high}")

        range_size = high - low

        if size is None:
            with self.lock:
                raw = self.bit_generator.random_raw()
                return low + (raw % range_size)

        if isinstance(size, int):
            size = (size,)

        n_total = int(np.prod(size))
        result = np.empty(n_total, dtype=dtype)

        with self.lock:
            for i in range(n_total):
                raw = self.bit_generator.random_raw()
                result[i] = low + (raw % range_size)

        return result.reshape(size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        """
        Normal distribution (Box-Muller transform).

        Parameters
        ----------
        loc : float, default=0.0
            Mean
        scale : float, default=1.0
            Standard deviation
        size : int or tuple, optional
            Output shape

        Returns
        -------
        float or ndarray
            Normal samples

        Examples
        --------
        >>> gen = KissGenerator(42)
        >>> gen.normal(0, 1, size=1000)
        array([...])
        """
        if size is None:
            u1, u2 = self.random(), self.random()
            z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
            return loc + scale * z

        if isinstance(size, int):
            size = (size,)

        n_total = int(np.prod(size))
        n_pairs = (n_total + 1) // 2

        samples = []
        for _ in range(n_pairs):
            u1, u2 = self.random(), self.random()
            z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
            z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
            samples.extend([z1, z2])

        result = loc + scale * np.array(samples[:n_total])
        return result.reshape(size)

    def permutation(self, x, axis=0):
        """
        Randomly permute sequence or return permuted range.

        Parameters
        ----------
        x : int or array-like
            If int, permute np.arange(x).
            If array-like, permute copy of array.
        axis : int, default=0
            Axis to permute along

        Returns
        -------
        ndarray
            Permuted sequence

        Notes
        -----
        Unlike shuffle(), this returns a permuted copy without
        modifying the input.

        See Also
        --------
        shuffle : In-place shuffle (destructive)
        choice : Random sampling

        Examples
        --------
        >>> gen = KissGenerator()
        >>> gen.permutation(10)  # Permuted [0, 1, ..., 9]
        array([3, 7, 1, 9, 2, 5, 8, 0, 6, 4])
        >>>
        >>> # Permute array (returns copy)
        >>> arr = np.array([1, 2, 3, 4])
        >>> gen.permutation(arr)
        array([3, 1, 4, 2])
        >>> arr  # Original unchanged
        array([1, 2, 3, 4])
        """
        if isinstance(x, int):
            arr = np.arange(x)
        else:
            arr = np.array(x, copy=True)

        self.shuffle(arr, axis=axis)
        return arr

    def random(self, size=None, dtype=np.float64, out=None):
        """
        Random floats in [0, 1).

        Parameters
        ----------
        size : int or tuple, optional
            Output shape
        dtype : dtype, default=np.float64
            Output data type
        out : ndarray, optional
            Output array

        Returns
        -------
        float or ndarray
            Random values

        Examples
        --------
        >>> gen = KissGenerator(42)
        >>> gen.random(5)
        array([...])
        """
        if size is None:
            return self.bit_generator.random_raw() / (2**64 - 1)

        raw = self.bit_generator.random_raw(size)

        if out is None:
            result = raw.astype(dtype) / (2**64 - 1)
        else:
            np.divide(raw, 2**64 - 1, out=out, casting="unsafe")
            result = out

        return result

    def shuffle(self, x):
        """
        Shuffle array in-place (Fisher-Yates algorithm).

        Parameters
        ----------
        x : ndarray
            Array to shuffle

        Examples
        --------
        >>> gen = KissGenerator(42)
        >>> arr = np.arange(10)
        >>> gen.shuffle(arr)
        >>> print(arr)  # shuffled
        """
        n = len(x)
        with self.lock:
            for i in range(n - 1, 0, -1):
                raw = self.bit_generator.random_raw()
                j = (raw % (i + 1))
                x[i], x[j] = x[j], x[i]

    def uniform(self, low=0.0, high=1.0, size=None):
        """
        Uniform distribution in [low, high).

        Parameters
        ----------
        low : float, default=0.0
            Lower bound (inclusive)
        high : float, default=1.0
            Upper bound (exclusive)
        size : int or tuple, optional
            Output shape

        Returns
        -------
        float or ndarray
            Uniform samples

        Examples
        --------
        >>> gen = KissGenerator(42)
        >>> gen.uniform(0, 10, size=5)
        array([...])
        """
        return low + (high - low) * self.random(size)


# ===========================================================================
# NumPy RandomState Compatible with Complete Serialization
# ===========================================================================

cdef class KissRandomState(KissGenerator):
    """
    NumPy RandomState-compatible interface with complete serialization.

    KissRandomState inherites from KissGenerator.
    Provides legacy numpy.random.RandomState API for backward compatibility.

    Parameters
    ----------
    seed : int or None, optional
        Random seed

    Attributes
    ----------
    bit_generator : KissBitGenerator
        Underlying bit generator (via get_bit_generator())

    See Also
    --------
    PyKiss32Random : 32-bit version for smaller datasets
    PyKiss64Random : 64-bit version for larger datasets
    PyKissRandom : Factory function for auto-detecting
    KissSeedSequence : Seed sequence for initialization
    KissBitGenerator : NumPy-compatible bit generator
    KissGenerator : High-level generator using this BitGenerator
    default_rng : Convenience function to create generator

    Notes
    -----
    This class provides the same interface as numpy.random.RandomState
    for users who prefer the legacy API. All serialization methods
    are inherited from KissGenerator.

    Examples
    --------
    >>> rs = KissRandomState(42)
    >>> rs.rand(5)  # Like np.random.rand
    >>> rs.randint(0, 100, size=10)  # Like np.random.randint
    >>> rs.randn(5)  # Like np.random.randn
    >>>
    >>> # Set/get bit generator
    >>> bg = rs.get_bit_generator()
    >>> rs.set_bit_generator(KissBitGenerator(123))
    >>>
    >>> # Serialization
    >>> import pickle
    >>> restored = pickle.loads(pickle.dumps(rs))
    """

    # Class attribute
    # __module__ = "scikitplot.cexternals._annoy._kissrandom.kissrandom"
    __module__ = "scikitplot.random"

    cdef object lock

    def __init__(self, seed=None):
        """Initialize RandomState."""
        super().__init__(bit_generator=seed)

        # Create lock
        self.lock = threading.RLock()

    def __repr__(self) -> str:
        return f"{self} at 0x{id(self):X}"

    def __str__(self) -> str:
        # return f'{self.__class__.__name__}({self.bit_generator.__class__.__name__})'
        return f"{self.__class__.__name__}"

    def __enter__(self):
        """Enter context manager (acquire lock)."""
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager (release lock)."""
        self.lock.release()
        return False

    # ===================================================================
    # RandomState-Specific Methods
    # ===================================================================

    def seed(self, seed=None):
        """
        Seed the generator.

        Parameters
        ----------
        seed : int or None, optional
            Random seed

        Examples
        --------
        >>> rs = KissRandomState(42)
        >>> rs.seed(123)  # Re-seed
        """
        if seed is None:
            self.bit_generator = KissBitGenerator()
        else:
            self.bit_generator = KissBitGenerator(seed=seed)

    def rand(self, *args):
        """
        Random values in [0, 1) with given shape.

        Parameters
        ----------
        *args : ints
            Shape dimensions

        Returns
        -------
        ndarray
            Random values

        Examples
        --------
        >>> rs = KissRandomState(42)
        >>> rs.rand(3, 4)  # 3x4 array
        """
        if not args:
            return self.random()
        return self.random(size=args)

    def randn(self, *args):
        """
        Standard normal distribution with given shape.

        Parameters
        ----------
        *args : ints
            Shape dimensions

        Returns
        -------
        ndarray
            Normal samples

        Examples
        --------
        >>> rs = KissRandomState(42)
        >>> rs.randn(3, 4)  # 3x4 array
        """
        if not args:
            return self.normal()
        return self.normal(size=args)

    def randint(self, low, high=None, size=None, dtype=np.int64):
        """
        Random integers in [low, high).

        Parameters
        ----------
        low : int
            Lowest value (inclusive)
        high : int, optional
            Highest value (exclusive)
        size : int or tuple, optional
            Output shape
        dtype : dtype, default=np.int64
            Data type

        Returns
        -------
        int or ndarray
            Random integers

        Examples
        --------
        >>> rs = KissRandomState(42)
        >>> rs.randint(0, 10, size=5)
        """
        return self.integers(low, high, size=size, dtype=dtype)

    def random_sample(self, size=None):
        """Random floats in [0, 1)."""
        return self.random(size=size)

    # (Not Overwrite) Aliases (NumPy RandomState compatibility)
    # random = random_sample
    # sample = random_sample
    # ranf = random_sample


# ===========================================================================
# Convenience Functions
# ===========================================================================

def default_rng(seed=None, bit_width=None):
    """
    Create default KISS random number generator.

    This is the recommended way to create an RNG, matching
    numpy.random.default_rng() signature.

    Parameters
    ----------
    seed : {None, int, KissSeedSequence, KissBitGenerator, KissGenerator, KissRandomState}, optional
        Random seed or generator

    Returns
    -------
    KissGenerator
        Initialized bit generator ready for use
    bit_width : int, default=64
        Generator bit width

    Raises
    ------
    ValueError
        If bit_width not in {32, 64}

    See Also
    --------
    PyKiss32Random : 32-bit version for smaller datasets
    PyKiss64Random : 64-bit version for larger datasets
    PyKissRandom : Factory function for auto-detecting
    KissSeedSequence : Seed sequence for initialization
    KissBitGenerator : NumPy-compatible bit generator
    KissGenerator : High-level generator using this BitGenerator
    KissRandomState inherites from KissGenerator

    Notes
    -----
    This is the recommended entry point for most users.
    Compatible with numpy.random.default_rng() interface.

    Examples
    --------
    >>> from scikitplot.cexternals._annoy._kissrandom.kissrandom import default_rng
    >>>
    >>> rng = default_rng(42)
    >>> rng.random(5)
    >>> rng.integers(0, 100, 10)
    >>> rng.normal(0, 1, 1000)
    >>>
    >>> # Context manager
    >>> with default_rng(42) as rng:
    ...     data = rng.random(1000)
    >>>
    >>> # Serialization
    >>> import pickle
    >>> restored = pickle.loads(pickle.dumps(rng))
    """
    if seed is None or isinstance(seed, int):
        bg = KissBitGenerator(seed=seed)
        return KissGenerator(bg)
    elif isinstance(seed, KissBitGenerator):
        return KissGenerator(seed)
    elif isinstance(seed, KissGenerator):
        return seed
    elif isinstance(seed, KissRandomState):
        return seed
    elif isinstance(seed, (KissSeedSequence, NumpySeedSequence)):
        bg = KissBitGenerator(seed=seed)
        return KissGenerator(bg)
    elif hasattr(seed, "bit_generator"):
        # NumPy Generator-like object
        return KissGenerator(seed.bit_generator)
    else:
        raise TypeError(
            f"seed must be None, int, SeedSequence, BitGenerator, or Generator; got {type(seed)}"
        )

@contextmanager
def kiss_context(seed=None, bit_width=None):
    """
    Context manager for temporary RNG.

    Parameters
    ----------
    seed : int or None
        Random seed
    bit_width : int, default=64
        Generator bit width

    Yields
    ------
    KissGenerator
        Bit generator instance

    See Also
    --------
    PyKiss32Random : 32-bit version for smaller datasets
    PyKiss64Random : 64-bit version for larger datasets
    PyKissRandom : Factory function for auto-detecting
    KissSeedSequence : Seed sequence for initialization
    KissBitGenerator : NumPy-compatible bit generator
    KissGenerator : High-level generator using this BitGenerator
    KissRandomState inherites from KissGenerator
    default_rng : Convenience function to create generator

    Notes
    -----
    Automatically acquires and releases lock for thread safety.

    Examples
    --------
    >>> # from contextlib import closing
    >>> from scikitplot.cexternals._annoy._kissrandom.kissrandom import kiss_context
    >>>
    >>> with kiss_context(42) as rng:
    ...     data = rng.random(1000)
    """
    rng = default_rng(seed)
    with rng:
        yield rng


# ===========================================================================
# (Optionally) Initialize instance and export method
# ===========================================================================

_rng = KissGenerator()

choice = _rng.choice
integers = _rng.integers
normal = _rng.normal
permutation = _rng.permutation
random = _rng.random
shuffle = _rng.shuffle
uniform = _rng.uniform

# ===========================================================================
# (Optionally) Module initialization check
# ===========================================================================

def _verify_numpy_compatibility():
    """
    Verify that NumPy Generator integration works.

    Returns
    -------
    bool
        True if compatible, False otherwise

    Notes
    -----
    This is called automatically on module import to ensure
    everything is working correctly.
    """
    try:
        bg = KissBitGenerator(seed=42)

        # Check required attributes
        required_attrs = ["lock", "capsule", "seed_seq", "state"]
        for attr in required_attrs:
            if not hasattr(bg, attr):
                print(f"Warning: KissBitGenerator missing '{attr}' attribute")
                return False

        # Try creating NumPy Generator
        try:
            gen = np.random.Generator(bg)
            # Try generating some values
            gen.random(5)
            return True
        except AttributeError as e:
            print(f"Warning: NumPy Generator integration failed: {e}")
            return False
    except Exception as e:
        print(f"Warning: Compatibility check failed: {e}")
        return False

# Run compatibility check on import (optional)
# _verify_numpy_compatibility()


# if __name__ == "__main__":
# _ = PyKiss32Random()
# _ = PyKiss64Random()
# _ = PyKissRandom()
# KissSeedSequence()
# KissBitGenerator()
# KissGenerator()
# # KissRandomState()
# default_rng()
# kiss_context()


# ===========================================================================
# KissRandomState - NumPy RandomState Compatible
# ===========================================================================

# def set_bit_generator(bitgen: BitGenerator) -> None: ...
#
# def get_bit_generator() -> BitGenerator: ...
#
#
# class RandomState:
#     _bit_generator: BitGenerator
#     def __init__(self, seed: _ArrayLikeInt_co | BitGenerator | None = ...) -> None: ...
#     def __repr__(self) -> str: ...
#     def __str__(self) -> str: ...
#     def __getstate__(self) -> dict[str, Any]: ...
#     def __setstate__(self, state: dict[str, Any]) -> None: ...
#     def __reduce__(self) -> tuple[Callable[[BitGenerator], RandomState], tuple[BitGenerator], dict[str, Any]]: ...  # noqa: E501
#     def seed(self, seed: _ArrayLikeFloat_co | None = None) -> None: ...
#     @overload
#     def get_state(self, legacy: Literal[False] = False) -> dict[str, Any]: ...
#     @overload
#     def get_state(
#         self, legacy: Literal[True] = True
#     ) -> dict[str, Any] | tuple[str, NDArray[uint32], int, int, float]: ...
#     def set_state(
#         self, state: dict[str, Any] | tuple[str, NDArray[uint32], int, int, float]
#     ) -> None: ...
#     @overload
#     def random_sample(self, size: None = None) -> float: ...  # type: ignore[misc]
#     @overload
#     def random_sample(self, size: _ShapeLike) -> NDArray[float64]: ...
#     @overload
#     def random(self, size: None = None) -> float: ...  # type: ignore[misc]
#     @overload
#     def random(self, size: _ShapeLike) -> NDArray[float64]: ...
#     @overload
#     def beta(self, a: float, b: float, size: None = None) -> float: ...  # type: ignore[misc]
#     @overload
#     def beta(
#         self,
#         a: _ArrayLikeFloat_co,
#         b: _ArrayLikeFloat_co,
#         size: _ShapeLike | None = None
#     ) -> NDArray[float64]: ...
#
# _rand: RandomState
#
# beta = _rand.beta
# bytes = _rand.bytes
# choice = _rand.choice
# normal = _rand.normal
# rand = _rand.rand
# randint = _rand.randint
# randn = _rand.randn
# random = _rand.random
# random_integers = _rand.random_integers
# random_sample = _rand.random_sample
# rayleigh = _rand.rayleigh
# seed = _rand.seed
# set_state = _rand.set_state
# shuffle = _rand.shuffle
# uniform = _rand.uniform
# # Two legacy that are trivial wrappers around random_sample
# sample = _rand.random_sample
# ranf = _rand.random_sample
