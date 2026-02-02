# scikitplot/cexternals/_annoy/_kissrandom/kissrandom.pyi
#
# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

"""
Type stubs for kissrandom module.

This file provides type hints for Python type checkers (mypy, pyright, etc.)
and IDE autocompletion. It describes the public Python API exposed by the
Cython extension module.

Design Principles
-----------------
- Follow PEP 484 (Type Hints) and PEP 561 (Distributing Type Information)
- Mirror the exact public API exposed in kissrandom.pyx
- Use precise types (no 'Any' unless unavoidable)
- Document all public classes and methods with docstrings
- Keep in sync with kissrandom.pyx implementation

Best Practices
--------------
- Use `int` for seed types (Python's arbitrary precision int)
- Use `Final` for constants that should not be reassigned
- Use `@overload` if methods have multiple signatures
- Keep stub file minimal - detailed docs go in .pyx docstrings

Notes for Maintainers
---------------------
- This file is NOT imported at runtime; it's only for static analysis
- Changes to public API in .pyx MUST be reflected here
- Test with `mypy --strict` and `pyright` before committing
- For Cython-specific types (uint32_t, etc.), use Python equivalents (int)

References
----------
- PEP 484: https://www.python.org/dev/peps/pep-0484/
- PEP 561: https://www.python.org/dev/peps/pep-0561/
- Typing documentation: https://docs.python.org/3/library/typing.html
"""

from typing import Final

__version__: Final[str]

class Kiss32Random:
    """
    32-bit KISS random number generator.

    A fast, high-quality pseudo-random number generator combining
    Linear Congruential Generator (LCG), Xorshift, and Multiply-With-Carry
    (MWC) algorithms.

    Attributes
    ----------
    default_seed : int
        Default seed value (123456789)

    Notes
    -----
    - Not cryptographically secure - do NOT use for security purposes
    - Period: approximately 2^121
    - Suitable for up to ~2^24 data points
    - For larger datasets, use Kiss64Random

    Examples
    --------
    >>> rng = Kiss32Random(42)
    >>> rng.kiss()  # Generate random uint32
    >>> rng.flip()  # Generate random 0 or 1
    >>> rng.index(10)  # Generate random int in [0, 9]
    """

    default_seed: Final[int]

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
        Getting the seed returns the normalized seed value that was last set.
        Setting the seed reinitializes the RNG with the new seed value.

        Examples
        --------
        >>> rng = Kiss32Random(42)
        >>> rng.seed  # Get current seed
        42
        >>> rng.seed = 123  # Set new seed (reinitializes RNG)
        >>> rng.seed
        123
        """
        ...

    @seed.setter
    def seed(self, value: int) -> None:
        """Set new seed and reinitialize RNG."""
        ...

    def __init__(self, seed: int = 123456789) -> None:
        """
        Initialize Kiss32Random with given seed.

        Parameters
        ----------
        seed : int, optional
            Initial seed value. If 0, uses default_seed (123456789).
            Default: 123456789

        Notes
        -----
        - Seed value 0 is automatically normalized to default_seed
        - All internal state is fully reset when seed is set
        """
        ...

    @staticmethod
    def get_default_seed() -> int:
        """
        Get the default seed value.

        Returns
        -------
        int
            Default seed value (123456789)
        """
        ...

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
        Maps seed==0 to default_seed to avoid degenerate RNG states.
        """
        ...

    def reset(self, seed: int) -> None:
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
        """
        ...

    def reset_default(self) -> None:
        """
        Reset RNG to default seed.

        Equivalent to reset(default_seed).
        """
        ...

    def set_seed(self, seed: int) -> None:
        """
        Set new seed (alias for reset).

        Parameters
        ----------
        seed : int
            New seed value. If 0, uses default_seed.
        """
        ...

    def kiss(self) -> int:
        """
        Generate next random 32-bit unsigned integer.

        Returns
        -------
        int
            Random value in [0, 2^32-1]

        Notes
        -----
        This is the core RNG method. Other methods (flip, index) are
        built on top of kiss().
        """
        ...

    def flip(self) -> int:
        """
        Generate random binary value (0 or 1).

        Returns
        -------
        int
            Either 0 or 1 with equal probability
        """
        ...

    def index(self, n: int) -> int:
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
        - Handles n==0 gracefully (returns 0)
        - Uses modulo bias for simplicity (suitable for non-crypto use)
        """
        ...


class Kiss64Random:
    """
    64-bit KISS random number generator.

    A fast, high-quality pseudo-random number generator for large datasets.
    Use this when you have more than ~2^24 data points.

    Attributes
    ----------
    default_seed : int
        Default seed value (1234567890987654321)

    Notes
    -----
    - Not cryptographically secure - do NOT use for security purposes
    - Period: approximately 2^250
    - Recommended for datasets larger than ~16 million points
    - Slightly slower than Kiss32Random but much longer period

    Examples
    --------
    >>> rng = Kiss64Random(12345678901234567890)
    >>> rng.kiss()  # Generate random uint64
    >>> rng.flip()  # Generate random 0 or 1
    >>> rng.index(1000000)  # Generate random int in [0, 999999]
    """

    default_seed: Final[int]

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
        Getting the seed returns the normalized seed value that was last set.
        Setting the seed reinitializes the RNG with the new seed value.

        Examples
        --------
        >>> rng = Kiss64Random(12345678901234567890)
        >>> rng.seed  # Get current seed
        12345678901234567890
        >>> rng.seed = 999  # Set new seed
        >>> rng.seed
        999
        """
        ...

    @seed.setter
    def seed(self, value: int) -> None:
        """Set new seed and reinitialize RNG."""
        ...

    def __init__(self, seed: int = 1234567890987654321) -> None:
        """
        Initialize Kiss64Random with given seed.

        Parameters
        ----------
        seed : int, optional
            Initial seed value. If 0, uses default_seed.
            Default: 1234567890987654321

        Notes
        -----
        - Seed value 0 is automatically normalized to default_seed
        - All internal state is fully reset when seed is set
        """
        ...

    @staticmethod
    def get_default_seed() -> int:
        """
        Get the default seed value.

        Returns
        -------
        int
            Default seed value (1234567890987654321)
        """
        ...

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
        Maps seed==0 to default_seed to avoid degenerate RNG states.
        """
        ...

    def reset(self, seed: int) -> None:
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
        """
        ...

    def reset_default(self) -> None:
        """
        Reset RNG to default seed.

        Equivalent to reset(default_seed).
        """
        ...

    def set_seed(self, seed: int) -> None:
        """
        Set new seed (alias for reset).

        Parameters
        ----------
        seed : int
            New seed value. If 0, uses default_seed.
        """
        ...

    def kiss(self) -> int:
        """
        Generate next random 64-bit unsigned integer.

        Returns
        -------
        int
            Random value in [0, 2^64-1]

        Notes
        -----
        This is the core RNG method. Other methods (flip, index) are
        built on top of kiss().
        """
        ...

    def flip(self) -> int:
        """
        Generate random binary value (0 or 1).

        Returns
        -------
        int
            Either 0 or 1 with equal probability
        """
        ...

    def index(self, n: int) -> int:
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
        - Handles n==0 gracefully (returns 0)
        - Uses modulo bias for simplicity (suitable for non-crypto use)
        """
        ...
