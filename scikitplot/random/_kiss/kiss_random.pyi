# scikitplot/random/_kiss/kiss_random.pyi
#
# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

"""
Type stubs for KISS random number generator module.

This file provides comprehensive type hints for all public classes and
functions in the kiss_random module, enabling static type checking with
mypy, pyright, and IDE autocompletion.

Design Principles
-----------------
- Follow PEP 484 (Type Hints) and PEP 561 (Distributing Type Information)
- Mirror the exact public API exposed in kiss_random.pyx
- Use precise types (no 'Any' unless unavoidable)
- Document all public classes and methods with docstrings
- Keep in sync with kiss_random.pyx implementation

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

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from threading import Lock, RLock
from typing import Any, Final, Literal, overload

import numpy as np
from numpy.random import SeedSequence as NumpySeedSequence
from numpy._typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _ShapeLike,
)

# ===========================================================================
# Module Metadata
# ===========================================================================

__version__: Final[str]

__all__: list[str]

# ===========================================================================
# Seed Sequence
# ===========================================================================

class KissSeedSequence:
    """
    Seed sequence compatible with numpy.random.SeedSequence.

    NumPy-compatible seed sequence for KISS RNG with complete serialization support.
    """

    __module__: str

    entropy: int | None
    spawn_key: tuple[int, ...]
    pool_size: int
    n_children_spawned: int

    def __init__(
        self,
        entropy: int | Sequence[int] | None = None,
        *,
        spawn_key: Sequence[int] = (),
        pool_size: int = 4,
        n_children_spawned: int = 0
    ) -> None: ...

    def generate_state(
        self,
        n_words: int,
        dtype: DTypeLike = np.uint32
    ) -> NDArray[np.uint32] | NDArray[np.uint64]: ...

    def spawn(self, n_children: int) -> list[KissSeedSequence]: ...

    # Serialization methods
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    def __reduce__(self) -> tuple[type[KissSeedSequence], tuple[()], dict[str, Any]]: ...
    def __reduce_ex__(self, protocol: int) -> tuple[type[KissSeedSequence], tuple[()], dict[str, Any]]: ...

    def get_state(self) -> dict[str, Any]: ...
    def set_state(self, state: dict[str, Any]) -> None: ...

    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def set_params(self, **params: Any) -> KissSeedSequence: ...

    def serialize(self) -> dict[str, Any]: ...
    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> KissSeedSequence: ...

    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KissSeedSequence: ...

    @property
    def state(self) -> dict[str, Any]: ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

# ===========================================================================
# Low-Level RNG Classes
# ===========================================================================

class Kiss32Random:
    """
    32-bit KISS random number generator with complete serialization support.

    Legacy 32-bit KISS RNG (use KissBitGenerator instead).
    Period: approximately 2^121
    """

    __module__: str

    lock: Lock

    def __init__(self, seed: int | None = None) -> None: ...

    @property
    def seed(self) -> int: ...

    @seed.setter
    def seed(self, value: int) -> None: ...

    @staticmethod
    def get_default_seed() -> int: ...

    @staticmethod
    def normalize_seed(seed: int) -> int: ...

    def reset(self, seed: int) -> None: ...
    def reset_default(self) -> None: ...
    def set_seed(self, seed: int) -> None: ...

    def kiss(self) -> int: ...
    def flip(self) -> int: ...
    def index(self, n: int) -> int: ...

    # Context manager
    def __enter__(self) -> Kiss32Random: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool: ...

    # Serialization methods
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    def __reduce__(self) -> tuple[type[Kiss32Random], tuple[int], None]: ...
    def __reduce_ex__(self, protocol: int) -> tuple[type[Kiss32Random], tuple[int], None]: ...

    def get_state(self) -> dict[str, Any]: ...
    def set_state(self, state: dict[str, Any]) -> None: ...

    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def set_params(self, **params: Any) -> Kiss32Random: ...

    def serialize(self) -> dict[str, Any]: ...
    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> Kiss32Random: ...

    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Kiss32Random: ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...


class Kiss64Random:
    """
    64-bit KISS random number generator with complete serialization support.

    Legacy 64-bit KISS RNG (use KissBitGenerator instead).
    Period: approximately 2^250
    """

    __module__: str

    lock: Lock

    def __init__(self, seed: int | None = None) -> None: ...

    @property
    def seed(self) -> int: ...

    @seed.setter
    def seed(self, value: int) -> None: ...

    @staticmethod
    def get_default_seed() -> int: ...

    @staticmethod
    def normalize_seed(seed: int) -> int: ...

    def reset(self, seed: int) -> None: ...
    def reset_default(self) -> None: ...
    def set_seed(self, seed: int) -> None: ...

    def kiss(self) -> int: ...
    def flip(self) -> int: ...
    def index(self, n: int) -> int: ...

    # Context manager
    def __enter__(self) -> Kiss64Random: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool: ...

    # Serialization methods
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    def __reduce__(self) -> tuple[type[Kiss64Random], tuple[int], None]: ...
    def __reduce_ex__(self, protocol: int) -> tuple[type[Kiss64Random], tuple[int], None]: ...

    def get_state(self) -> dict[str, Any]: ...
    def set_state(self, state: dict[str, Any]) -> None: ...

    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def set_params(self, **params: Any) -> Kiss64Random: ...

    def serialize(self) -> dict[str, Any]: ...
    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> Kiss64Random: ...

    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Kiss64Random: ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...


def KissRandom(
    seed: int | None = None,
    bit_width: int | Literal['auto'] | None = None
) -> Kiss32Random | Kiss64Random: ...

# ===========================================================================
# Bit Generator
# ===========================================================================

class KissBitGenerator:
    """
    NumPy-compatible BitGenerator using KISS algorithm with complete serialization.
    """

    __module__: str

    lock: Lock
    seed_seq: KissSeedSequence

    def __init__(
        self,
        seed: int | KissSeedSequence | NumpySeedSequence | None = None,
        *,
        bit_width: int = 64
    ) -> None: ...

    # Random generation
    @overload
    def random_raw(
        self, size: None = None, output: Literal[True] = True
    ) -> int: ...

    @overload
    def random_raw(
        self, size: int | tuple[int, ...], output: Literal[True] = True
    ) -> NDArray[np.uint64]: ...

    @overload
    def random_raw(
        self, size: int | tuple[int, ...] | None = None, *, output: Literal[False]
    ) -> None: ...

    def spawn(self, n_children: int) -> list[KissBitGenerator]: ...

    @property
    def capsule(self) -> Any: ...

    @property
    def state(self) -> dict[str, Any]: ...

    @state.setter
    def state(self, value: dict[str, Any]) -> None: ...

    # Context manager
    def __enter__(self) -> KissBitGenerator: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool: ...

    # Serialization methods
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    def __reduce__(self) -> tuple[type[KissBitGenerator], tuple[KissSeedSequence], None]: ...
    def __reduce_ex__(self, protocol: int) -> tuple[type[KissBitGenerator], tuple[KissSeedSequence], None]: ...

    def get_state(self) -> dict[str, Any]: ...
    def set_state(self, state: dict[str, Any]) -> None: ...

    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def set_params(self, **params: Any) -> KissBitGenerator: ...

    def serialize(self) -> dict[str, Any]: ...
    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> KissBitGenerator: ...

    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KissBitGenerator: ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

# ===========================================================================
# Generator
# ===========================================================================

class KissGenerator:
    """
    High-level random number generator using KISS algorithm.

    NumPy Generator-like high-level interface with complete serialization support.
    """

    __module__: str

    _bit_generator: KissBitGenerator
    lock: RLock

    def __init__(
        self, bit_generator: int | KissBitGenerator | None = None
    ) -> None: ...

    # BitGenerator access
    @property
    def bit_generator(self) -> KissBitGenerator: ...

    @bit_generator.setter
    def bit_generator(self, value: KissBitGenerator) -> None: ...

    def get_bit_generator(self) -> KissBitGenerator: ...
    def set_bit_generator(self, bit_generator: KissBitGenerator) -> None: ...

    # Random floats
    @overload
    def random(
        self,
        size: None = None,
        dtype: DTypeLike = np.float64,
        out: None = None
    ) -> float: ...

    @overload
    def random(
        self,
        size: int | tuple[int, ...],
        dtype: DTypeLike = np.float64,
        out: NDArray[np.floating[Any]] | None = None
    ) -> NDArray[np.floating[Any]]: ...

    # Random integers
    @overload
    def integers(
        self,
        low: int,
        high: int | None = None,
        size: None = None,
        dtype: DTypeLike = np.int64,
        endpoint: bool = False
    ) -> int: ...

    @overload
    def integers(
        self,
        low: int,
        high: int | None = None,
        size: int | tuple[int, ...] = ...,
        dtype: DTypeLike = np.int64,
        endpoint: bool = False
    ) -> NDArray[np.signedinteger[Any]]: ...

    # Normal distribution
    @overload
    def normal(
        self, loc: float = 0.0, scale: float = 1.0, size: None = None
    ) -> float: ...

    @overload
    def normal(
        self,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        size: int | tuple[int, ...] | None = ...
    ) -> NDArray[np.float64]: ...

    # Uniform distribution
    @overload
    def uniform(
        self, low: float = 0.0, high: float = 1.0, size: None = None
    ) -> float: ...

    @overload
    def uniform(
        self,
        low: ArrayLike = 0.0,
        high: ArrayLike = 1.0,
        size: int | tuple[int, ...] | None = ...
    ) -> NDArray[np.float64]: ...

    # Choice
    @overload
    def choice(
        self,
        a: int,
        size: None = None,
        replace: bool = True,
        p: ArrayLike | None = None
    ) -> int: ...

    @overload
    def choice(
        self,
        a: ArrayLike,
        size: None = None,
        replace: bool = True,
        p: ArrayLike | None = None
    ) -> Any: ...

    @overload
    def choice(
        self,
        a: int | ArrayLike,
        size: int | tuple[int, ...],
        replace: bool = True,
        p: ArrayLike | None = None
    ) -> NDArray[Any]: ...

    # Array operations
    def shuffle(self, x: NDArray[Any]) -> None: ...

    # Spawning
    def spawn(self, n_children: int) -> list[KissGenerator]: ...

    # Context manager
    def __enter__(self) -> KissGenerator: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool: ...

    # Serialization methods
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    def __reduce__(self) -> tuple[type[KissGenerator], tuple[()], dict[str, Any]]: ...
    def __reduce_ex__(self, protocol: int) -> tuple[type[KissGenerator], tuple[()], dict[str, Any]]: ...

    def get_state(self) -> dict[str, Any]: ...
    def set_state(self, state: dict[str, Any]) -> None: ...

    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def set_params(self, **params: Any) -> KissGenerator: ...

    def serialize(self) -> dict[str, Any]: ...
    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> KissGenerator: ...

    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KissGenerator: ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

# ===========================================================================
# RandomState (Legacy)
# ===========================================================================

class KissRandomState(KissGenerator):
    """
    NumPy RandomState-compatible interface.

    Provides legacy numpy.random.RandomState API for backward compatibility.
    Inherits all methods from KissGenerator.
    """

    __module__: str

    def __init__(self, seed: int | None = None) -> None: ...

    # RandomState-specific methods
    def seed(self, seed: int | None = None) -> None: ...

    def rand(self, *args: int) -> float | NDArray[np.float64]: ...
    def randn(self, *args: int) -> float | NDArray[np.float64]: ...

    def randint(
        self,
        low: int,
        high: int | None = None,
        size: int | tuple[int, ...] | None = None,
        dtype: DTypeLike = np.int64
    ) -> int | NDArray[np.signedinteger[Any]]: ...

    def random_sample(
        self, size: int | tuple[int, ...] | None = None
    ) -> float | NDArray[np.float64]: ...

    # Aliases
    def random(
        self, size: int | tuple[int, ...] | None = None
    ) -> float | NDArray[np.float64]: ...

    def sample(
        self, size: int | tuple[int, ...] | None = None
    ) -> float | NDArray[np.float64]: ...

    def ranf(
        self, size: int | tuple[int, ...] | None = None
    ) -> float | NDArray[np.float64]: ...

    # State management (RandomState-style)
    def get_state(self) -> dict[str, Any]: ...
    def set_state(self, state: dict[str, Any]) -> None: ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

# ===========================================================================
# Convenience Functions
# ===========================================================================

def default_rng(
    seed: int | KissSeedSequence | KissBitGenerator | KissGenerator | KissRandomState | None = None
) -> KissGenerator: ...

def kiss_context(
    seed: int | KissSeedSequence | None = None
) -> AbstractContextManager[KissGenerator]: ...
