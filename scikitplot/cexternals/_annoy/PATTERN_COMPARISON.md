# Pattern Comparison: kissrandom → mman

## File Structure (Identical Pattern)

```
kissrandom/                          mman/
├── kissrandom.h                     ├── mman.h
│   (C++ header - RNG logic)         │   (C header - mmap for Windows)
│                                    │
├── kissrandom.pxd                   ├── mman.pxd
│   (Cython declarations)            │   (Cython declarations)
│   - cdef extern from               │   - cdef extern from
│   - cppclass Kiss32Random          │   - void* mmap() nogil
│   - DEF constants                  │   - DEF constants
│                                    │
├── kissrandom.pyx                   ├── mman.pyx
│   (Python wrapper)                 │   (Python wrapper)
│   - cdef class Kiss32Random        │   - cdef class MemoryMap
│   - @property seed                 │   - @property addr, size
│   - cpdef methods                  │   - cpdef methods
│   - __cinit__/__dealloc__          │   - __cinit__/__dealloc__
│                                    │
├── kissrandom.pyi                   ├── mman.pyi
│   (Type stubs)                     │   (Type stubs)
│   - class Kiss32Random             │   - class MemoryMap
│   - Type hints for all methods     │   - Type hints for all methods
│                                    │
└── test_kissrandom.py               └── test_mman.py
    (50+ tests)                          (40+ tests)
```

---

## Code Pattern Comparison

### Pattern 1: Constants (DEF not IF)

**kissrandom.pxd:**
```cython
DEF VERSION_MAJOR = 1
DEF VERSION_MINOR = 0
```

**mman.pxd:**
```cython
DEF PROT_READ = 1
DEF PROT_WRITE = 2
DEF MAP_SHARED = 1
```

---

### Pattern 2: External C Declarations

**kissrandom.pxd:**
```cython
cdef extern from "kissrandom.h" namespace "Annoy" nogil:
    cdef cppclass Kiss32Random:
        uint32_t kiss()
        int flip()
        size_t index(size_t n)
```

**mman.pxd:**
```cython
cdef extern from "mman.h" nogil:
    void* mmap(void* addr, size_t len, int prot,
               int flags, int fildes, off_t off) nogil
    int munmap(void* addr, size_t len) nogil
    int mprotect(void* addr, size_t len, int prot) nogil
```

---

### Pattern 3: Memory Management

**kissrandom.pyx:**
```cython
cdef class Kiss32Random:
    cdef kr.Kiss32Random* _rng
    cdef uint32_t _current_seed

    def __cinit__(self, seed: int | None = None):
        cseed = validate_and_normalize(seed)
        self._current_seed = cseed
        self._rng = new kr.Kiss32Random(cseed)
        if self._rng is NULL:
            raise MemoryError("Allocation failed")

    def __dealloc__(self):
        if self._rng is not NULL:
            del self._rng
            self._rng = NULL
```

**mman.pyx:**
```cython
cdef class MemoryMap:
    cdef void* _addr
    cdef size_t _size
    cdef bint _is_valid

    def __cinit__(self):
        self._addr = NULL
        self._size = 0
        self._is_valid = False

    def __dealloc__(self):
        if self._is_valid and self._addr is not NULL:
            mm.munmap(self._addr, self._size)
            self._addr = NULL
```

---

### Pattern 4: Properties (Get/Set)

**kissrandom.pyx:**
```cython
@property
def seed(self) -> int:
    """Get current seed."""
    return self._current_seed

@seed.setter
def seed(self, value: int) -> None:
    """Set new seed and reinitialize."""
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"Invalid seed: {value}")
    self._current_seed = normalize(value)
    self._rng.reset(self._current_seed)
```

**mman.pyx:**
```cython
@property
def addr(self) -> int:
    """Get memory address."""
    if not self._is_valid:
        raise ValueError("Mapping is closed")
    return <size_t>self._addr

@property
def size(self) -> int:
    """Get mapping size."""
    if not self._is_valid:
        raise ValueError("Mapping is closed")
    return self._size
```

---

### Pattern 5: Input Validation

**kissrandom.pyx:**
```cython
def __cinit__(self, seed: int | None = None):
    if seed is None:
        cseed = default_seed
    elif seed < 0 or seed > 0xFFFFFFFF:
        raise ValueError(f"seed must be in [0, 2^32-1], got {seed}")
    else:
        cseed = <uint32_t>seed
```

**mman.pyx:**
```cython
def read(self, size: int, offset: int = 0) -> bytes:
    if not self._is_valid:
        raise ValueError("Mapping is closed")
    if size < 0:
        raise ValueError(f"Size must be non-negative, got {size}")
    if offset < 0:
        raise ValueError(f"Offset must be non-negative, got {offset}")
    if offset + size > self._size:
        raise ValueError("Read beyond mapping bounds")
```

---

### Pattern 6: cpdef Methods (Dual Access)

**kissrandom.pyx:**
```cython
cpdef uint32_t kiss(self) nogil:
    """Generate random number (callable from Python and Cython)."""
    return self._rng.kiss()

cpdef void reset(self, uint32_t seed):
    """Reset RNG (callable from Python and Cython)."""
    self._current_seed = normalize_seed(seed)
    self._rng.reset(self._current_seed)
```

**mman.pyx:**
```cython
# Note: mman doesn't use cpdef as extensively since most operations
# involve Python objects (bytes), but the pattern is available:

def close(self) -> None:  # Could be cpdef void close(self)
    """Close mapping (automatic cleanup)."""
    if not self._is_valid:
        return

    cdef int result
    with nogil:
        result = mm.munmap(self._addr, self._size)

    if result != 0:
        raise MMapError(f"Failed to unmap: errno={errno}")

    self._addr = NULL
    self._is_valid = False
```

---

### Pattern 7: Context Managers

**kissrandom:**
```python
# Not typically used as context manager, but could be:
rng = Kiss32Random(42)
rng.seed = 123  # Property-based management
```

**mman.pyx:**
```cython
def __enter__(self) -> "MemoryMap":
    """Context manager entry."""
    return self

def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    """Context manager exit."""
    self.close()

# Usage:
with MemoryMap.create_anonymous(4096) as m:
    m.write(b"data")
# Automatically closed
```

---

### Pattern 8: String Representation

**kissrandom.pyx:**
```cython
def __repr__(self) -> str:
    return f"Kiss32Random(seed={self._current_seed})"

def __str__(self) -> str:
    return f"Kiss32Random(seed={self._current_seed})"
```

**mman.pyx:**
```cython
def __repr__(self) -> str:
    if self._is_valid:
        return f"MemoryMap(addr=0x{<size_t>self._addr:x}, size={self._size})"
    else:
        return "MemoryMap(closed)"

def __str__(self) -> str:
    return self.__repr__()
```

---

### Pattern 9: Type Hints (.pyi)

**kissrandom.pyi:**
```python
class Kiss32Random:
    default_seed: Final[int]

    @property
    def seed(self) -> int: ...

    @seed.setter
    def seed(self, value: int) -> None: ...

    def __init__(self, seed: int | None = None) -> None: ...
    def kiss(self) -> int: ...
    def flip(self) -> int: ...
    def index(self, n: int) -> int: ...
```

**mman.pyi:**
```python
class MemoryMap:
    @property
    def addr(self) -> int: ...

    @property
    def size(self) -> int: ...

    @staticmethod
    def create_anonymous(size: int, prot: int = ..., flags: int = ...) -> MemoryMap: ...

    def read(self, size: int, offset: int = 0) -> bytes: ...
    def write(self, data: bytes, offset: int = 0) -> int: ...
    def close(self) -> None: ...
```

---

### Pattern 10: Documentation Style (NumPy)

**Both use identical NumPy-style docstrings:**

```python
def method(self, param: int) -> int:
    """
    Short description.

    Longer description with more details about behavior,
    edge cases, and usage patterns.

    Parameters
    ----------
    param : int
        Description of parameter

    Returns
    -------
    int
        Description of return value

    Raises
    ------
    ValueError
        When and why this is raised

    Notes
    -----
    - Additional information
    - Performance characteristics
    - Thread safety notes

    Examples
    --------
    >>> obj = MyClass()
    >>> result = obj.method(42)
    >>> print(result)
    42
    """
```

---

## Key Takeaways

### Patterns Successfully Applied from kissrandom to mman ✅

1. **File hierarchy**: .h → .pxd → .pyx → .pyi
2. **DEF constants**: Compile-time configuration
3. **Memory safety**: __cinit__/__dealloc__ with NULL checks
4. **Input validation**: Explicit checks with clear errors
5. **Properties**: Clean getter/setter interface
6. **cpdef methods**: Dual Python/Cython access
7. **nogil**: Thread-safe C-level operations
8. **Documentation**: NumPy-style docstrings
9. **Type hints**: Complete .pyi files
10. **Testing**: Comprehensive test suites

### Differences (Domain-Specific)

| Aspect | kissrandom | mman |
|--------|-----------|------|
| **Language** | C++ (namespace) | C (no namespace) |
| **Main Purpose** | RNG state machine | Resource wrapper |
| **State** | Internal RNG state | OS-level resource handle |
| **Primary Method** | `kiss()` generates numbers | `read()`/`write()` data access |
| **Context Manager** | Not typically needed | Essential for cleanup |
| **Platform** | Cross-platform | Windows-only |

---

## Visual: Same Pattern, Different Domain

```
┌─────────────────────────────────────────────────────────────┐
│                    COMMON PATTERN                           │
│                                                             │
│  C/C++ Header (.h)  ──┐                                     │
│                       │                                     │
│  Cython Decls (.pxd)  ├──> Modern Cython Best Practices     │
│                       │                                     │
│  Python Wrap (.pyx)   ├──> - DEF constants                  │
│                       │    - cpdef methods                  │
│  Type Stubs (.pyi)    ├──> - Properties                     │
│                       │    - Memory safety                  │
│  Tests (.py)         ─┘    - Input validation               │
│                            - Documentation                  │
└─────────────────────────────────────────────────────────────┘

    Applied to:              Applied to:
┌──────────────────┐    ┌──────────────────────┐
│   kissrandom     │    │        mman          │
├──────────────────┤    ├──────────────────────┤
│ Random numbers   │    │ Memory mapping       │
│ Kiss32Random     │    │ MemoryMap            │
│ kiss(), flip()   │    │ read(), write()      │
│ seed property    │    │ addr, size props     │
│ reset() methods  │    │ close() method       │
│ Cross-platform   │    │ Cross-platform       │
└──────────────────┘    └──────────────────────┘
```
