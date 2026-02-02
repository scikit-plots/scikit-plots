# Cython Tutorial: Understanding kissrandom

This tutorial explains how the C++ KISS random number generator is wrapped in Cython, aimed at beginners learning Cython.

## Table of Contents

1. [What is Cython?](#what-is-cython)
2. [File Structure Overview](#file-structure-overview)
3. [Understanding Each File](#understanding-each-file)
4. [Modern Cython Practices](#modern-cython-practices)
5. [Building and Testing](#building-and-testing)
6. [Common Pitfalls](#common-pitfalls)

## What is Cython?

Cython is a programming language that:
- Extends Python with C/C++ types and functions
- Compiles Python code to C/C++ for speed
- Allows seamless integration of Python and C/C++ code
- Maintains Python's ease of use while achieving C-level performance

**Key concept**: Cython code looks like Python but can be compiled to C/C++ for massive speedups.

## File Structure Overview

```
kissrandom/
├── kissrandom.h        # C++ header (original implementation)
├── kissrandom.pyi      # Python type hints (for IDEs/type checkers)
├── kissrandom.pxd      # Cython declarations (C-level interface)
├── kissrandom.pxi      # Cython shared code (avoid in modern code)
├── kissrandom.pyx      # Cython implementation (Python wrapper)
├── setup.py            # Build configuration
├── pyproject.toml      # Modern Python packaging
├── test_kissrandom.py  # Test suite
└── README.md           # Documentation
```

### File Purpose Summary

| File | Purpose | When to Edit |
|------|---------|--------------|
| `.h` | C++ implementation | Never (upstream dependency) |
| `.pyi` | Type hints for Python | When public API changes |
| `.pxd` | C/C++ declarations | When wrapping new C++ code |
| `.pxi` | Shared implementation | Avoid (deprecated pattern) |
| `.pyx` | Python-facing wrapper | Main implementation file |

## Understanding Each File

### 1. kissrandom.h (C++ Header)

**Purpose**: The original C++ implementation we're wrapping.

**Key Points**:
- Contains `Kiss32Random` and `Kiss64Random` structs
- Pure C++ code, not modified by Cython
- Located in C++ namespace `Annoy`

**What you need to know**:
```cpp
namespace Annoy {
  struct Kiss32Random {
    uint32_t x, y, z, c;  // State variables
    uint32_t kiss();       // Main RNG method
    int flip();            // Random 0/1
    size_t index(size_t n); // Random index
  };
}
```

### 2. kissrandom.pyi (Python Type Hints)

**Purpose**: Tell Python type checkers (mypy, pyright) about our API.

**Key Points**:
- Pure Python syntax (PEP 484 type hints)
- NOT imported at runtime
- Only for static type checking and IDE autocomplete

**Example**:
```python
class Kiss32Random:
    def kiss(self) -> int: ...  # Return type hint
    def flip(self) -> int: ...
    def index(self, n: int) -> int: ...
```

**When to edit**: When you change the public Python API.

### 3. kissrandom.pxd (Cython Declarations)

**Purpose**: Declare C/C++ types and functions for Cython to use.

**Key Points**:
- Like a C header file, but in Cython syntax
- Contains declarations, not implementations
- Other `.pyx` files can `cimport` from here

**Structure**:
```cython
# Import C++ types
from libc.stdint cimport uint32_t, uint64_t

# Declare C++ classes from the header
cdef extern from "kissrandom.h" namespace "Annoy" nogil:
    cdef cppclass Kiss32Random:
        uint32_t kiss()  # Declare methods
        int flip()
        # ... more methods
```

**Key Cython Keywords**:
- `cdef extern from`: Import from C/C++ header
- `namespace`: Specify C++ namespace
- `nogil`: Can be called without Python GIL (thread-safe)
- `cppclass`: Declare C++ class

**When to edit**: When you need to access new C++ functionality.

### 4. kissrandom.pxi (Shared Implementation)

**Purpose**: Share code between multiple `.pyx` files.

**⚠️ WARNING**: `.pxi` files are **discouraged** in modern Cython!

**Why it exists here**: For educational purposes only.

**Why avoid `.pxi`**:
- Uses textual inclusion (like C `#include`)
- Can cause duplicate symbols
- Makes dependencies hard to track
- Better alternatives exist (see below)

**Modern Alternatives**:

1. **For shared C-level code**: Use `.pxd` + `.pyx`
   ```cython
   # helpers.pxd
   cdef inline int helper_func(int x) nogil

   # helpers.pyx
   cdef inline int helper_func(int x) nogil:
       return x * 2

   # main.pyx
   from helpers cimport helper_func  # Clean import!
   ```

2. **For shared Python code**: Use regular `.py` files
   ```python
   # utils.py
   def python_helper():
       return 42

   # main.pyx
   from utils import python_helper  # Normal Python import
   ```

**When to edit**: Never in production code. Delete if possible.

### 5. kissrandom.pyx (Main Implementation)

**Purpose**: The actual Cython wrapper that exposes C++ to Python.

**This is the most important file!**

**Structure**:
```cython
# Compiler directives (at top)
# cython: language_level=3
# cython: boundscheck=False

# Imports
from libc.stdint cimport uint32_t
cimport kissrandom as kr  # Import from .pxd

# Module version
__version__ = "1.0.0"

# Cython class wrapping C++ class
cdef class Kiss32Random:
    cdef kr.Kiss32Random* _rng  # Pointer to C++ object

    def __cinit__(self, seed):
        self._rng = new kr.Kiss32Random(seed)  # Allocate

    def __dealloc__(self):
        del self._rng  # Free memory

    cpdef uint32_t kiss(self):  # Expose method to Python
        return self._rng.kiss()
```

**Key Cython Concepts**:

#### Compiler Directives
```cython
# cython: language_level=3        # Use Python 3 syntax
# cython: boundscheck=False       # Disable array bounds checking (faster)
# cython: wraparound=False        # Disable negative indexing (faster)
# cython: cdivision=True          # Use C division (faster, no zero-check)
```

#### Function Types

| Declaration | C-level | Python-level | When to use |
|-------------|---------|--------------|-------------|
| `def` | ❌ | ✅ | Normal Python functions |
| `cdef` | ✅ | ❌ | C-only, not callable from Python |
| `cpdef` | ✅ | ✅ | Both! Cython calls C version, Python calls wrapper |

**Example**:
```cython
def py_only(x):           # Callable from Python only
    return x * 2

cdef int c_only(int x):   # Callable from Cython only (fast!)
    return x * 2

cpdef int both(int x):    # Callable from both (best of both worlds)
    return x * 2
```

#### Memory Management

**Manual memory management** (like C++):
```cython
cdef class MyClass:
    cdef SomeCppClass* _obj

    def __cinit__(self):  # Called first
        self._obj = new SomeCppClass()  # Allocate

    def __dealloc__(self):  # Called last
        del self._obj  # Free (must not fail!)
```

**Important**:
- `__cinit__`: Guaranteed to be called exactly once
- `__dealloc__`: Must not raise exceptions
- Always check `if self._obj is not NULL` before deleting

#### Type Declarations

```cython
# C types
cdef int x = 5                    # C int
cdef uint32_t y = 42              # Explicit unsigned 32-bit
cdef double z = 3.14              # C double

# Python types
cdef object py_obj = "hello"      # Python object
cdef list py_list = [1, 2, 3]     # Python list

# Function parameters can be typed too
cpdef int add(int a, int b):      # Type checking at C level
    return a + b
```

### 6. setup.py (Build Configuration)

**Purpose**: Configure how Cython compiles to C++ and builds the extension.

**Key sections**:

```python
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="kissrandom",           # Module name
        sources=["kissrandom.pyx"],  # Source files
        language="c++",              # Use C++ compiler
        extra_compile_args=["-std=c++11", "-O3"],  # Compiler flags
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={        # Cython options
            "language_level": "3",
            "boundscheck": False,
        }
    )
)
```

**When to edit**: When changing build settings or adding new extensions.

## Modern Cython Practices

### 1. Use DEF Instead of IF

**❌ Old way (avoid)**:
```cython
IF UNAME_SYSNAME == "Linux":
    # Linux-specific code
ELSE:
    # Other platforms
```

**✅ Modern way**:
```cython
DEF IS_LINUX = True  # Set at compile time

cdef int platform_specific():
    if IS_LINUX:  # Optimized away at compile time
        return linux_code()
    else:
        return other_code()
```

**Why**: `DEF` creates true compile-time constants. `IF`/`ELSE` is for conditional compilation (rarely needed).

### 2. Use cpdef for Dual-Access Methods

**✅ Good**:
```cython
cpdef int public_method(self, int x):
    # Callable from both Python and Cython
    return self._helper(x)

cdef int _helper(self, int x):
    # Internal, C-only helper (faster)
    return x * 2
```

**Benefits**:
- Python code calls `public_method()` normally
- Cython code calls fast C version
- Single implementation

**Bonus: Python Properties in Cython**:
```cython
cdef class MyRNG:
    cdef uint32_t _current_seed  # Private storage

    @property
    def seed(self) -> int:
        """Get current seed."""
        return self._current_seed

    @seed.setter
    def seed(self, uint32_t value):
        """Set seed and reinitialize."""
        self._current_seed = value
        self._reinitialize()

# Usage from Python:
rng = MyRNG()
print(rng.seed)  # Getter
rng.seed = 42    # Setter
```

### 3. Use nogil for Thread Safety

**When to use `nogil`**:
```cython
cpdef uint32_t kiss(self) nogil:
    # No Python objects touched = safe without GIL
    return self._rng.kiss()
```

**Benefits**:
- Allows true parallel execution
- Much faster in multithreaded code
- C++ code can run freely

**When NOT to use `nogil`**:
- Touching Python objects
- Raising Python exceptions
- Calling Python functions

### 4. Type All Variables for Performance

**❌ Slow (Python types)**:
```cython
def slow_sum(n):
    total = 0
    for i in range(n):
        total += i
    return total
```

**✅ Fast (C types)**:
```cython
cpdef long fast_sum(long n):
    cdef long total = 0
    cdef long i
    for i in range(n):
        total += i
    return total
```

**Performance**: Fast version is ~100x faster!

## Building and Testing

### Building the Extension

```bash
# Development build (in-place)
python setup.py build_ext --inplace

# Install in development mode
pip install -e .

# Clean build
python setup.py clean --all
python setup.py build_ext --inplace
```

### Running Tests

```bash
# All tests
pytest test_kissrandom.py -v

# Specific test
pytest test_kissrandom.py::test_kiss32_basic

# With coverage
pytest test_kissrandom.py --cov=kissrandom --cov-report=html
```

### Debugging Compilation Issues

1. **Check Cython is installed**:
   ```bash
   python -c "import Cython; print(Cython.__version__)"
   ```

2. **Generate C++ code to inspect**:
   ```bash
   cython -a kissrandom.pyx  # Creates kissrandom.html
   # Open in browser to see Python/C interaction
   ```

3. **Enable debug symbols**:
   ```bash
   DEBUG=1 python setup.py build_ext --inplace
   ```

4. **Check compiler output**:
   ```bash
   python setup.py build_ext --inplace --verbose
   ```

## Common Pitfalls

### 1. Memory Leaks

**❌ BAD**:
```cython
cdef class LeakyClass:
    cdef SomeCppClass* _obj

    def __cinit__(self):
        self._obj = new SomeCppClass()

    # Missing __dealloc__!
    # Memory is never freed!
```

**✅ GOOD**:
```cython
cdef class SafeClass:
    cdef SomeCppClass* _obj

    def __cinit__(self):
        self._obj = new SomeCppClass()

    def __dealloc__(self):
        if self._obj is not NULL:
            del self._obj
```

### 2. Forgetting nogil Restrictions

**❌ BAD**:
```cython
cpdef int bad_function(self) nogil:
    py_list = [1, 2, 3]  # ERROR! Python object in nogil
    return len(py_list)
```

**✅ GOOD**:
```cython
cpdef int good_function(self) nogil:
    cdef int result = self._c_helper()
    return result

cdef int _c_helper(self) nogil:
    # No Python objects here
    return 42
```

### 3. Type Mismatches

**❌ BAD**:
```cython
cpdef int process(self):
    cdef uint32_t value = self._rng.kiss()
    return value - 1  # Might overflow! uint32 can be very large
```

**✅ GOOD**:
```cython
cpdef long process(self):
    cdef uint32_t value = self._rng.kiss()
    return <long>value - 1  # Explicit cast
```

### 4. Modifying .pxd Without Rebuilding

**Problem**: Changes to `.pxd` files require full rebuild.

**Solution**:
```bash
python setup.py clean --all
python setup.py build_ext --inplace
```

Or use `force=True` in setup.py:
```python
ext_modules = cythonize(extensions, force=True)
```

## Learning Resources

1. **Official Cython Docs**: https://cython.readthedocs.io/
2. **Cython Book**: "Cython: A Guide for Python Programmers" by Kurt W. Smith
3. **Modern Patterns**: https://github.com/cython/cython/issues/4310
4. **CPython Extension Guide**: https://docs.python.org/3/extending/extending.html

## Next Steps

1. **Experiment**: Modify `kissrandom.pyx` and rebuild
2. **Add features**: Try adding a new method to the wrapper
3. **Optimize**: Use `cython -a` to identify bottlenecks
4. **Profile**: Use `cProfile` to measure performance
5. **Read others' code**: Study popular Cython projects (NumPy, SciPy, pandas)

## Conclusion

Key takeaways:
- `.pxd` = declarations (like C headers)
- `.pyx` = implementation (like C source files)
- `.pyi` = type hints (for Python tooling)
- Use `cpdef` for public API, `cdef` for internal helpers
- Always manage memory explicitly (`__cinit__`/`__dealloc__`)
- Use `nogil` when possible for performance
- Avoid `.pxi` files in modern code

Happy Cython coding! 🐍⚡
